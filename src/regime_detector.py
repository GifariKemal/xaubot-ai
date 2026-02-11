"""
Market Regime Detection Module v3
==================================
HMM-based regime detection with feature scaling,
covariance regularization, and ATR fallback.

Fixes from v1:
- StandardScaler prevents feature magnitude bias
- min_covar prevents phantom/dead states
- Multi-seed fitting picks best non-degenerate model
- ATR-based fallback overrides HMM when clearly wrong
- Backward-compatible with v1 model files (triggers retrain)

Fixes from v2 (Phase 0 "Stable Regimes"):
- Diagonal-dominant transmat init (90% stay) prevents alternating-state local minima
- Transition quality scoring penalizes unstable transition matrices
- Min-duration smoothing eliminates 1-bar noise transitions
- Enhanced diagnostics: diag quality, duration stats, transition counts
- Feature flag: HMM_SMOOTHING_ENABLED env var (default on)
"""

import os
import polars as pl
import numpy as np
import pickle
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from loguru import logger

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")
    GaussianHMM = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logger.warning("sklearn not installed for StandardScaler")
    StandardScaler = None


class MarketRegime(Enum):
    """Market regime states."""
    LOW_VOLATILITY = "low_volatility"
    MEDIUM_VOLATILITY = "medium_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: MarketRegime
    confidence: float
    probabilities: Dict[str, float]
    volatility: float
    recommendation: str  # "TRADE", "REDUCE", "SLEEP"


class MarketRegimeDetector:
    """
    HMM-based market regime detector v3.

    Key improvements (v2):
    - Feature scaling (StandardScaler) — prevents magnitude bias
    - Covariance floor (min_covar=1e-2) — no phantom states
    - Multi-seed fitting (5 attempts) — picks best model
    - State validation — penalizes degenerate solutions
    - ATR fallback — overrides HMM when ATR percentile disagrees

    Phase 0 "Stable Regimes" (v3):
    - Diagonal-dominant transmat init (90% stay) — biases EM toward sticky regimes
    - Transition quality scoring — penalizes alternating/unstable matrices
    - Min-duration smoothing — eliminates short noise transitions
    - Enhanced diagnostics — diag quality, duration, transition counts
    """

    def __init__(
        self,
        n_regimes: int = 3,
        lookback_periods: int = 500,
        retrain_frequency: int = 20,
        model_path: Optional[str] = None,
        covariance_type: str = "diag",
        random_state: int = 42,
        min_covar: float = 1e-2,
        n_fit_attempts: int = 5,
        smoothing_min_duration: int = 5,
    ):
        if GaussianHMM is None:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")

        self.n_regimes = n_regimes
        self.lookback_periods = lookback_periods
        self.retrain_frequency = retrain_frequency
        self.model_path = Path(model_path) if model_path else None
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.min_covar = min_covar
        self.n_fit_attempts = n_fit_attempts
        self.smoothing_min_duration = smoothing_min_duration
        self.smoothing_enabled = os.getenv("HMM_SMOOTHING_ENABLED", "1") in ("1", "true", "yes")

        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional["StandardScaler"] = StandardScaler() if StandardScaler else None
        self.fitted = False
        self.last_train_idx = 0
        self.regime_mapping: Dict[int, MarketRegime] = {}
        self._train_metrics: Dict = {}

    def _create_model(self, seed: int) -> GaussianHMM:
        """Create a fresh HMM with diagonal-dominant transmat init."""
        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            min_covar=self.min_covar,
            n_iter=500,
            tol=1e-4,
            random_state=seed,
            verbose=False,
            init_params="smc",   # Don't random-init transmat (we set it)
            params="stmc",       # But DO update transmat during EM
        )
        # 90% stay in same state, uniform off-diagonal
        off_diag = 0.10 / max(1, self.n_regimes - 1)
        transmat_init = np.full((self.n_regimes, self.n_regimes), off_diag)
        np.fill_diagonal(transmat_init, 0.90)
        model.transmat_ = transmat_init
        return model

    def prepare_features(self, df: pl.DataFrame) -> np.ndarray:
        """
        Prepare 8 features for HMM.
        Returns raw (unscaled) features — scaling is done in fit/predict.
        """
        # 1-2: Log returns + short-term volatility
        df_features = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns"),
        ])
        df_features = df_features.with_columns([
            pl.col("log_returns").rolling_std(window_size=20).alias("volatility_20"),
            pl.col("log_returns").rolling_std(window_size=100).alias("volatility_100"),
        ])

        # 3-4: Range and ATR features
        df_features = df_features.with_columns([
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_norm"),
        ])

        if "atr" not in df_features.columns:
            df_features = df_features.with_columns([
                pl.max_horizontal([
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs()
                ]).rolling_mean(window_size=14).alias("atr")
            ])

        df_features = df_features.with_columns([
            (pl.col("range_norm") * pl.col("close") / pl.col("atr")).alias("range_atr_ratio"),
        ])

        # 5: Trend strength (SMA distance / ATR)
        df_features = df_features.with_columns([
            pl.col("close").rolling_mean(window_size=9).alias("sma_9"),
            pl.col("close").rolling_mean(window_size=21).alias("sma_21"),
        ])
        df_features = df_features.with_columns([
            ((pl.col("sma_9") - pl.col("sma_21")).abs() / pl.col("atr")).alias("trend_strength"),
        ])

        # 6: RSI deviation (simple momentum proxy)
        df_features = df_features.with_columns([
            (pl.col("close") - pl.col("close").shift(1)).alias("delta"),
        ])
        df_features = df_features.with_columns([
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).rolling_mean(window_size=14).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).rolling_mean(window_size=14).alias("loss"),
        ])
        df_features = df_features.with_columns([
            (100 - (100 / (1 + pl.col("gain") / pl.col("loss")))).alias("rsi_calc"),
        ])
        df_features = df_features.with_columns([
            ((pl.col("rsi_calc") - 50).abs() / 50).alias("rsi_deviation"),
        ])

        # 7: Autocorrelation proxy
        df_features = df_features.with_columns([
            (pl.col("log_returns") * pl.col("log_returns").shift(1)).rolling_mean(window_size=20).alias("autocorr"),
        ])

        # 8: Volatility regime (ATR zscore)
        df_features = df_features.with_columns([
            pl.col("atr").rolling_mean(window_size=100).alias("atr_mean"),
            pl.col("atr").rolling_std(window_size=100).alias("atr_std"),
        ])
        df_features = df_features.with_columns([
            ((pl.col("atr") - pl.col("atr_mean")) / pl.col("atr_std")).alias("vol_regime"),
        ])

        # Select 8 features and clean
        feature_cols = ["log_returns", "volatility_20", "volatility_100", "range_atr_ratio",
                       "trend_strength", "rsi_deviation", "autocorr", "vol_regime"]

        df_features = df_features.drop_nulls(subset=feature_cols)
        features = df_features.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

        return features

    def fit(self, df: pl.DataFrame) -> "MarketRegimeDetector":
        """
        Fit HMM with feature scaling + multi-seed + validation.

        Process:
        1. Extract 8 raw features
        2. Fit StandardScaler on features
        3. Try N random seeds, pick best non-degenerate model
        4. Validate all states are populated (>3% each)
        5. Map states to regime names by volatility_20 mean
        """
        features_raw = self.prepare_features(df)

        if len(features_raw) < 200:
            logger.warning(f"Insufficient data for HMM training: {len(features_raw)} samples (need 200+)")
            return self

        # Step 1: Fit scaler on raw features
        if self.scaler is not None:
            features = self.scaler.fit_transform(features_raw)
            logger.info(f"Features scaled: {len(features)} samples, 8 features")
        else:
            features = features_raw
            logger.warning("No scaler available, using raw features")

        # Step 2: Multi-seed fitting — pick best non-degenerate model
        best_model = None
        best_score = -np.inf
        best_seed = self.random_state
        best_fracs = None

        for attempt in range(self.n_fit_attempts):
            seed = self.random_state + attempt * 17
            try:
                model = self._create_model(seed)
                model.fit(features)
                score = model.score(features)

                # Check state population
                predictions = model.predict(features)
                state_counts = np.bincount(predictions, minlength=self.n_regimes)
                state_fracs = state_counts / len(predictions)
                min_frac = state_fracs.min()

                # Check covariance health (no default 1000.0 values)
                has_phantom = False
                for s in range(self.n_regimes):
                    if np.any(model.covars_[s] > 100):
                        has_phantom = True
                        break

                # Penalize degenerate models
                effective_score = score
                if min_frac < 0.03:
                    effective_score -= 10000  # Heavy penalty
                if has_phantom:
                    effective_score -= 5000   # Phantom state penalty

                # Transition matrix quality scoring
                diag_min = float(np.min(np.diag(model.transmat_)))
                diag_mean = float(np.mean(np.diag(model.transmat_)))
                if diag_min < 0.50:
                    effective_score -= 3000   # Alternating pattern
                elif diag_min < 0.70:
                    effective_score -= 1000   # Unstable
                effective_score += diag_mean * 100  # Bonus for sticky regimes

                if min_frac < 0.15 and min_frac >= 0.03:
                    effective_score -= 500    # Uneven distribution

                logger.debug(
                    f"  HMM seed {seed}: score={score:.1f}, effective={effective_score:.1f}, "
                    f"fracs=[{', '.join(f'{f:.1%}' for f in state_fracs)}], phantom={has_phantom}, "
                    f"diag_min={diag_min:.3f}, diag_mean={diag_mean:.3f}"
                )

                if effective_score > best_score:
                    best_score = effective_score
                    best_model = model
                    best_seed = seed
                    best_fracs = state_fracs

            except Exception as e:
                logger.debug(f"  HMM seed {seed} failed: {e}")
                continue

        if best_model is None:
            logger.error("All HMM fitting attempts failed")
            return self

        self.model = best_model
        self.fitted = True
        self._map_regimes()

        # Compute stability diagnostics on best model
        transmat = best_model.transmat_
        diag_min = float(np.min(np.diag(transmat)))
        diag_mean = float(np.mean(np.diag(transmat)))

        # Count transitions and avg duration on training predictions
        train_preds = best_model.predict(features)
        if self.smoothing_enabled and self.smoothing_min_duration > 1:
            train_preds_smoothed = self._smooth_predictions(train_preds, self.smoothing_min_duration)
        else:
            train_preds_smoothed = train_preds
        transitions = np.sum(train_preds_smoothed[1:] != train_preds_smoothed[:-1])
        total_bars = len(train_preds_smoothed)
        transitions_per_1000 = (transitions / max(1, total_bars)) * 1000
        avg_duration = total_bars / max(1, transitions + 1)

        # Store metrics
        self._train_metrics = {
            "samples": len(features),
            "n_regimes": self.n_regimes,
            "log_likelihood": float(best_model.score(features)),
            "best_seed": best_seed,
            "state_distribution": {
                self.regime_mapping.get(i, MarketRegime.MEDIUM_VOLATILITY).value: f"{frac:.1%}"
                for i, frac in enumerate(best_fracs)
            },
            "diag_min": diag_min,
            "diag_mean": diag_mean,
            "transition_matrix": transmat.tolist(),
            "avg_duration_bars": round(avg_duration, 1),
            "transitions_per_1000": round(transitions_per_1000, 1),
            "total_transitions": int(transitions),
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_min_duration": self.smoothing_min_duration,
            "version": 3,
        }

        logger.info(f"HMM v3 fitted: {len(features)} samples, score={best_model.score(features):.1f}, seed={best_seed}")
        logger.info(f"  State distribution: {self._train_metrics['state_distribution']}")
        logger.info(f"  Transmat diag: min={diag_min:.3f}, mean={diag_mean:.3f}")
        logger.info(f"  Stability: avg_duration={avg_duration:.1f} bars, transitions={transitions}/{total_bars} ({transitions_per_1000:.1f}/1000)")
        logger.info(f"  Smoothing: enabled={self.smoothing_enabled}, min_duration={self.smoothing_min_duration}")

        # Warn if still degenerate (even best model)
        if best_fracs.min() < 0.03:
            logger.warning(f"  Best model still degenerate: min state fraction = {best_fracs.min():.1%}")

        # Quality warnings against targets
        if diag_min < 0.70:
            logger.warning(f"  Transition matrix below target: diag_min={diag_min:.3f} (target > 0.70)")
        if avg_duration < 10:
            logger.warning(f"  Regime duration below target: {avg_duration:.1f} bars (target > 10)")
        if transitions_per_1000 > 50:
            logger.warning(f"  Too many transitions: {transitions_per_1000:.1f}/1000 (target < 50)")

        # Show transition matrix
        for i, regime in self.regime_mapping.items():
            probs = [f"{p:.3f}" for p in transmat[i]]
            logger.info(f"  {regime.value}: [{', '.join(probs)}]")

        # Auto-save
        if self.model_path:
            self.save()

        return self

    def _map_regimes(self):
        """Map HMM states to regime names based on volatility_20 mean."""
        if not self.fitted:
            return

        # volatility_20 is feature index 1 (even after scaling, ordering preserved)
        means = self.model.means_[:, 1]
        sorted_indices = np.argsort(means)

        regimes = [
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.MEDIUM_VOLATILITY,
            MarketRegime.HIGH_VOLATILITY,
        ]

        if self.n_regimes == 4:
            regimes.append(MarketRegime.CRISIS)

        self.regime_mapping = {
            sorted_indices[i]: regimes[min(i, len(regimes) - 1)]
            for i in range(self.n_regimes)
        }

    def _smooth_predictions(self, regimes: np.ndarray, min_duration: int = 5) -> np.ndarray:
        """
        Replace regime segments shorter than min_duration with preceding regime.

        Multi-pass: converges when no short segments remain. Max 10 passes for safety.
        """
        smoothed = regimes.copy()
        for _ in range(10):
            changed = False
            i = 0
            while i < len(smoothed):
                # Find segment start/end
                seg_start = i
                current = smoothed[i]
                while i < len(smoothed) and smoothed[i] == current:
                    i += 1
                seg_len = i - seg_start

                # Replace short segment with preceding regime
                if seg_len < min_duration and seg_start > 0:
                    prev_regime = smoothed[seg_start - 1]
                    smoothed[seg_start:i] = prev_regime
                    changed = True

            if not changed:
                break

        return smoothed

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict regime for each data point (with optional smoothing)."""
        if not self.fitted:
            logger.warning("Model not fitted, returning with neutral regime")
            return df.with_columns([
                pl.lit(1).alias("regime"),
                pl.lit("medium_volatility").alias("regime_name"),
                pl.lit(1.0).alias("regime_confidence"),
            ])

        features_raw = self.prepare_features(df)

        if len(features_raw) == 0:
            return df

        # Scale with fitted scaler
        if self.scaler is not None:
            features = self.scaler.transform(features_raw)
        else:
            features = features_raw

        regimes = self.model.predict(features)
        proba = self.model.predict_proba(features)

        # Apply min-duration smoothing to eliminate noise transitions
        if self.smoothing_enabled and self.smoothing_min_duration > 1:
            regimes = self._smooth_predictions(regimes, self.smoothing_min_duration)

        regime_names = [
            self.regime_mapping.get(r, MarketRegime.MEDIUM_VOLATILITY).value
            for r in regimes
        ]

        confidences = [proba[i, regimes[i]] for i in range(len(regimes))]

        n_dropped = len(df) - len(regimes)

        regimes_padded = [None] * n_dropped + list(regimes)
        names_padded = [None] * n_dropped + regime_names
        conf_padded = [None] * n_dropped + confidences

        df = df.with_columns([
            pl.Series("regime", regimes_padded),
            pl.Series("regime_name", names_padded),
            pl.Series("regime_confidence", conf_padded),
        ])

        return df

    def get_current_state(self, df: pl.DataFrame) -> RegimeState:
        """Get current regime state with ATR fallback."""
        if not self.fitted:
            return RegimeState(
                regime=MarketRegime.MEDIUM_VOLATILITY,
                confidence=0.5,
                probabilities={r.value: 1/self.n_regimes for r in MarketRegime},
                volatility=0.0,
                recommendation="TRADE",
            )

        df_pred = self.predict(df)
        latest = df_pred.tail(1)

        regime_name = latest["regime_name"].item()
        regime = MarketRegime(regime_name) if regime_name else MarketRegime.MEDIUM_VOLATILITY
        confidence = latest["regime_confidence"].item() or 0.5

        probabilities = {}
        for i in range(self.n_regimes):
            r_name = self.regime_mapping.get(i, MarketRegime.MEDIUM_VOLATILITY).value
            probabilities[r_name] = 1.0 / self.n_regimes

        # Calculate volatility
        if "atr_percent" in df.columns:
            volatility = df["atr_percent"].tail(1).item() or 0.0
        else:
            returns = (df["close"] / df["close"].shift(1) - 1).drop_nulls()
            volatility = returns.tail(20).std() * 100 if len(returns) > 0 else 0.0

        # ATR-based fallback: override HMM when ATR percentile clearly disagrees
        regime = self._atr_fallback(df, regime)

        # Recommendation
        if regime == MarketRegime.LOW_VOLATILITY:
            recommendation = "TRADE"
        elif regime == MarketRegime.MEDIUM_VOLATILITY:
            recommendation = "TRADE"
        elif regime == MarketRegime.HIGH_VOLATILITY:
            recommendation = "REDUCE"
        else:
            recommendation = "SLEEP"

        return RegimeState(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            volatility=volatility,
            recommendation=recommendation,
        )

    def _atr_fallback(self, df: pl.DataFrame, hmm_regime: MarketRegime) -> MarketRegime:
        """
        ATR-based fallback — override HMM when ATR percentile clearly disagrees.

        Uses ATR percentile over last 200 candles:
        - ATR >= P90 -> force HIGH_VOLATILITY
        - ATR >= P75 -> at least MEDIUM_VOLATILITY
        - ATR <= P25 -> at most LOW_VOLATILITY
        """
        try:
            if "atr" not in df.columns:
                return hmm_regime

            atr_series = df["atr"].drop_nulls()
            if len(atr_series) < 50:
                return hmm_regime

            current_atr = atr_series.tail(1).item()
            if current_atr is None or current_atr <= 0:
                return hmm_regime

            # Use last 200 candles for percentile baseline
            window = atr_series.tail(200)
            atr_p25 = window.quantile(0.25)
            atr_p75 = window.quantile(0.75)
            atr_p90 = window.quantile(0.90)

            # Override rules
            if current_atr >= atr_p90 and hmm_regime == MarketRegime.LOW_VOLATILITY:
                logger.debug(f"ATR fallback: LOW->HIGH (ATR={current_atr:.2f} >= P90={atr_p90:.2f})")
                return MarketRegime.HIGH_VOLATILITY

            if current_atr >= atr_p75 and hmm_regime == MarketRegime.LOW_VOLATILITY:
                logger.debug(f"ATR fallback: LOW->MEDIUM (ATR={current_atr:.2f} >= P75={atr_p75:.2f})")
                return MarketRegime.MEDIUM_VOLATILITY

            if current_atr <= atr_p25 and hmm_regime == MarketRegime.HIGH_VOLATILITY:
                logger.debug(f"ATR fallback: HIGH->LOW (ATR={current_atr:.2f} <= P25={atr_p25:.2f})")
                return MarketRegime.LOW_VOLATILITY

            return hmm_regime

        except Exception as e:
            logger.debug(f"ATR fallback error: {e}")
            return hmm_regime

    def should_trade(self, df: pl.DataFrame) -> Tuple[bool, str]:
        """Check if trading is allowed in current regime."""
        state = self.get_current_state(df)

        if state.recommendation == "SLEEP":
            return False, f"Market in {state.regime.value} - sleeping"

        if state.recommendation == "REDUCE":
            return True, f"Market in {state.regime.value} - reduce position size"

        return True, f"Market in {state.regime.value} - normal trading"

    def get_position_multiplier(self, df: pl.DataFrame) -> float:
        """Get position size multiplier based on regime."""
        state = self.get_current_state(df)

        multipliers = {
            MarketRegime.LOW_VOLATILITY: 1.0,
            MarketRegime.MEDIUM_VOLATILITY: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.CRISIS: 0.0,
        }

        return multipliers.get(state.regime, 0.5)

    def get_transition_matrix(self) -> np.ndarray:
        """Get the HMM transition probability matrix."""
        if not self.fitted:
            return np.eye(self.n_regimes)
        return self.model.transmat_

    def save(self, path: Optional[str] = None):
        """Save model + scaler to .pkl file."""
        save_path = Path(path) if path else self.model_path

        if save_path is None:
            logger.warning("No save path provided")
            return

        save_path = save_path.with_suffix(".pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "n_regimes": self.n_regimes,
            "lookback_periods": self.lookback_periods,
            "regime_mapping": self.regime_mapping,
            "train_metrics": self._train_metrics,
            "fitted": self.fitted,
            "smoothing_min_duration": self.smoothing_min_duration,
            "version": 3,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"HMM model v3 saved to {save_path}")

    def load(self, path: Optional[str] = None) -> "MarketRegimeDetector":
        """Load model + scaler from .pkl file. Backward-compatible with v1/v2."""
        load_path = Path(path) if path else self.model_path

        if load_path is None:
            logger.warning("No load path provided")
            return self

        load_path = load_path.with_suffix(".pkl")

        if not load_path.exists():
            logger.warning(f"Model file not found: {load_path}")
            return self

        try:
            with open(load_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data.get("model")
            self.n_regimes = model_data.get("n_regimes", 3)
            self.lookback_periods = model_data.get("lookback_periods", 500)
            self.regime_mapping = model_data.get("regime_mapping", {})
            self._train_metrics = model_data.get("train_metrics", {})
            self.fitted = model_data.get("fitted", self.model is not None)

            # Load smoothing config (v3+)
            self.smoothing_min_duration = model_data.get("smoothing_min_duration", 5)

            # Load scaler (v2+)
            version = model_data.get("version", 1)
            if "scaler" in model_data and model_data["scaler"] is not None:
                self.scaler = model_data["scaler"]
            else:
                logger.warning("Loaded v1 model (no scaler). Retrain recommended for v3 features.")
                self.scaler = None

            if version < 3:
                logger.warning(f"Loaded v{version} model — missing v3 features (diagonal-dominant init, smoothing). Retrain recommended.")

            logger.info(f"HMM model v{version} loaded from {load_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        return self


class FlashCrashDetector:
    """Detector for flash crash / extreme volatility events."""

    def __init__(
        self,
        threshold_percent: float = 1.0,
        window_minutes: int = 1,
    ):
        self.threshold_percent = threshold_percent
        self.window_minutes = window_minutes

    def detect(self, df: pl.DataFrame) -> Tuple[bool, float]:
        """Detect flash crash condition."""
        if len(df) < 2:
            return False, 0.0

        latest_close = df["close"].tail(1).item()
        first_close = df["close"].head(1).item()

        if first_close == 0:
            return False, 0.0

        move_percent = abs((latest_close / first_close) - 1) * 100
        is_flash = move_percent >= self.threshold_percent

        if is_flash:
            logger.warning(f"FLASH CRASH DETECTED: {move_percent:.2f}% move")

        return is_flash, move_percent


if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)
    n = 2000  # More data for testing

    base_price = 2000.0
    prices = [base_price]
    for i in range(1, n):
        # Simulate regime changes
        if i < 600:
            vol = 0.001  # Low vol
        elif i < 1200:
            vol = 0.005  # High vol
        else:
            vol = 0.002  # Medium vol
        ret = np.random.randn() * vol
        prices.append(prices[-1] * (1 + ret))

    df = pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
        "open": prices,
        "high": [p * (1 + np.abs(np.random.randn()) * 0.001) for p in prices],
        "low": [p * (1 - np.abs(np.random.randn()) * 0.001) for p in prices],
        "close": [p * (1 + np.random.randn() * 0.0005) for p in prices],
        "volume": np.random.randint(1000, 10000, n),
    })

    detector = MarketRegimeDetector(
        n_regimes=3,
        model_path="models/hmm_regime.pkl"
    )
    detector.fit(df)

    state = detector.get_current_state(df)
    print(f"\nCurrent Regime: {state.regime.value}")
    print(f"Confidence: {state.confidence:.2%}")
    print(f"Recommendation: {state.recommendation}")
    print(f"Volatility: {state.volatility:.4f}")
