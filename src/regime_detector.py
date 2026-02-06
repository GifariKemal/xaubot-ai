"""
Market Regime Detection Module
==============================
HMM-based regime detection for market state classification.
Saves/loads as .pkl format.

Detects:
- Low Volatility (Safe to trade)
- Medium Volatility (Normal trading)
- High Volatility / Crisis (Sleep mode)
"""

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
    HMM-based market regime detector.
    Saves/loads models as .pkl files.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_periods: int = 500,
        retrain_frequency: int = 20,
        model_path: Optional[str] = None,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        """
        Initialize regime detector.
        """
        if GaussianHMM is None:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_regimes = n_regimes
        self.lookback_periods = lookback_periods
        self.retrain_frequency = retrain_frequency
        self.model_path = Path(model_path) if model_path else None
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",  # Use diagonal for stability
            n_iter=200,
            random_state=random_state,
            verbose=False,
        )
        
        self.fitted = False
        self.last_train_idx = 0
        self.regime_mapping: Dict[int, MarketRegime] = {}
        self._train_metrics: Dict = {}
        
    def prepare_features(self, df: pl.DataFrame) -> np.ndarray:
        """Prepare features for HMM training/prediction."""
        df_features = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns"),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("normalized_range"),
        ])
        
        df_features = df_features.with_columns([
            pl.col("log_returns")
                .rolling_std(window_size=20)
                .alias("volatility"),
        ])
        
        df_features = df_features.drop_nulls(subset=["log_returns", "volatility"])
        features = df_features.select(["log_returns", "volatility"]).to_numpy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def fit(self, df: pl.DataFrame) -> "MarketRegimeDetector":
        """Fit the HMM model on historical data."""
        features = self.prepare_features(df)
        
        if len(features) < 100:
            logger.warning(f"Insufficient data for HMM training: {len(features)} samples")
            return self
        
        try:
            self.model.fit(features)
            self.fitted = True
            self._map_regimes()
            
            # Store metrics
            self._train_metrics = {
                "samples": len(features),
                "n_regimes": self.n_regimes,
                "log_likelihood": float(self.model.score(features)),
            }
            
            logger.info(f"HMM fitted with {len(features)} samples, log-likelihood: {self._train_metrics['log_likelihood']:.2f}")
            
            # Auto-save if path provided
            if self.model_path:
                self.save()
                
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            
        return self
    
    def _map_regimes(self):
        """Map HMM states to regime names based on volatility."""
        if not self.fitted:
            return
        
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
    
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict regime for each data point."""
        if not self.fitted:
            logger.warning("Model not fitted, returning with neutral regime")
            return df.with_columns([
                pl.lit(1).alias("regime"),
                pl.lit("medium_volatility").alias("regime_name"),
                pl.lit(1.0).alias("regime_confidence"),
            ])
        
        features = self.prepare_features(df)
        
        if len(features) == 0:
            return df
        
        regimes = self.model.predict(features)
        proba = self.model.predict_proba(features)
        
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
        """Get current regime state with trading recommendation."""
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
        """Save model to .pkl file."""
        save_path = Path(path) if path else self.model_path
        
        if save_path is None:
            logger.warning("No save path provided")
            return
        
        save_path = save_path.with_suffix(".pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "lookback_periods": self.lookback_periods,
            "regime_mapping": self.regime_mapping,
            "train_metrics": self._train_metrics,
            "fitted": self.fitted,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"HMM model saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> "MarketRegimeDetector":
        """Load model from .pkl file."""
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
            
            logger.info(f"HMM model loaded from {load_path}")
            
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
    n = 500
    
    base_price = 2000.0
    prices = [base_price]
    for _ in range(1, n):
        vol = 0.002 + np.random.random() * 0.005
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
