"""
Machine Learning Model Module
=============================
XGBoost-based signal prediction with Polars support.

Features:
- Native Polars DataFrame support
- Walk-forward training
- Feature importance analysis
- Model persistence (.pkl format)
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    logger.warning("xgboost not installed. Install with: pip install xgboost")
    xgb = None


@dataclass
class PredictionResult:
    """Model prediction result."""
    signal: str  # "BUY", "SELL", "HOLD"
    probability: float
    confidence: float
    feature_importance: Dict[str, float]


class TradingModel:
    """
    XGBoost-based trading signal model.
    
    Features:
    - Works with Polars DataFrames natively
    - Binary classification (up/down)
    - Walk-forward retraining
    - Feature importance tracking
    - Saves/loads as .pkl
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        model_path: Optional[str] = None,
        params: Optional[Dict] = None,
    ):
        """
        Initialize trading model.
        
        Args:
            confidence_threshold: Minimum confidence for signal
            model_path: Path to save/load model (.pkl)
            params: XGBoost parameters
        """
        if xgb is None:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        
        self.confidence_threshold = confidence_threshold
        self.model_path = Path(model_path) if model_path else None
        
        # Default XGBoost parameters - TUNED TO PREVENT OVERFITTING
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 3,            # Reduced from 6 to prevent overfitting
            "learning_rate": 0.05,     # Reduced from 0.1 for smoother learning
            "tree_method": "hist",
            "device": "cpu",
            "min_child_weight": 10,    # Increased from 1 to require more samples per leaf
            "subsample": 0.7,          # Reduced from 0.8 for more regularization
            "colsample_bytree": 0.6,   # Reduced from 0.8 for more regularization
            "reg_alpha": 1.0,          # Increased L1 regularization (was 0.1)
            "reg_lambda": 5.0,         # Increased L2 regularization (was 1.0)
            "gamma": 1.0,              # Added minimum loss reduction for split
            "max_delta_step": 1,       # Added to help with imbalanced classes
        }
        
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.fitted = False
        self._feature_importance: Dict[str, float] = {}
        self._train_metrics: Dict[str, float] = {}
    
    def fit(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        train_ratio: float = 0.8,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
    ) -> "TradingModel":
        """
        Train the XGBoost model on Polars DataFrame.
        
        Args:
            df: Polars DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            train_ratio: Train/test split ratio
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Self for chaining
        """
        # Drop rows with nulls in features or target
        available_features = [f for f in feature_cols if f in df.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features (will be skipped): {missing}")
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found")
            return self
        
        df_clean = df.select(available_features + [target_col]).drop_nulls()
        
        if len(df_clean) < 100:
            logger.warning(f"Insufficient data for training: {len(df_clean)} samples")
            return self
        
        self.feature_names = available_features
        
        # Extract features and target
        X = df_clean.select(available_features).to_numpy()
        y = df_clean.select(target_col).to_numpy().ravel()
        
        # Handle any NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train/test split (time-series aware - no shuffle)
        # FIX: Add GAP between train and test to prevent temporal leakage
        # Gap of 50 bars (~12.5 hours on M15) breaks autocorrelation
        gap_size = 50
        split_idx = int(len(X) * train_ratio)

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        # Skip 'gap_size' bars between train and test
        test_start_idx = split_idx + gap_size
        if test_start_idx >= len(X):
            # Not enough data for gap, use smaller gap
            test_start_idx = min(split_idx + 10, len(X) - 1)
        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]

        logger.info(f"Train/Test gap: {test_start_idx - split_idx} bars to prevent temporal leakage")
        
        logger.info(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=available_features)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=available_features)
        
        # Train model
        evals = [(dtrain, "train"), (dtest, "eval")]
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10,
        )
        
        self.fitted = True
        
        # Store feature importance
        importance = self.model.get_score(importance_type="gain")
        self._feature_importance = {
            feat: importance.get(feat, 0) for feat in available_features
        }
        
        # Calculate and log training results
        train_auc = self._evaluate(dtrain)
        test_auc = self._evaluate(dtest)
        
        self._train_metrics = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(available_features),
        }
        
        logger.info(f"Training complete: Train AUC={train_auc:.4f}, Test AUC={test_auc:.4f}")
        
        # Auto-save if path provided
        if self.model_path:
            self.save()
        
        return self
    
    def _evaluate(self, dmatrix: xgb.DMatrix) -> float:
        """Evaluate model on DMatrix."""
        if self.model is None:
            return 0.0
        
        try:
            from sklearn.metrics import roc_auc_score
            preds = self.model.predict(dmatrix)
            labels = dmatrix.get_label()
            return roc_auc_score(labels, preds)
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return 0.5
    
    def predict(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        Predict trading signal for latest data point.
        
        Args:
            df: Polars DataFrame with features
            feature_cols: Feature columns (uses stored if None)
            
        Returns:
            PredictionResult with signal and confidence
        """
        if not self.fitted or self.model is None:
            logger.warning("Model not fitted, returning HOLD")
            return PredictionResult(
                signal="HOLD",
                probability=0.5,
                confidence=0.0,
                feature_importance={},
            )
        
        # ALWAYS use the model's feature names to avoid mismatch
        features = self.feature_names

        # Get latest row
        latest = df.tail(1)

        # Check for missing features - must be exact match
        available_features = [f for f in features if f in latest.columns]
        missing_features = [f for f in features if f not in latest.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features}")

        # Must use EXACTLY the model's features in same order
        if len(available_features) != len(features):
            logger.error(f"Feature mismatch: expected {len(features)}, got {len(available_features)}")
            logger.error(f"Missing: {missing_features}")
            return PredictionResult(
                signal="HOLD",
                probability=0.5,
                confidence=0.0,
                feature_importance={},
            )

        # Extract features
        try:
            X = latest.select(features).to_numpy()

            # Handle nulls
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Create DMatrix with exact feature names from model
            dmatrix = xgb.DMatrix(X, feature_names=features)
            
            # Predict probability
            prob_up = float(self.model.predict(dmatrix)[0])
            prob_down = 1 - prob_up
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                signal="HOLD",
                probability=0.5,
                confidence=0.0,
                feature_importance={},
            )
        
        # Determine signal based on probability
        if prob_up > self.confidence_threshold:
            signal = "BUY"
            confidence = prob_up
        elif prob_down > self.confidence_threshold:
            signal = "SELL"
            confidence = prob_down
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down)
        
        return PredictionResult(
            signal=signal,
            probability=prob_up,
            confidence=confidence,
            feature_importance=self._feature_importance,
        )
    
    def predict_proba(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Add prediction probabilities to DataFrame.
        """
        if not self.fitted or self.model is None:
            return df.with_columns([
                pl.lit(0.5).alias("pred_prob_up"),
                pl.lit("HOLD").alias("pred_signal"),
            ])
        
        features = feature_cols or self.feature_names
        available_features = [f for f in features if f in df.columns]
        
        # Extract features
        X = df.select(available_features).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(X, feature_names=available_features)
        probs = self.model.predict(dmatrix)
        
        # Add to DataFrame
        df = df.with_columns([
            pl.Series("pred_prob_up", probs),
        ])
        
        # Add signal column
        df = df.with_columns([
            pl.when(pl.col("pred_prob_up") > self.confidence_threshold)
                .then(pl.lit("BUY"))
                .when(pl.col("pred_prob_up") < (1 - self.confidence_threshold))
                .then(pl.lit("SELL"))
                .otherwise(pl.lit("HOLD"))
                .alias("pred_signal"),
        ])
        
        return df
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get top N important features."""
        if not self._feature_importance:
            return {}
        
        sorted_importance = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_importance[:top_n])
    
    def save(self, path: Optional[str] = None):
        """Save model to .pkl file."""
        save_path = Path(path) if path else self.model_path
        
        if save_path is None:
            logger.warning("No save path provided")
            return
        
        # Ensure .pkl extension
        save_path = save_path.with_suffix(".pkl")
        
        # Create directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save everything as pickle
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "confidence_threshold": self.confidence_threshold,
            "params": self.params,
            "feature_importance": self._feature_importance,
            "train_metrics": self._train_metrics,
            "fitted": self.fitted,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> "TradingModel":
        """Load model from .pkl file."""
        load_path = Path(path) if path else self.model_path
        
        if load_path is None:
            logger.warning("No load path provided")
            return self
        
        # Ensure .pkl extension
        load_path = load_path.with_suffix(".pkl")
        
        if not load_path.exists():
            logger.warning(f"Model file not found: {load_path}")
            return self
        
        try:
            with open(load_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get("model")
            self.feature_names = model_data.get("feature_names", [])
            self.confidence_threshold = model_data.get("confidence_threshold", 0.65)
            self.params = model_data.get("params", self.params)
            self._feature_importance = model_data.get("feature_importance", {})
            self._train_metrics = model_data.get("train_metrics", {})
            self.fitted = model_data.get("fitted", self.model is not None)
            
            logger.info(f"Model loaded from {load_path}")
            if self._train_metrics:
                logger.info(f"  Train AUC: {self._train_metrics.get('train_auc', 'N/A')}")
                logger.info(f"  Test AUC: {self._train_metrics.get('test_auc', 'N/A')}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
        
        return self
    
    def walk_forward_train(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        train_window: int = 500,
        test_window: int = 50,
        step: int = 20,
    ) -> List[Tuple[float, float]]:
        """Walk-forward optimization and validation."""
        results = []
        n = len(df)
        
        for start in range(0, n - train_window - test_window, step):
            train_end = start + train_window
            test_end = train_end + test_window
            
            train_df = df.slice(start, train_window)
            test_df = df.slice(train_end, test_window)
            
            # Train on this fold
            self.fit(
                train_df,
                feature_cols,
                target_col,
                train_ratio=1.0,
                num_boost_round=50,
                early_stopping_rounds=None,
            )
            
            if not self.fitted:
                continue
            
            # Evaluate
            available_features = [f for f in feature_cols if f in train_df.columns]
            
            X_train = train_df.select(available_features).to_numpy()
            y_train = train_df.select(target_col).to_numpy().ravel()
            X_test = test_df.select(available_features).to_numpy()
            y_test = test_df.select(target_col).to_numpy().ravel()
            
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=available_features)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=available_features)
            
            train_auc = self._evaluate(dtrain)
            test_auc = self._evaluate(dtest)
            
            results.append((train_auc, test_auc))
        
        if results:
            avg_train = np.mean([r[0] for r in results])
            avg_test = np.mean([r[1] for r in results])
            logger.info(f"Walk-forward: Avg Train AUC={avg_train:.4f}, Avg Test AUC={avg_test:.4f}")
        
        return results


def get_default_feature_columns() -> List[str]:
    """Get default feature columns for ML model."""
    return [
        # Technical indicators
        "rsi", "atr", "atr_percent",
        "macd", "macd_signal", "macd_histogram",
        "bb_percent_b", "bb_width",
        "ema_9", "ema_21",
        
        # Returns and momentum
        "returns_1", "returns_5", "returns_20",
        "log_returns",
        
        # Volatility
        "volatility_20", "normalized_range", "avg_normalized_range",
        
        # Price position
        "price_position", "dist_from_sma_20",
        
        # Trend
        "higher_high", "lower_low",
        "hh_count_5", "ll_count_5",
        
        # Volume
        "volume_ratio", "high_volume",
        
        # SMC signals (numeric)
        "swing_high", "swing_low",
        "fvg_signal",
        "ob",
        "bos", "choch",
        "market_structure",
        
        # Time features
        "hour", "weekday",
        "london_session", "ny_session",
        
        # Regime
        "regime",
    ]


if __name__ == "__main__":
    # Test ML model
    import numpy as np
    
    np.random.seed(42)
    n = 500
    
    df = pl.DataFrame({
        "rsi": np.random.uniform(20, 80, n),
        "atr": np.random.uniform(0.5, 2.0, n),
        "macd": np.random.randn(n) * 0.001,
        "returns_1": np.random.randn(n) * 0.01,
    })
    
    target = ((df["rsi"].to_numpy() > 50).astype(int) * 0.5 +
              np.random.randint(0, 2, n) * 0.5)
    target = (target > 0.5).astype(int)
    df = df.with_columns([pl.Series("target", target)])
    
    model = TradingModel(
        confidence_threshold=0.65,
        model_path="models/test_model.pkl"
    )
    
    feature_cols = ["rsi", "atr", "macd", "returns_1"]
    model.fit(df, feature_cols, "target")
    
    # Test save/load
    model.save()
    
    model2 = TradingModel(model_path="models/test_model.pkl")
    model2.load()
    
    prediction = model2.predict(df, feature_cols)
    print(f"Prediction: {prediction.signal} ({prediction.confidence:.2%})")
