"""
ML V2 Model Module
===================
Multi-model support: XGBoost, LightGBM, Ensemble.

Backward compatible with V1 TradingModel for easy comparison.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
from loguru import logger

try:
    import xgboost as xgb
except ImportError:
    logger.warning("xgboost not installed")
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    logger.warning("lightgbm not installed (optional for ensemble)")
    lgb = None


class ModelType(Enum):
    """Supported model types."""
    XGBOOST_BINARY = "xgboost_binary"
    XGBOOST_3CLASS = "xgboost_3class"
    LIGHTGBM_BINARY = "lightgbm_binary"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResultV2:
    """Model prediction result."""
    signal: str  # "BUY", "SELL", "HOLD"
    probability: float  # Probability of UP (binary) or class probabilities (3-class)
    confidence: float
    probabilities: Optional[Dict[str, float]] = None  # For 3-class


class TradingModelV2:
    """
    V2 Trading Model with multi-model support.

    Features:
    - XGBoost (binary or 3-class)
    - LightGBM (binary)
    - Ensemble (average XGBoost + LightGBM)
    - Backward compatible with V1 TradingModel
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.XGBOOST_BINARY,
        confidence_threshold: float = 0.65,
        model_path: Optional[str] = None,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
    ):
        """
        Initialize V2 trading model.

        Args:
            model_type: Type of model to use
            confidence_threshold: Minimum confidence for signal
            model_path: Path to save/load model (.pkl)
            xgb_params: XGBoost parameters (optional)
            lgb_params: LightGBM parameters (optional)
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model_path = Path(model_path) if model_path else None

        # XGBoost params (anti-overfitting philosophy from V1)
        self.xgb_params = xgb_params or self._get_default_xgb_params()

        # LightGBM params (equivalent to XGBoost)
        self.lgb_params = lgb_params or self._get_default_lgb_params()

        # Models
        self.xgb_model: Optional[xgb.Booster] = None
        self.lgb_model: Optional[lgb.Booster] = None

        # Metadata
        self.feature_names: List[str] = []
        self.fitted = False
        self._feature_importance: Dict[str, float] = {}
        self._train_metrics: Dict[str, float] = {}

    def _get_default_xgb_params(self) -> Dict:
        """Get default XGBoost params (same anti-overfitting as V1)."""
        if self.model_type == ModelType.XGBOOST_3CLASS:
            return {
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "max_depth": 3,
                "learning_rate": 0.05,
                "tree_method": "hist",
                "device": "cpu",
                "min_child_weight": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "gamma": 1.0,
            }
        else:
            return {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 3,
                "learning_rate": 0.05,
                "tree_method": "hist",
                "device": "cpu",
                "min_child_weight": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.6,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "gamma": 1.0,
                "max_delta_step": 1,
            }

    def _get_default_lgb_params(self) -> Dict:
        """Get default LightGBM params (equivalent to XGBoost)."""
        return {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 8,  # Equivalent to max_depth=3
            "learning_rate": 0.05,
            "min_child_weight": 10,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "min_split_gain": 1.0,  # Equivalent to gamma
            "verbose": -1,
        }

    def fit(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str = "multi_bar_target",
        train_ratio: float = 0.8,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
    ) -> "TradingModelV2":
        """
        Train the model on Polars DataFrame.

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
        # Validate features
        available_features = [f for f in feature_cols if f in df.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features (will be skipped): {missing}")

        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found")
            return self

        # Drop nulls
        df_clean = df.select(available_features + [target_col]).drop_nulls()

        if len(df_clean) < 100:
            logger.warning(f"Insufficient data for training: {len(df_clean)} samples")
            return self

        self.feature_names = available_features

        # Extract features and target
        X = df_clean.select(available_features).to_numpy()
        y = df_clean.select(target_col).to_numpy().ravel()

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Train/test split with gap (prevent temporal leakage)
        gap_size = 50
        split_idx = int(len(X) * train_ratio)

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        test_start_idx = min(split_idx + gap_size, len(X) - 1)
        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]

        logger.info(f"Training {self.model_type.value} with {len(X_train)} samples, testing with {len(X_test)} samples")

        # Train based on model type
        if self.model_type in [ModelType.XGBOOST_BINARY, ModelType.XGBOOST_3CLASS]:
            self._fit_xgboost(X_train, y_train, X_test, y_test, num_boost_round, early_stopping_rounds)

        elif self.model_type == ModelType.LIGHTGBM_BINARY:
            if lgb is None:
                logger.error("LightGBM not installed. Install with: pip install lightgbm")
                return self
            self._fit_lightgbm(X_train, y_train, X_test, y_test, num_boost_round, early_stopping_rounds)

        elif self.model_type == ModelType.ENSEMBLE:
            # Train both models
            self._fit_xgboost(X_train, y_train, X_test, y_test, num_boost_round, early_stopping_rounds)
            if lgb is not None:
                self._fit_lightgbm(X_train, y_train, X_test, y_test, num_boost_round, early_stopping_rounds)
            else:
                logger.warning("LightGBM not available, ensemble will use XGBoost only")

        self.fitted = True

        # Auto-save
        if self.model_path:
            self.save()

        return self

    def _fit_xgboost(self, X_train, y_train, X_test, y_test, num_boost_round, early_stopping_rounds):
        """Fit XGBoost model."""
        if xgb is None:
            logger.error("XGBoost not installed")
            return

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)

        evals = [(dtrain, "train"), (dtest, "eval")]

        self.xgb_model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10,
        )

        # Feature importance
        importance = self.xgb_model.get_score(importance_type="gain")
        self._feature_importance = {
            feat: importance.get(feat, 0) for feat in self.feature_names
        }

        # Evaluate
        train_score = self._evaluate_xgb(dtrain)
        test_score = self._evaluate_xgb(dtest)

        self._train_metrics["xgb_train_score"] = train_score
        self._train_metrics["xgb_test_score"] = test_score
        self._train_metrics["train_samples"] = len(X_train)
        self._train_metrics["test_samples"] = len(X_test)

        logger.info(f"XGBoost: Train={train_score:.4f}, Test={test_score:.4f}")

    def _fit_lightgbm(self, X_train, y_train, X_test, y_test, num_boost_round, early_stopping_rounds):
        """Fit LightGBM model."""
        if lgb is None:
            return

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=self.feature_names)

        self.lgb_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, test_data],
            valid_names=["train", "eval"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=10),
            ],
        )

        # Evaluate
        train_score = self._evaluate_lgb(X_train, y_train)
        test_score = self._evaluate_lgb(X_test, y_test)

        self._train_metrics["lgb_train_score"] = train_score
        self._train_metrics["lgb_test_score"] = test_score

        logger.info(f"LightGBM: Train={train_score:.4f}, Test={test_score:.4f}")

    def _evaluate_xgb(self, dmatrix: xgb.DMatrix) -> float:
        """Evaluate XGBoost model."""
        if self.xgb_model is None:
            return 0.0

        try:
            from sklearn.metrics import roc_auc_score, log_loss

            preds = self.xgb_model.predict(dmatrix)
            labels = dmatrix.get_label()

            if self.model_type == ModelType.XGBOOST_3CLASS:
                # Multi-class: use log loss
                return -log_loss(labels, preds)  # Negative so higher is better
            else:
                # Binary: use AUC
                return roc_auc_score(labels, preds)
        except Exception as e:
            logger.warning(f"XGBoost evaluation error: {e}")
            return 0.5

    def _evaluate_lgb(self, X, y) -> float:
        """Evaluate LightGBM model."""
        if self.lgb_model is None:
            return 0.0

        try:
            from sklearn.metrics import roc_auc_score

            preds = self.lgb_model.predict(X)
            return roc_auc_score(y, preds)
        except Exception as e:
            logger.warning(f"LightGBM evaluation error: {e}")
            return 0.5

    def predict(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> PredictionResultV2:
        """
        Predict trading signal for latest data point.

        Args:
            df: Polars DataFrame with features
            feature_cols: Feature columns (uses stored if None)

        Returns:
            PredictionResultV2 with signal and confidence
        """
        if not self.fitted:
            logger.warning("Model not fitted, returning HOLD")
            return PredictionResultV2(
                signal="HOLD",
                probability=0.5,
                confidence=0.0,
            )

        features = feature_cols or self.feature_names
        latest = df.tail(1)

        # Extract features
        try:
            X = latest.select(features).to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return PredictionResultV2(signal="HOLD", probability=0.5, confidence=0.0)

        # Predict based on model type
        if self.model_type == ModelType.ENSEMBLE:
            prob_up = self._predict_ensemble(X, features)
        elif self.model_type in [ModelType.XGBOOST_BINARY, ModelType.XGBOOST_3CLASS]:
            prob_up = self._predict_xgboost(X, features)
        elif self.model_type == ModelType.LIGHTGBM_BINARY:
            prob_up = self._predict_lightgbm(X)
        else:
            prob_up = 0.5

        # Determine signal
        if isinstance(prob_up, dict):  # 3-class
            # prob_up = {"BUY": 0.4, "SELL": 0.3, "HOLD": 0.3}
            max_class = max(prob_up, key=prob_up.get)
            confidence = prob_up[max_class]

            if confidence > self.confidence_threshold:
                signal = max_class
            else:
                signal = "HOLD"

            return PredictionResultV2(
                signal=signal,
                probability=prob_up.get("BUY", 0.0),
                confidence=confidence,
                probabilities=prob_up,
            )
        else:  # Binary
            prob_down = 1 - prob_up

            if prob_up > self.confidence_threshold:
                signal = "BUY"
                confidence = prob_up
            elif prob_down > self.confidence_threshold:
                signal = "SELL"
                confidence = prob_down
            else:
                signal = "HOLD"
                confidence = max(prob_up, prob_down)

            return PredictionResultV2(
                signal=signal,
                probability=prob_up,
                confidence=confidence,
            )

    def _predict_xgboost(self, X, feature_names: Optional[List[str]] = None) -> float:
        """Predict with XGBoost."""
        if self.xgb_model is None:
            return 0.5

        names = feature_names or self.feature_names
        dmatrix = xgb.DMatrix(X, feature_names=names)
        preds = self.xgb_model.predict(dmatrix)

        if self.model_type == ModelType.XGBOOST_3CLASS:
            # Multi-class: return dict
            return {
                "BUY": float(preds[0][0]),
                "SELL": float(preds[0][1]),
                "HOLD": float(preds[0][2]),
            }
        else:
            # Binary
            return float(preds[0])

    def _predict_lightgbm(self, X) -> float:
        """Predict with LightGBM."""
        if self.lgb_model is None:
            return 0.5

        preds = self.lgb_model.predict(X)
        return float(preds[0])

    def _predict_ensemble(self, X, feature_names: Optional[List[str]] = None) -> float:
        """Predict with ensemble (average of XGBoost + LightGBM)."""
        preds = []

        if self.xgb_model is not None:
            xgb_pred = self._predict_xgboost(X, feature_names)
            if isinstance(xgb_pred, dict):
                # Can't ensemble 3-class easily, just use XGBoost
                return xgb_pred
            preds.append(xgb_pred)

        if self.lgb_model is not None:
            lgb_pred = self._predict_lightgbm(X)
            preds.append(lgb_pred)

        if not preds:
            return 0.5

        # Average
        return float(np.mean(preds))

    def save(self, path: Optional[str] = None):
        """Save model to .pkl file."""
        save_path = Path(path) if path else self.model_path

        if save_path is None:
            logger.warning("No save path provided")
            return

        save_path = save_path.with_suffix(".pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_type": self.model_type,
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "feature_names": self.feature_names,
            "confidence_threshold": self.confidence_threshold,
            "xgb_params": self.xgb_params,
            "lgb_params": self.lgb_params,
            "feature_importance": self._feature_importance,
            "train_metrics": self._train_metrics,
            "fitted": self.fitted,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {save_path}")

    def load(self, path: Optional[str] = None) -> "TradingModelV2":
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

            self.model_type = model_data.get("model_type", ModelType.XGBOOST_BINARY)
            self.xgb_model = model_data.get("xgb_model")
            self.lgb_model = model_data.get("lgb_model")
            self.feature_names = model_data.get("feature_names", [])
            self.confidence_threshold = model_data.get("confidence_threshold", 0.65)
            self.xgb_params = model_data.get("xgb_params", {})
            self.lgb_params = model_data.get("lgb_params", {})
            self._feature_importance = model_data.get("feature_importance", {})
            self._train_metrics = model_data.get("train_metrics", {})
            self.fitted = model_data.get("fitted", False)

            logger.info(f"Model loaded from {load_path}")
            logger.info(f"  Type: {self.model_type.value}")
            if self._train_metrics:
                for key, val in self._train_metrics.items():
                    if isinstance(val, float):
                        logger.info(f"  {key}: {val:.4f}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        return self

    def load_legacy_v1(self, path: str) -> "TradingModelV2":
        """
        Load V1 TradingModel and convert to V2.

        Args:
            path: Path to V1 model .pkl file

        Returns:
            Self with V1 model loaded as XGBoost binary
        """
        load_path = Path(path).with_suffix(".pkl")

        if not load_path.exists():
            logger.warning(f"V1 model file not found: {load_path}")
            return self

        try:
            with open(load_path, "rb") as f:
                v1_data = pickle.load(f)

            # V1 structure: {"model": xgb.Booster, "feature_names": [], ...}
            self.model_type = ModelType.XGBOOST_BINARY
            self.xgb_model = v1_data.get("model")
            self.feature_names = v1_data.get("feature_names", [])
            self.confidence_threshold = v1_data.get("confidence_threshold", 0.65)
            self.xgb_params = v1_data.get("params", {})
            self._feature_importance = v1_data.get("feature_importance", {})
            self._train_metrics = v1_data.get("train_metrics", {})
            self.fitted = v1_data.get("fitted", self.xgb_model is not None)

            logger.info(f"V1 model loaded and converted from {load_path}")

        except Exception as e:
            logger.error(f"Failed to load V1 model: {e}")

        return self


if __name__ == "__main__":
    # Test V2 model
    import numpy as np

    np.random.seed(42)
    n = 500

    # Synthetic features
    df = pl.DataFrame({
        "rsi": np.random.uniform(20, 80, n),
        "atr": np.random.uniform(0.5, 2.0, n),
        "macd": np.random.randn(n) * 0.001,
        "returns_1": np.random.randn(n) * 0.01,
    })

    # Binary target
    target_binary = ((df["rsi"].to_numpy() > 50).astype(int) * 0.5 +
                     np.random.randint(0, 2, n) * 0.5)
    target_binary = (target_binary > 0.5).astype(int)
    df = df.with_columns([pl.Series("multi_bar_target", target_binary)])

    # 3-class target
    target_3class = np.random.choice([0, 1, 2], n)
    df = df.with_columns([pl.Series("target_3class", target_3class)])

    feature_cols = ["rsi", "atr", "macd", "returns_1"]

    # Test XGBoost binary
    print("\n=== Testing XGBoost Binary ===")
    model_xgb = TradingModelV2(
        model_type=ModelType.XGBOOST_BINARY,
        model_path="models/test_v2_xgb.pkl"
    )
    model_xgb.fit(df, feature_cols, "multi_bar_target")
    pred = model_xgb.predict(df, feature_cols)
    print(f"Prediction: {pred.signal} ({pred.confidence:.2%})")

    # Test XGBoost 3-class
    print("\n=== Testing XGBoost 3-Class ===")
    model_3class = TradingModelV2(
        model_type=ModelType.XGBOOST_3CLASS,
        model_path="models/test_v2_3class.pkl"
    )
    model_3class.fit(df, feature_cols, "target_3class")
    pred = model_3class.predict(df, feature_cols)
    print(f"Prediction: {pred.signal} ({pred.confidence:.2%})")
    if pred.probabilities:
        print(f"Probabilities: {pred.probabilities}")

    # Test LightGBM (if available)
    if lgb is not None:
        print("\n=== Testing LightGBM Binary ===")
        model_lgb = TradingModelV2(
            model_type=ModelType.LIGHTGBM_BINARY,
            model_path="models/test_v2_lgb.pkl"
        )
        model_lgb.fit(df, feature_cols, "multi_bar_target")
        pred = model_lgb.predict(df, feature_cols)
        print(f"Prediction: {pred.signal} ({pred.confidence:.2%})")

        # Test Ensemble
        print("\n=== Testing Ensemble ===")
        model_ensemble = TradingModelV2(
            model_type=ModelType.ENSEMBLE,
            model_path="models/test_v2_ensemble.pkl"
        )
        model_ensemble.fit(df, feature_cols, "multi_bar_target")
        pred = model_ensemble.predict(df, feature_cols)
        print(f"Prediction: {pred.signal} ({pred.confidence:.2%})")
    else:
        print("\n[SKIP] LightGBM not installed")
