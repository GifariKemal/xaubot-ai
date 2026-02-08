"""
ML V2 Training Pipeline
========================
Training pipeline with purged walk-forward CV and experiment configs.

Experiment Configs:
- Baseline: 1-bar target + 37 base features + XGBoost (V1 reproduction)
- A: 3-bar target + 37 base features + XGBoost
- B: 3-bar target + 45 features (37 + 8 H1) + XGBoost
- C: 3-bar target + 52 features (45 + 7 SMC) + XGBoost
- D: 3-bar target + 60 features (52 + 8 regime/PA) + XGBoost
- E: 3-bar target + 60 features + Ensemble (XGB + LGBM)
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .ml_v2_target import TargetBuilder
from .ml_v2_feature_eng import MLV2FeatureEngineer
from .ml_v2_model import TradingModelV2, ModelType


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    name: str
    target_col: str  # Which target to use
    feature_cols: List[str]  # Which features to use
    model_type: ModelType
    lookahead: int = 3  # For target creation
    threshold_atr_mult: float = 0.3  # For target creation


class MLV2Trainer:
    """
    ML V2 Training Pipeline.

    Features:
    - Purged walk-forward CV (5 folds)
    - Gap between folds to prevent temporal leakage
    - Experiment comparison
    - Optional Boruta feature selection
    """

    def __init__(
        self,
        train_size: int = 5000,
        test_size: int = 1000,
        gap_size: int = 50,
        n_folds: int = 5,
    ):
        """
        Initialize trainer.

        Args:
            train_size: Samples per training fold
            test_size: Samples per test fold
            gap_size: Gap between train and test (prevent leakage)
            n_folds: Number of CV folds
        """
        self.train_size = train_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.n_folds = n_folds

    def purged_walk_forward_cv(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_type: ModelType,
    ) -> Dict[str, float]:
        """
        Purged walk-forward cross-validation.

        Splits data into `n_folds` sequential folds with:
        - `train_size` samples for training
        - `gap_size` samples skipped (purge)
        - `test_size` samples for testing

        Args:
            df: DataFrame with features and target
            feature_cols: Feature columns
            target_col: Target column
            model_type: Model type to train

        Returns:
            Dict with mean/std of train/test AUC and overfitting ratio
        """
        logger.info(f"Starting purged walk-forward CV ({self.n_folds} folds)...")

        # Drop nulls
        df_clean = df.select(feature_cols + [target_col]).drop_nulls()

        if len(df_clean) < self.train_size + self.gap_size + self.test_size:
            logger.error(f"Insufficient data for CV: {len(df_clean)} samples")
            return {}

        train_scores = []
        test_scores = []

        fold_step = self.train_size + self.gap_size + self.test_size

        for fold in range(self.n_folds):
            start_idx = fold * fold_step

            if start_idx + fold_step > len(df_clean):
                logger.warning(f"Fold {fold+1}: Not enough data, skipping")
                break

            train_end = start_idx + self.train_size
            test_start = train_end + self.gap_size
            test_end = test_start + self.test_size

            # Extract fold data
            df_train = df_clean.slice(start_idx, self.train_size)
            df_test = df_clean.slice(test_start, self.test_size)

            logger.info(f"  Fold {fold+1}/{self.n_folds}: Train [{start_idx}:{train_end}], Test [{test_start}:{test_end}]")

            # Train model
            model = TradingModelV2(model_type=model_type)
            model.fit(
                df_train,
                feature_cols,
                target_col,
                train_ratio=1.0,  # Use all training data
                num_boost_round=100,
                early_stopping_rounds=None,  # No early stopping in CV
            )

            if not model.fitted:
                logger.warning(f"  Fold {fold+1}: Model failed to fit")
                continue

            # Extract features and targets
            X_train = df_train.select(feature_cols).to_numpy()
            y_train = df_train.select(target_col).to_numpy().ravel()
            X_test = df_test.select(feature_cols).to_numpy()
            y_test = df_test.select(target_col).to_numpy().ravel()

            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            # Evaluate
            train_score = self._evaluate_binary(model, X_train, y_train)
            test_score = self._evaluate_binary(model, X_test, y_test)

            train_scores.append(train_score)
            test_scores.append(test_score)

            logger.info(f"  Fold {fold+1}: Train AUC={train_score:.4f}, Test AUC={test_score:.4f}")

        if not train_scores:
            logger.error("No folds completed successfully")
            return {}

        # Compute statistics
        mean_train = np.mean(train_scores)
        std_train = np.std(train_scores)
        mean_test = np.mean(test_scores)
        std_test = np.std(test_scores)
        overfitting_ratio = mean_train / mean_test if mean_test > 0 else 999.0

        results = {
            "mean_train_auc": mean_train,
            "std_train_auc": std_train,
            "mean_test_auc": mean_test,
            "std_test_auc": std_test,
            "overfitting_ratio": overfitting_ratio,
            "n_folds": len(train_scores),
        }

        logger.info(f"CV Results: Train AUC={mean_train:.4f}±{std_train:.4f}, "
                   f"Test AUC={mean_test:.4f}±{std_test:.4f}, "
                   f"Overfit Ratio={overfitting_ratio:.2f}")

        return results

    def _evaluate_binary(self, model: TradingModelV2, X, y) -> float:
        """Evaluate binary classification model."""
        try:
            from sklearn.metrics import roc_auc_score
            import xgboost as xgb

            if model.xgb_model is not None:
                dmatrix = xgb.DMatrix(X, feature_names=model.feature_names)
                preds = model.xgb_model.predict(dmatrix)
                return roc_auc_score(y, preds)
            elif model.lgb_model is not None:
                preds = model.lgb_model.predict(X)
                return roc_auc_score(y, preds)
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            return 0.5

    def train_experiment(
        self,
        config: ExperimentConfig,
        df: pl.DataFrame,
        save_path: Optional[str] = None,
        run_cv: bool = True,
    ) -> Tuple[TradingModelV2, Dict]:
        """
        Train a single experiment config.

        Args:
            config: Experiment configuration
            df: DataFrame with all features and targets
            save_path: Path to save trained model
            run_cv: Whether to run cross-validation

        Returns:
            Tuple of (trained model, CV results dict)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Experiment: {config.name}")
        logger.info(f"  Target: {config.target_col}")
        logger.info(f"  Features: {len(config.feature_cols)}")
        logger.info(f"  Model: {config.model_type.value}")
        logger.info(f"{'='*60}")

        # Cross-validation
        cv_results = {}
        if run_cv:
            cv_results = self.purged_walk_forward_cv(
                df,
                config.feature_cols,
                config.target_col,
                config.model_type,
            )

        # Train final model on all data
        logger.info("Training final model on all data...")
        model = TradingModelV2(
            model_type=config.model_type,
            model_path=save_path,
        )

        model.fit(
            df,
            config.feature_cols,
            config.target_col,
            train_ratio=0.8,
            num_boost_round=100,
            early_stopping_rounds=10,
        )

        logger.info(f"Experiment {config.name} complete!")

        return model, cv_results


def get_baseline_config(base_features: List[str]) -> ExperimentConfig:
    """Get baseline (V1) experiment config."""
    return ExperimentConfig(
        name="Baseline (V1)",
        target_col="baseline_target",
        feature_cols=base_features,
        model_type=ModelType.XGBOOST_BINARY,
        lookahead=1,
        threshold_atr_mult=0.0,
    )


def get_config_a(base_features: List[str]) -> ExperimentConfig:
    """Config A: Better target, same features."""
    return ExperimentConfig(
        name="A: Better Target",
        target_col="multi_bar_target",
        feature_cols=base_features,
        model_type=ModelType.XGBOOST_BINARY,
        lookahead=3,
        threshold_atr_mult=0.3,
    )


def get_config_b(base_features: List[str], h1_features: List[str]) -> ExperimentConfig:
    """Config B: Better target + H1 features."""
    features = base_features + h1_features
    return ExperimentConfig(
        name="B: +H1 Features",
        target_col="multi_bar_target",
        feature_cols=features,
        model_type=ModelType.XGBOOST_BINARY,
        lookahead=3,
        threshold_atr_mult=0.3,
    )


def get_config_c(base_features: List[str], h1_features: List[str], smc_features: List[str]) -> ExperimentConfig:
    """Config C: Better target + H1 + continuous SMC."""
    features = base_features + h1_features + smc_features
    return ExperimentConfig(
        name="C: +Continuous SMC",
        target_col="multi_bar_target",
        feature_cols=features,
        model_type=ModelType.XGBOOST_BINARY,
        lookahead=3,
        threshold_atr_mult=0.3,
    )


def get_config_d(
    base_features: List[str],
    h1_features: List[str],
    smc_features: List[str],
    regime_features: List[str],
    pa_features: List[str],
) -> ExperimentConfig:
    """Config D: All features."""
    features = base_features + h1_features + smc_features + regime_features + pa_features
    return ExperimentConfig(
        name="D: All 60 Features",
        target_col="multi_bar_target",
        feature_cols=features,
        model_type=ModelType.XGBOOST_BINARY,
        lookahead=3,
        threshold_atr_mult=0.3,
    )


def get_config_e(
    base_features: List[str],
    h1_features: List[str],
    smc_features: List[str],
    regime_features: List[str],
    pa_features: List[str],
) -> ExperimentConfig:
    """Config E: All features + ensemble."""
    features = base_features + h1_features + smc_features + regime_features + pa_features
    return ExperimentConfig(
        name="E: Ensemble",
        target_col="multi_bar_target",
        feature_cols=features,
        model_type=ModelType.ENSEMBLE,
        lookahead=3,
        threshold_atr_mult=0.3,
    )


if __name__ == "__main__":
    # Test training pipeline
    import numpy as np

    np.random.seed(42)
    n = 10000

    # Synthetic data with 40 features
    feature_data = {}
    for i in range(40):
        feature_data[f"feat_{i}"] = np.random.randn(n)

    df = pl.DataFrame(feature_data)

    # Add target
    target = (df["feat_0"].to_numpy() + df["feat_1"].to_numpy() > 0).astype(int)
    df = df.with_columns([pl.Series("multi_bar_target", target)])

    # Test config
    config = ExperimentConfig(
        name="Test",
        target_col="multi_bar_target",
        feature_cols=[f"feat_{i}" for i in range(10)],
        model_type=ModelType.XGBOOST_BINARY,
    )

    # Train
    trainer = MLV2Trainer(train_size=1000, test_size=200, gap_size=50, n_folds=3)
    model, cv_results = trainer.train_experiment(config, df, run_cv=True)

    print(f"\nCV Results: {cv_results}")
    print(f"Model fitted: {model.fitted}")
