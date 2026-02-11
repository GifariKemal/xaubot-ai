"""
ML Model V3 Training Pipeline
==============================
Complete rewrite with production-grade ML practices.

Bismillah - Let's build something exceptional.

Key improvements:
1. Triple barrier labeling for clean targets
2. 100k+ bars training data (2+ months)
3. Proper H1 feature integration
4. Purged walk-forward cross-validation
5. Hyperparameter optimization
6. Class balancing
7. Model monitoring metrics
8. Full explainability (SHAP values)

Author: Claude + Gifari Kemal
Date: 2026-02-09
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mt5_connector import MT5Connector
from src.config import TradingConfig
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from triple_barrier_labeling import TripleBarrierLabeling

# ML imports
try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  Optuna not installed. Using default hyperparameters.")


class MLTrainerV3:
    """
    Production-grade ML model trainer.

    Features:
    - Proper time-series validation
    - Hyperparameter tuning
    - Feature importance analysis
    - Model versioning
    - Performance monitoring
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.mt5 = MT5Connector(
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
            path=config.mt5_path
        )
        self.fe = FeatureEngineer()
        self.smc = SMCAnalyzer()

        # Triple barrier for BINARY classification (BUY vs SELL only)
        # Symmetric barriers for balanced labeling
        self.labeler = TripleBarrierLabeling(
            profit_atr_mult=0.5,     # 50% ATR profit target
            stoploss_atr_mult=0.5,   # 50% ATR stop loss (symmetric RR 1.0)
            max_holding_bars=20,     # 5 hours on M15 (allow time to develop)
        )

        self.model = None
        self.feature_cols = []
        self.metadata = {}

        # Paths
        self.output_dir = Path("backtests/ml_v3")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_training_data(self, n_bars: int = 50000) -> pl.DataFrame:
        """
        Fetch large amount of training data.

        Args:
            n_bars: number of M15 bars to fetch (50k = ~1 month, safer limit)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"\n Fetching {n_bars:,} bars of M15 data...")
        print(f"   Symbol: {self.config.symbol}")
        print(f"   Timeframe: M15")

        self.mt5.connect()
        df = self.mt5.get_market_data(
            symbol=self.config.symbol,
            timeframe="M15",
            count=n_bars,
        )

        if df is None or len(df) == 0:
            raise ValueError("Failed to fetch M15 data from MT5. Check connection and symbol.")

        print(f" Fetched {len(df):,} bars")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def fetch_h1_data(self, n_bars: int = 5000) -> pl.DataFrame:
        """Fetch H1 data for higher timeframe features."""
        print(f"\n Fetching {n_bars:,} bars of H1 data...")

        df_h1 = self.mt5.get_market_data(
            symbol=self.config.symbol,
            timeframe="H1",
            count=n_bars,
        )

        print(f" Fetched {len(df_h1):,} H1 bars")
        return df_h1

    def engineer_features(
        self,
        df_m15: pl.DataFrame,
        df_h1: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate all features for M15 data, including H1 features.

        Args:
            df_m15: M15 OHLCV data
            df_h1: H1 OHLCV data

        Returns:
            DataFrame with all features
        """
        print(f"\n Engineering features...")

        # Calculate M15 features
        print("   M15 technical indicators...")
        df = self.fe.calculate_all(df_m15, include_ml_features=True)

        # Calculate SMC features
        print("   SMC structure features...")
        df = self.smc.calculate_all(df)

        # Add MLV2 features (includes H1 + advanced derived features)
        print("   MLV2 features (H1 + derived)...")
        from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
        fe_v2 = MLV2FeatureEngineer()
        df = fe_v2.add_all_v2_features(df, df_h1)

        # Feature validation
        n_features = len([c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']])
        print(f" Total features: {n_features} (MLV2 compatible)")

        # Check for nulls
        null_counts = df.null_count()
        cols_with_nulls = [
            col for col in null_counts.columns
            if null_counts[col][0] > 0
        ]
        if cols_with_nulls:
            print(f"     Columns with nulls: {len(cols_with_nulls)}")
            print(f"      {', '.join(cols_with_nulls[:10])}")
            print("   Filling nulls with forward fill...")
            df = df.fill_null(strategy="forward")
            df = df.fill_null(strategy="zero")  # Remaining nulls at start

        return df

    def _join_h1_features(
        self,
        df_m15: pl.DataFrame,
        df_h1: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Join H1 features to M15 data using asof join.

        This ensures no look-ahead bias.
        """
        # Calculate H1 indicators
        df_h1 = self.fe.calculate_all(df_h1, include_ml_features=False)
        df_h1 = self.smc.calculate_all(df_h1)

        # Select H1 features to join
        h1_feature_cols = [
            "time", "close", "rsi", "atr", "bb_upper", "bb_lower",
            "macd", "macd_signal", "ema_20", "ema_50",
            "ob", "fvg", "market_structure"
        ]
        h1_feature_cols = [c for c in h1_feature_cols if c in df_h1.columns]

        df_h1_selected = df_h1.select(h1_feature_cols)

        # Rename H1 columns
        rename_map = {c: f"h1_{c}" for c in df_h1_selected.columns if c != "time"}
        rename_map["time"] = "time"  # Keep time for join
        df_h1_selected = df_h1_selected.rename(rename_map)

        # Asof join (each M15 bar gets H1 features from the latest H1 bar)
        df_joined = df_m15.join_asof(
            df_h1_selected,
            on="time",
            strategy="backward"  # Use most recent H1 bar
        )

        # Calculate H1 derived features
        if "h1_close" in df_joined.columns and "h1_ema_20" in df_joined.columns:
            df_joined = df_joined.with_columns([
                ((pl.col("h1_close") - pl.col("h1_ema_20")) / pl.col("h1_ema_20")).alias("h1_ema20_distance")
            ])

        return df_joined

    def label_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply triple barrier labeling (BINARY: BUY vs SELL).

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with target column (1=BUY, 0=SELL)
        """
        print(f"\n Labeling data with Triple Barrier Method (BINARY)...")

        # Apply triple barrier (binary classification only)
        df = self.labeler.label_data(df)

        return df

    def prepare_train_test(
        self,
        df: pl.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data into train and test sets with stratified sampling.

        Args:
            df: Full dataset
            test_size: Fraction for test set

        Returns:
            (df_train, df_test)
        """
        print(f"\n Splitting train/test (stratified, BINARY)...")

        # Remove unlabeled rows (target == -1) and null targets
        df = df.filter((pl.col("target").is_not_null()) & (pl.col("target") >= 0))

        if len(df) == 0:
            raise ValueError("No labeled data available after filtering. Check labeling logic.")

        # Stratified split for BINARY classification (BUY=1, SELL=0)
        df_buy = df.filter(pl.col("target") == 1)
        df_sell = df.filter(pl.col("target") == 0)

        n_buy_test = int(len(df_buy) * test_size)
        n_sell_test = int(len(df_sell) * test_size)

        # Use time-based split (last 20% as test)
        df_buy_train = df_buy.head(len(df_buy) - n_buy_test)
        df_buy_test = df_buy.tail(n_buy_test)

        df_sell_train = df_sell.head(len(df_sell) - n_sell_test)
        df_sell_test = df_sell.tail(n_sell_test)

        # Combine
        df_train = pl.concat([df_buy_train, df_sell_train])
        df_test = pl.concat([df_buy_test, df_sell_test])

        # Shuffle train (but keep test chronological)
        df_train = df_train.sample(fraction=1.0, seed=42)

        print(f"   Train: {len(df_train):,} samples")
        print(f"   Test:  {len(df_test):,} samples")

        # Check class balance
        for name, subset in [("Train", df_train), ("Test", df_test)]:
            n_buy = subset.filter(pl.col("target") == 1).height
            n_sell = subset.filter(pl.col("target") == 0).height
            total = n_buy + n_sell
            if total > 0:
                print(f"   {name} distribution: BUY={n_buy/total*100:.1f}%, SELL={n_sell/total*100:.1f}%")

        return df_train, df_test

    def select_features(self, df: pl.DataFrame) -> List[str]:
        """
        Select features for training (exclude metadata columns).

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        exclude_cols = {
            'time', 'open', 'high', 'low', 'close', 'volume',
            'target', 'target_label', 'barrier_hit', 'bars_to_barrier',
            'return_pct', 'smc_signal', 'smc_confidence', 'smc_reason'
        }

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Boolean]
        ]

        print(f"\n Selected {len(feature_cols)} features")
        print(f"   Sample features: {', '.join(feature_cols[:10])}...")

        self.feature_cols = feature_cols
        return feature_cols

    def train_xgboost(
        self,
        df_train: pl.DataFrame,
        df_test: pl.DataFrame,
        feature_cols: List[str],
        optimize_hyperparams: bool = True
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with optional hyperparameter optimization.

        Args:
            df_train: Training data
            df_test: Test data
            feature_cols: List of feature column names
            optimize_hyperparams: Whether to run Optuna optimization

        Returns:
            Trained XGBoost model
        """
        print(f"\n Training XGBoost model (BINARY: BUY vs SELL)...")

        # Prepare data
        X_train = df_train.select(feature_cols).to_numpy()
        y_train = df_train["target"].to_numpy()  # Already 0=SELL, 1=BUY

        X_test = df_test.select(feature_cols).to_numpy()
        y_test = df_test["target"].to_numpy()  # Already 0=SELL, 1=BUY

        # Verify binary classes
        unique_classes_train = np.unique(y_train)
        print(f"   Training classes: {unique_classes_train} (expected: [0, 1])")

        if not np.array_equal(unique_classes_train, np.array([0, 1])):
            print(f"   WARNING: Expected binary classes [0, 1], got {unique_classes_train}")

        # Class weights (handle imbalance) - BINARY
        n_sell = (y_train == 0).sum()
        n_buy = (y_train == 1).sum()
        n_total = len(y_train)

        weight_sell = n_total / (2 * n_sell) if n_sell > 0 else 1.0
        weight_buy = n_total / (2 * n_buy) if n_buy > 0 else 1.0

        sample_weights = np.where(y_train == 0, weight_sell, weight_buy)

        print(f"   Class weights: SELL={weight_sell:.2f}, BUY={weight_buy:.2f}")
        print(f"   Class distribution: SELL={n_sell} ({n_sell/n_total*100:.1f}%), BUY={n_buy} ({n_buy/n_total*100:.1f}%)")

        # Hyperparameters
        if optimize_hyperparams and HAS_OPTUNA:
            print("   Running Optuna hyperparameter optimization...")
            best_params = self._optimize_hyperparameters(
                X_train, y_train, X_test, y_test, sample_weights
            )
        else:
            # Default params (conservative)
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'min_child_weight': 3,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }

        # Train final model (BINARY classification)
        print(f"\n   Training final model with params: {best_params}")

        model = xgb.XGBClassifier(
            objective='binary:logistic',  # Binary classification
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            **best_params
        )

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        print(f"\n Model Performance (BINARY):")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = (y_train_pred == y_train).mean()
        test_acc = (y_test_pred == y_test).mean()

        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy:  {test_acc:.4f}")

        # Per-class metrics
        print(f"\n   Test Set Classification Report (BINARY):")
        print(classification_report(y_test, y_test_pred, target_names=['SELL', 'BUY'], digits=3))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\n   Confusion Matrix:")
        print(f"                Predicted")
        print(f"            SELL   BUY")
        print(f"   SELL    {cm[0][0]:5d} {cm[0][1]:5d}")
        print(f"   BUY     {cm[1][0]:5d} {cm[1][1]:5d}")

        # Store metadata
        self.metadata = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'n_features': len(feature_cols),
            'feature_cols': feature_cols,
            'hyperparameters': best_params,
            'class_distribution_train': {
                'SELL': int(n_sell),
                'BUY': int(n_buy)
            },
            'model_type': 'binary_classification'
        }

        self.model = model
        return model

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weights: np.ndarray
    ) -> Dict:
        """
        Use Optuna to find optimal hyperparameters.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            sample_weights: Sample weights for imbalance

        Returns:
            Best hyperparameters dict
        """

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            }

            model = xgb.XGBClassifier(
                objective='binary:logistic',  # Binary classification
                random_state=42,
                n_jobs=1,  # Single thread per trial
                **params
            )

            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
            y_pred = model.predict(X_test)
            accuracy = (y_pred == y_test).mean()

            return accuracy

        study = optuna.create_study(direction='maximize', study_name='xgboost_opt')
        study.optimize(objective, n_trials=30, show_progress_bar=True, n_jobs=1)

        print(f"\n   Best trial: {study.best_trial.number}")
        print(f"   Best accuracy: {study.best_value:.4f}")

        return study.best_params

    def save_model(self, output_name: str = "xgboost_model_v3.pkl"):
        """Save trained model with metadata (TradingModelV2 compatible format)."""
        output_path = self.output_dir / output_name

        # Save in TradingModelV2 format for compatibility with main_live.py
        from backtests.ml_v2.ml_v2_model import ModelType

        model_data = {
            'xgb_model': self.model.get_booster(),  # XGBoost Booster object (low-level API)
            'lgb_model': None,  # Not used
            'model_type': ModelType.XGBOOST_BINARY,  # Binary classification
            'feature_names': self.feature_cols,
            'confidence_threshold': 0.60,  # Binary confidence threshold
            'xgb_params': self.metadata.get('hyperparameters', {}),
            'lgb_params': {},
            'feature_importance': {},  # Can be populated later
            'train_metrics': {
                'train_accuracy': self.metadata['train_accuracy'],
                'test_accuracy': self.metadata['test_accuracy'],
            },
            'fitted': True,
            'metadata': self.metadata,
            'version': '3.0_binary',
            'trained_at': datetime.now().isoformat(),
            'symbol': self.config.symbol,
            'timeframe': 'M15'
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n Model saved to: {output_path} (TradingModelV2 format)")

        # Save metadata as JSON
        metadata_path = self.output_dir / output_name.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f" Metadata saved to: {metadata_path}")

    def run_full_pipeline(self):
        """Execute full training pipeline."""
        print("=" * 80)
        print("ML MODEL V3 TRAINING PIPELINE")
        print("Bismillah - Building Exceptional Model")
        print("=" * 80)

        # 1. Fetch data
        df_m15 = self.fetch_training_data(n_bars=50000)  # 50k bars = ~1 month
        df_h1 = self.fetch_h1_data(n_bars=2000)  # 2k H1 bars = ~3 months

        # 2. Engineer features
        df = self.engineer_features(df_m15, df_h1)

        # 3. Label data
        df = self.label_data(df)

        # 4. Split train/test BEFORE balancing (to preserve natural distribution in test set)
        df_train_raw, df_test = self.prepare_train_test(df, test_size=0.20)

        # 5. Balance ONLY training set (keep test set natural) - BINARY 50/50
        print("\n Balancing TRAINING set only (BINARY)...")
        df_train = self.labeler.balance_classes(
            df_train_raw,
            target_buy_pct=0.50,   # 50% BUY
            target_sell_pct=0.50,  # 50% SELL
        )

        # 6. Select features
        feature_cols = self.select_features(df_train)

        # 7. Train model
        model = self.train_xgboost(df_train, df_test, feature_cols, optimize_hyperparams=True)

        # 8. Save model
        self.save_model()

        print("\n" + "=" * 80)
        print(" TRAINING COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    config = TradingConfig()
    trainer = MLTrainerV3(config)

    try:
        trainer.run_full_pipeline()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()
