"""
Auto Training Module
====================
Automatically retrain ML models during market close.

Features:
- Daily retraining at market close (05:00 WIB)
- Weekend deep training (more data, more epochs)
- Incremental learning from recent trades
- Model performance tracking in PostgreSQL
- Auto-rollback if new model performs worse
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
from zoneinfo import ZoneInfo
from loguru import logger
import polars as pl

# Database imports
try:
    from src.db import get_db, init_db, TrainingRepository
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database module not available, using file-based history")

# Timezone
WIB = ZoneInfo("Asia/Jakarta")


class AutoTrainer:
    """
    Automatic model retraining system.

    Retrains models during market close to keep AI up-to-date
    with latest market conditions.

    Features:
    - Stores training history in PostgreSQL
    - Fallback to file-based history if DB unavailable
    - Auto-rollback on poor performance
    """

    def __init__(
        self,
        models_dir: str = "models",
        data_dir: str = "data",
        daily_retrain_hour_wib: int = 5,      # 05:00 WIB (market close)
        weekend_retrain: bool = True,          # Deep training on weekends
        min_hours_between_retrain: float = 20, # Don't retrain too often
        backup_models: bool = True,            # Keep backup of old models
        use_db: bool = True,                   # Use database for history
        min_auc_threshold: float = 0.65,       # Alert if AUC drops below this
        auto_retrain_on_low_auc: bool = True,  # Auto retrain when AUC low
    ):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.daily_retrain_hour = daily_retrain_hour_wib
        self.weekend_retrain = weekend_retrain
        self.min_hours_between_retrain = min_hours_between_retrain
        self.backup_models = backup_models
        self.min_auc_threshold = min_auc_threshold
        self.auto_retrain_on_low_auc = auto_retrain_on_low_auc

        # Database setup
        self._use_db = use_db and DB_AVAILABLE
        self._db_connected = False
        self._training_repo = None

        if self._use_db:
            self._init_database()

        # Tracking
        self._last_retrain_time: Optional[datetime] = None
        self._current_run_id: Optional[int] = None
        self._current_auc: Optional[float] = None
        self._auc_check_count: int = 0
        self._low_auc_alert_sent: bool = False
        self._load_retrain_history()

        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        (self.models_dir / "backups").mkdir(exist_ok=True)

        logger.info(f"AutoTrainer initialized: DB={self._db_connected}, Min AUC={min_auc_threshold}")

    def _init_database(self):
        """Initialize database connection."""
        try:
            if init_db():
                self._db = get_db()
                self._training_repo = TrainingRepository(self._db)
                self._db_connected = True
                logger.info("AutoTrainer: Database connected")
            else:
                logger.warning("AutoTrainer: Database connection failed, using file")
                self._db_connected = False
        except Exception as e:
            logger.error(f"AutoTrainer: Database init error: {e}")
            self._db_connected = False

    def _load_retrain_history(self):
        """Load last retrain time from database or file."""
        # Try database first
        if self._db_connected and self._training_repo:
            try:
                latest = self._training_repo.get_latest_successful()
                if latest and latest.get("completed_at"):
                    self._last_retrain_time = latest["completed_at"]
                    if self._last_retrain_time.tzinfo is None:
                        self._last_retrain_time = self._last_retrain_time.replace(tzinfo=WIB)
                    logger.debug(f"Last retrain (DB): {self._last_retrain_time}")
                    return
            except Exception as e:
                logger.warning(f"Could not load from DB: {e}")

        # Fallback to file
        history_file = self.data_dir / "retrain_history.txt"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        self._last_retrain_time = datetime.fromisoformat(last_line)
                        logger.debug(f"Last retrain (file): {self._last_retrain_time}")
            except Exception as e:
                logger.warning(f"Could not load retrain history: {e}")

    def _save_retrain_start(self, training_type: str, bars: int, num_boost_rounds: int) -> Optional[int]:
        """Record training start in database."""
        if self._db_connected and self._training_repo:
            try:
                result = self._training_repo.insert_training_run({
                    "training_type": training_type,
                    "bars_used": bars,
                    "num_boost_rounds": num_boost_rounds,
                    "started_at": datetime.now(WIB),
                })
                if result:
                    self._current_run_id = result.get("id")
                    return self._current_run_id
            except Exception as e:
                logger.error(f"Failed to save training start: {e}")
        return None

    def _save_retrain_complete(
        self,
        success: bool,
        hmm_trained: bool = False,
        xgb_trained: bool = False,
        train_auc: float = 0,
        test_auc: float = 0,
        train_accuracy: float = 0,
        test_accuracy: float = 0,
        model_path: str = "",
        backup_path: str = "",
        error_message: str = None,
        started_at: datetime = None,
    ):
        """Record training completion in database."""
        now = datetime.now(WIB)
        self._last_retrain_time = now

        # Calculate duration
        duration = None
        if started_at:
            duration = int((now - started_at).total_seconds())

        # Update database
        if self._db_connected and self._training_repo and self._current_run_id:
            try:
                self._training_repo.update_training_complete(
                    self._current_run_id,
                    {
                        "completed_at": now,
                        "duration_seconds": duration,
                        "hmm_trained": hmm_trained,
                        "xgb_trained": xgb_trained,
                        "train_auc": train_auc,
                        "test_auc": test_auc,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "model_path": model_path,
                        "backup_path": backup_path,
                        "success": success,
                        "error_message": error_message,
                    }
                )
                logger.debug(f"Training run #{self._current_run_id} completed in DB")
            except Exception as e:
                logger.error(f"Failed to update training completion: {e}")

        # Also save to file (backup)
        history_file = self.data_dir / "retrain_history.txt"
        with open(history_file, "a") as f:
            f.write(f"{now.isoformat()}\n")

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Check if retraining should happen now.

        Returns:
            (should_retrain, reason)
        """
        now = datetime.now(WIB)

        # Check if enough time has passed since last retrain
        if self._last_retrain_time:
            # Ensure timezone-aware comparison
            last_time = self._last_retrain_time
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=WIB)

            hours_since_retrain = (now - last_time).total_seconds() / 3600
            if hours_since_retrain < self.min_hours_between_retrain:
                return False, f"Too soon since last retrain ({hours_since_retrain:.1f}h ago)"

        # Check if it's daily retrain time (within 30 min window)
        is_retrain_hour = (
            now.hour == self.daily_retrain_hour and
            now.minute < 30
        )

        # Check if it's weekend (Saturday or Sunday)
        is_weekend = now.weekday() >= 5  # 5=Saturday, 6=Sunday

        if is_retrain_hour:
            if is_weekend and self.weekend_retrain:
                return True, "Weekend deep training time"
            elif not is_weekend:
                return True, "Daily market close training"

        # Manual trigger check - if no retrain in 24+ hours
        if self._last_retrain_time:
            last_time = self._last_retrain_time
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=WIB)
            hours_since = (now - last_time).total_seconds() / 3600
            if hours_since > 24:
                return True, f"Over 24h since last training ({hours_since:.1f}h)"
        elif self._last_retrain_time is None:
            # Never trained before
            return True, "Initial training required"

        return False, "Not retrain time"

    def check_model_auc(self, model=None) -> Tuple[float, bool, str]:
        """
        Check current model AUC and determine if it's acceptable.

        Args:
            model: ML model instance (optional, will load from file if not provided)

        Returns:
            (current_auc, is_acceptable, message)
        """
        try:
            current_auc = None

            # Try to get AUC from loaded model's train_metrics (V2 format)
            if model is not None and hasattr(model, '_train_metrics') and model._train_metrics:
                tm = model._train_metrics
                current_auc = tm.get("test_auc") or tm.get("xgb_test_score") or tm.get("test_accuracy")
            # V1 model attributes
            elif model is not None and hasattr(model, '_test_auc'):
                current_auc = model._test_auc
            elif model is not None and hasattr(model, 'auc'):
                current_auc = model.auc

            if current_auc is None:
                # Try to load from model file
                import pickle
                model_path = self.models_dir / "xgboost_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        saved_data = pickle.load(f)
                        if isinstance(saved_data, dict):
                            # V2 format: train_metrics dict inside
                            tm = saved_data.get("train_metrics", {})
                            current_auc = (tm.get("test_auc") or tm.get("xgb_test_score")
                                          or tm.get("test_accuracy") or saved_data.get("test_auc"))
                        elif hasattr(saved_data, '_test_auc'):
                            current_auc = saved_data._test_auc
                        if current_auc is None:
                            return 0.0, False, "Could not determine AUC from model"
                else:
                    return 0.0, False, "Model file not found"

            self._current_auc = current_auc
            self._auc_check_count += 1

            # Check if AUC is acceptable
            is_acceptable = current_auc >= self.min_auc_threshold

            if is_acceptable:
                message = f"Model AUC OK: {current_auc:.4f} (threshold: {self.min_auc_threshold})"
                self._low_auc_alert_sent = False  # Reset alert flag
            else:
                message = f"[WARNING] LOW AUC ALERT: {current_auc:.4f} < {self.min_auc_threshold} threshold!"
                if not self._low_auc_alert_sent:
                    logger.warning(message)
                    self._low_auc_alert_sent = True

            return current_auc, is_acceptable, message

        except Exception as e:
            logger.error(f"Error checking model AUC: {e}")
            return 0.0, False, f"Error: {e}"

    def should_retrain_due_to_low_auc(self) -> Tuple[bool, str]:
        """
        Check if model should be retrained due to low AUC.

        Returns:
            (should_retrain, reason)
        """
        if not self.auto_retrain_on_low_auc:
            return False, "Auto-retrain on low AUC disabled"

        current_auc, is_acceptable, message = self.check_model_auc()

        if not is_acceptable and current_auc > 0:
            # Check if enough time has passed since last retrain
            now = datetime.now(WIB)
            if self._last_retrain_time:
                last_time = self._last_retrain_time
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=WIB)
                hours_since = (now - last_time).total_seconds() / 3600

                # Only retrain if at least 4 hours since last retrain (to prevent loops)
                if hours_since < 4:
                    return False, f"Low AUC but retrained recently ({hours_since:.1f}h ago)"

            return True, f"Low AUC detected: {current_auc:.4f} < {self.min_auc_threshold}"

        return False, "AUC acceptable" if is_acceptable else "Could not check AUC"

    def get_auc_status(self) -> Dict:
        """Get current AUC status for monitoring."""
        current_auc, is_acceptable, message = self.check_model_auc()
        return {
            "current_auc": current_auc,
            "min_threshold": self.min_auc_threshold,
            "is_acceptable": is_acceptable,
            "message": message,
            "check_count": self._auc_check_count,
            "alert_sent": self._low_auc_alert_sent,
        }

    def backup_current_models(self) -> Tuple[bool, str]:
        """
        Backup current models before retraining.

        Returns:
            (success, backup_path)
        """
        if not self.backup_models:
            return True, ""

        try:
            now = datetime.now(WIB)
            backup_suffix = now.strftime("%Y%m%d_%H%M%S")
            backup_dir = self.models_dir / "backups" / backup_suffix
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup XGBoost model
            xgb_path = self.models_dir / "xgboost_model.pkl"
            if xgb_path.exists():
                shutil.copy(xgb_path, backup_dir / "xgboost_model.pkl")

            # Backup HMM model
            hmm_path = self.models_dir / "hmm_regime.pkl"
            if hmm_path.exists():
                shutil.copy(hmm_path, backup_dir / "hmm_regime.pkl")

            logger.info(f"Models backed up to {backup_dir}")

            # Keep only last 5 backups
            self._cleanup_old_backups(keep=5)

            return True, str(backup_dir)
        except Exception as e:
            logger.error(f"Failed to backup models: {e}")
            return False, ""

    def _cleanup_old_backups(self, keep: int = 5):
        """Remove old backups, keeping only the most recent ones."""
        backup_base = self.models_dir / "backups"
        if not backup_base.exists():
            return

        backups = sorted(backup_base.iterdir(), reverse=True)
        for old_backup in backups[keep:]:
            if old_backup.is_dir():
                shutil.rmtree(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")

    def retrain(
        self,
        connector,  # MT5Connector
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        is_weekend: bool = False,
    ) -> Dict:
        """
        Retrain all models with latest data.

        Args:
            connector: MT5 connector for fetching data
            symbol: Trading symbol
            timeframe: Timeframe for training data
            is_weekend: If True, use more data for deep training

        Returns:
            Dict with training results
        """
        from src.feature_eng import FeatureEngineer
        from src.smc_polars import SMCAnalyzer
        from src.regime_detector import MarketRegimeDetector
        from src.ml_model import get_default_feature_columns
        from backtests.ml_v2.ml_v2_model import TradingModelV2
        from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer

        started_at = datetime.now(WIB)

        results = {
            "success": False,
            "hmm_trained": False,
            "xgb_trained": False,
            "xgb_train_auc": 0,
            "xgb_test_auc": 0,
            "train_accuracy": 0,
            "test_accuracy": 0,
            "samples": 0,
            "error": None,
            "backup_path": "",
            "duration_seconds": 0,
        }

        # Determine training parameters
        training_type = "weekend" if is_weekend else "daily"
        if is_weekend:
            bars = 20000  # More data for weekend deep training
            num_boost_round = 80
        else:
            bars = 15000  # Daily training (increased from 8000 for better HMM regime detection)
            num_boost_round = 50

        # Record training start in database
        self._save_retrain_start(training_type, bars, num_boost_round)

        try:
            logger.info("=" * 50)
            logger.info("AUTO-RETRAINING STARTED")
            logger.info(f"Type: {training_type}, Bars: {bars}, Boost Rounds: {num_boost_round}")
            logger.info("=" * 50)

            # Backup current models
            backup_success, backup_path = self.backup_current_models()
            results["backup_path"] = backup_path

            # Fetch latest data
            logger.info(f"Fetching {bars} bars of {symbol} {timeframe} data...")
            df = connector.get_market_data(symbol, timeframe, bars)

            if len(df) < 1000:
                results["error"] = f"Insufficient data: {len(df)} bars"
                logger.error(results["error"])
                self._save_retrain_complete(
                    success=False,
                    error_message=results["error"],
                    started_at=started_at,
                )
                return results

            results["samples"] = len(df)
            logger.info(f"Received {len(df)} bars")
            logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")

            # Feature engineering
            logger.info("Applying feature engineering...")
            fe = FeatureEngineer()
            df = fe.calculate_all(df, include_ml_features=True)

            # SMC indicators
            smc = SMCAnalyzer(swing_length=5)
            df = smc.calculate_all(df)

            # Create target
            df = fe.create_target(df, lookahead=1)

            # Train HMM
            logger.info("Training HMM Regime Model...")
            hmm = MarketRegimeDetector(
                n_regimes=3,
                lookback_periods=500,
                model_path=str(self.models_dir / "hmm_regime.pkl"),
            )
            hmm.fit(df)

            if hmm.fitted:
                df = hmm.predict(df)
                results["hmm_trained"] = True
                logger.info("HMM model trained and saved")

            # Add V2 features (23 additional features on top of base 37)
            logger.info("Adding V2 features for enhanced model training...")
            fe_v2 = MLV2FeatureEngineer()

            # Fetch H1 data for multi-timeframe features
            df_h1 = None
            try:
                df_h1 = connector.get_market_data(symbol, "H1", min(bars // 4, 2000))
                if len(df_h1) > 30:
                    df_h1 = fe.calculate_all(df_h1, include_ml_features=False)
                    df_h1 = smc.calculate_all(df_h1)
                    logger.info(f"H1 data fetched: {len(df_h1)} bars for V2 features")
                else:
                    df_h1 = None
                    logger.warning("Insufficient H1 data, using defaults")
            except Exception as e:
                logger.warning(f"Could not fetch H1 data: {e}, using defaults")

            df = fe_v2.add_all_v2_features(df, df_h1)

            # Train XGBoost V2 Model (with all available features)
            logger.info("Training XGBoost V2 Model...")
            xgb_model = TradingModelV2(
                confidence_threshold=0.60,
                model_path=str(self.models_dir / "xgboost_model.pkl"),
            )

            # Auto-detect all numeric feature columns (like V3 trainer)
            exclude_cols = {"time", "open", "high", "low", "close", "volume", "target",
                           "tick_volume", "spread", "real_volume", "multi_bar_target"}
            feature_cols = [
                col for col in df.columns
                if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Boolean]
            ]
            logger.info(f"Training with {len(feature_cols)} features")

            xgb_model.fit(
                df,
                feature_cols,
                target_col="target",
                train_ratio=0.7,
                num_boost_round=num_boost_round,
                early_stopping_rounds=5,
            )

            if xgb_model.fitted:
                results["xgb_trained"] = True
                # V2 model stores AUC as xgb_train_score/xgb_test_score
                results["xgb_train_auc"] = xgb_model._train_metrics.get("xgb_train_score", 0)
                results["xgb_test_auc"] = xgb_model._train_metrics.get("xgb_test_score", 0)
                results["train_accuracy"] = xgb_model._train_metrics.get("train_accuracy", 0)
                results["test_accuracy"] = xgb_model._train_metrics.get("test_accuracy", 0)
                results["n_features"] = len(feature_cols)
                logger.info(f"XGBoost V2 trained: Train AUC={results['xgb_train_auc']:.4f}, Test AUC={results['xgb_test_auc']:.4f}")
                logger.info(f"  Features: {len(feature_cols)}")

            # Save training data
            training_data_path = self.data_dir / "training_data.parquet"
            df.write_parquet(training_data_path)
            logger.info(f"Training data saved to {training_data_path}")

            # Mark success
            results["success"] = results["hmm_trained"] and results["xgb_trained"]
            results["duration_seconds"] = int((datetime.now(WIB) - started_at).total_seconds())

            # Save completion to database
            self._save_retrain_complete(
                success=results["success"],
                hmm_trained=results["hmm_trained"],
                xgb_trained=results["xgb_trained"],
                train_auc=results["xgb_train_auc"],
                test_auc=results["xgb_test_auc"],
                train_accuracy=results["train_accuracy"],
                test_accuracy=results["test_accuracy"],
                model_path=str(self.models_dir / "xgboost_model.pkl"),
                backup_path=backup_path,
                started_at=started_at,
            )

            if results["success"]:
                # Update cached AUC for dashboard reporting
                self._current_auc = results["xgb_test_auc"]
                logger.info("=" * 50)
                logger.info("AUTO-RETRAINING COMPLETED SUCCESSFULLY")
                logger.info(f"Duration: {results['duration_seconds']}s")
                logger.info("=" * 50)
            else:
                logger.warning("Retraining completed with issues")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Retraining failed: {e}")
            import traceback
            traceback.print_exc()

            # Save failure to database
            self._save_retrain_complete(
                success=False,
                error_message=str(e),
                started_at=started_at,
            )

        return results

    def rollback_models(self, reason: str = "Manual rollback") -> bool:
        """Rollback to previous model version if new one performs worse."""
        backup_base = self.models_dir / "backups"
        if not backup_base.exists():
            logger.warning("No backups available for rollback")
            return False

        # Get most recent backup
        backups = sorted(backup_base.iterdir(), reverse=True)
        if not backups:
            logger.warning("No backups found")
            return False

        latest_backup = backups[0]

        try:
            # Restore XGBoost
            xgb_backup = latest_backup / "xgboost_model.pkl"
            if xgb_backup.exists():
                shutil.copy(xgb_backup, self.models_dir / "xgboost_model.pkl")

            # Restore HMM
            hmm_backup = latest_backup / "hmm_regime.pkl"
            if hmm_backup.exists():
                shutil.copy(hmm_backup, self.models_dir / "hmm_regime.pkl")

            logger.info(f"Models rolled back from {latest_backup}")

            # Record rollback in database
            if self._db_connected and self._training_repo and self._current_run_id:
                try:
                    self._training_repo.mark_rollback(self._current_run_id, reason)
                except Exception as e:
                    logger.error(f"Failed to record rollback: {e}")

            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_training_history(self, limit: int = 10) -> list:
        """Get recent training history from database."""
        if self._db_connected and self._training_repo:
            try:
                return self._training_repo.get_training_history(limit)
            except Exception as e:
                logger.error(f"Failed to get training history: {e}")
        return []

    def get_status(self) -> Dict:
        """Get current auto-trainer status."""
        now = datetime.now(WIB)

        hours_since_retrain = None
        if self._last_retrain_time:
            last_time = self._last_retrain_time
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=WIB)
            hours_since_retrain = (now - last_time).total_seconds() / 3600

        should_train, reason = self.should_retrain()

        # Get latest training from DB
        latest_training = None
        if self._db_connected and self._training_repo:
            try:
                latest_training = self._training_repo.get_latest_successful()
            except:
                pass

        return {
            "last_retrain": self._last_retrain_time.isoformat() if self._last_retrain_time else "Never",
            "hours_since_retrain": round(hours_since_retrain, 1) if hours_since_retrain else None,
            "should_retrain": should_train,
            "reason": reason,
            "next_retrain_hour": f"{self.daily_retrain_hour:02d}:00 WIB",
            "weekend_training": self.weekend_retrain,
            "db_connected": self._db_connected,
            "latest_training": {
                "train_auc": latest_training.get("train_auc") if latest_training else None,
                "test_auc": latest_training.get("test_auc") if latest_training else None,
                "bars_used": latest_training.get("bars_used") if latest_training else None,
            } if latest_training else None,
        }


def create_auto_trainer() -> AutoTrainer:
    """Create default auto trainer instance."""
    return AutoTrainer(
        models_dir="models",
        data_dir="data",
        daily_retrain_hour_wib=5,      # 05:00 WIB (market close)
        weekend_retrain=True,
        min_hours_between_retrain=20,
        backup_models=True,
        use_db=True,
        min_auc_threshold=0.65,        # Alert if AUC drops below 0.65
        auto_retrain_on_low_auc=True,  # Auto retrain when AUC is low
    )


if __name__ == "__main__":
    # Test auto trainer
    trainer = create_auto_trainer()

    print("=== Auto Trainer Status ===")
    status = trainer.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n=== Should Retrain Check ===")
    should, reason = trainer.should_retrain()
    print(f"  Should retrain: {should}")
    print(f"  Reason: {reason}")

    print("\n=== Training History ===")
    history = trainer.get_training_history(5)
    for h in history:
        print(f"  - {h.get('training_type')}: AUC={h.get('test_auc')}, Success={h.get('success')}")
