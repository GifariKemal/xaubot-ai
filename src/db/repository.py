"""
Database Repository Module
==========================
Data access layer for PostgreSQL operations.

Repositories:
- TradeRepository: Trade CRUD operations
- TrainingRepository: ML training history
- SignalRepository: Signal logging
- MarketSnapshotRepository: Market state snapshots
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, date
from decimal import Decimal
import json

from loguru import logger

from .connection import DatabaseConnection


class TradeRepository:
    """Repository for trade operations."""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def insert_trade(self, trade_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Insert a new trade record.

        Args:
            trade_data: Trade information dict

        Returns:
            Inserted trade record with ID
        """
        query = """
        INSERT INTO trades (
            ticket, symbol, direction,
            entry_price, stop_loss, take_profit, lot_size,
            opened_at,
            entry_regime, entry_volatility, entry_session, entry_spread, entry_atr,
            smc_signal, smc_confidence, smc_reason,
            smc_fvg_detected, smc_ob_detected, smc_bos_detected, smc_choch_detected,
            ml_signal, ml_confidence,
            market_quality, market_score, dynamic_threshold,
            balance_before, equity_at_entry,
            features_entry, bot_version, trade_mode
        ) VALUES (
            %(ticket)s, %(symbol)s, %(direction)s,
            %(entry_price)s, %(stop_loss)s, %(take_profit)s, %(lot_size)s,
            %(opened_at)s,
            %(entry_regime)s, %(entry_volatility)s, %(entry_session)s, %(entry_spread)s, %(entry_atr)s,
            %(smc_signal)s, %(smc_confidence)s, %(smc_reason)s,
            %(smc_fvg_detected)s, %(smc_ob_detected)s, %(smc_bos_detected)s, %(smc_choch_detected)s,
            %(ml_signal)s, %(ml_confidence)s,
            %(market_quality)s, %(market_score)s, %(dynamic_threshold)s,
            %(balance_before)s, %(equity_at_entry)s,
            %(features_entry)s, %(bot_version)s, %(trade_mode)s
        )
        RETURNING *
        """

        # Set defaults
        defaults = {
            'symbol': 'XAUUSD',
            'stop_loss': 0,
            'take_profit': 0,
            'opened_at': datetime.now(),
            'entry_regime': None,
            'entry_volatility': None,
            'entry_session': None,
            'entry_spread': None,
            'entry_atr': None,
            'smc_signal': None,
            'smc_confidence': None,
            'smc_reason': None,
            'smc_fvg_detected': False,
            'smc_ob_detected': False,
            'smc_bos_detected': False,
            'smc_choch_detected': False,
            'ml_signal': None,
            'ml_confidence': None,
            'market_quality': None,
            'market_score': None,
            'dynamic_threshold': None,
            'balance_before': None,
            'equity_at_entry': None,
            'features_entry': '{}',
            'bot_version': '2.1',
            'trade_mode': 'SMC-ONLY',
        }

        # Merge defaults with provided data
        params = {**defaults, **trade_data}

        # Convert features to JSON string if dict
        if isinstance(params.get('features_entry'), dict):
            params['features_entry'] = json.dumps(params['features_entry'])

        try:
            result = self.db.insert_returning(query, params)
            logger.info(f"Trade inserted: ticket={params['ticket']}")
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            raise

    def update_trade_close(self, ticket: int, close_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Update trade with close information.

        Args:
            ticket: Trade ticket number
            close_data: Close information dict

        Returns:
            Updated trade record
        """
        query = """
        UPDATE trades SET
            exit_price = %(exit_price)s,
            profit_usd = %(profit_usd)s,
            profit_pips = %(profit_pips)s,
            closed_at = %(closed_at)s,
            duration_seconds = %(duration_seconds)s,
            exit_reason = %(exit_reason)s,
            exit_regime = %(exit_regime)s,
            exit_ml_signal = %(exit_ml_signal)s,
            exit_ml_confidence = %(exit_ml_confidence)s,
            balance_after = %(balance_after)s,
            features_exit = %(features_exit)s
        WHERE ticket = %(ticket)s
        RETURNING *
        """

        defaults = {
            'closed_at': datetime.now(),
            'duration_seconds': None,
            'exit_reason': None,
            'exit_regime': None,
            'exit_ml_signal': None,
            'exit_ml_confidence': None,
            'balance_after': None,
            'features_exit': '{}',
        }

        params = {**defaults, **close_data, 'ticket': ticket}

        # Convert features to JSON string if dict
        if isinstance(params.get('features_exit'), dict):
            params['features_exit'] = json.dumps(params['features_exit'])

        try:
            result = self.db.insert_returning(query, params)
            logger.info(f"Trade closed: ticket={ticket}, profit={close_data.get('profit_usd')}")
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to update trade close: {e}")
            raise

    def get_trade_by_ticket(self, ticket: int) -> Optional[Dict]:
        """Get trade by ticket number."""
        query = "SELECT * FROM trades WHERE ticket = %s"
        result = self.db.execute(query, (ticket,), fetch=True)
        return dict(result[0]) if result else None

    def get_open_trades(self) -> List[Dict]:
        """Get all trades that haven't been closed."""
        query = """
        SELECT * FROM trades
        WHERE closed_at IS NULL
        ORDER BY opened_at DESC
        """
        result = self.db.execute(query, fetch=True)
        return [dict(r) for r in result] if result else []

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent closed trades."""
        query = """
        SELECT * FROM trades
        WHERE closed_at IS NOT NULL
        ORDER BY closed_at DESC
        LIMIT %s
        """
        result = self.db.execute(query, (limit,), fetch=True)
        return [dict(r) for r in result] if result else []

    def get_trades_for_training(self, days: int = 30) -> List[Dict]:
        """
        Get trades suitable for ML training.

        Args:
            days: Number of days to look back

        Returns:
            List of closed trades with features
        """
        query = """
        SELECT * FROM trades
        WHERE closed_at IS NOT NULL
          AND closed_at >= NOW() - INTERVAL '%s days'
          AND features_entry IS NOT NULL
        ORDER BY closed_at ASC
        """
        result = self.db.execute(query, (days,), fetch=True)
        return [dict(r) for r in result] if result else []

    def get_daily_stats(self, trade_date: date) -> Dict[str, Any]:
        """Get statistics for a specific date."""
        query = """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_usd) as net_profit,
            AVG(profit_usd) as avg_profit,
            MAX(profit_usd) as max_profit,
            MIN(profit_usd) as min_profit
        FROM trades
        WHERE DATE(closed_at) = %s
        """
        result = self.db.execute(query, (trade_date,), fetch=True)
        return dict(result[0]) if result else {}

    def get_session_stats(self, session: str, days: int = 30) -> Dict[str, Any]:
        """Get statistics for a specific trading session."""
        query = """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END) as wins,
            SUM(profit_usd) as net_profit,
            AVG(profit_usd) as avg_profit
        FROM trades
        WHERE entry_session = %s
          AND closed_at >= NOW() - INTERVAL '%s days'
        """
        result = self.db.execute(query, (session, days), fetch=True)
        return dict(result[0]) if result else {}

    def get_smc_pattern_stats(self, days: int = 30) -> List[Dict]:
        """Get statistics grouped by SMC pattern."""
        query = """
        SELECT
            CASE
                WHEN smc_fvg_detected THEN 'FVG'
                WHEN smc_ob_detected THEN 'OB'
                WHEN smc_bos_detected THEN 'BOS'
                WHEN smc_choch_detected THEN 'CHoCH'
                ELSE 'OTHER'
            END as pattern,
            COUNT(*) as total,
            SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END) as wins,
            SUM(profit_usd) as profit
        FROM trades
        WHERE closed_at IS NOT NULL
          AND closed_at >= NOW() - INTERVAL '%s days'
        GROUP BY pattern
        ORDER BY total DESC
        """
        result = self.db.execute(query, (days,), fetch=True)
        return [dict(r) for r in result] if result else []


class TrainingRepository:
    """Repository for ML training history."""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def insert_training_run(self, training_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Record a new training run.

        Args:
            training_data: Training run information

        Returns:
            Inserted record with ID
        """
        query = """
        INSERT INTO training_runs (
            training_type, bars_used, num_boost_rounds,
            hmm_trained, hmm_n_regimes,
            xgb_trained, train_auc, test_auc, train_accuracy, test_accuracy,
            model_path, backup_path,
            success, error_message,
            started_at, completed_at, duration_seconds
        ) VALUES (
            %(training_type)s, %(bars_used)s, %(num_boost_rounds)s,
            %(hmm_trained)s, %(hmm_n_regimes)s,
            %(xgb_trained)s, %(train_auc)s, %(test_auc)s, %(train_accuracy)s, %(test_accuracy)s,
            %(model_path)s, %(backup_path)s,
            %(success)s, %(error_message)s,
            %(started_at)s, %(completed_at)s, %(duration_seconds)s
        )
        RETURNING *
        """

        defaults = {
            'training_type': 'manual',
            'bars_used': None,
            'num_boost_rounds': None,
            'hmm_trained': False,
            'hmm_n_regimes': 3,
            'xgb_trained': False,
            'train_auc': None,
            'test_auc': None,
            'train_accuracy': None,
            'test_accuracy': None,
            'model_path': None,
            'backup_path': None,
            'success': False,
            'error_message': None,
            'started_at': datetime.now(),
            'completed_at': None,
            'duration_seconds': None,
        }

        params = {**defaults, **training_data}

        try:
            result = self.db.insert_returning(query, params)
            logger.info(f"Training run recorded: type={params['training_type']}")
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to insert training run: {e}")
            raise

    def update_training_complete(self, run_id: int, result_data: Dict[str, Any]) -> Optional[Dict]:
        """Update training run with completion data."""
        query = """
        UPDATE training_runs SET
            completed_at = %(completed_at)s,
            duration_seconds = %(duration_seconds)s,
            hmm_trained = %(hmm_trained)s,
            xgb_trained = %(xgb_trained)s,
            train_auc = %(train_auc)s,
            test_auc = %(test_auc)s,
            train_accuracy = %(train_accuracy)s,
            test_accuracy = %(test_accuracy)s,
            model_path = %(model_path)s,
            backup_path = %(backup_path)s,
            success = %(success)s,
            error_message = %(error_message)s
        WHERE id = %(id)s
        RETURNING *
        """

        params = {**result_data, 'id': run_id}

        try:
            result = self.db.insert_returning(query, params)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to update training run: {e}")
            raise

    def mark_rollback(self, run_id: int, reason: str) -> bool:
        """Mark a training run as rolled back."""
        query = """
        UPDATE training_runs SET
            rolled_back = TRUE,
            rollback_reason = %s,
            rollback_at = NOW()
        WHERE id = %s
        """
        try:
            self.db.execute(query, (reason, run_id))
            return True
        except Exception as e:
            logger.error(f"Failed to mark rollback: {e}")
            return False

    def get_latest_successful(self) -> Optional[Dict]:
        """Get the most recent successful training run."""
        query = """
        SELECT * FROM training_runs
        WHERE success = TRUE AND rolled_back = FALSE
        ORDER BY completed_at DESC
        LIMIT 1
        """
        result = self.db.execute(query, fetch=True)
        return dict(result[0]) if result else None

    def get_training_history(self, limit: int = 20) -> List[Dict]:
        """Get recent training runs."""
        query = """
        SELECT * FROM training_runs
        ORDER BY started_at DESC
        LIMIT %s
        """
        result = self.db.execute(query, (limit,), fetch=True)
        return [dict(r) for r in result] if result else []


class SignalRepository:
    """Repository for trading signals."""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def insert_signal(self, signal_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Record a trading signal.

        Args:
            signal_data: Signal information

        Returns:
            Inserted record
        """
        query = """
        INSERT INTO signals (
            signal_time, symbol, price,
            signal_type, signal_source, combined_confidence,
            smc_signal, smc_confidence, smc_fvg, smc_ob, smc_bos, smc_choch, smc_reason,
            ml_signal, ml_confidence,
            regime, session, volatility, market_score, dynamic_threshold,
            executed, execution_reason, trade_ticket
        ) VALUES (
            %(signal_time)s, %(symbol)s, %(price)s,
            %(signal_type)s, %(signal_source)s, %(combined_confidence)s,
            %(smc_signal)s, %(smc_confidence)s, %(smc_fvg)s, %(smc_ob)s, %(smc_bos)s, %(smc_choch)s, %(smc_reason)s,
            %(ml_signal)s, %(ml_confidence)s,
            %(regime)s, %(session)s, %(volatility)s, %(market_score)s, %(dynamic_threshold)s,
            %(executed)s, %(execution_reason)s, %(trade_ticket)s
        )
        RETURNING *
        """

        defaults = {
            'signal_time': datetime.now(),
            'symbol': 'XAUUSD',
            'signal_source': 'SMC-ONLY',
            'combined_confidence': None,
            'smc_signal': None,
            'smc_confidence': None,
            'smc_fvg': False,
            'smc_ob': False,
            'smc_bos': False,
            'smc_choch': False,
            'smc_reason': None,
            'ml_signal': None,
            'ml_confidence': None,
            'regime': None,
            'session': None,
            'volatility': None,
            'market_score': None,
            'dynamic_threshold': None,
            'executed': False,
            'execution_reason': None,
            'trade_ticket': None,
        }

        params = {**defaults, **signal_data}

        try:
            result = self.db.insert_returning(query, params)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to insert signal: {e}")
            raise

    def mark_executed(self, signal_id: int, ticket: int) -> bool:
        """Mark a signal as executed with trade ticket."""
        query = """
        UPDATE signals SET
            executed = TRUE,
            trade_ticket = %s
        WHERE id = %s
        """
        try:
            self.db.execute(query, (ticket, signal_id))
            return True
        except Exception as e:
            logger.error(f"Failed to mark signal executed: {e}")
            return False

    def get_recent_signals(self, limit: int = 100) -> List[Dict]:
        """Get recent signals."""
        query = """
        SELECT * FROM signals
        ORDER BY signal_time DESC
        LIMIT %s
        """
        result = self.db.execute(query, (limit,), fetch=True)
        return [dict(r) for r in result] if result else []

    def get_signal_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get signal statistics for recent period."""
        query = """
        SELECT
            COUNT(*) as total_signals,
            SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
            SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
            SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_signals,
            AVG(smc_confidence) as avg_smc_confidence,
            AVG(ml_confidence) as avg_ml_confidence
        FROM signals
        WHERE signal_time >= NOW() - INTERVAL '%s hours'
        """
        result = self.db.execute(query, (hours,), fetch=True)
        return dict(result[0]) if result else {}


class MarketSnapshotRepository:
    """Repository for market state snapshots."""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def insert_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[Dict]:
        """
        Record a market snapshot.

        Args:
            snapshot_data: Market state information

        Returns:
            Inserted record
        """
        query = """
        INSERT INTO market_snapshots (
            snapshot_time, symbol, price,
            open_price, high_price, low_price, close_price,
            regime, volatility, session, atr, spread,
            ml_signal, ml_confidence, smc_signal, smc_confidence,
            open_positions, floating_pnl,
            features
        ) VALUES (
            %(snapshot_time)s, %(symbol)s, %(price)s,
            %(open_price)s, %(high_price)s, %(low_price)s, %(close_price)s,
            %(regime)s, %(volatility)s, %(session)s, %(atr)s, %(spread)s,
            %(ml_signal)s, %(ml_confidence)s, %(smc_signal)s, %(smc_confidence)s,
            %(open_positions)s, %(floating_pnl)s,
            %(features)s
        )
        ON CONFLICT (snapshot_time, symbol) DO UPDATE SET
            price = EXCLUDED.price,
            regime = EXCLUDED.regime,
            ml_signal = EXCLUDED.ml_signal,
            ml_confidence = EXCLUDED.ml_confidence
        RETURNING *
        """

        defaults = {
            'snapshot_time': datetime.now(),
            'symbol': 'XAUUSD',
            'open_price': None,
            'high_price': None,
            'low_price': None,
            'close_price': None,
            'regime': None,
            'volatility': None,
            'session': None,
            'atr': None,
            'spread': None,
            'ml_signal': None,
            'ml_confidence': None,
            'smc_signal': None,
            'smc_confidence': None,
            'open_positions': 0,
            'floating_pnl': 0,
            'features': '{}',
        }

        params = {**defaults, **snapshot_data}

        # Convert features to JSON string if dict
        if isinstance(params.get('features'), dict):
            params['features'] = json.dumps(params['features'])

        try:
            result = self.db.insert_returning(query, params)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to insert snapshot: {e}")
            raise

    def get_recent_snapshots(self, minutes: int = 60) -> List[Dict]:
        """Get snapshots from recent period."""
        query = """
        SELECT * FROM market_snapshots
        WHERE snapshot_time >= NOW() - INTERVAL '%s minutes'
        ORDER BY snapshot_time DESC
        """
        result = self.db.execute(query, (minutes,), fetch=True)
        return [dict(r) for r in result] if result else []


class BotStatusRepository:
    """Repository for bot health status."""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def insert_status(self, status_data: Dict[str, Any]) -> Optional[Dict]:
        """Record bot status."""
        query = """
        INSERT INTO bot_status (
            status_time, is_running, status,
            loop_count, avg_execution_ms, uptime_seconds,
            balance, equity, margin_used,
            open_positions, floating_pnl,
            daily_pnl, risk_mode,
            current_session, is_golden_time,
            last_error, last_error_at
        ) VALUES (
            %(status_time)s, %(is_running)s, %(status)s,
            %(loop_count)s, %(avg_execution_ms)s, %(uptime_seconds)s,
            %(balance)s, %(equity)s, %(margin_used)s,
            %(open_positions)s, %(floating_pnl)s,
            %(daily_pnl)s, %(risk_mode)s,
            %(current_session)s, %(is_golden_time)s,
            %(last_error)s, %(last_error_at)s
        )
        RETURNING *
        """

        defaults = {
            'status_time': datetime.now(),
            'is_running': True,
            'status': 'active',
            'loop_count': 0,
            'avg_execution_ms': None,
            'uptime_seconds': 0,
            'balance': None,
            'equity': None,
            'margin_used': None,
            'open_positions': 0,
            'floating_pnl': 0,
            'daily_pnl': None,
            'risk_mode': 'normal',
            'current_session': None,
            'is_golden_time': False,
            'last_error': None,
            'last_error_at': None,
        }

        params = {**defaults, **status_data}

        try:
            result = self.db.insert_returning(query, params)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to insert bot status: {e}")
            raise

    def get_latest_status(self) -> Optional[Dict]:
        """Get most recent bot status."""
        query = """
        SELECT * FROM bot_status
        ORDER BY status_time DESC
        LIMIT 1
        """
        result = self.db.execute(query, fetch=True)
        return dict(result[0]) if result else None


class DailySummaryRepository:
    """Repository for daily performance summaries."""

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def upsert_summary(self, summary_date: date, summary_data: Dict[str, Any]) -> Optional[Dict]:
        """Insert or update daily summary."""
        query = """
        INSERT INTO daily_summaries (
            summary_date,
            total_trades, winning_trades, losing_trades, breakeven_trades,
            gross_profit, gross_loss, net_profit,
            start_balance, end_balance,
            win_rate, profit_factor, average_win, average_loss,
            largest_win, largest_loss,
            trades_sydney, trades_tokyo, trades_london, trades_ny, trades_golden,
            fvg_trades, fvg_wins, ob_trades, ob_wins
        ) VALUES (
            %(summary_date)s,
            %(total_trades)s, %(winning_trades)s, %(losing_trades)s, %(breakeven_trades)s,
            %(gross_profit)s, %(gross_loss)s, %(net_profit)s,
            %(start_balance)s, %(end_balance)s,
            %(win_rate)s, %(profit_factor)s, %(average_win)s, %(average_loss)s,
            %(largest_win)s, %(largest_loss)s,
            %(trades_sydney)s, %(trades_tokyo)s, %(trades_london)s, %(trades_ny)s, %(trades_golden)s,
            %(fvg_trades)s, %(fvg_wins)s, %(ob_trades)s, %(ob_wins)s
        )
        ON CONFLICT (summary_date) DO UPDATE SET
            total_trades = EXCLUDED.total_trades,
            winning_trades = EXCLUDED.winning_trades,
            losing_trades = EXCLUDED.losing_trades,
            net_profit = EXCLUDED.net_profit,
            end_balance = EXCLUDED.end_balance,
            win_rate = EXCLUDED.win_rate,
            updated_at = NOW()
        RETURNING *
        """

        defaults = {
            'summary_date': summary_date,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'start_balance': None,
            'end_balance': None,
            'win_rate': None,
            'profit_factor': None,
            'average_win': None,
            'average_loss': None,
            'largest_win': None,
            'largest_loss': None,
            'trades_sydney': 0,
            'trades_tokyo': 0,
            'trades_london': 0,
            'trades_ny': 0,
            'trades_golden': 0,
            'fvg_trades': 0,
            'fvg_wins': 0,
            'ob_trades': 0,
            'ob_wins': 0,
        }

        params = {**defaults, **summary_data}

        try:
            result = self.db.insert_returning(query, params)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to upsert daily summary: {e}")
            raise

    def get_summary(self, summary_date: date) -> Optional[Dict]:
        """Get summary for specific date."""
        query = "SELECT * FROM daily_summaries WHERE summary_date = %s"
        result = self.db.execute(query, (summary_date,), fetch=True)
        return dict(result[0]) if result else None

    def get_recent_summaries(self, days: int = 30) -> List[Dict]:
        """Get recent daily summaries."""
        query = """
        SELECT * FROM daily_summaries
        ORDER BY summary_date DESC
        LIMIT %s
        """
        result = self.db.execute(query, (days,), fetch=True)
        return [dict(r) for r in result] if result else []
