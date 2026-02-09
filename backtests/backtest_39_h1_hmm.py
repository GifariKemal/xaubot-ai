#!/usr/bin/env python3
"""
Backtest #39: H1 HMM Regime Detector

Test moving HMM from M15 to H1 timeframe for more stable regime detection.

Hypothesis:
- H1 HMM will have 4-8x longer regime duration
- Fewer regime transitions = more stable risk management
- Better regime classification due to less noise

Expected Impact: +15-20% Sharpe improvement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.mt5_connector import MT5Connector
from src.config import TradingConfig, get_config
from src.feature_eng import FeatureEngineer
from src.smc_polars import SMCAnalyzer
from src.regime_detector import MarketRegimeDetector
from backtests.ml_v2.ml_v2_model import TradingModelV2
from src.smart_risk_manager import SmartRiskManager
from src.session_filter import SessionFilter
from src.dynamic_confidence import DynamicConfidenceManager

# Import V2 features
from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    symbol: str = "XAUUSD"
    m15_bars: int = 5000
    h1_bars: int = 1500
    initial_balance: float = 500.0
    results_dir: str = "backtests/39_h1_hmm_results"


class H1HMMBacktest:
    """Backtest with H1-based HMM regime detector"""

    def __init__(self, config: BacktestConfig, use_h1_hmm: bool = True):
        self.config = config
        self.use_h1_hmm = use_h1_hmm
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.trades = []
        self.equity_curve = []

        # Trading config
        self.trading_config = TradingConfig()

        # Components
        self.mt5 = None
        self.features = FeatureEngineer()
        self.smc = SMCAnalyzer()
        self.ml_model = TradingModelV2()
        self.fe_v2 = MLV2FeatureEngineer()

        # Skip risk manager and session filter for backtest simplicity
        self.risk_manager = None
        self.session_filter = None
        self.dynamic_conf = None

        # Regime detectors - pass individual params (API changed)
        self.regime_m15 = MarketRegimeDetector(
            n_regimes=3,
            lookback_periods=500,
            retrain_frequency=20
        )
        self.regime_h1 = MarketRegimeDetector(
            n_regimes=3,
            lookback_periods=500,
            retrain_frequency=20
        )

        # Stats
        self.regime_changes_m15 = []
        self.regime_changes_h1 = []

    def connect_mt5(self):
        """Connect to MT5"""
        logger.info("Connecting to MT5...")
        config = get_config()
        self.mt5 = MT5Connector(
            login=config.mt5_login,
            password=config.mt5_password,
            server=config.mt5_server,
            path=config.mt5_path
        )
        if not self.mt5.connect():
            raise RuntimeError("Failed to connect to MT5")
        logger.info("✓ MT5 connected")

    def fetch_data(self):
        """Fetch M15 and H1 data"""
        logger.info(f"Fetching {self.config.m15_bars} M15 bars...")
        df_m15 = self.mt5.get_market_data(
            self.config.symbol, "M15", self.config.m15_bars
        )

        logger.info(f"Fetching {self.config.h1_bars} H1 bars...")
        df_h1 = self.mt5.get_market_data(
            self.config.symbol, "H1", self.config.h1_bars
        )

        logger.info(f"✓ M15: {len(df_m15)} bars, H1: {len(df_h1)} bars")
        return df_m15, df_h1

    def prepare_data(self, df_m15: pl.DataFrame, df_h1: pl.DataFrame):
        """Calculate all features"""
        logger.info("Calculating M15 features...")
        df_m15 = self.features.calculate_all(df_m15, include_ml_features=True)
        df_m15 = self.smc.calculate_all(df_m15)

        logger.info("Calculating H1 features...")
        df_h1 = self.features.calculate_all(df_h1, include_ml_features=True)
        df_h1 = self.smc.calculate_all(df_h1)

        logger.info("Adding V2 features...")
        df_m15 = self.fe_v2.add_all_v2_features(df_m15, df_h1)

        return df_m15, df_h1

    def fit_regimes(self, df_m15: pl.DataFrame, df_h1: pl.DataFrame):
        """Train both M15 and H1 HMM models"""
        logger.info("Training M15 HMM (baseline)...")
        self.regime_m15.fit(df_m15.slice(0, 500))

        logger.info("Training H1 HMM (test)...")
        self.regime_h1.fit(df_h1.slice(0, 500))

        logger.info("✓ Both HMM models fited")

    def detect_regime(self, df_m15: pl.DataFrame, df_h1: pl.DataFrame, m15_idx: int):
        """Detect regime using M15 or H1 and return updated df with regime columns"""
        if self.use_h1_hmm:
            # Use H1 regime (map M15 index to H1)
            h1_idx = m15_idx // 4  # 4 M15 bars = 1 H1 bar
            if h1_idx >= len(df_h1):
                h1_idx = len(df_h1) - 1

            df_h1_slice = df_h1.slice(max(0, h1_idx - 100), h1_idx + 1)
            df_h1_pred = self.regime_h1.predict(df_h1_slice)
            regime_state = self.regime_h1.get_current_state(df_h1_pred)

            # Track H1 regime changes
            if hasattr(self, '_last_h1_regime') and self._last_h1_regime != regime_state.regime.value:
                self.regime_changes_h1.append({
                    'm15_idx': m15_idx,
                    'old': self._last_h1_regime,
                    'new': regime_state.regime.value
                })
            self._last_h1_regime = regime_state.regime.value
        else:
            # Use M15 regime (baseline)
            df_m15_slice = df_m15.slice(max(0, m15_idx - 100), m15_idx + 1)
            df_m15_pred = self.regime_m15.predict(df_m15_slice)
            regime_state = self.regime_m15.get_current_state(df_m15_pred)

            # Track M15 regime changes
            if hasattr(self, '_last_m15_regime') and self._last_m15_regime != regime_state.regime.value:
                self.regime_changes_m15.append({
                    'm15_idx': m15_idx,
                    'old': self._last_m15_regime,
                    'new': regime_state.regime.value
                })
            self._last_m15_regime = regime_state.regime.value

        return regime_state, df_m15_pred if not self.use_h1_hmm else df_h1_pred

    def run_backtest(self, df_m15: pl.DataFrame, df_h1: pl.DataFrame):
        """Run backtest loop"""
        logger.info(f"Running backtest ({'H1 HMM' if self.use_h1_hmm else 'M15 HMM'})...")

        # Load ML model
        self.ml_model.load("models/xgboost_model_v2d.pkl")

        # Start from bar 600 (after fiting window)
        for idx in range(600, len(df_m15)):
            # Get current bar
            row = df_m15.row(idx, named=True)
            timestamp = row['time']
            price = row['close']

            # Detect regime
            regime_state, df_regime_pred = self.detect_regime(df_m15, df_h1, idx)

            # Add regime columns to current df slice for ML prediction
            if 'regime' not in df_m15.columns:
                # Initialize regime columns in df_m15
                df_m15 = df_m15.with_columns([
                    pl.lit(0).alias('regime'),
                    pl.lit(0.0).alias('regime_confidence')
                ])

            # Check if regime blocks trading
            if regime_state.recommendation == "SLEEP":
                continue

            # Get ML prediction (use model's stored feature names)
            df_slice = df_m15.slice(0, idx + 1)
            if hasattr(self.ml_model, 'feature_names') and self.ml_model.feature_names:
                feature_cols = self.ml_model.feature_names
            else:
                # Fallback: filter out OHLC and intermediate columns
                exclude_cols = {'time', 'open', 'high', 'low', 'close', 'spread', 'real_volume', 'volume',
                               'swing_high_level', 'swing_low_level', 'last_swing_high', 'last_swing_low',
                               'fvg_top', 'fvg_bottom', 'fvg_mid', 'ob_top', 'ob_bottom'}
                feature_cols = [c for c in df_slice.columns if c not in exclude_cols]

            ml_pred = self.ml_model.predict(df_slice, feature_cols)

            if ml_pred.signal == "HOLD":
                continue

            # Skip session check for simplicity
            # session_ok, _, session_mult = self.session_filter.can_trade(timestamp)
            # if not session_ok:
            #     continue

            # Entry filters (simplified)
            if ml_pred.confidence < 0.50:
                continue

            # SMC signal
            smc_signal = row.get('ob', 0)
            if smc_signal == 0:
                continue

            if (ml_pred.signal == "BUY" and smc_signal < 0) or \
               (ml_pred.signal == "SELL" and smc_signal > 0):
                continue

            # Execute trade (simplified)
            direction = ml_pred.signal
            entry_price = price

            # Calculate SL/TP (simplified)
            atr = row.get('atr', 12.0)
            if direction == "BUY":
                sl = entry_price - (2.0 * atr)
                tp = entry_price + (3.0 * atr)
            else:
                sl = entry_price + (2.0 * atr)
                tp = entry_price - (3.0 * atr)

            # Simulate trade (find exit)
            exit_price = None
            exit_reason = None
            exit_idx = None

            for future_idx in range(idx + 1, min(idx + 100, len(df_m15))):
                future_row = df_m15.row(future_idx, named=True)
                future_high = future_row['high']
                future_low = future_row['low']

                if direction == "BUY":
                    if future_low <= sl:
                        exit_price = sl
                        exit_reason = "SL"
                        exit_idx = future_idx
                        break
                    elif future_high >= tp:
                        exit_price = tp
                        exit_reason = "TP"
                        exit_idx = future_idx
                        break
                else:  # SELL
                    if future_high >= sl:
                        exit_price = sl
                        exit_reason = "SL"
                        exit_idx = future_idx
                        break
                    elif future_low <= tp:
                        exit_price = tp
                        exit_reason = "TP"
                        exit_idx = future_idx
                        break

            if exit_price is None:
                # No exit found, close at last bar
                exit_price = df_m15.row(min(idx + 99, len(df_m15) - 1), named=True)['close']
                exit_reason = "EOD"
                exit_idx = min(idx + 99, len(df_m15) - 1)

            # Calculate P&L
            if direction == "BUY":
                profit = exit_price - entry_price
            else:
                profit = entry_price - exit_price

            profit_usd = profit * 0.01  # 0.01 lot

            # Update balance
            self.balance += profit_usd
            self.equity = self.balance

            # Log trade
            self.trades.append({
                'entry_time': timestamp,
                'exit_time': df_m15.row(exit_idx, named=True)['time'],
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_usd': profit_usd,
                'exit_reason': exit_reason,
                'regime': regime_state.regime.value,
                'ml_conf': ml_pred.confidence,
            })

            # Record equity
            self.equity_curve.append({
                'time': df_m15.row(exit_idx, named=True)['time'],
                'equity': self.equity
            })

        logger.info(f"✓ Backtest complete: {len(self.trades)} trades")

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {}

        df_trades = pl.DataFrame(self.trades)

        total_trades = len(self.trades)
        wins = df_trades.filter(pl.col('profit_usd') > 0).shape[0]
        losses = df_trades.filter(pl.col('profit_usd') < 0).shape[0]
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        net_profit = df_trades['profit_usd'].sum()
        gross_profit = df_trades.filter(pl.col('profit_usd') > 0)['profit_usd'].sum()
        gross_loss = abs(df_trades.filter(pl.col('profit_usd') < 0)['profit_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        equity_curve = [self.config.initial_balance] + [e['equity'] for e in self.equity_curve]
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        max_dd = np.max(drawdown)
        max_dd_pct = max_dd / self.config.initial_balance * 100

        # Sharpe ratio (annualized)
        returns = df_trades['profit_usd'].to_numpy()
        if len(returns) > 1 and np.std(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            # Assume ~5 trades/day, 252 trading days/year
            sharpe = (avg_return / std_return) * np.sqrt(5 * 252)
        else:
            sharpe = 0

        # Regime changes
        regime_changes_m15 = len(self.regime_changes_m15)
        regime_changes_h1 = len(self.regime_changes_h1)

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'net_profit': net_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_dd': max_dd,
            'max_dd_pct': max_dd_pct,
            'sharpe': sharpe,
            'final_balance': self.balance,
            'regime_changes_m15': regime_changes_m15,
            'regime_changes_h1': regime_changes_h1,
        }

    def save_results(self, variant_name: str, metrics: dict):
        """Save results to file"""
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.config.results_dir) / f"h1_hmm_{timestamp}.log"

        with open(log_file, 'w') as f:
            f.write(f"#39 H1 HMM Results\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Variant: {variant_name}\n\n")

            f.write("--- PERFORMANCE SUMMARY ---\n")
            f.write(f"  Total Trades:    {metrics['total_trades']}\n")
            f.write(f"  Win Rate:        {metrics['win_rate']:.1f}%\n")
            f.write(f"  Net PnL:         ${metrics['net_profit']:.2f}\n")
            f.write(f"  Profit Factor:   {metrics['profit_factor']:.2f}\n")
            f.write(f"  Max Drawdown:    ${metrics['max_dd']:.2f} ({metrics['max_dd_pct']:.1f}%)\n")
            f.write(f"  Sharpe Ratio:    {metrics['sharpe']:.2f}\n")
            f.write(f"  Final Balance:   ${metrics['final_balance']:.2f}\n\n")

            f.write("--- REGIME STABILITY ---\n")
            f.write(f"  M15 Regime Changes: {metrics['regime_changes_m15']}\n")
            f.write(f"  H1 Regime Changes:  {metrics['regime_changes_h1']}\n")
            if metrics['regime_changes_h1'] > 0:
                improvement = (metrics['regime_changes_m15'] - metrics['regime_changes_h1']) / metrics['regime_changes_m15'] * 100
                f.write(f"  Improvement:        {improvement:.1f}% fewer changes\n\n")

            f.write("--- TRADE LOG ---\n")
            f.write(f"{'#':>4} {'Entry Time':>19} {'Dir':>4} {'Entry':>8} {'Exit':>8} {'P/L':>7} {'Result':>6} {'Exit_Reason':>12} {'Regime':>15} {'ML_Conf':>7}\n")
            f.write("-" * 120 + "\n")

            for i, trade in enumerate(self.trades, 1):
                result = "WIN" if trade['profit_usd'] > 0 else "LOSS"
                f.write(f"{i:4d} {trade['entry_time']:%Y-%m-%d %H:%M:%S} "
                       f"{trade['direction']:>4} {trade['entry_price']:8.2f} {trade['exit_price']:8.2f} "
                       f"{trade['profit_usd']:7.2f} {result:>6} {trade['exit_reason']:>12} "
                       f"{trade['regime']:>15} {trade['ml_conf']:7.3f}\n")

        logger.info(f"✓ Results saved to {log_file}")


def main():
    """Run both M15 and H1 HMM backtests"""
    config = BacktestConfig()

    # Baseline: M15 HMM
    logger.info("=" * 80)
    logger.info("BASELINE: M15 HMM")
    logger.info("=" * 80)

    bt_m15 = H1HMMBacktest(config, use_h1_hmm=False)
    bt_m15.connect_mt5()
    df_m15, df_h1 = bt_m15.fetch_data()
    df_m15, df_h1 = bt_m15.prepare_data(df_m15, df_h1)
    bt_m15.fit_regimes(df_m15, df_h1)
    bt_m15.run_backtest(df_m15, df_h1)
    metrics_m15 = bt_m15.calculate_metrics()
    bt_m15.save_results("M15_HMM", metrics_m15)

    logger.info(f"\nBASELINE Results:")
    logger.info(f"  Trades: {metrics_m15['total_trades']}, WR: {metrics_m15['win_rate']:.1f}%, "
                f"PnL: ${metrics_m15['net_profit']:.2f}, Sharpe: {metrics_m15['sharpe']:.2f}")
    logger.info(f"  M15 Regime Changes: {metrics_m15['regime_changes_m15']}")

    # Test: H1 HMM
    logger.info("\n" + "=" * 80)
    logger.info("TEST: H1 HMM")
    logger.info("=" * 80)

    bt_h1 = H1HMMBacktest(config, use_h1_hmm=True)
    bt_h1.connect_mt5()
    # Reuse same data
    bt_h1.fit_regimes(df_m15, df_h1)
    bt_h1.run_backtest(df_m15, df_h1)
    metrics_h1 = bt_h1.calculate_metrics()
    bt_h1.save_results("H1_HMM", metrics_h1)

    logger.info(f"\nH1 HMM Results:")
    logger.info(f"  Trades: {metrics_h1['total_trades']}, WR: {metrics_h1['win_rate']:.1f}%, "
                f"PnL: ${metrics_h1['net_profit']:.2f}, Sharpe: {metrics_h1['sharpe']:.2f}")
    logger.info(f"  H1 Regime Changes: {metrics_h1['regime_changes_h1']}")

    # Comparison
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON")
    logger.info("=" * 80)

    pnl_diff = metrics_h1['net_profit'] - metrics_m15['net_profit']
    sharpe_diff = metrics_h1['sharpe'] - metrics_m15['sharpe']
    regime_reduction = (metrics_m15['regime_changes_m15'] - metrics_h1['regime_changes_h1']) / metrics_m15['regime_changes_m15'] * 100 if metrics_m15['regime_changes_m15'] > 0 else 0

    logger.info(f"PnL Difference:      ${pnl_diff:+.2f} ({pnl_diff/metrics_m15['net_profit']*100:+.1f}%)")
    logger.info(f"Sharpe Difference:   {sharpe_diff:+.2f} ({sharpe_diff/metrics_m15['sharpe']*100:+.1f}%)")
    logger.info(f"Regime Stability:    {regime_reduction:.1f}% fewer regime changes")

    bt_m15.mt5.disconnect()
    bt_h1.mt5.disconnect()


if __name__ == "__main__":
    main()
