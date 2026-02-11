"""
Backtest Comparison: H1 Bias vs M5 Confirmation
================================================
Compare the performance of:
1. Current H1 Bias system (lagging)
2. New M5 Confirmation system (fast)

Author: Claude Opus 4.6
Date: 2026-02-09
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Tuple

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer
from src.feature_eng import FeatureEngineer
from src.ml_model import TradingModel
from src.regime_detector import MarketRegimeDetector
from src.m5_confirmation import M5ConfirmationAnalyzer, get_m5_confirmation_summary


class BacktestComparison:
    """Compare H1 Bias vs M5 Confirmation backtest."""

    def __init__(self):
        """Initialize backtest comparison."""
        logger.info("=" * 60)
        logger.info("BACKTEST COMPARISON: H1 Bias vs M5 Confirmation")
        logger.info("=" * 60)

        # Initialize components
        self.features = FeatureEngineer()
        self.smc = SMCAnalyzer()
        self.regime = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
        self.regime.load()

        # ML Model
        self.ml = TradingModel(model_path="backtests/ml_v3/xgboost_model_v3.pkl")
        self.ml.load()

        # M5 Confirmation
        self.m5_analyzer = M5ConfirmationAnalyzer(
            smc_analyzer=self.smc,
            feature_engineer=self.features
        )

        # Config
        self.initial_capital = 5000
        self.risk_per_trade = 0.015  # 1.5%
        self.lot_size = 0.02  # Fixed lot for comparison

        logger.info(f"Initial Capital: ${self.initial_capital}")
        logger.info(f"Risk per Trade: {self.risk_per_trade:.1%}")
        logger.info(f"Lot Size: {self.lot_size}")

    def fetch_data(self, days: int = 30) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Fetch M15 and M5 data for backtest.

        Args:
            days: Number of days to backtest

        Returns:
            (df_m15, df_m5) tuple
        """
        logger.info(f"Fetching {days} days of data...")

        mt5 = MT5Connector(
            login=int(os.getenv("MT5_LOGIN")),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=os.getenv("MT5_PATH")
        )
        mt5.connect()

        # Calculate bars needed
        bars_m15 = days * 24 * 4  # 4 bars per hour
        bars_m5 = days * 24 * 12  # 12 bars per hour

        df_m15 = mt5.get_market_data(symbol="XAUUSD", timeframe="M15", count=bars_m15)
        df_m5 = mt5.get_market_data(symbol="XAUUSD", timeframe="M5", count=bars_m5)

        mt5.disconnect()

        logger.info(f"M15 bars: {len(df_m15)}")
        logger.info(f"M5 bars: {len(df_m5)}")

        return df_m15, df_m5

    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare data with features and SMC."""
        df = self.features.calculate_all(df, include_ml_features=True)
        df = self.smc.calculate_all(df)
        df = self.regime.predict(df)
        return df

    def get_h1_bias(self, df_h1: pl.DataFrame) -> str:
        """
        Get H1 bias using old EMA20 method.

        Args:
            df_h1: H1 OHLCV data

        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"
        """
        if len(df_h1) < 20:
            return "NEUTRAL"

        closes = df_h1["close"].to_list()
        current_price = closes[-1]

        # Calculate EMA20
        period = 20
        multiplier = 2 / (period + 1)
        ema = np.mean(closes[:period])
        for val in closes[period:]:
            ema = (val - ema) * multiplier + ema

        # Determine bias with 0.1% buffer
        if current_price > ema * 1.001:
            return "BULLISH"
        elif current_price < ema * 0.999:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def run_backtest_h1(self, df_m15: pl.DataFrame, df_h1: pl.DataFrame) -> Dict:
        """
        Run backtest with H1 Bias filter.

        Args:
            df_m15: M15 prepared data
            df_h1: H1 OHLCV data

        Returns:
            Backtest results dict
        """
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST 1: H1 Bias (Current System)")
        logger.info("=" * 60)

        trades = []
        capital = self.initial_capital
        equity_curve = []

        # Get H1 bias (update every 4 M15 candles = 1 hour)
        h1_bias = "NEUTRAL"
        h1_update_interval = 4

        for i in range(100, len(df_m15)):
            # Update H1 bias every 4 candles
            if i % h1_update_interval == 0:
                # Get corresponding H1 data
                m15_time = df_m15["time"][i]
                h1_idx = int(i / 4)  # M15 to H1 conversion
                if h1_idx < len(df_h1):
                    df_h1_slice = df_h1[:h1_idx+1]
                    h1_bias = self.get_h1_bias(df_h1_slice)

            # Get M15 signal
            row = df_m15.row(i, named=True)

            # SMC Signal
            smc_signal = row.get("smc_signal", "HOLD")
            smc_confidence = row.get("smc_confidence", 0.5)

            # ML Signal
            ml_features = self.ml.prepare_features(df_m15[:i+1])
            if ml_features is not None and len(ml_features) > 0:
                ml_pred = self.ml.predict(ml_features[-1:])
                ml_signal = "BUY" if ml_pred["prediction"][0] == 1 else "SELL"
                ml_confidence = ml_pred["probability"][0]
            else:
                ml_signal = "HOLD"
                ml_confidence = 0.5

            # Check if SMC + ML agree
            if smc_signal == "HOLD" or ml_signal == "HOLD":
                continue

            if smc_signal != ml_signal:
                continue

            # --- H1 BIAS FILTER ---
            signal_blocked = False
            override_triggered = False

            if h1_bias != "NEUTRAL":
                # Check if signal conflicts with H1
                if (smc_signal == "BUY" and h1_bias != "BULLISH") or \
                   (smc_signal == "SELL" and h1_bias != "BEARISH"):

                    # Check for override (SMC >= 80% + ML >= 65%)
                    if smc_confidence >= 0.80 and ml_confidence >= 0.65:
                        override_triggered = True
                    else:
                        signal_blocked = True
                        continue

            # --- Execute Trade ---
            entry_price = row["close"]
            atr = row.get("atr", 15.0)

            # Calculate SL/TP
            sl_distance = atr * 1.5
            tp_distance = sl_distance * 1.5  # RR 1.5:1

            if smc_signal == "BUY":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
                direction = 1
            else:  # SELL
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
                direction = -1

            # Simulate trade exit
            exit_price = None
            exit_reason = None
            exit_idx = None

            for j in range(i+1, min(i+100, len(df_m15))):  # Max 100 candles (25 hours)
                candle = df_m15.row(j, named=True)

                if direction == 1:  # BUY
                    if candle["low"] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                        exit_idx = j
                        break
                    elif candle["high"] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                        exit_idx = j
                        break
                else:  # SELL
                    if candle["high"] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                        exit_idx = j
                        break
                    elif candle["low"] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                        exit_idx = j
                        break

            # Default exit at 100 candles
            if exit_price is None:
                exit_idx = min(i+100, len(df_m15)-1)
                exit_price = df_m15["close"][exit_idx]
                exit_reason = "TIME"

            # Calculate P/L
            pnl = (exit_price - entry_price) * direction * self.lot_size * 100  # 1 lot = 100oz

            capital += pnl
            equity_curve.append(capital)

            trades.append({
                "entry_time": row["time"],
                "entry_price": entry_price,
                "exit_time": df_m15["time"][exit_idx],
                "exit_price": exit_price,
                "direction": "BUY" if direction == 1 else "SELL",
                "pnl": pnl,
                "exit_reason": exit_reason,
                "smc_confidence": smc_confidence,
                "ml_confidence": ml_confidence,
                "h1_bias": h1_bias,
                "override": override_triggered
            })

        # Calculate metrics
        results = self._calculate_metrics(trades, equity_curve)
        results["method"] = "H1_BIAS"

        return results

    def run_backtest_m5(self, df_m15: pl.DataFrame, df_m5: pl.DataFrame) -> Dict:
        """
        Run backtest with M5 Confirmation.

        Args:
            df_m15: M15 prepared data
            df_m5: M5 prepared data

        Returns:
            Backtest results dict
        """
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST 2: M5 Confirmation (New System)")
        logger.info("=" * 60)

        trades = []
        capital = self.initial_capital
        equity_curve = []

        # Prepare M5 data
        df_m5 = self.prepare_data(df_m5)

        for i in range(100, len(df_m15)):
            # Get M15 signal
            row = df_m15.row(i, named=True)

            # SMC Signal
            smc_signal = row.get("smc_signal", "HOLD")
            smc_confidence = row.get("smc_confidence", 0.5)

            # ML Signal
            ml_features = self.ml.prepare_features(df_m15[:i+1])
            if ml_features is not None and len(ml_features) > 0:
                ml_pred = self.ml.predict(ml_features[-1:])
                ml_signal = "BUY" if ml_pred["prediction"][0] == 1 else "SELL"
                ml_confidence = ml_pred["probability"][0]
            else:
                ml_signal = "HOLD"
                ml_confidence = 0.5

            # Check if SMC + ML agree
            if smc_signal == "HOLD" or ml_signal == "HOLD":
                continue

            if smc_signal != ml_signal:
                continue

            # --- M5 CONFIRMATION ---
            # Get corresponding M5 data (3x more candles than M15)
            m5_idx = i * 3
            if m5_idx >= len(df_m5):
                continue

            df_m5_slice = df_m5[:m5_idx+1].tail(100)  # Last 100 M5 candles

            m5_confirmation = self.m5_analyzer.analyze(
                df_m5=df_m5_slice,
                m15_signal=smc_signal,
                m15_confidence=smc_confidence
            )

            # Check M5 confirmation
            if m5_confirmation.signal == "NEUTRAL":
                # M5 conflicts → skip trade
                continue

            # Use M5-adjusted confidence
            final_confidence = m5_confirmation.confidence

            # --- Execute Trade ---
            entry_price = row["close"]
            atr = row.get("atr", 15.0)

            # Calculate SL/TP
            sl_distance = atr * 1.5
            tp_distance = sl_distance * 1.5  # RR 1.5:1

            if smc_signal == "BUY":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
                direction = 1
            else:  # SELL
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
                direction = -1

            # Simulate trade exit (same logic as H1 backtest)
            exit_price = None
            exit_reason = None
            exit_idx = None

            for j in range(i+1, min(i+100, len(df_m15))):
                candle = df_m15.row(j, named=True)

                if direction == 1:  # BUY
                    if candle["low"] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                        exit_idx = j
                        break
                    elif candle["high"] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                        exit_idx = j
                        break
                else:  # SELL
                    if candle["high"] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                        exit_idx = j
                        break
                    elif candle["low"] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                        exit_idx = j
                        break

            if exit_price is None:
                exit_idx = min(i+100, len(df_m15)-1)
                exit_price = df_m15["close"][exit_idx]
                exit_reason = "TIME"

            # Calculate P/L
            pnl = (exit_price - entry_price) * direction * self.lot_size * 100

            capital += pnl
            equity_curve.append(capital)

            trades.append({
                "entry_time": row["time"],
                "entry_price": entry_price,
                "exit_time": df_m15["time"][exit_idx],
                "exit_price": exit_price,
                "direction": "BUY" if direction == 1 else "SELL",
                "pnl": pnl,
                "exit_reason": exit_reason,
                "smc_confidence": smc_confidence,
                "ml_confidence": ml_confidence,
                "m5_trend": m5_confirmation.trend,
                "m5_confidence": final_confidence,
                "m5_aligned": m5_confirmation.smc_alignment
            })

        # Calculate metrics
        results = self._calculate_metrics(trades, equity_curve)
        results["method"] = "M5_CONFIRMATION"

        return results

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculate backtest performance metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "trades": trades
            }

        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] < 0]

        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t["pnl"] for t in trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Sharpe ratio (simplified)
        returns = [t["pnl"] for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "final_capital": equity_curve[-1] if equity_curve else self.initial_capital,
            "roi": ((equity_curve[-1] - self.initial_capital) / self.initial_capital * 100) if equity_curve else 0,
            "trades": trades
        }

    def print_comparison(self, results_h1: Dict, results_m5: Dict):
        """Print comparison table."""
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPARISON RESULTS")
        logger.info("=" * 80)

        # Create comparison table
        metrics = [
            ("Total Trades", "total_trades", ""),
            ("Winning Trades", "winning_trades", ""),
            ("Losing Trades", "losing_trades", ""),
            ("Win Rate", "win_rate", "%"),
            ("Total P/L", "total_pnl", "$"),
            ("Avg Win", "avg_win", "$"),
            ("Avg Loss", "avg_loss", "$"),
            ("Largest Win", "largest_win", "$"),
            ("Largest Loss", "largest_loss", "$"),
            ("Profit Factor", "profit_factor", ""),
            ("Sharpe Ratio", "sharpe_ratio", ""),
            ("Max Drawdown", "max_drawdown", "%"),
            ("Final Capital", "final_capital", "$"),
            ("ROI", "roi", "%"),
        ]

        print("\n{:<20} {:<20} {:<20} {:<15}".format("Metric", "H1 Bias", "M5 Confirmation", "Improvement"))
        print("-" * 80)

        for label, key, unit in metrics:
            val_h1 = results_h1.get(key, 0)
            val_m5 = results_m5.get(key, 0)

            if unit == "%":
                str_h1 = f"{val_h1:.2f}%"
                str_m5 = f"{val_m5:.2f}%"
                improvement = f"{val_m5 - val_h1:+.2f}%"
            elif unit == "$":
                str_h1 = f"${val_h1:.2f}"
                str_m5 = f"${val_m5:.2f}"
                improvement = f"${val_m5 - val_h1:+.2f}"
            else:
                str_h1 = f"{val_h1:.2f}"
                str_m5 = f"{val_m5:.2f}"
                if val_h1 != 0:
                    pct = (val_m5 - val_h1) / abs(val_h1) * 100
                    improvement = f"{pct:+.1f}%"
                else:
                    improvement = "N/A"

            print(f"{label:<20} {str_h1:<20} {str_m5:<20} {improvement:<15}")

        print("=" * 80)

    def run_comparison(self, days: int = 30):
        """Run full comparison backtest."""
        import os
        from dotenv import load_dotenv
        load_dotenv()

        # Fetch data
        df_m15, df_m5 = self.fetch_data(days=days)

        # Prepare M15 data
        logger.info("Preparing M15 data...")
        df_m15 = self.prepare_data(df_m15)

        # Create H1 data from M15 (resample)
        logger.info("Creating H1 data from M15...")
        df_h1 = df_m15.group_by_dynamic(
            "time",
            every="1h",
            period="1h",
        ).agg([
            pl.first("open").alias("open"),
            pl.max("high").alias("high"),
            pl.min("low").alias("low"),
            pl.last("close").alias("close"),
            pl.sum("tick_volume").alias("tick_volume"),
        ])

        # Run backtests
        results_h1 = self.run_backtest_h1(df_m15, df_h1)
        results_m5 = self.run_backtest_m5(df_m15, df_m5)

        # Print comparison
        self.print_comparison(results_h1, results_m5)

        # Save results
        self._save_results(results_h1, results_m5)

        return results_h1, results_m5

    def _save_results(self, results_h1: Dict, results_m5: Dict):
        """Save results to file."""
        output_dir = Path("backtests/comparison_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        import json
        output_file = output_dir / f"h1_vs_m5_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "h1_bias": {k: v for k, v in results_h1.items() if k != "trades"},
                "m5_confirmation": {k: v for k, v in results_m5.items() if k != "trades"},
            }, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare H1 Bias vs M5 Confirmation")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")

    args = parser.parse_args()

    # Run comparison
    comparison = BacktestComparison()
    comparison.run_comparison(days=args.days)

    logger.info("\n✅ BACKTEST COMPARISON COMPLETE!")
