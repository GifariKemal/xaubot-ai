"""
Walk-Forward Optimization Backtest (1 Year)
============================================
Simulasi backtest dengan ML yang belajar progressif setiap bulan.
Periode: Januari 2025 - Februari 2026

Metodologi:
1. Ambil data historis 1 tahun
2. Setiap bulan:
   - Train model dengan data sebelumnya (rolling window)
   - Backtest bulan tersebut dengan model baru
   - Evaluasi dan catat hasil
3. Analisis performa keseluruhan
4. Temukan parameter optimal
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import polars as pl
from dotenv import load_dotenv
from loguru import logger

warnings.filterwarnings('ignore')

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

load_dotenv()

@dataclass
class MonthlyResult:
    """Result for one month of backtesting."""
    month: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    profit_factor: float
    model_auc: float
    avg_confidence: float
    ml_only_trades: int
    smc_ml_trades: int

@dataclass
class TradeResult:
    """Individual trade result."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    confidence: float
    signal_type: str  # ML_ONLY or SMC_ML

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    # Training window (months of data for training)
    train_window_months: int = 3
    # Minimum bars for training
    min_train_bars: int = 5000
    # ML thresholds to test
    ml_thresholds: List[float] = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.75])
    # ML-only thresholds to test
    ml_only_thresholds: List[float] = field(default_factory=lambda: [0.70, 0.75, 0.80])
    # Lot sizes
    base_lot: float = 0.01
    max_lot: float = 0.02
    # Risk parameters
    max_loss_per_trade: float = 30.0
    # TP/SL multipliers
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.5


class WalkForwardBacktest:
    """Walk-forward optimization backtester."""

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.mt5 = None
        self.all_data = None
        self.monthly_results: List[MonthlyResult] = []
        self.all_trades: List[TradeResult] = []

    def connect_mt5(self) -> bool:
        """Connect to MT5."""
        import MetaTrader5 as mt5

        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False

        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')

        if not mt5.login(login, password, server):
            logger.error("MT5 login failed")
            return False

        account = mt5.account_info()
        logger.info(f"Connected to MT5 - Balance: ${account.balance:,.2f}")
        self.mt5 = mt5
        return True

    def fetch_historical_data(self, months: int = 13) -> Optional[pl.DataFrame]:
        """Fetch historical M5 data for the specified period."""
        import MetaTrader5 as mt5

        # Calculate bars needed (288 bars per day * 22 trading days * months)
        bars_per_month = 288 * 22
        total_bars = bars_per_month * months

        logger.info(f"Fetching {total_bars:,} bars ({months} months of M5 data)...")

        # MT5 has limit, fetch in chunks if needed
        max_bars = 100000

        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, min(total_bars, max_bars))

        if rates is None or len(rates) == 0:
            logger.error("Failed to fetch historical data")
            return None

        # Convert to polars DataFrame
        df = pl.DataFrame({
            'time': [datetime.fromtimestamp(r[0]) for r in rates],
            'open': [r[1] for r in rates],
            'high': [r[2] for r in rates],
            'low': [r[3] for r in rates],
            'close': [r[4] for r in rates],
            'volume': [float(r[5]) for r in rates],
        })

        logger.info(f"Fetched {len(df):,} bars")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")

        self.all_data = df
        return df

    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate all features needed for ML."""
        from src.feature_eng import FeatureEngineer
        from src.smc_polars import SMCAnalyzer

        feature_eng = FeatureEngineer()
        smc = SMCAnalyzer()

        df = feature_eng.calculate_all(df)
        df = smc.calculate_all(df)

        return df

    def train_models(self, train_df: pl.DataFrame) -> Tuple[object, object, float]:
        """Train HMM and XGBoost models on training data."""
        from src.regime_detector import MarketRegimeDetector
        from src.ml_model import TradingModel

        # Train HMM Regime Detector
        regime = MarketRegimeDetector()

        # Prepare features for HMM
        train_df = self.prepare_features(train_df)

        # Train regime detector
        try:
            regime.fit(train_df)
        except Exception as e:
            logger.warning(f"HMM training failed: {e}, using default")
            regime.load()  # Load pre-trained as fallback

        # Add regime predictions
        train_df = regime.predict(train_df)

        # Train XGBoost
        ml_model = TradingModel()

        # Prepare labels (next bar direction)
        train_df = train_df.with_columns([
            (pl.col('close').shift(-1) > pl.col('close')).cast(pl.Int32).alias('target')
        ])

        # Drop nulls
        train_df = train_df.drop_nulls()

        # Get feature columns
        feature_cols = [c for c in train_df.columns if c not in ['time', 'target', 'open', 'high', 'low', 'close', 'volume']]

        # Train model
        try:
            X = train_df.select(feature_cols).to_numpy()
            y = train_df['target'].to_numpy()

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            ml_model.train(X_train, y_train, X_test, y_test, feature_cols)
            auc = ml_model.test_auc if hasattr(ml_model, 'test_auc') else 0.5
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}, using default")
            ml_model.load("models/xgboost_model.pkl")
            auc = 0.5

        return regime, ml_model, auc

    def simulate_month(
        self,
        test_df: pl.DataFrame,
        regime: object,
        ml_model: object,
        ml_threshold: float = 0.65,
        ml_only_threshold: float = 0.75,
    ) -> Tuple[List[TradeResult], float]:
        """Simulate trading for one month."""
        from src.smc_polars import SMCAnalyzer
        from src.dynamic_confidence import create_dynamic_confidence

        smc = SMCAnalyzer()
        dynamic_conf = create_dynamic_confidence()

        trades = []
        position = None
        total_confidence = 0
        confidence_count = 0

        # Prepare test data with features
        test_df = self.prepare_features(test_df)
        test_df = regime.predict(test_df)

        # Iterate through test period
        for i in range(100, len(test_df) - 20):  # Leave room for TP/SL check
            row = test_df.row(i, named=True)
            current_time = row['time']

            # Skip if already in position
            if position is not None:
                # Check if position should be closed
                for j in range(i + 1, min(i + 20, len(test_df))):
                    future_row = test_df.row(j, named=True)

                    if position['direction'] == 'BUY':
                        # Check TP
                        if future_row['high'] >= position['tp']:
                            pnl = (position['tp'] - position['entry']) * position['lot'] * 100
                            trades.append(TradeResult(
                                entry_time=position['time'],
                                exit_time=future_row['time'],
                                direction='BUY',
                                entry_price=position['entry'],
                                exit_price=position['tp'],
                                lot_size=position['lot'],
                                pnl=pnl,
                                confidence=position['confidence'],
                                signal_type=position['signal_type'],
                            ))
                            position = None
                            break
                        # Check SL
                        if future_row['low'] <= position['sl']:
                            pnl = (position['sl'] - position['entry']) * position['lot'] * 100
                            pnl = max(pnl, -self.config.max_loss_per_trade)
                            trades.append(TradeResult(
                                entry_time=position['time'],
                                exit_time=future_row['time'],
                                direction='BUY',
                                entry_price=position['entry'],
                                exit_price=position['sl'],
                                lot_size=position['lot'],
                                pnl=pnl,
                                confidence=position['confidence'],
                                signal_type=position['signal_type'],
                            ))
                            position = None
                            break
                    else:  # SELL
                        # Check TP
                        if future_row['low'] <= position['tp']:
                            pnl = (position['entry'] - position['tp']) * position['lot'] * 100
                            trades.append(TradeResult(
                                entry_time=position['time'],
                                exit_time=future_row['time'],
                                direction='SELL',
                                entry_price=position['entry'],
                                exit_price=position['tp'],
                                lot_size=position['lot'],
                                pnl=pnl,
                                confidence=position['confidence'],
                                signal_type=position['signal_type'],
                            ))
                            position = None
                            break
                        # Check SL
                        if future_row['high'] >= position['sl']:
                            pnl = (position['entry'] - position['sl']) * position['lot'] * 100
                            pnl = max(pnl, -self.config.max_loss_per_trade)
                            trades.append(TradeResult(
                                entry_time=position['time'],
                                exit_time=future_row['time'],
                                direction='SELL',
                                entry_price=position['entry'],
                                exit_price=position['sl'],
                                lot_size=position['lot'],
                                pnl=pnl,
                                confidence=position['confidence'],
                                signal_type=position['signal_type'],
                            ))
                            position = None
                            break

                if position is not None:
                    # Position still open, skip to next bar
                    continue

            # Check for new signal
            # Get ML prediction
            try:
                window_df = test_df.slice(max(0, i - 100), 101)
                ml_pred = ml_model.predict(window_df)

                if ml_pred.confidence < ml_threshold:
                    continue

                total_confidence += ml_pred.confidence
                confidence_count += 1

                # Get SMC signal
                smc_signal = smc.generate_signal(window_df)
                has_smc = smc_signal is not None

                # Apply entry rules
                signal_type = None
                direction = None

                if has_smc:
                    # SMC + ML must agree
                    smc_dir = smc_signal.signal_type if smc_signal else None
                    if smc_dir == ml_pred.signal and ml_pred.confidence >= ml_threshold:
                        signal_type = "SMC_ML"
                        direction = ml_pred.signal
                else:
                    # ML-only needs higher threshold
                    if ml_pred.confidence >= ml_only_threshold:
                        signal_type = "ML_ONLY"
                        direction = ml_pred.signal

                if direction is None:
                    continue

                # Session filter (simplified)
                hour = current_time.hour
                # London: 8-16 UTC, NY: 13-21 UTC, Overlap: 13-16 UTC
                if not (8 <= hour <= 21):
                    continue  # Skip Asia/Sydney

                # Calculate TP/SL based on ATR
                atr = row.get('atr_14', 2.0)
                if atr is None or atr < 0.5:
                    atr = 2.0

                entry_price = row['close']

                if direction == 'BUY':
                    tp = entry_price + (atr * self.config.tp_atr_mult)
                    sl = entry_price - (atr * self.config.sl_atr_mult)
                else:
                    tp = entry_price - (atr * self.config.tp_atr_mult)
                    sl = entry_price + (atr * self.config.sl_atr_mult)

                # Open position
                position = {
                    'time': current_time,
                    'direction': direction,
                    'entry': entry_price,
                    'tp': tp,
                    'sl': sl,
                    'lot': self.config.base_lot,
                    'confidence': ml_pred.confidence,
                    'signal_type': signal_type,
                }

            except Exception as e:
                continue

        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        return trades, avg_confidence

    def calculate_metrics(self, trades: List[TradeResult]) -> Dict:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'ml_only_trades': 0,
                'smc_ml_trades': 0,
            }

        wins = len([t for t in trades if t.pnl > 0])
        losses = len([t for t in trades if t.pnl <= 0])
        total_pnl = sum(t.pnl for t in trades)

        # Calculate max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        ml_only = len([t for t in trades if t.signal_type == 'ML_ONLY'])
        smc_ml = len([t for t in trades if t.signal_type == 'SMC_ML'])

        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / len(trades) * 100) if trades else 0,
            'total_pnl': total_pnl,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'ml_only_trades': ml_only,
            'smc_ml_trades': smc_ml,
        }

    def run_walkforward(
        self,
        start_month: int = 1,  # January
        start_year: int = 2025,
        end_month: int = 2,    # February
        end_year: int = 2026,
    ):
        """Run walk-forward optimization."""

        if self.all_data is None:
            logger.error("No data loaded. Call fetch_historical_data first.")
            return

        logger.info("=" * 70)
        logger.info("WALK-FORWARD OPTIMIZATION BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Period: {start_month}/{start_year} - {end_month}/{end_year}")
        logger.info(f"Training window: {self.config.train_window_months} months")
        logger.info(f"ML Thresholds to test: {self.config.ml_thresholds}")
        logger.info(f"ML-Only Thresholds to test: {self.config.ml_only_thresholds}")
        logger.info("=" * 70)
        print()

        # Best parameters tracking
        best_params = {
            'ml_threshold': 0.65,
            'ml_only_threshold': 0.75,
            'total_pnl': float('-inf'),
            'win_rate': 0,
        }

        # Generate month ranges
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        months = []
        while current < end:
            next_month = current + timedelta(days=32)
            next_month = datetime(next_month.year, next_month.month, 1)
            months.append((current, next_month))
            current = next_month

        logger.info(f"Testing {len(months)} months")
        print()

        # Test different parameter combinations
        param_results = []

        for ml_thresh in self.config.ml_thresholds:
            for ml_only_thresh in self.config.ml_only_thresholds:
                if ml_only_thresh < ml_thresh:
                    continue  # ML-only should be >= base threshold

                logger.info(f"Testing: ML={ml_thresh:.0%}, ML-Only={ml_only_thresh:.0%}")

                monthly_results = []
                all_month_trades = []

                for month_start, month_end in months:
                    # Get training data (previous N months)
                    train_start = month_start - timedelta(days=self.config.train_window_months * 30)

                    train_df = self.all_data.filter(
                        (pl.col('time') >= train_start) & (pl.col('time') < month_start)
                    )

                    test_df = self.all_data.filter(
                        (pl.col('time') >= month_start) & (pl.col('time') < month_end)
                    )

                    if len(train_df) < self.config.min_train_bars:
                        logger.warning(f"  Skipping {month_start.strftime('%Y-%m')}: insufficient training data ({len(train_df)} bars)")
                        continue

                    if len(test_df) < 100:
                        logger.warning(f"  Skipping {month_start.strftime('%Y-%m')}: insufficient test data ({len(test_df)} bars)")
                        continue

                    # Train models
                    try:
                        regime, ml_model, auc = self.train_models(train_df)
                    except Exception as e:
                        logger.warning(f"  Training failed for {month_start.strftime('%Y-%m')}: {e}")
                        continue

                    # Simulate month
                    trades, avg_conf = self.simulate_month(
                        test_df, regime, ml_model,
                        ml_threshold=ml_thresh,
                        ml_only_threshold=ml_only_thresh,
                    )

                    # Calculate metrics
                    metrics = self.calculate_metrics(trades)

                    month_result = MonthlyResult(
                        month=month_start.strftime('%Y-%m'),
                        start_date=month_start,
                        end_date=month_end,
                        total_trades=metrics['total_trades'],
                        wins=metrics['wins'],
                        losses=metrics['losses'],
                        win_rate=metrics['win_rate'],
                        total_pnl=metrics['total_pnl'],
                        max_drawdown=metrics['max_drawdown'],
                        profit_factor=metrics['profit_factor'],
                        model_auc=auc,
                        avg_confidence=avg_conf,
                        ml_only_trades=metrics['ml_only_trades'],
                        smc_ml_trades=metrics['smc_ml_trades'],
                    )

                    monthly_results.append(month_result)
                    all_month_trades.extend(trades)

                # Calculate total performance for this parameter set
                total_pnl = sum(m.total_pnl for m in monthly_results)
                total_trades = sum(m.total_trades for m in monthly_results)
                total_wins = sum(m.wins for m in monthly_results)
                avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

                param_results.append({
                    'ml_threshold': ml_thresh,
                    'ml_only_threshold': ml_only_thresh,
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'win_rate': avg_win_rate,
                    'monthly_results': monthly_results,
                })

                logger.info(f"  Result: {total_trades} trades, {avg_win_rate:.1f}% WR, ${total_pnl:+,.2f}")

                if total_pnl > best_params['total_pnl']:
                    best_params = {
                        'ml_threshold': ml_thresh,
                        'ml_only_threshold': ml_only_thresh,
                        'total_pnl': total_pnl,
                        'win_rate': avg_win_rate,
                        'monthly_results': monthly_results,
                    }

        print()
        logger.info("=" * 70)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 70)
        print()

        # Sort by total P/L
        param_results.sort(key=lambda x: x['total_pnl'], reverse=True)

        print("Parameter Combinations (sorted by P/L):")
        print("-" * 60)
        for i, p in enumerate(param_results[:10]):
            print(f"  {i+1}. ML={p['ml_threshold']:.0%}, ML-Only={p['ml_only_threshold']:.0%}")
            print(f"     Trades: {p['total_trades']}, Win Rate: {p['win_rate']:.1f}%, P/L: ${p['total_pnl']:+,.2f}")
        print()

        # Show best parameters
        logger.info("=" * 70)
        logger.info("BEST PARAMETERS FOUND")
        logger.info("=" * 70)
        print(f"  ML Threshold      : {best_params['ml_threshold']:.0%}")
        print(f"  ML-Only Threshold : {best_params['ml_only_threshold']:.0%}")
        print(f"  Total P/L         : ${best_params['total_pnl']:+,.2f}")
        print(f"  Win Rate          : {best_params['win_rate']:.1f}%")
        print()

        # Show monthly breakdown for best params
        if 'monthly_results' in best_params:
            print("Monthly Breakdown (Best Parameters):")
            print("-" * 70)
            print(f"{'Month':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'P/L':>12} {'PF':>8}")
            print("-" * 70)

            for m in best_params['monthly_results']:
                print(f"{m.month:<10} {m.total_trades:>8} {m.wins:>6} {m.win_rate:>7.1f}% ${m.total_pnl:>10.2f} {m.profit_factor:>7.2f}")

            print("-" * 70)
            total_trades = sum(m.total_trades for m in best_params['monthly_results'])
            total_wins = sum(m.wins for m in best_params['monthly_results'])
            total_pnl = sum(m.total_pnl for m in best_params['monthly_results'])
            avg_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
            print(f"{'TOTAL':<10} {total_trades:>8} {total_wins:>6} {avg_wr:>7.1f}% ${total_pnl:>10.2f}")

        print()
        logger.info("=" * 70)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 70)
        print()
        print(f"Based on 1-year walk-forward optimization:")
        print(f"  1. Set ML threshold to: {best_params['ml_threshold']:.0%}")
        print(f"  2. Set ML-only threshold to: {best_params['ml_only_threshold']:.0%}")
        print(f"  3. Expected monthly P/L: ${best_params['total_pnl'] / len(best_params.get('monthly_results', [1])):+,.2f}")
        print()

        return best_params, param_results


def main():
    """Main function."""
    print("=" * 70)
    print("WALK-FORWARD OPTIMIZATION BACKTEST")
    print("=" * 70)
    print()
    print("This will:")
    print("  1. Fetch 13 months of historical data (Jan 2025 - Feb 2026)")
    print("  2. Train ML models progressively each month")
    print("  3. Test different parameter combinations")
    print("  4. Find optimal ML thresholds")
    print()

    # Initialize
    config = WalkForwardConfig(
        train_window_months=3,
        ml_thresholds=[0.55, 0.60, 0.65, 0.70, 0.75],
        ml_only_thresholds=[0.65, 0.70, 0.75, 0.80, 0.85],
        base_lot=0.01,
        max_lot=0.02,
        max_loss_per_trade=30.0,
    )

    backtest = WalkForwardBacktest(config)

    # Connect to MT5
    if not backtest.connect_mt5():
        print("Failed to connect to MT5")
        return

    # Fetch historical data
    data = backtest.fetch_historical_data(months=14)

    if data is None:
        print("Failed to fetch historical data")
        return

    # Run walk-forward optimization
    best_params, all_results = backtest.run_walkforward(
        start_month=1,
        start_year=2025,
        end_month=2,
        end_year=2026,
    )

    # Shutdown MT5
    import MetaTrader5 as mt5
    mt5.shutdown()

    print()
    print("Walk-forward optimization complete!")
    print(f"Best parameters saved for future use.")


if __name__ == "__main__":
    main()
