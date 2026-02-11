"""
Advanced Target Labeling for ML Model V3 - BINARY CLASSIFICATION
=================================================================
Implements Triple Barrier Method for high-quality BUY vs SELL signals.

Key improvements over V2:
1. Triple barrier: profit target, stop loss, time limit
2. Binary classification: BUY (1) vs SELL (0) only - no HOLD class
3. ATR-adaptive thresholds for balanced labeling
4. Class balancing to 50/50 distribution
5. Time barrier labels by final direction (always directional)

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

import polars as pl
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TripleBarrierLabeling:
    """
    Binary classification using triple barrier method.

    For each bar, we define:
    - Upper barrier (profit target): +profit_atr_mult * ATR
    - Lower barrier (stop loss): -stoploss_atr_mult * ATR
    - Vertical barrier (time limit): max_holding_bars

    Label = BUY (1) if upper barrier hit first or time barrier with positive return
            SELL (0) if lower barrier hit first or time barrier with negative return
    """

    def __init__(
        self,
        profit_atr_mult: float = 0.5,     # 50% of ATR for TP (balanced)
        stoploss_atr_mult: float = 0.5,   # 50% of ATR for SL (symmetric RR 1.0)
        max_holding_bars: int = 20,       # 5 hours on M15 (allow time to develop)
    ):
        self.profit_atr_mult = profit_atr_mult
        self.stoploss_atr_mult = stoploss_atr_mult
        self.max_holding_bars = max_holding_bars

    def label_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply triple barrier labeling to DataFrame (BINARY classification).

        Args:
            df: DataFrame with columns ['close', 'high', 'low', 'atr']

        Returns:
            DataFrame with additional columns:
            - target: 1 (BUY), 0 (SELL) - BINARY only, no HOLD
            - target_label: "BUY" or "SELL"
            - barrier_hit: which barrier was hit first
            - bars_to_barrier: how many bars until barrier hit
            - return_pct: actual return achieved (ATR-normalized)
        """
        print(f" Starting Triple Barrier Labeling (BINARY: BUY vs SELL)...")
        print(f"   Profit target: {self.profit_atr_mult} ATR")
        print(f"   Stop loss: {self.stoploss_atr_mult} ATR")
        print(f"   Max holding: {self.max_holding_bars} bars")

        # Convert to numpy for speed
        closes = df["close"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        atrs = df["atr"].to_numpy()

        n = len(df)
        targets = np.zeros(n, dtype=np.int8)
        barriers_hit = np.zeros(n, dtype='U10')  # 'profit', 'stoploss', 'time', 'none'
        bars_to_barrier = np.zeros(n, dtype=np.int32)
        returns_pct = np.zeros(n, dtype=np.float32)

        # For each bar, scan forward to find first barrier hit
        for i in range(n - self.max_holding_bars):
            entry_price = closes[i]
            entry_atr = atrs[i]

            if entry_atr == 0 or np.isnan(entry_atr):
                barriers_hit[i] = 'none'
                continue

            # Define barriers
            upper_barrier = entry_price + (self.profit_atr_mult * entry_atr)
            lower_barrier = entry_price - (self.stoploss_atr_mult * entry_atr)

            # Scan forward
            barrier_found = False
            for j in range(1, self.max_holding_bars + 1):
                if i + j >= n:
                    break

                future_high = highs[i + j]
                future_low = lows[i + j]
                future_close = closes[i + j]

                # Check upper barrier (BUY signal if hit first)
                if future_high >= upper_barrier:
                    targets[i] = 1  # BUY
                    barriers_hit[i] = 'profit_long'
                    bars_to_barrier[i] = j
                    returns_pct[i] = (upper_barrier - entry_price) / entry_atr
                    barrier_found = True
                    break

                # Check lower barrier (SELL signal if hit first)
                if future_low <= lower_barrier:
                    targets[i] = 0  # SELL (binary: 0)
                    barriers_hit[i] = 'profit_short'
                    bars_to_barrier[i] = j
                    returns_pct[i] = (entry_price - lower_barrier) / entry_atr
                    barrier_found = True
                    break

            # If no barrier hit within time limit - use time barrier
            if not barrier_found:
                final_price = closes[min(i + self.max_holding_bars, n - 1)]
                return_atr = (final_price - entry_price) / entry_atr

                # Time barrier: ALWAYS label by final direction (no HOLD for binary)
                targets[i] = 1 if return_atr >= 0 else 0  # BUY if positive, SELL if negative
                barriers_hit[i] = 'time_up' if return_atr >= 0 else 'time_down'
                bars_to_barrier[i] = self.max_holding_bars
                returns_pct[i] = return_atr

        # Last few bars cannot be labeled (no forward data) - mark as unlabeled (-1)
        targets[-self.max_holding_bars:] = -1
        barriers_hit[-self.max_holding_bars:] = 'no_data'

        # Add to DataFrame
        df = df.with_columns([
            pl.Series("target", targets),
            pl.Series("barrier_hit", barriers_hit),
            pl.Series("bars_to_barrier", bars_to_barrier),
            pl.Series("return_pct", returns_pct),
        ])

        # Add text labels (binary: BUY=1, SELL=0, unlabeled=-1)
        df = df.with_columns([
            pl.when(pl.col("target") == 1).then(pl.lit("BUY"))
              .when(pl.col("target") == 0).then(pl.lit("SELL"))
              .otherwise(pl.lit("UNLABELED"))
              .alias("target_label")
        ])

        # Stats (exclude unlabeled from distribution)
        labeled_mask = targets >= 0
        n_buy = (targets[labeled_mask] == 1).sum()
        n_sell = (targets[labeled_mask] == 0).sum()
        n_unlabeled = (targets == -1).sum()
        n_total = n_buy + n_sell

        print(f"\n Target Distribution (BINARY):")
        print(f"   BUY:  {n_buy:6d} ({n_buy/n_total*100:5.2f}%)")
        print(f"   SELL: {n_sell:6d} ({n_sell/n_total*100:5.2f}%)")
        print(f"   Unlabeled: {n_unlabeled:6d} (last {self.max_holding_bars} bars)")

        # Quality metrics
        profit_barriers = (barriers_hit == 'profit_long') | (barriers_hit == 'profit_short')
        avg_bars_profit = bars_to_barrier[profit_barriers].mean() if profit_barriers.sum() > 0 else 0
        avg_return_profit = returns_pct[profit_barriers].mean() if profit_barriers.sum() > 0 else 0

        print(f"\n Quality Metrics:")
        print(f"   Profit barriers hit: {profit_barriers.sum():6d} ({profit_barriers.sum()/n_total*100:5.2f}%)")
        print(f"   Avg bars to profit:  {avg_bars_profit:.1f}")
        print(f"   Avg return (ATR):    {avg_return_profit:.3f}")

        return df

    def apply_meta_labeling(
        self,
        df: pl.DataFrame,
        smc_signal_col: str = "smc_signal",
        smc_confidence_col: str = "smc_confidence",
        min_smc_confidence: float = 0.65,
    ) -> pl.DataFrame:
        """
        Meta-labeling: refine targets using SMC signal quality.

        If triple-barrier says BUY but SMC says SELL (or vice versa) with high confidence,
        flip to HOLD (conflicting signals = don't trade).

        Args:
            df: DataFrame with target column
            smc_signal_col: column with SMC signal ("BUY", "SELL", or "")
            smc_confidence_col: column with SMC confidence (0-1)
            min_smc_confidence: min confidence to trust SMC signal

        Returns:
            DataFrame with refined target column
        """
        print(f"\n Applying Meta-Labeling (SMC signal quality)...")

        if smc_signal_col not in df.columns or smc_confidence_col not in df.columns:
            print("     SMC columns not found, skipping meta-labeling")
            return df

        # Count conflicts before
        conflicts_before = 0

        # Refine targets
        refined_targets = []
        for row in df.iter_rows(named=True):
            target = row["target"]
            target_label = row["target_label"]
            smc_signal = row.get(smc_signal_col, "")
            smc_conf = row.get(smc_confidence_col, 0.0)

            # If no strong SMC signal, keep original target
            if not smc_signal or smc_conf < min_smc_confidence:
                refined_targets.append(target)
                continue

            # Check for conflict
            if target_label == "BUY" and smc_signal == "SELL":
                conflicts_before += 1
                refined_targets.append(0)  # HOLD (conflicting signals)
            elif target_label == "SELL" and smc_signal == "BUY":
                conflicts_before += 1
                refined_targets.append(0)  # HOLD (conflicting signals)
            else:
                refined_targets.append(target)  # Keep original

        df = df.with_columns([
            pl.Series("target", refined_targets)
        ])

        # Recalculate target_label
        df = df.with_columns([
            pl.when(pl.col("target") == 1).then(pl.lit("BUY"))
              .when(pl.col("target") == -1).then(pl.lit("SELL"))
              .otherwise(pl.lit("HOLD"))
              .alias("target_label")
        ])

        print(f"   Conflicts resolved: {conflicts_before} (BUYSELL  HOLD)")

        # New distribution
        n_buy = df.filter(pl.col("target") == 1).height
        n_sell = df.filter(pl.col("target") == -1).height
        n_hold = df.filter(pl.col("target") == 0).height
        n_total = n_buy + n_sell + n_hold

        print(f"\n Refined Target Distribution:")
        print(f"   BUY:  {n_buy:6d} ({n_buy/n_total*100:5.2f}%)")
        print(f"   SELL: {n_sell:6d} ({n_sell/n_total*100:5.2f}%)")
        print(f"   HOLD: {n_hold:6d} ({n_hold/n_total*100:5.2f}%)")

        return df

    def balance_classes(
        self,
        df: pl.DataFrame,
        target_buy_pct: float = 0.50,
        target_sell_pct: float = 0.50,
        random_seed: int = 42,
    ) -> pl.DataFrame:
        """
        Balance target classes via stratified downsampling (BINARY: BUY vs SELL).

        Args:
            df: DataFrame with target column (1=BUY, 0=SELL)
            target_buy_pct: desired % of BUY samples (default 50%)
            target_sell_pct: desired % of SELL samples (default 50%)
            random_seed: for reproducibility

        Returns:
            Balanced DataFrame
        """
        print(f"\n  Balancing Classes (BINARY)...")
        print(f"   Target distribution: BUY={target_buy_pct*100:.0f}%, SELL={target_sell_pct*100:.0f}%")

        # Filter labeled data only (exclude -1 = unlabeled)
        df_labeled = df.filter(pl.col("target") >= 0)

        df_buy = df_labeled.filter(pl.col("target") == 1)
        df_sell = df_labeled.filter(pl.col("target") == 0)

        n_buy = df_buy.height
        n_sell = df_sell.height

        # Find minority class size
        min_count = min(n_buy, n_sell)

        # Calculate target counts to achieve desired distribution
        # Use minority class as anchor
        total_target = int(min_count / min(target_buy_pct, target_sell_pct))
        n_buy_target = int(total_target * target_buy_pct)
        n_sell_target = int(total_target * target_sell_pct)

        # Sample each class
        if n_buy > n_buy_target:
            df_buy = df_buy.sample(n=n_buy_target, seed=random_seed)
        if n_sell > n_sell_target:
            df_sell = df_sell.sample(n=n_sell_target, seed=random_seed)

        # Combine
        df_balanced = pl.concat([df_buy, df_sell])

        # Shuffle
        df_balanced = df_balanced.sample(fraction=1.0, seed=random_seed)

        print(f"   Before: BUY={n_buy}, SELL={n_sell}")
        print(f"   After:  BUY={df_buy.height}, SELL={df_sell.height}")
        print(f"   Total samples: {df_balanced.height}")

        return df_balanced


if __name__ == "__main__":
    # Test on sample data
    from src.mt5_connector import MT5Connector
    from src.config import TradingConfig
    from src.feature_eng import FeatureEngineer

    config = TradingConfig()
    mt5 = MT5Connector(config)
    mt5.connect()

    # Fetch data
    df = mt5.get_market_data(symbol="XAUUSD", timeframe="M15", count=10000)
    print(f"Fetched {len(df)} bars")

    # Calculate features (need ATR)
    fe = FeatureEngineer()
    df = fe.calculate_all(df, include_ml_features=False)

    # Apply labeling
    labeler = TripleBarrierLabeling(
        profit_atr_mult=0.20,
        stoploss_atr_mult=0.15,
        max_holding_bars=8,
        min_move_threshold=0.10,
    )

    df = labeler.label_data(df)

    # Save
    output_path = Path("backtests/ml_v3/labeled_data_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"\n Saved to {output_path}")
