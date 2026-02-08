"""
ML V2 Target Builder
====================
Better target variables to reduce noise and improve ML predictive power.

Problem with current target (src/feature_eng.py::create_target):
- Predicts 1-bar ahead price movement with threshold=0 → too noisy
- Captures noise, not tradeable moves

Solutions:
A. Multi-bar + ATR threshold (primary)
B. 3-class target (BUY/SELL/HOLD)
C. Baseline (current method for comparison)
"""

import polars as pl
import numpy as np
from typing import Tuple, Optional
from loguru import logger


class TargetBuilder:
    """
    Builder for improved target variables.

    Key improvements:
    1. Multi-bar lookahead (reduces noise)
    2. ATR-based threshold (filters small moves)
    3. 3-class option (explicit HOLD class)
    """

    def __init__(self):
        """Initialize target builder."""
        pass

    def create_multi_bar_target(
        self,
        df: pl.DataFrame,
        lookahead: int = 3,
        threshold_atr_mult: float = 0.3,
    ) -> pl.DataFrame:
        """
        Create multi-bar binary target with ATR-based threshold.

        Logic:
        - Look at next `lookahead` bars (default 3 = 45 min on M15)
        - Find max close in that window
        - UP (1) if: max_future_close - current_close > threshold * ATR
        - DOWN (0) if: current_close - min_future_close > threshold * ATR
        - Filtered out: moves smaller than threshold (noise)

        Why this works:
        - Multi-bar: reduces bar-to-bar noise
        - ATR threshold: filters moves too small to trade profitably
        - For XAUUSD @ ATR ~$12, threshold=0.3 means $3.6 minimum move

        Args:
            df: DataFrame with OHLCV and ATR
            lookahead: Number of bars to look ahead (default 3)
            threshold_atr_mult: ATR multiplier for minimum move (default 0.3)

        Returns:
            DataFrame with multi_bar_target column (1=UP, 0=DOWN, null=HOLD/filtered)
        """
        # Ensure ATR exists
        if "atr" not in df.columns:
            logger.error("ATR column required for multi-bar target")
            return df

        # Calculate future max/min close in lookahead window
        df = df.with_columns([
            # Rolling max of future closes (reverse window)
            pl.col("close").shift(-lookahead).alias("_future_start_close"),
            pl.col("close").shift(-1).alias("_future_1"),
            pl.col("close").shift(-2).alias("_future_2") if lookahead >= 2 else pl.col("close").alias("_future_2"),
            pl.col("close").shift(-3).alias("_future_3") if lookahead >= 3 else pl.col("close").alias("_future_3"),
        ])

        # Get max and min across future window
        if lookahead == 1:
            df = df.with_columns([
                pl.col("_future_1").alias("_max_future_close"),
                pl.col("_future_1").alias("_min_future_close"),
            ])
        elif lookahead == 2:
            df = df.with_columns([
                pl.max_horizontal("_future_1", "_future_2").alias("_max_future_close"),
                pl.min_horizontal("_future_1", "_future_2").alias("_min_future_close"),
            ])
        else:  # lookahead >= 3
            df = df.with_columns([
                pl.max_horizontal("_future_1", "_future_2", "_future_3").alias("_max_future_close"),
                pl.min_horizontal("_future_1", "_future_2", "_future_3").alias("_min_future_close"),
            ])

        # Calculate move sizes
        df = df.with_columns([
            (pl.col("_max_future_close") - pl.col("close")).alias("_up_move"),
            (pl.col("close") - pl.col("_min_future_close")).alias("_down_move"),
        ])

        # Calculate threshold (ATR * multiplier)
        df = df.with_columns([
            (pl.col("atr") * threshold_atr_mult).alias("_threshold"),
        ])

        # Create target:
        # - UP (1): if up_move > threshold AND up_move > down_move
        # - DOWN (0): if down_move > threshold AND down_move > up_move
        # - null: otherwise (filtered as noise)
        df = df.with_columns([
            pl.when(
                (pl.col("_up_move") > pl.col("_threshold")) &
                (pl.col("_up_move") > pl.col("_down_move"))
            )
                .then(1)
                .when(
                    (pl.col("_down_move") > pl.col("_threshold")) &
                    (pl.col("_down_move") > pl.col("_up_move"))
                )
                .then(0)
                .otherwise(None)  # Filter out noise
                .alias("multi_bar_target")
                .cast(pl.Int32),
        ])

        # Drop temporary columns
        df = df.drop([
            "_future_start_close", "_future_1", "_future_2", "_future_3",
            "_max_future_close", "_min_future_close",
            "_up_move", "_down_move", "_threshold"
        ])

        # Log statistics
        total = len(df)
        ups = df.filter(pl.col("multi_bar_target") == 1).height
        downs = df.filter(pl.col("multi_bar_target") == 0).height
        filtered = total - ups - downs

        logger.info(
            f"Multi-bar target (lookahead={lookahead}, threshold={threshold_atr_mult}*ATR): "
            f"{ups} UP ({ups/total*100:.1f}%), "
            f"{downs} DOWN ({downs/total*100:.1f}%), "
            f"{filtered} filtered ({filtered/total*100:.1f}%)"
        )

        return df

    def create_3class_target(
        self,
        df: pl.DataFrame,
        lookahead: int = 3,
        threshold_atr_mult: float = 0.3,
    ) -> pl.DataFrame:
        """
        Create 3-class target: BUY (0), SELL (1), HOLD (2).

        Same logic as multi_bar_target but keeps HOLD as explicit class
        instead of filtering it out.

        Use with XGBoost multi:softprob objective.

        Args:
            df: DataFrame with OHLCV and ATR
            lookahead: Number of bars to look ahead
            threshold_atr_mult: ATR multiplier for threshold

        Returns:
            DataFrame with target_3class column (0=BUY, 1=SELL, 2=HOLD)
        """
        # Reuse multi_bar logic but map null to HOLD (2)
        df = self.create_multi_bar_target(df, lookahead, threshold_atr_mult)

        # Convert to 3-class: 0=BUY, 1=SELL, 2=HOLD
        df = df.with_columns([
            pl.when(pl.col("multi_bar_target") == 1)
                .then(0)  # UP → BUY
                .when(pl.col("multi_bar_target") == 0)
                .then(1)  # DOWN → SELL
                .otherwise(2)  # null → HOLD
                .alias("target_3class")
                .cast(pl.Int32),
        ])

        # Log distribution
        total = len(df)
        buys = df.filter(pl.col("target_3class") == 0).height
        sells = df.filter(pl.col("target_3class") == 1).height
        holds = df.filter(pl.col("target_3class") == 2).height

        logger.info(
            f"3-class target: "
            f"{buys} BUY ({buys/total*100:.1f}%), "
            f"{sells} SELL ({sells/total*100:.1f}%), "
            f"{holds} HOLD ({holds/total*100:.1f}%)"
        )

        return df

    def create_baseline_target(
        self,
        df: pl.DataFrame,
        lookahead: int = 1,
        threshold: float = 0.0,
    ) -> pl.DataFrame:
        """
        Create baseline target (mirrors current FeatureEngineer.create_target()).

        For comparison with V1 model.

        Args:
            df: DataFrame with price data
            lookahead: Bars to look ahead (default 1)
            threshold: Minimum return threshold (default 0.0)

        Returns:
            DataFrame with baseline_target column
        """
        df = df.with_columns([
            pl.col("close").shift(-lookahead).alias("_future_close"),
        ])

        df = df.with_columns([
            ((pl.col("_future_close") / pl.col("close") - 1) > threshold)
                .cast(pl.Int32)
                .alias("baseline_target"),
        ])

        df = df.drop(["_future_close"])

        ups = df.filter(pl.col("baseline_target") == 1).height
        total = len(df)
        logger.info(
            f"Baseline target (lookahead={lookahead}, threshold={threshold}): "
            f"{ups} UP ({ups/total*100:.1f}%), {total-ups} DOWN ({(total-ups)/total*100:.1f}%)"
        )

        return df

    def create_all_targets(
        self,
        df: pl.DataFrame,
        lookahead: int = 3,
        threshold_atr_mult: float = 0.3,
    ) -> pl.DataFrame:
        """
        Create all target variants for comparison.

        Args:
            df: DataFrame with OHLCV and ATR
            lookahead: Lookahead for multi-bar targets
            threshold_atr_mult: ATR threshold multiplier

        Returns:
            DataFrame with all target columns added
        """
        logger.info(f"Creating all target variants (lookahead={lookahead}, threshold={threshold_atr_mult}*ATR)...")

        # Baseline (V1)
        df = self.create_baseline_target(df, lookahead=1, threshold=0.0)

        # Multi-bar binary
        df = self.create_multi_bar_target(df, lookahead=lookahead, threshold_atr_mult=threshold_atr_mult)

        # 3-class
        df = self.create_3class_target(df, lookahead=lookahead, threshold_atr_mult=threshold_atr_mult)

        return df


if __name__ == "__main__":
    # Test target builder
    import numpy as np
    from datetime import datetime, timedelta

    # Create synthetic OHLCV data with trend
    np.random.seed(42)
    n = 500

    base_price = 2000.0
    # Add uptrend
    trend = np.linspace(0, 50, n)
    noise = np.random.randn(n) * 5
    prices = base_price + trend + noise

    # Create ATR (realistic for XAUUSD)
    atr_values = np.random.uniform(10, 14, n)

    df = pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)) * 2,
        "low": prices - np.abs(np.random.randn(n)) * 2,
        "close": prices + np.random.randn(n) * 1,
        "volume": np.random.randint(1000, 10000, n),
        "atr": atr_values,
    })

    # Build targets
    builder = TargetBuilder()
    df = builder.create_all_targets(df, lookahead=3, threshold_atr_mult=0.3)

    # Show comparison
    print("\n=== Target Builder Test ===")
    print(f"Total bars: {len(df)}")
    print(f"\nTarget columns created:")
    print(f"  - baseline_target (1-bar, threshold=0)")
    print(f"  - multi_bar_target (3-bar, 0.3*ATR threshold)")
    print(f"  - target_3class (3-class version)")

    # Sample
    print("\n=== Sample Data (Last 10 Rows) ===")
    cols = ["time", "close", "atr", "baseline_target", "multi_bar_target", "target_3class"]
    print(df.select([c for c in cols if c in df.columns]).tail(10))

    # Class distribution comparison
    print("\n=== Class Distribution ===")

    baseline_up = df.filter(pl.col("baseline_target") == 1).height
    baseline_down = len(df) - baseline_up
    print(f"Baseline: {baseline_up} UP, {baseline_down} DOWN")

    multi_up = df.filter(pl.col("multi_bar_target") == 1).height
    multi_down = df.filter(pl.col("multi_bar_target") == 0).height
    multi_filtered = len(df) - multi_up - multi_down
    print(f"Multi-bar: {multi_up} UP, {multi_down} DOWN, {multi_filtered} filtered")

    class3_buy = df.filter(pl.col("target_3class") == 0).height
    class3_sell = df.filter(pl.col("target_3class") == 1).height
    class3_hold = df.filter(pl.col("target_3class") == 2).height
    print(f"3-class: {class3_buy} BUY, {class3_sell} SELL, {class3_hold} HOLD")
