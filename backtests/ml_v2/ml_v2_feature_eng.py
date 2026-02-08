"""
ML V2 Feature Engineering
==========================
23 new features on top of the base 37 features.

New feature categories:
1. H1 Multi-Timeframe (8 features) - Higher timeframe context
2. Continuous SMC (7 features) - SMC as continuous values instead of binary
3. Regime Conditioning (4 features) - Regime-based features
4. Price Action (4 features) - Candle patterns and momentum

Total: 37 (base) + 23 (new) = 60 features
"""

import polars as pl
import numpy as np
from typing import List, Optional
from loguru import logger


class MLV2FeatureEngineer:
    """
    V2 Feature Engineer with 23 additional features.

    Builds on top of base FeatureEngineer (37 features).
    """

    def __init__(self):
        """Initialize V2 feature engineer."""
        pass

    # =========================================================================
    # H1 MULTI-TIMEFRAME FEATURES (8 features)
    # =========================================================================

    def add_h1_features(
        self,
        df_m15: pl.DataFrame,
        df_h1: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add H1 (higher timeframe) features to M15 data.

        Uses join_asof to merge H1 data into M15 without lookahead bias.

        Features added (8 total):
        - h1_market_structure: H1 BOS-based trend (1/-1/0)
        - h1_ema20_distance: (M15 close - H1 EMA20) / ATR
        - h1_trend_strength: Count of H1 BOS in same direction
        - h1_swing_proximity: Distance to nearest H1 swing / ATR
        - h1_fvg_active: 1 if price inside H1 FVG zone
        - h1_ob_proximity: Distance to H1 order block / ATR
        - h1_atr_ratio: H1 ATR / M15 ATR
        - h1_rsi: H1 RSI value

        Args:
            df_m15: M15 DataFrame (must have 'time', 'close', 'atr')
            df_h1: H1 DataFrame (must have indicators calculated)

        Returns:
            M15 DataFrame with H1 features added
        """
        if df_h1 is None or len(df_h1) == 0:
            logger.warning("H1 data empty, skipping H1 features")
            return df_m15

        # Ensure both have time column
        if "time" not in df_m15.columns or "time" not in df_h1.columns:
            logger.error("Both DataFrames must have 'time' column")
            return df_m15

        # Calculate H1 EMA20
        if "close" in df_h1.columns:
            df_h1 = df_h1.with_columns([
                pl.col("close")
                    .ewm_mean(span=20, adjust=False)
                    .alias("h1_ema20"),
            ])

        # Select H1 columns to join
        h1_cols = ["time"]
        h1_features = {}

        # H1 market structure
        if "market_structure" in df_h1.columns:
            h1_cols.append("market_structure")
            h1_features["h1_market_structure"] = "market_structure"

        # H1 EMA20
        if "h1_ema20" in df_h1.columns:
            h1_cols.append("h1_ema20")

        # H1 ATR
        if "atr" in df_h1.columns:
            h1_cols.append("atr")
            h1_features["h1_atr"] = "atr"

        # H1 RSI
        if "rsi" in df_h1.columns:
            h1_cols.append("rsi")
            h1_features["h1_rsi"] = "rsi"

        # H1 swing levels
        if "last_swing_high" in df_h1.columns and "last_swing_low" in df_h1.columns:
            h1_cols.extend(["last_swing_high", "last_swing_low"])

        # H1 FVG
        if "fvg_top" in df_h1.columns and "fvg_bottom" in df_h1.columns:
            h1_cols.extend(["fvg_top", "fvg_bottom"])

        # H1 OB
        if "ob_top" in df_h1.columns and "ob_bottom" in df_h1.columns:
            h1_cols.extend(["ob_top", "ob_bottom"])

        # H1 BOS for trend strength
        if "bos" in df_h1.columns:
            h1_cols.append("bos")

        # Prepare H1 data for join
        df_h1_join = df_h1.select([c for c in h1_cols if c in df_h1.columns])

        # Join H1 to M15 using join_asof (backward looking, no lookahead)
        df_m15 = df_m15.join_asof(
            df_h1_join,
            on="time",
            strategy="backward",  # Use most recent H1 bar
            suffix="_h1",
        )

        # === Feature 1: H1 Market Structure ===
        if "market_structure_h1" in df_m15.columns:
            df_m15 = df_m15.rename({"market_structure_h1": "h1_market_structure"})
        elif "h1_market_structure" not in df_m15.columns:
            df_m15 = df_m15.with_columns([
                pl.lit(0).alias("h1_market_structure"),
            ])

        # === Feature 2: H1 EMA20 Distance (normalized by ATR) ===
        if "h1_ema20" in df_m15.columns and "atr" in df_m15.columns:
            df_m15 = df_m15.with_columns([
                ((pl.col("close") - pl.col("h1_ema20")) / pl.col("atr"))
                    .alias("h1_ema20_distance"),
            ])
        else:
            df_m15 = df_m15.with_columns([pl.lit(0.0).alias("h1_ema20_distance")])

        # === Feature 3: H1 Trend Strength (BOS count in last 10 H1 bars) ===
        # We can't do rolling sum on joined data, so use a proxy:
        # Check if H1 BOS is present
        if "bos_h1" in df_m15.columns:
            # Simplification: just use current H1 BOS value as proxy
            df_m15 = df_m15.with_columns([
                pl.col("bos_h1").fill_null(0).alias("h1_trend_strength"),
            ])
        else:
            df_m15 = df_m15.with_columns([pl.lit(0).alias("h1_trend_strength")])

        # === Feature 4: H1 Swing Proximity ===
        if "last_swing_high_h1" in df_m15.columns and "last_swing_low_h1" in df_m15.columns:
            df_m15 = df_m15.with_columns([
                # Distance to nearest swing (high or low)
                pl.min_horizontal(
                    (pl.col("last_swing_high_h1") - pl.col("close")).abs(),
                    (pl.col("close") - pl.col("last_swing_low_h1")).abs()
                ).alias("_swing_dist"),
            ])

            # Normalize by ATR and create final feature
            if "atr" in df_m15.columns:
                df_m15 = df_m15.with_columns([
                    (pl.col("_swing_dist") / pl.col("atr")).alias("h1_swing_proximity"),
                ]).drop(["_swing_dist"])
            else:
                df_m15 = df_m15.rename({"_swing_dist": "h1_swing_proximity"})
        else:
            df_m15 = df_m15.with_columns([pl.lit(0.0).alias("h1_swing_proximity")])

        # === Feature 5: H1 FVG Active ===
        if "fvg_top_h1" in df_m15.columns and "fvg_bottom_h1" in df_m15.columns:
            df_m15 = df_m15.with_columns([
                pl.when(
                    (pl.col("close") >= pl.col("fvg_bottom_h1")) &
                    (pl.col("close") <= pl.col("fvg_top_h1"))
                )
                    .then(1)
                    .otherwise(0)
                    .alias("h1_fvg_active"),
            ])
        else:
            df_m15 = df_m15.with_columns([pl.lit(0).alias("h1_fvg_active")])

        # === Feature 6: H1 OB Proximity ===
        if "ob_top_h1" in df_m15.columns and "ob_bottom_h1" in df_m15.columns:
            df_m15 = df_m15.with_columns([
                # Distance to OB center
                (((pl.col("ob_top_h1") + pl.col("ob_bottom_h1")) / 2 - pl.col("close")).abs())
                    .alias("_ob_dist"),
            ])

            if "atr" in df_m15.columns:
                df_m15 = df_m15.with_columns([
                    (pl.col("_ob_dist") / pl.col("atr")).alias("h1_ob_proximity"),
                ])
            else:
                df_m15 = df_m15.rename({"_ob_dist": "h1_ob_proximity"})

            df_m15 = df_m15.drop(["_ob_dist"])
        else:
            df_m15 = df_m15.with_columns([pl.lit(0.0).alias("h1_ob_proximity")])

        # === Feature 7: H1 ATR Ratio ===
        if "atr_h1" in df_m15.columns and "atr" in df_m15.columns:
            df_m15 = df_m15.with_columns([
                (pl.col("atr_h1") / pl.col("atr")).fill_null(1.0).alias("h1_atr_ratio"),
            ])
        else:
            df_m15 = df_m15.with_columns([pl.lit(1.0).alias("h1_atr_ratio")])

        # === Feature 8: H1 RSI ===
        if "rsi_h1" in df_m15.columns:
            df_m15 = df_m15.rename({"rsi_h1": "h1_rsi"})
        else:
            df_m15 = df_m15.with_columns([pl.lit(50.0).alias("h1_rsi")])

        # Clean up temporary H1 columns
        cols_to_drop = [
            c for c in df_m15.columns
            if c.endswith("_h1") and c not in ["h1_market_structure", "h1_rsi"]
        ]
        if cols_to_drop:
            df_m15 = df_m15.drop(cols_to_drop)

        logger.debug("H1 features added (8 features)")
        return df_m15

    # =========================================================================
    # CONTINUOUS SMC FEATURES (7 features)
    # =========================================================================

    def add_continuous_smc_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert binary SMC signals to continuous features.

        Features added (7 total):
        - fvg_gap_size_atr: FVG gap size / ATR
        - fvg_age_bars: Bars since last FVG (fresher = better)
        - ob_width_atr: OB width / ATR
        - ob_distance_atr: Distance to nearest OB / ATR
        - bos_recency: Bars since last BOS
        - confluence_score: Count of SMC signals in last 10 bars
        - swing_distance_atr: Distance to swing level / ATR

        Args:
            df: DataFrame with SMC columns

        Returns:
            DataFrame with continuous SMC features added
        """
        # === Feature 1: FVG Gap Size ===
        if "fvg_top" in df.columns and "fvg_bottom" in df.columns and "atr" in df.columns:
            df = df.with_columns([
                ((pl.col("fvg_top") - pl.col("fvg_bottom")) / pl.col("atr"))
                    .fill_null(0.0)
                    .alias("fvg_gap_size_atr"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("fvg_gap_size_atr")])

        # === Feature 2: FVG Age (bars since last FVG) ===
        if "fvg_signal" in df.columns:
            # Create row number index
            df = df.with_row_count("_row_idx")

            # Find last FVG index for each row
            df = df.with_columns([
                pl.when(pl.col("fvg_signal") != 0)
                    .then(pl.col("_row_idx"))
                    .otherwise(None)
                    .alias("_last_fvg_idx"),
            ])

            # Forward fill last FVG index
            df = df.with_columns([
                pl.col("_last_fvg_idx").forward_fill().alias("_last_fvg_idx_ff"),
            ])

            # Calculate age
            df = df.with_columns([
                (pl.col("_row_idx") - pl.col("_last_fvg_idx_ff"))
                    .fill_null(999)
                    .alias("fvg_age_bars"),
            ])

            df = df.drop(["_row_idx", "_last_fvg_idx", "_last_fvg_idx_ff"])
        else:
            df = df.with_columns([pl.lit(999).alias("fvg_age_bars")])

        # === Feature 3: OB Width ===
        if "ob_top" in df.columns and "ob_bottom" in df.columns and "atr" in df.columns:
            df = df.with_columns([
                ((pl.col("ob_top") - pl.col("ob_bottom")) / pl.col("atr"))
                    .fill_null(0.0)
                    .alias("ob_width_atr"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("ob_width_atr")])

        # === Feature 4: OB Distance ===
        if "ob_top" in df.columns and "ob_bottom" in df.columns and "atr" in df.columns:
            df = df.with_columns([
                # Distance to OB center
                (((pl.col("ob_top") + pl.col("ob_bottom")) / 2 - pl.col("close")).abs() / pl.col("atr"))
                    .fill_null(999.0)
                    .alias("ob_distance_atr"),
            ])
        else:
            df = df.with_columns([pl.lit(999.0).alias("ob_distance_atr")])

        # === Feature 5: BOS Recency ===
        if "bos" in df.columns:
            df = df.with_row_count("_row_idx")

            df = df.with_columns([
                pl.when(pl.col("bos") != 0)
                    .then(pl.col("_row_idx"))
                    .otherwise(None)
                    .alias("_last_bos_idx"),
            ])

            df = df.with_columns([
                pl.col("_last_bos_idx").forward_fill().alias("_last_bos_idx_ff"),
            ])

            df = df.with_columns([
                (pl.col("_row_idx") - pl.col("_last_bos_idx_ff"))
                    .fill_null(999)
                    .alias("bos_recency"),
            ])

            df = df.drop(["_row_idx", "_last_bos_idx", "_last_bos_idx_ff"])
        else:
            df = df.with_columns([pl.lit(999).alias("bos_recency")])

        # === Feature 6: Confluence Score ===
        # Count OB + FVG + BOS + CHoCH in last 10 bars
        smc_signals = []
        if "ob" in df.columns:
            smc_signals.append("_ob_signal")
            df = df.with_columns([
                (pl.col("ob").abs() > 0).cast(pl.Int8).alias("_ob_signal"),
            ])
        if "fvg_signal" in df.columns or "is_fvg_bull" in df.columns:
            if "fvg_signal" in df.columns:
                smc_signals.append("_fvg_signal")
                df = df.with_columns([
                    (pl.col("fvg_signal").abs() > 0).cast(pl.Int8).alias("_fvg_signal"),
                ])
            else:
                smc_signals.append("_fvg_signal")
                df = df.with_columns([
                    (pl.col("is_fvg_bull") | pl.col("is_fvg_bear")).cast(pl.Int8).alias("_fvg_signal"),
                ])
        if "bos" in df.columns:
            smc_signals.append("_bos_signal")
            df = df.with_columns([
                (pl.col("bos").abs() > 0).cast(pl.Int8).alias("_bos_signal"),
            ])
        if "choch" in df.columns:
            smc_signals.append("_choch_signal")
            df = df.with_columns([
                (pl.col("choch").abs() > 0).cast(pl.Int8).alias("_choch_signal"),
            ])

        if smc_signals:
            # Sum all signals in rolling window
            total_expr = pl.lit(0)
            for sig in smc_signals:
                total_expr = total_expr + pl.col(sig).rolling_sum(window_size=10, min_periods=1)

            df = df.with_columns([
                total_expr.alias("confluence_score"),
            ])

            # Drop temp columns
            df = df.drop(smc_signals)
        else:
            df = df.with_columns([pl.lit(0).alias("confluence_score")])

        # === Feature 7: Swing Distance ===
        # Skip if h1_swing_proximity already exists (from H1 features)
        if "h1_swing_proximity" not in df.columns:
            if "last_swing_high" in df.columns and "last_swing_low" in df.columns and "atr" in df.columns:
                df = df.with_columns([
                    # Distance to nearest swing
                    (pl.min_horizontal(
                        (pl.col("last_swing_high") - pl.col("close")).abs(),
                        (pl.col("close") - pl.col("last_swing_low")).abs()
                    ) / pl.col("atr"))
                        .fill_null(999.0)
                        .alias("swing_distance_atr"),
                ])
            else:
                df = df.with_columns([pl.lit(999.0).alias("swing_distance_atr")])
        else:
            # Use existing h1_swing_proximity as swing_distance_atr
            df = df.with_columns([
                pl.col("h1_swing_proximity").alias("swing_distance_atr"),
            ])

        logger.debug("Continuous SMC features added (7 features)")
        return df

    # =========================================================================
    # REGIME CONDITIONING FEATURES (4 features)
    # =========================================================================

    def add_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add regime-based conditioning features.

        Features added (4 total):
        - regime_duration_bars: Consecutive bars in current regime
        - regime_transition_prob: 1 / duration (proxy for change probability)
        - volatility_zscore: (ATR - mean50) / std50
        - crisis_proximity: ATR / (mean_ATR * 2.5)

        Args:
            df: DataFrame with regime and ATR columns

        Returns:
            DataFrame with regime features added
        """
        # === Feature 1 & 2: Regime Duration & Transition Prob ===
        if "regime" in df.columns:
            # Calculate consecutive bars in same regime
            df = df.with_columns([
                # Create regime change flag
                (pl.col("regime") != pl.col("regime").shift(1)).alias("_regime_change"),
            ])

            # Cumsum of changes to create regime groups
            df = df.with_columns([
                pl.col("_regime_change").cum_sum().alias("_regime_group"),
            ])

            # Count bars in each group
            df = df.with_columns([
                pl.col("_regime_group").count().over("_regime_group").alias("regime_duration_bars"),
            ])

            # Transition probability (inverse of duration)
            df = df.with_columns([
                (1.0 / pl.col("regime_duration_bars")).alias("regime_transition_prob"),
            ])

            df = df.drop(["_regime_change", "_regime_group"])
        else:
            df = df.with_columns([
                pl.lit(1).alias("regime_duration_bars"),
                pl.lit(1.0).alias("regime_transition_prob"),
            ])

        # === Feature 3: Volatility Z-Score ===
        if "atr" in df.columns:
            df = df.with_columns([
                pl.col("atr").rolling_mean(window_size=50, min_periods=1).alias("_atr_mean50"),
                pl.col("atr").rolling_std(window_size=50, min_periods=1).alias("_atr_std50"),
            ])

            df = df.with_columns([
                ((pl.col("atr") - pl.col("_atr_mean50")) / pl.col("_atr_std50"))
                    .fill_null(0.0)
                    .alias("volatility_zscore"),
            ])

            df = df.drop(["_atr_mean50", "_atr_std50"])
        else:
            df = df.with_columns([pl.lit(0.0).alias("volatility_zscore")])

        # === Feature 4: Crisis Proximity ===
        if "atr" in df.columns:
            df = df.with_columns([
                pl.col("atr").rolling_mean(window_size=50, min_periods=1).alias("_atr_mean"),
            ])

            # Crisis threshold: 2.5x mean ATR
            df = df.with_columns([
                (pl.col("atr") / (pl.col("_atr_mean") * 2.5))
                    .fill_null(0.0)
                    .alias("crisis_proximity"),
            ])

            df = df.drop(["_atr_mean"])
        else:
            df = df.with_columns([pl.lit(0.0).alias("crisis_proximity")])

        logger.debug("Regime features added (4 features)")
        return df

    # =========================================================================
    # PRICE ACTION FEATURES (4 features)
    # =========================================================================

    def add_price_action_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add price action pattern features.

        Features added (4 total):
        - wick_ratio: (upper + lower wick) / range
        - body_ratio: |close - open| / range
        - gap_from_prev_close: (open - prev close) / ATR
        - consecutive_direction: # candles in same direction

        Args:
            df: DataFrame with OHLCV and ATR

        Returns:
            DataFrame with price action features added
        """
        # === Feature 1: Wick Ratio ===
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            df = df.with_columns([
                # Upper wick
                (pl.max_horizontal("open", "close") - pl.col("high")).abs().alias("_upper_wick"),
                # Lower wick
                (pl.col("low") - pl.min_horizontal("open", "close")).abs().alias("_lower_wick"),
                # Range
                (pl.col("high") - pl.col("low")).alias("_range"),
            ])

            df = df.with_columns([
                ((pl.col("_upper_wick") + pl.col("_lower_wick")) / pl.col("_range"))
                    .fill_null(0.0)
                    .alias("wick_ratio"),
            ])

            # === Feature 2: Body Ratio ===
            df = df.with_columns([
                ((pl.col("close") - pl.col("open")).abs() / pl.col("_range"))
                    .fill_null(0.0)
                    .alias("body_ratio"),
            ])

            df = df.drop(["_upper_wick", "_lower_wick", "_range"])
        else:
            df = df.with_columns([
                pl.lit(0.0).alias("wick_ratio"),
                pl.lit(0.0).alias("body_ratio"),
            ])

        # === Feature 3: Gap from Previous Close ===
        if "open" in df.columns and "close" in df.columns and "atr" in df.columns:
            df = df.with_columns([
                ((pl.col("open") - pl.col("close").shift(1)) / pl.col("atr"))
                    .fill_null(0.0)
                    .alias("gap_from_prev_close"),
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("gap_from_prev_close")])

        # === Feature 4: Consecutive Direction ===
        if "close" in df.columns and "open" in df.columns:
            # Direction: 1 if bullish, -1 if bearish
            df = df.with_columns([
                pl.when(pl.col("close") > pl.col("open"))
                    .then(1)
                    .when(pl.col("close") < pl.col("open"))
                    .then(-1)
                    .otherwise(0)
                    .alias("_direction"),
            ])

            # Count consecutive bars in same direction
            # Create change flag
            df = df.with_columns([
                (pl.col("_direction") != pl.col("_direction").shift(1)).alias("_dir_change"),
            ])

            # Cumsum to create groups
            df = df.with_columns([
                pl.col("_dir_change").cum_sum().alias("_dir_group"),
            ])

            # Count within each group
            df = df.with_columns([
                pl.col("_dir_group").count().over("_dir_group").alias("consecutive_direction"),
            ])

            df = df.drop(["_direction", "_dir_change", "_dir_group"])
        else:
            df = df.with_columns([pl.lit(1).alias("consecutive_direction")])

        logger.debug("Price action features added (4 features)")
        return df

    # =========================================================================
    # MAIN INTERFACE
    # =========================================================================

    def add_all_v2_features(
        self,
        df_m15: pl.DataFrame,
        df_h1: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Add all 23 V2 features to M15 data.

        Args:
            df_m15: M15 DataFrame with base features (37) already calculated
            df_h1: H1 DataFrame with indicators (optional)

        Returns:
            M15 DataFrame with all 60 features (37 base + 23 V2)
        """
        logger.info("Adding all V2 features (23 new features)...")

        # H1 features (8)
        if df_h1 is not None:
            df_m15 = self.add_h1_features(df_m15, df_h1)
        else:
            logger.warning("No H1 data provided, using default H1 features")
            df_m15 = df_m15.with_columns([
                pl.col("close").alias("h1_ema20"),  # Use M15 close as proxy
                pl.lit(0).alias("h1_market_structure"),
                pl.lit(0.0).alias("h1_ema20_distance"),
                pl.lit(0).alias("h1_trend_strength"),
                pl.lit(0.0).alias("h1_swing_proximity"),
                pl.lit(0).alias("h1_fvg_active"),
                pl.lit(0.0).alias("h1_ob_proximity"),
                pl.lit(1.0).alias("h1_atr_ratio"),
                pl.lit(50.0).alias("h1_rsi"),
            ])

        # Continuous SMC (7)
        df_m15 = self.add_continuous_smc_features(df_m15)

        # Regime conditioning (4)
        df_m15 = self.add_regime_features(df_m15)

        # Price action (4)
        df_m15 = self.add_price_action_features(df_m15)

        logger.info("All V2 features added (23 total)")
        return df_m15

    def get_v2_feature_columns(self) -> List[str]:
        """
        Get list of V2 feature column names (23 features).

        Returns:
            List of V2 feature names
        """
        return [
            # H1 features (8)
            "h1_market_structure",
            "h1_ema20_distance",
            "h1_trend_strength",
            "h1_swing_proximity",
            "h1_fvg_active",
            "h1_ob_proximity",
            "h1_atr_ratio",
            "h1_rsi",
            # Continuous SMC (7)
            "fvg_gap_size_atr",
            "fvg_age_bars",
            "ob_width_atr",
            "ob_distance_atr",
            "bos_recency",
            "confluence_score",
            "swing_distance_atr",
            # Regime (4)
            "regime_duration_bars",
            "regime_transition_prob",
            "volatility_zscore",
            "crisis_proximity",
            # Price action (4)
            "wick_ratio",
            "body_ratio",
            "gap_from_prev_close",
            "consecutive_direction",
        ]


if __name__ == "__main__":
    # Test V2 features
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)
    n_m15 = 500
    n_h1 = 100

    # M15 data
    prices_m15 = 2000 + np.cumsum(np.random.randn(n_m15) * 2)
    df_m15 = pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n_m15-1, -1, -1)],
        "open": prices_m15,
        "high": prices_m15 + np.abs(np.random.randn(n_m15)) * 2,
        "low": prices_m15 - np.abs(np.random.randn(n_m15)) * 2,
        "close": prices_m15 + np.random.randn(n_m15),
        "atr": np.random.uniform(10, 14, n_m15),
        "regime": np.random.randint(0, 3, n_m15),
        "fvg_top": np.where(np.random.random(n_m15) > 0.9, prices_m15 + 5, None),
        "fvg_bottom": np.where(np.random.random(n_m15) > 0.9, prices_m15 - 5, None),
        "fvg_signal": np.where(np.random.random(n_m15) > 0.95, np.random.choice([-1, 1]), 0),
        "ob_top": np.where(np.random.random(n_m15) > 0.9, prices_m15 + 3, None),
        "ob_bottom": np.where(np.random.random(n_m15) > 0.9, prices_m15 - 3, None),
        "ob": np.where(np.random.random(n_m15) > 0.95, np.random.choice([-1, 1]), 0),
        "bos": np.where(np.random.random(n_m15) > 0.95, np.random.choice([-1, 1]), 0),
        "choch": np.where(np.random.random(n_m15) > 0.98, np.random.choice([-1, 1]), 0),
        "last_swing_high": prices_m15 + 10,
        "last_swing_low": prices_m15 - 10,
    })

    # H1 data
    prices_h1 = 2000 + np.cumsum(np.random.randn(n_h1) * 5)
    df_h1 = pl.DataFrame({
        "time": [datetime.now() - timedelta(hours=i) for i in range(n_h1-1, -1, -1)],
        "close": prices_h1,
        "atr": np.random.uniform(11, 13, n_h1),
        "rsi": np.random.uniform(30, 70, n_h1),
        "market_structure": np.random.choice([-1, 0, 1], n_h1),
        "last_swing_high": prices_h1 + 15,
        "last_swing_low": prices_h1 - 15,
        "fvg_top": np.where(np.random.random(n_h1) > 0.9, prices_h1 + 8, None),
        "fvg_bottom": np.where(np.random.random(n_h1) > 0.9, prices_h1 - 8, None),
        "ob_top": np.where(np.random.random(n_h1) > 0.9, prices_h1 + 5, None),
        "ob_bottom": np.where(np.random.random(n_h1) > 0.9, prices_h1 - 5, None),
        "bos": np.where(np.random.random(n_h1) > 0.95, np.random.choice([-1, 1]), 0),
    })

    # Add V2 features
    fe_v2 = MLV2FeatureEngineer()
    df_m15 = fe_v2.add_all_v2_features(df_m15, df_h1)

    print("\n=== ML V2 Feature Engineering Test ===")
    print(f"Total columns: {len(df_m15.columns)}")
    print(f"\nV2 feature columns (23):")
    v2_cols = fe_v2.get_v2_feature_columns()
    for i, col in enumerate(v2_cols, 1):
        print(f"  {i:2d}. {col}")

    # Show sample
    print("\n=== Sample Data (Last 5 Rows) ===")
    sample_cols = ["time", "close", "h1_ema20_distance", "confluence_score", "volatility_zscore", "wick_ratio"]
    available = [c for c in sample_cols if c in df_m15.columns]
    print(df_m15.select(available).tail(5))

    # Check for nulls
    null_counts = {col: df_m15[col].null_count() for col in v2_cols if col in df_m15.columns}
    print("\n=== Null Counts in V2 Features ===")
    for col, count in null_counts.items():
        if count > 0:
            print(f"  {col}: {count} nulls ({count/len(df_m15)*100:.1f}%)")
    if not any(null_counts.values()):
        print("  No nulls found! ✓")
