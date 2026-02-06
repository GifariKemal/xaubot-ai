"""
Smart Money Concepts (SMC) Implementation - Pure Polars
========================================================
Native implementation of SMC concepts using Polars expressions.

NO PANDAS. NO smartmoneyconcepts library.

Implements:
- Fair Value Gaps (FVG)
- Swing Points (Fractal High/Low)
- Order Blocks
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Liquidity Zones
"""

import polars as pl
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from loguru import logger


@dataclass
class SMCSignal:
    """SMC trading signal."""
    signal_type: str          # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    
    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer using Pure Polars.
    
    All calculations are vectorized using Polars expressions.
    No loops, no Pandas, maximum performance.
    """
    
    def __init__(
        self,
        swing_length: int = 5,
        fvg_min_gap_pips: float = 2.0,
        ob_lookback: int = 10,
    ):
        """
        Initialize SMC Analyzer.
        
        Args:
            swing_length: Number of bars for swing detection
            fvg_min_gap_pips: Minimum FVG gap size in pips
            ob_lookback: Order block lookback period
        """
        self.swing_length = swing_length
        self.fvg_min_gap_pips = fvg_min_gap_pips
        self.ob_lookback = ob_lookback
    
    def calculate_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all SMC indicators.
        
        Args:
            df: Polars DataFrame with OHLCV data
            
        Returns:
            DataFrame with all SMC columns added
        """
        df = self.calculate_swing_points(df)
        df = self.calculate_fvg(df)
        df = self.calculate_order_blocks(df)
        df = self.calculate_bos_choch(df)
        return df
    
    def calculate_fvg(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Fair Value Gaps (FVG) using Polars expressions.
        
        Bullish FVG: Current Low > Previous-2 High (gap up)
        Bearish FVG: Current High < Previous-2 Low (gap down)
        
        This is a vectorized implementation - no loops.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with FVG columns:
            - is_fvg_bull: Boolean for bullish FVG
            - is_fvg_bear: Boolean for bearish FVG
            - fvg_top: Top of FVG zone
            - fvg_bottom: Bottom of FVG zone
            - fvg_mid: Midpoint of FVG (50% retracement target)
        """
        # Get shifted values using Polars expressions
        # FIX: NO LOOKAHEAD - detect FVG on the THIRD candle (after it's confirmed)
        # We only use PAST data (shift positive values)
        df = df.with_columns([
            # Previous candle values (t-1)
            pl.col("high").shift(1).alias("_prev_high"),
            pl.col("low").shift(1).alias("_prev_low"),
            # Candle before previous (t-2) - this is the FIRST candle of FVG pattern
            pl.col("high").shift(2).alias("_prev2_high"),
            pl.col("low").shift(2).alias("_prev2_low"),
            # Current candle is the THIRD candle - NO shift(-1) needed!
        ])

        # Calculate FVG conditions - detected on THIRD candle (current)
        # Bullish FVG: First candle high < Third candle low (gap up)
        # Bearish FVG: First candle low > Third candle high (gap down)
        # NO LOOKAHEAD: we detect AFTER the pattern is complete

        df = df.with_columns([
            # Bullish FVG: gap between candle 1's high and current candle's low
            (pl.col("_prev2_high") < pl.col("low")).alias("is_fvg_bull"),

            # Bearish FVG: gap between candle 1's low and current candle's high
            (pl.col("_prev2_low") > pl.col("high")).alias("is_fvg_bear"),
        ])
        
        # Calculate FVG zones using CURRENT candle (no lookahead)
        df = df.with_columns([
            # Bullish FVG zone: from prev2_high (bottom) to current_low (top)
            pl.when(pl.col("is_fvg_bull"))
                .then(pl.col("low"))  # Current candle low is FVG top
                .when(pl.col("is_fvg_bear"))
                .then(pl.col("_prev2_low"))  # First candle low is FVG top for bearish
                .otherwise(None)
                .alias("fvg_top"),

            pl.when(pl.col("is_fvg_bull"))
                .then(pl.col("_prev2_high"))  # First candle high is FVG bottom for bullish
                .when(pl.col("is_fvg_bear"))
                .then(pl.col("high"))  # Current candle high is FVG bottom
                .otherwise(None)
                .alias("fvg_bottom"),
        ])
        
        # Calculate FVG midpoint (50% retracement)
        df = df.with_columns([
            ((pl.col("fvg_top") + pl.col("fvg_bottom")) / 2).alias("fvg_mid"),
        ])
        
        # Combined FVG signal: 1 for bullish, -1 for bearish, 0 for none
        df = df.with_columns([
            pl.when(pl.col("is_fvg_bull"))
                .then(1)
                .when(pl.col("is_fvg_bear"))
                .then(-1)
                .otherwise(0)
                .alias("fvg_signal"),
        ])
        
        # Drop temporary columns (no _next columns since we removed lookahead)
        df = df.drop([
            "_prev_high", "_prev_low", "_prev2_high", "_prev2_low"
        ])
        
        logger.debug(f"FVG calculation complete. Bullish: {df['is_fvg_bull'].sum()}, Bearish: {df['is_fvg_bear'].sum()}")
        return df
    
    def calculate_swing_points(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Swing Points (Fractal Highs/Lows) using rolling windows.
        
        A Swing High is when the current high is the highest in the window.
        A Swing Low is when the current low is the lowest in the window.
        
        Uses centered rolling window for look-ahead detection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with swing point columns:
            - swing_high: 1 if swing high, 0 otherwise
            - swing_low: -1 if swing low, 0 otherwise
            - swing_high_level: Price level of swing high
            - swing_low_level: Price level of swing low
        """
        window_size = 2 * self.swing_length + 1

        # Calculate rolling max/min WITHOUT LOOKAHEAD
        # FIX: We detect swing points AFTER they're confirmed (swing_length bars later)
        # This means swing detection is delayed but NO FUTURE DATA is used
        #
        # Strategy: A swing high at bar [i] is confirmed at bar [i + swing_length]
        # when we can verify bar [i] was the highest in window
        # We use shift(swing_length) to look back at the confirmed swing point
        df = df.with_columns([
            # Look at past window_size bars only
            pl.col("high")
                .rolling_max(window_size=window_size, center=False)
                .alias("_roll_max"),
            pl.col("low")
                .rolling_min(window_size=window_size, center=False)
                .alias("_roll_min"),
            # Get the high/low from swing_length bars ago (the "center" point)
            pl.col("high").shift(self.swing_length).alias("_center_high"),
            pl.col("low").shift(self.swing_length).alias("_center_low"),
        ])
        
        # Detect swing points: the CENTER point equals rolling extreme
        # This detects swing points swing_length bars LATE (after confirmation)
        # NO LOOKAHEAD: we only confirm after seeing bars on both sides
        df = df.with_columns([
            # Swing High: center high equals rolling max (confirmed swing high)
            pl.when(pl.col("_center_high") == pl.col("_roll_max"))
                .then(1)
                .otherwise(0)
                .alias("swing_high"),

            # Swing Low: center low equals rolling min (confirmed swing low)
            pl.when(pl.col("_center_low") == pl.col("_roll_min"))
                .then(-1)
                .otherwise(0)
                .alias("swing_low"),
        ])
        
        # Store swing levels (use center values, not current values)
        df = df.with_columns([
            pl.when(pl.col("swing_high") == 1)
                .then(pl.col("_center_high"))
                .otherwise(None)
                .alias("swing_high_level"),

            pl.when(pl.col("swing_low") == -1)
                .then(pl.col("_center_low"))
                .otherwise(None)
                .alias("swing_low_level"),
        ])
        
        # Forward fill last swing levels for reference
        df = df.with_columns([
            pl.col("swing_high_level")
                .forward_fill()
                .alias("last_swing_high"),
            pl.col("swing_low_level")
                .forward_fill()
                .alias("last_swing_low"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_roll_max", "_roll_min", "_center_high", "_center_low"])
        
        swing_highs = (df["swing_high"] == 1).sum()
        swing_lows = (df["swing_low"] == -1).sum()
        logger.debug(f"Swing points: {swing_highs} highs, {swing_lows} lows")
        
        return df
    
    def calculate_order_blocks(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Order Blocks using vectorized Polars operations.
        
        Bullish Order Block: Last bearish candle before a bullish impulse
        that creates a swing low and breaks structure.
        
        Bearish Order Block: Last bullish candle before a bearish impulse
        that creates a swing high and breaks structure.
        
        This implementation uses numpy for the complex lookback logic,
        then converts back to Polars for performance.
        
        Args:
            df: DataFrame with OHLCV and swing point data
            
        Returns:
            DataFrame with Order Block columns:
            - ob: 1 for bullish OB, -1 for bearish OB, 0 for none
            - ob_top: Top of order block zone
            - ob_bottom: Bottom of order block zone
            - ob_mitigated: True if OB has been mitigated
        """
        # Ensure swing points are calculated
        if "swing_high" not in df.columns:
            df = self.calculate_swing_points(df)
        
        # Extract numpy arrays for complex logic
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        swing_highs = df["swing_high"].to_numpy()
        swing_lows = df["swing_low"].to_numpy()
        
        n = len(df)
        ob = np.zeros(n, dtype=np.int8)
        ob_top = np.full(n, np.nan)
        ob_bottom = np.full(n, np.nan)
        
        for i in range(self.ob_lookback, n):
            # Check for swing low -> Bullish Order Block
            # FIX: NO LOOKAHEAD - validate OB at CURRENT bar, not future bar
            if swing_lows[i] == -1:
                # Look for last bearish candle before swing low
                for j in range(i - 1, max(0, i - self.ob_lookback), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        # FIX: Validate OB using CURRENT bar (closes[i]) not future bar
                        # OB is valid if current close is above OB high (structure broken)
                        if closes[i] > highs[j]:
                            ob[j] = 1  # Bullish OB
                            ob_top[j] = highs[j]
                            ob_bottom[j] = lows[j]
                            break

            # Check for swing high -> Bearish Order Block
            if swing_highs[i] == 1:
                # Look for last bullish candle before swing high
                for j in range(i - 1, max(0, i - self.ob_lookback), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        # FIX: Validate OB using CURRENT bar (closes[i]) not future bar
                        # OB is valid if current close is below OB low (structure broken)
                        if closes[i] < lows[j]:
                            ob[j] = -1  # Bearish OB
                            ob_top[j] = highs[j]
                            ob_bottom[j] = lows[j]
                            break
        
        # Add to DataFrame
        df = df.with_columns([
            pl.Series("ob", ob),
            pl.Series("ob_top", ob_top),
            pl.Series("ob_bottom", ob_bottom),
        ])
        
        # Calculate OB mitigation (price has revisited the OB zone)
        df = df.with_columns([
            # Forward fill OB zones for mitigation checking
            pl.col("ob_top").forward_fill().alias("_ob_top_ff"),
            pl.col("ob_bottom").forward_fill().alias("_ob_bottom_ff"),
            pl.col("ob").forward_fill().alias("_ob_ff"),
        ])
        
        # Check if current price has entered OB zone (mitigation)
        df = df.with_columns([
            pl.when(
                (pl.col("_ob_ff") == 1) &
                (pl.col("low") <= pl.col("_ob_top_ff")) &
                (pl.col("high") >= pl.col("_ob_bottom_ff"))
            )
                .then(True)
                .when(
                    (pl.col("_ob_ff") == -1) &
                    (pl.col("high") >= pl.col("_ob_bottom_ff")) &
                    (pl.col("low") <= pl.col("_ob_top_ff"))
                )
                .then(True)
                .otherwise(False)
                .alias("ob_mitigated"),
        ])
        
        # Drop temporary columns
        df = df.drop(["_ob_top_ff", "_ob_bottom_ff", "_ob_ff"])
        
        bullish_obs = (df["ob"] == 1).sum()
        bearish_obs = (df["ob"] == -1).sum()
        logger.debug(f"Order Blocks: {bullish_obs} bullish, {bearish_obs} bearish")
        
        return df
    
    def calculate_bos_choch(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Break of Structure (BOS) and Change of Character (CHoCH).
        
        BOS: Structure break in the direction of the trend (continuation)
        CHoCH: Structure break against the trend (reversal signal)
        
        Uses numpy for stateful trend tracking, then converts to Polars.
        
        Args:
            df: DataFrame with OHLCV and swing point data
            
        Returns:
            DataFrame with BOS/CHoCH columns:
            - bos: 1 for bullish BOS, -1 for bearish BOS
            - choch: 1 for bullish CHoCH, -1 for bearish CHoCH
            - market_structure: Current market structure (1=bullish, -1=bearish)
        """
        # Ensure swing points are calculated
        if "swing_high" not in df.columns:
            df = self.calculate_swing_points(df)
        
        # Extract arrays
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        swing_highs = df["swing_high"].to_numpy()
        swing_lows = df["swing_low"].to_numpy()
        swing_high_levels = df["swing_high_level"].to_numpy() if "swing_high_level" in df.columns else np.full(len(df), np.nan)
        swing_low_levels = df["swing_low_level"].to_numpy() if "swing_low_level" in df.columns else np.full(len(df), np.nan)
        
        n = len(df)
        bos = np.zeros(n, dtype=np.int8)
        choch = np.zeros(n, dtype=np.int8)
        market_structure = np.zeros(n, dtype=np.int8)
        
        # Track last significant swing levels
        last_swing_high = np.nan
        last_swing_low = np.nan
        trend = 0  # 0=neutral, 1=bullish, -1=bearish
        
        for i in range(self.swing_length, n):
            # Update last swing levels
            if swing_highs[i] == 1 and not np.isnan(swing_high_levels[i]):
                last_swing_high = swing_high_levels[i]
            if swing_lows[i] == -1 and not np.isnan(swing_low_levels[i]):
                last_swing_low = swing_low_levels[i]
            
            market_structure[i] = trend
            
            # Check for break of swing high (bullish break)
            if not np.isnan(last_swing_high):
                if closes[i] > last_swing_high:
                    if trend == 1:  # Continuing bullish trend
                        bos[i] = 1  # Bullish BOS
                    elif trend == -1:  # Was bearish, now breaking up
                        choch[i] = 1  # Bullish CHoCH (reversal)
                    trend = 1
                    last_swing_high = np.nan  # Reset after break
            
            # Check for break of swing low (bearish break)
            if not np.isnan(last_swing_low):
                if closes[i] < last_swing_low:
                    if trend == -1:  # Continuing bearish trend
                        bos[i] = -1  # Bearish BOS
                    elif trend == 1:  # Was bullish, now breaking down
                        choch[i] = -1  # Bearish CHoCH (reversal)
                    trend = -1
                    last_swing_low = np.nan  # Reset after break
            
            market_structure[i] = trend
        
        # Add to DataFrame
        df = df.with_columns([
            pl.Series("bos", bos),
            pl.Series("choch", choch),
            pl.Series("market_structure", market_structure),
        ])
        
        bullish_bos = (df["bos"] == 1).sum()
        bearish_bos = (df["bos"] == -1).sum()
        bullish_choch = (df["choch"] == 1).sum()
        bearish_choch = (df["choch"] == -1).sum()
        
        logger.debug(f"BOS: {bullish_bos} bullish, {bearish_bos} bearish")
        logger.debug(f"CHoCH: {bullish_choch} bullish, {bearish_choch} bearish")
        
        return df
    
    def calculate_liquidity_zones(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Liquidity Zones (Equal Highs/Lows and BSL/SSL).

        OPTIMIZED: Uses native Polars expressions instead of rolling_map for better performance.

        Buy Side Liquidity (BSL): Clusters of equal highs (stop losses of shorts)
        Sell Side Liquidity (SSL): Clusters of equal lows (stop losses of longs)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with liquidity columns:
            - bsl_level: Buy side liquidity level
            - ssl_level: Sell side liquidity level
            - liquidity_sweep: True when liquidity is swept
        """
        window_size = 20

        # OPTIMIZED: Use rolling_std to detect price clusters
        # Low standard deviation = prices are similar (potential liquidity zone)
        # This is much faster than rolling_map with lambda
        df = df.with_columns([
            # Rolling std of highs - low std means similar prices (cluster)
            pl.col("high")
                .rolling_std(window_size=window_size)
                .alias("_high_std"),

            # Rolling std of lows
            pl.col("low")
                .rolling_std(window_size=window_size)
                .alias("_low_std"),

            # Rolling mean for reference
            pl.col("high")
                .rolling_mean(window_size=window_size)
                .alias("_high_mean"),

            pl.col("low")
                .rolling_mean(window_size=window_size)
                .alias("_low_mean"),
        ])

        # Calculate coefficient of variation (std/mean) - lower = more clustered
        # Threshold: if CV < 0.001 (0.1%), prices are very similar
        cv_threshold = 0.001

        df = df.with_columns([
            # High cluster detection
            pl.when(
                (pl.col("_high_std") / pl.col("_high_mean")) < cv_threshold
            )
                .then(pl.col("high"))
                .otherwise(None)
                .alias("bsl_level"),

            # Low cluster detection
            pl.when(
                (pl.col("_low_std") / pl.col("_low_mean")) < cv_threshold
            )
                .then(pl.col("low"))
                .otherwise(None)
                .alias("ssl_level"),
        ])

        # Forward fill liquidity levels
        df = df.with_columns([
            pl.col("bsl_level").forward_fill().alias("_bsl_ff"),
            pl.col("ssl_level").forward_fill().alias("_ssl_ff"),
        ])

        # Detect liquidity sweeps
        df = df.with_columns([
            # BSL sweep: high goes above BSL then closes below
            pl.when(
                (pl.col("high") > pl.col("_bsl_ff").shift(1)) &
                (pl.col("close") < pl.col("_bsl_ff").shift(1))
            )
                .then(pl.lit("BSL"))
                .when(
                    (pl.col("low") < pl.col("_ssl_ff").shift(1)) &
                    (pl.col("close") > pl.col("_ssl_ff").shift(1))
                )
                .then(pl.lit("SSL"))
                .otherwise(None)
                .alias("liquidity_sweep"),
        ])

        # Drop temporary columns
        df = df.drop(["_high_std", "_low_std", "_high_mean", "_low_mean", "_bsl_ff", "_ssl_ff"])

        return df
    
    def generate_signal(self, df: pl.DataFrame) -> Optional[SMCSignal]:
        """
        Generate trading signal based on SMC analysis.

        Signal Logic (RELAXED for active trading):
        1. Check market structure (BOS/CHoCH) - extended lookback
        2. Find valid FVG OR Order Block in recent candles
        3. Generate signal based on best available setup

        Args:
            df: DataFrame with all SMC indicators

        Returns:
            SMCSignal if valid setup found, None otherwise
        """
        # Get latest row
        if len(df) < 10:
            return None

        latest = df.tail(1)
        current_close = latest["close"].item()
        current_high = latest["high"].item()
        current_low = latest["low"].item()
        market_structure = latest["market_structure"].item() if "market_structure" in df.columns else 0

        # Check for recent BOS/CHoCH (extended to 10 candles)
        recent_df = df.tail(10)
        recent_bos = recent_df["bos"].to_list() if "bos" in df.columns else []
        recent_choch = recent_df["choch"].to_list() if "choch" in df.columns else []

        # Check for FVG in recent candles (not just current)
        recent_fvg_bull = recent_df["is_fvg_bull"].to_list() if "is_fvg_bull" in df.columns else []
        recent_fvg_bear = recent_df["is_fvg_bear"].to_list() if "is_fvg_bear" in df.columns else []

        # Get FVG zones from recent candles
        fvg_bottoms = recent_df["fvg_bottom"].to_list() if "fvg_bottom" in df.columns else []
        fvg_tops = recent_df["fvg_top"].to_list() if "fvg_top" in df.columns else []

        # Check for Order Block in recent candles
        recent_obs = recent_df["ob"].to_list() if "ob" in df.columns else []
        ob_tops = recent_df["ob_top"].to_list() if "ob_top" in df.columns else []
        ob_bottoms = recent_df["ob_bottom"].to_list() if "ob_bottom" in df.columns else []

        # Get swing levels for SL
        last_swing_high = latest["last_swing_high"].item() if "last_swing_high" in df.columns else None
        last_swing_low = latest["last_swing_low"].item() if "last_swing_low" in df.columns else None

        signal = None

        # Determine if there's a recent bullish/bearish setup
        has_bullish_break = 1 in recent_bos or 1 in recent_choch
        has_bearish_break = -1 in recent_bos or -1 in recent_choch
        has_bullish_fvg = any(recent_fvg_bull)
        has_bearish_fvg = any(recent_fvg_bear)
        has_bullish_ob = 1 in recent_obs
        has_bearish_ob = -1 in recent_obs

        # Get valid FVG/OB zone for entry
        def get_valid_bullish_zone():
            # Find most recent bullish FVG or OB
            for i in range(len(recent_fvg_bull) - 1, -1, -1):
                if recent_fvg_bull[i] and fvg_bottoms[i] is not None:
                    return fvg_bottoms[i], "FVG"
            for i in range(len(recent_obs) - 1, -1, -1):
                if recent_obs[i] == 1 and ob_bottoms[i] is not None:
                    return ob_bottoms[i], "OB"
            return None, None

        def get_valid_bearish_zone():
            # Find most recent bearish FVG or OB
            for i in range(len(recent_fvg_bear) - 1, -1, -1):
                if recent_fvg_bear[i] and fvg_tops[i] is not None:
                    return fvg_tops[i], "FVG"
            for i in range(len(recent_obs) - 1, -1, -1):
                if recent_obs[i] == -1 and ob_tops[i] is not None:
                    return ob_tops[i], "OB"
            return None, None

        # Get ATR for dynamic SL/TP calculation
        # FIX: Realistic ATR fallback for XAUUSD (~$12-15 typical)
        if "atr" in df.columns:
            atr = latest["atr"].item()
            if atr is None or atr <= 0 or atr > current_close * 0.05:  # Sanity check
                atr = 12.0  # Default realistic ATR for XAUUSD
        else:
            atr = 12.0  # Default realistic ATR for XAUUSD

        # SL: 1.5-2 ATR distance (protects against noise)
        min_sl_distance = 1.5 * atr
        # TP: Must be at least 2x risk (RR 1:2 minimum)
        # With 1.5 ATR SL, TP should be at least 3 ATR
        min_rr_ratio = 2.0  # ENFORCED: Minimum Risk:Reward 1:2

        # BULLISH SIGNAL CONDITIONS
        # Need: bullish structure OR recent bullish break, AND (FVG OR OB)
        if ((market_structure == 1 or has_bullish_break) and
            (has_bullish_fvg or has_bullish_ob)):

            entry_zone, zone_type = get_valid_bullish_zone()
            # FIX: ALWAYS use current_close as entry (no stale prices)
            # FVG/OB zone is just for confirmation, not entry price
            entry = current_close

            # SL below swing low or ATR-based (use the FURTHER one to prevent whipsaw)
            swing_sl = last_swing_low if last_swing_low and last_swing_low < entry else None
            atr_sl = entry - min_sl_distance

            if swing_sl:
                # Use the further SL (more protection)
                sl = min(swing_sl, atr_sl)
            else:
                sl = atr_sl

            # Ensure SL is at least min_sl_distance away
            if entry - sl < min_sl_distance:
                sl = entry - min_sl_distance

            # FIX: TP at EXACTLY min_rr_ratio (1:2) - ENFORCED
            risk = entry - sl
            tp = entry + (risk * min_rr_ratio)

            # VALIDATE RR before creating signal
            actual_rr = (tp - entry) / risk if risk > 0 else 0
            if actual_rr < min_rr_ratio:
                logger.debug(f"Skipping BUY signal: RR {actual_rr:.2f} < {min_rr_ratio}")
                signal = None
            else:
                # Confidence based on confirmations
                conf = 0.55  # Base
                if has_bullish_break:
                    conf += 0.1
                if has_bullish_fvg:
                    conf += 0.1
                if has_bullish_ob:
                    conf += 0.1

                reason_parts = []
                if has_bullish_break:
                    reason_parts.append("BOS/CHoCH")
                if zone_type == "FVG":
                    reason_parts.append("FVG")
                if zone_type == "OB":
                    reason_parts.append("OB")

                signal = SMCSignal(
                    signal_type="BUY",
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=min(conf, 0.85),
                    reason="Bullish " + " + ".join(reason_parts),
                )

        # BEARISH SIGNAL CONDITIONS
        elif ((market_structure == -1 or has_bearish_break) and
              (has_bearish_fvg or has_bearish_ob)):

            entry_zone, zone_type = get_valid_bearish_zone()
            # FIX: ALWAYS use current_close as entry (no stale prices)
            entry = current_close

            # SL above swing high or ATR-based (use the FURTHER one to prevent whipsaw)
            swing_sl = last_swing_high if last_swing_high and last_swing_high > entry else None
            atr_sl = entry + min_sl_distance

            if swing_sl:
                # Use the further SL (more protection)
                sl = max(swing_sl, atr_sl)
            else:
                sl = atr_sl

            # Ensure SL is at least min_sl_distance away
            if sl - entry < min_sl_distance:
                sl = entry + min_sl_distance

            # FIX: TP at EXACTLY min_rr_ratio (1:2) - ENFORCED
            risk = sl - entry
            tp = entry - (risk * min_rr_ratio)

            # VALIDATE RR before creating signal
            actual_rr = (entry - tp) / risk if risk > 0 else 0
            if actual_rr < min_rr_ratio:
                logger.debug(f"Skipping SELL signal: RR {actual_rr:.2f} < {min_rr_ratio}")
                signal = None
            else:
                # Confidence based on confirmations
                conf = 0.55  # Base
                if has_bearish_break:
                    conf += 0.1
                if has_bearish_fvg:
                    conf += 0.1
                if has_bearish_ob:
                    conf += 0.1

                reason_parts = []
                if has_bearish_break:
                    reason_parts.append("BOS/CHoCH")
                if zone_type == "FVG":
                    reason_parts.append("FVG")
                if zone_type == "OB":
                    reason_parts.append("OB")

                signal = SMCSignal(
                    signal_type="SELL",
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=min(conf, 0.85),
                    reason="Bearish " + " + ".join(reason_parts),
                )

        if signal:
            logger.info(f"SMC Signal: {signal.signal_type} @ {signal.entry_price:.5f}, "
                       f"SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}, "
                       f"RR: {signal.risk_reward:.2f}, Confidence: {signal.confidence:.2f}")

        return signal


def calculate_smc_summary(df: pl.DataFrame) -> Dict:
    """
    Calculate summary statistics for SMC analysis.
    
    Args:
        df: DataFrame with SMC indicators
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_bars": len(df),
        "swing_highs": (df["swing_high"] == 1).sum() if "swing_high" in df.columns else 0,
        "swing_lows": (df["swing_low"] == -1).sum() if "swing_low" in df.columns else 0,
        "bullish_fvg": df["is_fvg_bull"].sum() if "is_fvg_bull" in df.columns else 0,
        "bearish_fvg": df["is_fvg_bear"].sum() if "is_fvg_bear" in df.columns else 0,
        "bullish_ob": (df["ob"] == 1).sum() if "ob" in df.columns else 0,
        "bearish_ob": (df["ob"] == -1).sum() if "ob" in df.columns else 0,
        "bullish_bos": (df["bos"] == 1).sum() if "bos" in df.columns else 0,
        "bearish_bos": (df["bos"] == -1).sum() if "bos" in df.columns else 0,
        "bullish_choch": (df["choch"] == 1).sum() if "choch" in df.columns else 0,
        "bearish_choch": (df["choch"] == -1).sum() if "choch" in df.columns else 0,
    }
    
    # Current market structure
    if "market_structure" in df.columns:
        current_structure = df["market_structure"].tail(1).item()
        summary["current_structure"] = "BULLISH" if current_structure == 1 else "BEARISH" if current_structure == -1 else "NEUTRAL"
    
    return summary


if __name__ == "__main__":
    # Test SMC analyzer with synthetic data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 500
    
    base_price = 2000.0
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
        "open": prices,
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.001),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.001),
        "close": prices * (1 + np.random.randn(n) * 0.0005),
        "volume": np.random.randint(1000, 10000, n),
    })
    
    # Initialize analyzer
    analyzer = SMCAnalyzer(swing_length=5)
    
    # Calculate all SMC indicators
    df = analyzer.calculate_all(df)
    
    # Print summary
    summary = calculate_smc_summary(df)
    print("\n=== SMC Analysis Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Generate signal
    signal = analyzer.generate_signal(df)
    if signal:
        print(f"\n=== Trading Signal ===")
        print(f"Type: {signal.signal_type}")
        print(f"Entry: {signal.entry_price:.2f}")
        print(f"SL: {signal.stop_loss:.2f}")
        print(f"TP: {signal.take_profit:.2f}")
        print(f"R:R: {signal.risk_reward:.2f}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Reason: {signal.reason}")
    else:
        print("\nNo valid signal")
    
    # Show columns
    print(f"\n=== DataFrame Columns ===")
    print(df.columns)
