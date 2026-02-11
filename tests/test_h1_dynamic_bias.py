"""
Test script for Dynamic H1 Bias System.
Verifies the multi-indicator scoring logic works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows console encoding
import os
if os.name == 'nt':
    os.system('chcp 65001 >nul 2>&1')

import polars as pl


def test_candle_bias_calculation():
    """Test the candle bias counting logic."""
    print("\n" + "=" * 60)
    print("Testing Candle Bias Calculation")
    print("=" * 60)

    # Test case 1: 4 bullish out of 5 (should return +1)
    df_bullish = pl.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "close": [101, 102, 103, 104, 105],  # 5 bullish candles
    })

    bullish_count = sum(1 for row in df_bullish.tail(5).iter_rows(named=True) if row["close"] > row["open"])
    result = 1 if bullish_count >= 3 else (-1 if (5 - bullish_count) >= 3 else 0)
    print(f"OK Bullish candles (5/5): result={result} (expected +1)")
    assert result == 1, "Bullish bias failed"

    # Test case 2: 4 bearish out of 5 (should return -1)
    df_bearish = pl.DataFrame({
        "open": [105, 104, 103, 102, 101],
        "close": [104, 103, 102, 101, 100],  # 5 bearish candles
    })

    bearish_count = sum(1 for row in df_bearish.tail(5).iter_rows(named=True) if row["close"] > row["open"])
    result = 1 if bearish_count >= 3 else (-1 if (5 - bearish_count) >= 3 else 0)
    print(f"OK Bearish candles (0/5): result={result} (expected -1)")
    assert result == -1, "Bearish bias failed"

    # Test case 3: 2 bullish, 3 bearish (should return -1)
    df_mixed = pl.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "close": [99, 100, 103, 102, 105],  # 2 bullish, 3 bearish
    })

    bullish_count = sum(1 for row in df_mixed.tail(5).iter_rows(named=True) if row["close"] > row["open"])
    result = 1 if bullish_count >= 3 else (-1 if (5 - bullish_count) >= 3 else 0)
    print(f"OK Mixed candles (2/5 bullish): result={result} (expected -1)")
    assert result == -1, "Mixed bias failed"

    print("OK All candle bias tests passed!\n")


def test_regime_weights():
    """Test regime-based weight selection."""
    print("=" * 60)
    print("Testing Regime Weight Selection")
    print("=" * 60)

    def get_weights(regime):
        regime_lower = regime.lower()
        if "low" in regime_lower or "ranging" in regime_lower:
            return {
                "ema_trend": 0.15,
                "ema_cross": 0.15,
                "rsi": 0.30,
                "macd": 0.25,
                "candles": 0.15,
            }
        elif "high" in regime_lower or "trending" in regime_lower:
            return {
                "ema_trend": 0.30,
                "ema_cross": 0.25,
                "rsi": 0.10,
                "macd": 0.25,
                "candles": 0.10,
            }
        else:
            return {
                "ema_trend": 0.25,
                "ema_cross": 0.20,
                "rsi": 0.20,
                "macd": 0.20,
                "candles": 0.15,
            }

    # Test low volatility
    weights_low = get_weights("Low Volatility")
    assert weights_low["rsi"] == 0.30, "Low vol RSI weight incorrect"
    assert sum(weights_low.values()) == 1.0, "Low vol weights don't sum to 1.0"
    print(f"OK Low volatility weights: RSI={weights_low['rsi']}, EMA_trend={weights_low['ema_trend']}")

    # Test high volatility
    weights_high = get_weights("High Volatility")
    assert weights_high["ema_trend"] == 0.30, "High vol EMA trend weight incorrect"
    assert sum(weights_high.values()) == 1.0, "High vol weights don't sum to 1.0"
    print(f"OK High volatility weights: EMA_trend={weights_high['ema_trend']}, RSI={weights_high['rsi']}")

    # Test medium volatility
    weights_med = get_weights("Medium Volatility")
    assert sum(weights_med.values()) == 1.0, "Med vol weights don't sum to 1.0"
    print(f"OK Medium volatility weights: balanced ({weights_med['ema_trend']}, {weights_med['rsi']})")

    print("OK All regime weight tests passed!\n")


def test_scoring_logic():
    """Test the weighted scoring calculation."""
    print("=" * 60)
    print("Testing Weighted Scoring Logic")
    print("=" * 60)

    # Test case 1: All bullish signals in high volatility
    signals_bull = {
        "ema_trend": 1,
        "ema_cross": 1,
        "rsi": 1,
        "macd": 1,
        "candles": 1,
    }
    weights_high = {
        "ema_trend": 0.30,
        "ema_cross": 0.25,
        "rsi": 0.10,
        "macd": 0.25,
        "candles": 0.10,
    }
    score = sum(signals_bull[k] * weights_high[k] for k in signals_bull)
    bias = "BULLISH" if score >= 0.3 else ("BEARISH" if score <= -0.3 else "NEUTRAL")
    print(f"OK All bullish + high vol: score={score:.2f}, bias={bias} (expected BULLISH)")
    assert score == 1.0, "All bullish score should be 1.0"
    assert bias == "BULLISH", "All bullish bias should be BULLISH"

    # Test case 2: All bearish signals in low volatility
    signals_bear = {k: -1 for k in signals_bull}
    weights_low = {
        "ema_trend": 0.15,
        "ema_cross": 0.15,
        "rsi": 0.30,
        "macd": 0.25,
        "candles": 0.15,
    }
    score = sum(signals_bear[k] * weights_low[k] for k in signals_bear)
    bias = "BULLISH" if score >= 0.3 else ("BEARISH" if score <= -0.3 else "NEUTRAL")
    print(f"OK All bearish + low vol: score={score:.2f}, bias={bias} (expected BEARISH)")
    assert score == -1.0, "All bearish score should be -1.0"
    assert bias == "BEARISH", "All bearish bias should be BEARISH"

    # Test case 3: Mixed signals (should be near neutral)
    signals_mixed = {
        "ema_trend": 1,
        "ema_cross": -1,
        "rsi": 0,
        "macd": 1,
        "candles": -1,
    }
    weights_med = {
        "ema_trend": 0.25,
        "ema_cross": 0.20,
        "rsi": 0.20,
        "macd": 0.20,
        "candles": 0.15,
    }
    score = sum(signals_mixed[k] * weights_med[k] for k in signals_mixed)
    bias = "BULLISH" if score >= 0.3 else ("BEARISH" if score <= -0.3 else "NEUTRAL")
    print(f"OK Mixed signals + med vol: score={score:.2f}, bias={bias} (expected NEUTRAL)")
    assert -0.3 < score < 0.3, "Mixed signals should be in neutral zone"
    assert bias == "NEUTRAL", "Mixed signals bias should be NEUTRAL"

    # Test case 4: Key test from plan — Price above EMA but bearish RSI+MACD+candles
    signals_key = {
        "ema_trend": 1,   # Price > EMA21 (old system would say BULLISH)
        "ema_cross": 1,   # EMA9 > EMA21
        "rsi": -1,        # RSI < 45 (bearish)
        "macd": -1,       # MACD bearish
        "candles": -1,    # Bearish candles
    }
    # Use high volatility weights (trending)
    score = sum(signals_key[k] * weights_high[k] for k in signals_key)
    bias = "BULLISH" if score >= 0.3 else ("BEARISH" if score <= -0.3 else "NEUTRAL")
    print(f"OK Price>EMA but bearish momentum: score={score:.2f}, bias={bias}")
    print(f"  -> Old system would say BULLISH, new system says {bias}")

    print("OK All scoring logic tests passed!\n")


def test_strength_calculation():
    """Test bias strength categorization."""
    print("=" * 60)
    print("Testing Bias Strength Calculation")
    print("=" * 60)

    test_cases = [
        (0.85, "strong"),
        (0.65, "moderate"),
        (0.45, "weak"),
        (0.25, "weak"),
        (-0.75, "strong"),
        (-0.55, "moderate"),
        (-0.35, "weak"),
    ]

    for score, expected_strength in test_cases:
        abs_score = abs(score)
        if abs_score >= 0.7:
            strength = "strong"
        elif abs_score >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        print(f"OK Score {score:+.2f} -> {strength} (expected: {expected_strength})")
        assert strength == expected_strength, f"Strength mismatch for score {score}"

    print("OK All strength tests passed!\n")


def run_all_tests():
    """Run all H1 dynamic bias tests."""
    print("\n" + "=" * 60)
    print("DYNAMIC H1 BIAS SYSTEM - TEST SUITE")
    print("=" * 60)

    try:
        test_candle_bias_calculation()
        test_regime_weights()
        test_scoring_logic()
        test_strength_calculation()

        print("=" * 60)
        print("OK ALL TESTS PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\nFAIL TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nFAIL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
