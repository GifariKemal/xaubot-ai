"""
Module Test Script
==================
Tests all modules to ensure they work correctly.
"""

import sys
import polars as pl
import numpy as np
from datetime import datetime, timedelta


def create_test_data(n: int = 500) -> pl.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    
    base_price = 2000.0
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pl.DataFrame({
        "time": [datetime.now() - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
        "open": prices,
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.001),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.001),
        "close": prices * (1 + np.random.randn(n) * 0.0005),
        "volume": np.random.randint(1000, 10000, n),
    })


def test_config():
    """Test configuration module."""
    print("\n" + "="*60)
    print("Testing: src/config.py")
    print("="*60)
    
    from src.config import TradingConfig, CapitalMode
    
    # Test small account
    config_small = TradingConfig(capital=5000)
    assert config_small.capital_mode == CapitalMode.SMALL
    assert config_small.risk.risk_per_trade == 1.5
    print(f"✓ Small account config: {config_small.capital_mode.value}")
    
    # Test medium account
    config_medium = TradingConfig(capital=50000)
    assert config_medium.capital_mode == CapitalMode.MEDIUM
    assert config_medium.risk.risk_per_trade == 0.5
    print(f"✓ Medium account config: {config_medium.capital_mode.value}")
    
    # Test position sizing
    lot = config_small.calculate_position_size(2000, 1995)
    assert lot > 0
    print(f"✓ Position sizing: {lot} lots")
    
    print("✓ Config module: PASSED")


def test_smc_polars():
    """Test SMC Polars module."""
    print("\n" + "="*60)
    print("Testing: src/smc_polars.py")
    print("="*60)
    
    from src.smc_polars import SMCAnalyzer, calculate_smc_summary
    
    df = create_test_data(500)
    analyzer = SMCAnalyzer(swing_length=5)
    
    # Test swing points
    df = analyzer.calculate_swing_points(df)
    assert "swing_high" in df.columns
    assert "swing_low" in df.columns
    print(f"✓ Swing points calculated")
    
    # Test FVG
    df = analyzer.calculate_fvg(df)
    assert "is_fvg_bull" in df.columns
    assert "is_fvg_bear" in df.columns
    print(f"✓ FVG calculated")
    
    # Test Order Blocks
    df = analyzer.calculate_order_blocks(df)
    assert "ob" in df.columns
    print(f"✓ Order Blocks calculated")
    
    # Test BOS/CHoCH
    df = analyzer.calculate_bos_choch(df)
    assert "bos" in df.columns
    assert "choch" in df.columns
    print(f"✓ BOS/CHoCH calculated")
    
    # Summary
    summary = calculate_smc_summary(df)
    print(f"  - Swing Highs: {summary['swing_highs']}")
    print(f"  - Swing Lows: {summary['swing_lows']}")
    print(f"  - Bullish FVG: {summary['bullish_fvg']}")
    print(f"  - Bearish FVG: {summary['bearish_fvg']}")
    
    print("✓ SMC Polars module: PASSED")


def test_feature_eng():
    """Test feature engineering module."""
    print("\n" + "="*60)
    print("Testing: src/feature_eng.py")
    print("="*60)
    
    from src.feature_eng import FeatureEngineer
    
    df = create_test_data(200)
    fe = FeatureEngineer()
    
    # Test RSI
    df = fe.calculate_rsi(df, period=14)
    assert "rsi" in df.columns
    rsi_range = df["rsi"].drop_nulls()
    assert rsi_range.min() >= 0 and rsi_range.max() <= 100
    print(f"✓ RSI calculated (range: {rsi_range.min():.1f} - {rsi_range.max():.1f})")
    
    # Test ATR
    df = fe.calculate_atr(df, period=14)
    assert "atr" in df.columns
    assert df["atr"].drop_nulls().min() >= 0
    print(f"✓ ATR calculated")
    
    # Test MACD
    df = fe.calculate_macd(df)
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    print(f"✓ MACD calculated")
    
    # Test Bollinger Bands
    df = fe.calculate_bollinger_bands(df)
    assert "bb_upper" in df.columns
    assert "bb_lower" in df.columns
    print(f"✓ Bollinger Bands calculated")
    
    # Test ML features
    df = fe.calculate_ml_features(df)
    assert "returns_1" in df.columns
    assert "volatility_20" in df.columns
    print(f"✓ ML features calculated")
    
    # Get feature columns
    feature_cols = fe.get_feature_columns(df)
    print(f"  - Total features: {len(feature_cols)}")
    
    print("✓ Feature Engineering module: PASSED")


def test_regime_detector():
    """Test regime detector module."""
    print("\n" + "="*60)
    print("Testing: src/regime_detector.py")
    print("="*60)
    
    from src.regime_detector import MarketRegimeDetector, FlashCrashDetector
    
    df = create_test_data(500)
    
    # Test HMM detector
    detector = MarketRegimeDetector(n_regimes=3)
    detector.fit(df.head(400))
    
    assert detector.fitted
    print(f"✓ HMM fitted")
    
    # Predict
    df_pred = detector.predict(df)
    assert "regime_name" in df_pred.columns
    print(f"✓ Regime prediction")
    
    # Get current state
    state = detector.get_current_state(df)
    print(f"  - Current regime: {state.regime.value}")
    print(f"  - Confidence: {state.confidence:.2%}")
    print(f"  - Recommendation: {state.recommendation}")
    
    # Test flash crash detector
    fc_detector = FlashCrashDetector(threshold_percent=1.0)
    is_flash, move = fc_detector.detect(df.tail(10))
    print(f"✓ Flash crash detector (flash={is_flash}, move={move:.2f}%)")
    
    print("✓ Regime Detector module: PASSED")


def test_risk_engine():
    """Test risk engine module."""
    print("\n" + "="*60)
    print("Testing: src/risk_engine.py")
    print("="*60)
    
    from src.config import TradingConfig
    from src.risk_engine import RiskEngine
    
    config = TradingConfig(capital=5000)
    engine = RiskEngine(config)
    
    # Test position sizing
    result = engine.calculate_position_size(
        entry_price=2000.0,
        stop_loss_price=1995.0,
        take_profit_price=2010.0,
        account_balance=5000.0,
        win_rate=0.55,
        avg_win_loss_ratio=2.0,
    )
    
    assert result.lot_size > 0
    print(f"✓ Position sizing: {result.lot_size} lots")
    print(f"  - Risk: ${result.risk_amount:.2f} ({result.risk_percent:.2f}%)")
    
    # Test order validation
    valid, reason = engine.validate_order(
        order_type="BUY",
        entry_price=2000.0,
        stop_loss=1995.0,
        take_profit=2010.0,
        lot_size=result.lot_size,
        current_price=2000.0,
        account_balance=5000.0,
    )
    
    assert valid
    print(f"✓ Order validation: {reason}")
    
    # Test risk check
    metrics = engine.check_risk(
        account_balance=5000.0,
        account_equity=4950.0,
        open_positions=pl.DataFrame({"ticket": [], "volume": [], "symbol": []}),
        current_price=2000.0,
    )
    
    print(f"✓ Risk check: can_trade={metrics.can_trade}")
    
    print("✓ Risk Engine module: PASSED")


def test_ml_model():
    """Test ML model module."""
    print("\n" + "="*60)
    print("Testing: src/ml_model.py")
    print("="*60)
    
    from src.ml_model import TradingModel
    
    # Create synthetic data with features
    np.random.seed(42)
    n = 500
    
    df = pl.DataFrame({
        "rsi": np.random.uniform(20, 80, n),
        "atr": np.random.uniform(0.5, 2.0, n),
        "macd": np.random.randn(n) * 0.001,
        "returns_1": np.random.randn(n) * 0.01,
    })
    
    # Create target
    target = ((df["rsi"].to_numpy() > 50).astype(int) * 0.5 +
              np.random.randint(0, 2, n) * 0.5)
    target = (target > 0.5).astype(int)
    df = df.with_columns([pl.Series("target", target)])
    
    # Test model
    model = TradingModel(confidence_threshold=0.6)
    feature_cols = ["rsi", "atr", "macd", "returns_1"]
    
    model.fit(df, feature_cols, "target")
    assert model.fitted
    print(f"✓ Model trained")
    
    # Test prediction
    prediction = model.predict(df, feature_cols)
    print(f"✓ Prediction: {prediction.signal} ({prediction.confidence:.2%})")
    
    # Test feature importance
    importance = model.get_feature_importance(3)
    print(f"✓ Feature importance: {list(importance.keys())}")
    
    print("✓ ML Model module: PASSED")


def test_utils():
    """Test utility module."""
    print("\n" + "="*60)
    print("Testing: src/utils.py")
    print("="*60)
    
    from src.utils import (
        validate_ohlcv_data,
        resample_ohlcv,
        calculate_trade_statistics,
        create_synthetic_data,
    )
    
    # Test synthetic data
    df = create_synthetic_data(100)
    assert len(df) == 100
    print(f"✓ Synthetic data created")
    
    # Test validation
    is_valid, issues = validate_ohlcv_data(df)
    assert is_valid
    print(f"✓ Data validation: valid={is_valid}")
    
    # Test resampling
    df_resampled = resample_ohlcv(df, "1h")
    assert len(df_resampled) < len(df)
    print(f"✓ Resampling: {len(df)} -> {len(df_resampled)} bars")
    
    # Test trade statistics
    trades = [
        {"pnl": 100, "is_win": True},
        {"pnl": -50, "is_win": False},
        {"pnl": 75, "is_win": True},
    ]
    stats = calculate_trade_statistics(trades)
    assert stats["total_trades"] == 3
    print(f"✓ Trade statistics: win_rate={stats['win_rate']:.2%}")
    
    print("✓ Utils module: PASSED")


def run_all_tests():
    """Run all module tests."""
    print("\n" + "="*60)
    print("SMART AUTOMATIC TRADING BOT + AI - MODULE TESTS")
    print("="*60)
    
    tests = [
        ("Config", test_config),
        ("SMC Polars", test_smc_polars),
        ("Feature Engineering", test_feature_eng),
        ("Regime Detector", test_regime_detector),
        ("Risk Engine", test_risk_engine),
        ("ML Model", test_ml_model),
        ("Utils", test_utils),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} module: FAILED - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
