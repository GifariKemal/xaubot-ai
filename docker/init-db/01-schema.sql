-- ============================================================
-- Trading Bot Database Schema
-- ============================================================
-- Version: 1.0.0
-- Created: 2026-02-05
-- Description: PostgreSQL schema for AI Trading Bot data persistence
-- ============================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TRADES TABLE - Main trade history
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    ticket BIGINT UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'XAUUSD',
    direction VARCHAR(4) NOT NULL CHECK (direction IN ('BUY', 'SELL')),

    -- Prices
    entry_price DECIMAL(12,5) NOT NULL,
    exit_price DECIMAL(12,5),
    stop_loss DECIMAL(12,5) DEFAULT 0,
    take_profit DECIMAL(12,5) DEFAULT 0,

    -- Size & Result
    lot_size DECIMAL(8,4) NOT NULL,
    profit_usd DECIMAL(12,2),
    profit_pips DECIMAL(10,2),

    -- Timing
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    duration_seconds INT,

    -- Market Context at Entry
    entry_regime VARCHAR(30),
    entry_volatility VARCHAR(20),
    entry_session VARCHAR(30),
    entry_spread DECIMAL(8,2),
    entry_atr DECIMAL(10,5),

    -- SMC Signals
    smc_signal VARCHAR(10),
    smc_confidence DECIMAL(5,4),
    smc_reason TEXT,
    smc_fvg_detected BOOLEAN DEFAULT FALSE,
    smc_ob_detected BOOLEAN DEFAULT FALSE,
    smc_bos_detected BOOLEAN DEFAULT FALSE,
    smc_choch_detected BOOLEAN DEFAULT FALSE,

    -- ML Signals
    ml_signal VARCHAR(10),
    ml_confidence DECIMAL(5,4),

    -- Dynamic Analysis
    market_quality VARCHAR(20),
    market_score INT,
    dynamic_threshold DECIMAL(5,4),

    -- Exit Details
    exit_reason VARCHAR(50),
    exit_regime VARCHAR(30),
    exit_ml_signal VARCHAR(10),
    exit_ml_confidence DECIMAL(5,4),

    -- Balance Tracking
    balance_before DECIMAL(12,2),
    balance_after DECIMAL(12,2),
    equity_at_entry DECIMAL(12,2),

    -- Features (JSONB for flexibility)
    features_entry JSONB DEFAULT '{}',
    features_exit JSONB DEFAULT '{}',

    -- Meta
    bot_version VARCHAR(20) DEFAULT '2.1',
    trade_mode VARCHAR(30) DEFAULT 'SMC-ONLY',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- TRAINING_RUNS TABLE - ML model training history
-- ============================================================
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID DEFAULT uuid_generate_v4(),

    -- Timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds INT,

    -- Configuration
    training_type VARCHAR(20) NOT NULL CHECK (training_type IN ('daily', 'weekend', 'manual', 'initial')),
    bars_used INT,
    num_boost_rounds INT,

    -- Results - HMM
    hmm_trained BOOLEAN DEFAULT FALSE,
    hmm_n_regimes INT,

    -- Results - XGBoost
    xgb_trained BOOLEAN DEFAULT FALSE,
    train_auc DECIMAL(6,5),
    test_auc DECIMAL(6,5),
    train_accuracy DECIMAL(6,5),
    test_accuracy DECIMAL(6,5),

    -- Model Paths
    model_path VARCHAR(255),
    backup_path VARCHAR(255),

    -- Status
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT,

    -- Rollback
    rolled_back BOOLEAN DEFAULT FALSE,
    rollback_reason TEXT,
    rollback_at TIMESTAMPTZ,

    -- Meta
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- MARKET_SNAPSHOTS TABLE - Periodic market state (time-series)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'XAUUSD',

    -- Price
    price DECIMAL(12,5) NOT NULL,

    -- OHLC (current candle)
    open_price DECIMAL(12,5),
    high_price DECIMAL(12,5),
    low_price DECIMAL(12,5),
    close_price DECIMAL(12,5),

    -- Market State
    regime VARCHAR(30),
    volatility VARCHAR(20),
    session VARCHAR(30),
    atr DECIMAL(10,5),
    spread DECIMAL(8,2),

    -- Signals
    ml_signal VARCHAR(10),
    ml_confidence DECIMAL(5,4),
    smc_signal VARCHAR(10),
    smc_confidence DECIMAL(5,4),

    -- Position State
    open_positions INT DEFAULT 0,
    floating_pnl DECIMAL(12,2) DEFAULT 0,

    -- Features Snapshot
    features JSONB DEFAULT '{}',

    -- Meta
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Composite unique constraint
    UNIQUE (snapshot_time, symbol)
);

-- ============================================================
-- SIGNALS TABLE - Every signal generated
-- ============================================================
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    signal_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL DEFAULT 'XAUUSD',
    price DECIMAL(12,5) NOT NULL,

    -- Signal Details
    signal_type VARCHAR(10) CHECK (signal_type IN ('BUY', 'SELL', 'NONE', 'HOLD')),
    signal_source VARCHAR(20),  -- 'SMC', 'ML', 'COMBINED', 'SMC-ONLY'
    combined_confidence DECIMAL(5,4),

    -- SMC Analysis
    smc_signal VARCHAR(10),
    smc_confidence DECIMAL(5,4),
    smc_fvg BOOLEAN DEFAULT FALSE,
    smc_ob BOOLEAN DEFAULT FALSE,
    smc_bos BOOLEAN DEFAULT FALSE,
    smc_choch BOOLEAN DEFAULT FALSE,
    smc_reason TEXT,

    -- ML Analysis
    ml_signal VARCHAR(10),
    ml_confidence DECIMAL(5,4),

    -- Market Context
    regime VARCHAR(30),
    session VARCHAR(30),
    volatility VARCHAR(20),
    market_score INT,
    dynamic_threshold DECIMAL(5,4),

    -- Execution
    executed BOOLEAN DEFAULT FALSE,
    execution_reason VARCHAR(100),
    trade_ticket BIGINT,

    -- Meta
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- DAILY_SUMMARIES TABLE - Daily performance tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_summaries (
    id SERIAL PRIMARY KEY,
    summary_date DATE UNIQUE NOT NULL,

    -- Trades
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    breakeven_trades INT DEFAULT 0,

    -- P/L
    gross_profit DECIMAL(12,2) DEFAULT 0,
    gross_loss DECIMAL(12,2) DEFAULT 0,
    net_profit DECIMAL(12,2) DEFAULT 0,

    -- Balance
    start_balance DECIMAL(12,2),
    end_balance DECIMAL(12,2),

    -- Metrics
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(8,4),
    average_win DECIMAL(12,2),
    average_loss DECIMAL(12,2),
    largest_win DECIMAL(12,2),
    largest_loss DECIMAL(12,2),

    -- Sessions
    trades_sydney INT DEFAULT 0,
    trades_tokyo INT DEFAULT 0,
    trades_london INT DEFAULT 0,
    trades_ny INT DEFAULT 0,
    trades_golden INT DEFAULT 0,

    -- SMC Performance
    fvg_trades INT DEFAULT 0,
    fvg_wins INT DEFAULT 0,
    ob_trades INT DEFAULT 0,
    ob_wins INT DEFAULT 0,

    -- Meta
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- BOT_STATUS TABLE - Bot health and status tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS bot_status (
    id SERIAL PRIMARY KEY,
    status_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Status
    is_running BOOLEAN DEFAULT TRUE,
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'paused', 'stopped', 'error'

    -- Performance
    loop_count INT DEFAULT 0,
    avg_execution_ms DECIMAL(10,2),
    uptime_seconds INT DEFAULT 0,

    -- Account
    balance DECIMAL(12,2),
    equity DECIMAL(12,2),
    margin_used DECIMAL(12,2),

    -- Positions
    open_positions INT DEFAULT 0,
    floating_pnl DECIMAL(12,2),

    -- Risk State
    daily_pnl DECIMAL(12,2),
    risk_mode VARCHAR(20),  -- 'normal', 'recovery', 'protected', 'stopped'

    -- Session
    current_session VARCHAR(30),
    is_golden_time BOOLEAN DEFAULT FALSE,

    -- Error
    last_error TEXT,
    last_error_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- INDEXES for fast queries
-- ============================================================

-- Trades indexes
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades(opened_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades(direction);
CREATE INDEX IF NOT EXISTS idx_trades_profit ON trades(profit_usd);
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason);
CREATE INDEX IF NOT EXISTS idx_trades_entry_session ON trades(entry_session);

-- Training runs indexes
CREATE INDEX IF NOT EXISTS idx_training_started_at ON training_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_type ON training_runs(training_type);

-- Snapshots indexes
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(snapshot_time DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_time ON market_snapshots(symbol, snapshot_time DESC);

-- Signals indexes
CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(signal_time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_executed ON signals(executed);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);

-- Daily summaries index
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summaries(summary_date DESC);

-- Bot status index
CREATE INDEX IF NOT EXISTS idx_bot_status_time ON bot_status(status_time DESC);

-- ============================================================
-- FUNCTIONS for automatic updates
-- ============================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for trades table
DROP TRIGGER IF EXISTS update_trades_updated_at ON trades;
CREATE TRIGGER update_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for daily_summaries table
DROP TRIGGER IF EXISTS update_daily_summaries_updated_at ON daily_summaries;
CREATE TRIGGER update_daily_summaries_updated_at
    BEFORE UPDATE ON daily_summaries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- VIEWS for easy querying
-- ============================================================

-- Recent trades view
CREATE OR REPLACE VIEW v_recent_trades AS
SELECT
    ticket,
    direction,
    entry_price,
    exit_price,
    lot_size,
    profit_usd,
    profit_pips,
    duration_seconds,
    entry_session,
    smc_reason,
    ml_confidence,
    exit_reason,
    opened_at,
    closed_at
FROM trades
WHERE closed_at IS NOT NULL
ORDER BY closed_at DESC
LIMIT 100;

-- Daily performance view
CREATE OR REPLACE VIEW v_daily_performance AS
SELECT
    DATE(closed_at) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN profit_usd < 0 THEN 1 ELSE 0 END) as losses,
    SUM(profit_usd) as net_profit,
    ROUND(AVG(profit_usd)::numeric, 2) as avg_profit,
    ROUND((SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END)::numeric /
           NULLIF(COUNT(*), 0) * 100), 1) as win_rate
FROM trades
WHERE closed_at IS NOT NULL
GROUP BY DATE(closed_at)
ORDER BY trade_date DESC;

-- SMC performance view
CREATE OR REPLACE VIEW v_smc_performance AS
SELECT
    CASE
        WHEN smc_fvg_detected THEN 'FVG'
        WHEN smc_ob_detected THEN 'OB'
        WHEN smc_bos_detected THEN 'BOS'
        ELSE 'OTHER'
    END as pattern_type,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END) as wins,
    SUM(profit_usd) as total_profit,
    ROUND(AVG(profit_usd)::numeric, 2) as avg_profit,
    ROUND((SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END)::numeric /
           NULLIF(COUNT(*), 0) * 100), 1) as win_rate
FROM trades
WHERE closed_at IS NOT NULL
GROUP BY pattern_type
ORDER BY total_trades DESC;

-- Session performance view
CREATE OR REPLACE VIEW v_session_performance AS
SELECT
    entry_session,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END) as wins,
    SUM(profit_usd) as total_profit,
    ROUND(AVG(profit_usd)::numeric, 2) as avg_profit,
    ROUND((SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END)::numeric /
           NULLIF(COUNT(*), 0) * 100), 1) as win_rate
FROM trades
WHERE closed_at IS NOT NULL AND entry_session IS NOT NULL
GROUP BY entry_session
ORDER BY total_trades DESC;

-- ============================================================
-- INITIAL DATA / SEED
-- ============================================================

-- Insert initial bot status
INSERT INTO bot_status (status, is_running, risk_mode)
VALUES ('initialized', false, 'normal')
ON CONFLICT DO NOTHING;

-- ============================================================
-- GRANTS (if needed for specific users)
-- ============================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_bot;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_bot;

-- ============================================================
-- Done!
-- ============================================================
