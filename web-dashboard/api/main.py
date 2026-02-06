"""
FastAPI Backend for Web Dashboard
=================================
Serves trading bot status data to the web frontend.
"""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
import asyncio
from typing import Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Import bot components
try:
    from src.mt5_connector import MT5Connector
    from src.smc_polars import SMCAnalyzer
    from src.ml_model import TradingModel
    from src.regime_detector import MarketRegimeDetector
    from src.session_filter import SessionFilter
    from src.feature_eng import FeatureEngineer
    from src.config import TradingConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")

app = FastAPI(title="Trading Bot API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class BotState:
    def __init__(self):
        self.mt5: Optional[MT5Connector] = None
        self.smc: Optional[SMCAnalyzer] = None
        self.ml: Optional[TradingModel] = None
        self.hmm: Optional[MarketRegimeDetector] = None
        self.session: Optional[SessionFilter] = None
        self.feature_eng: Optional[FeatureEngineer] = None
        self.config: Optional[TradingConfig] = None
        self.connected = False

        # History buffers
        self.price_history = deque(maxlen=120)
        self.equity_history = deque(maxlen=120)
        self.balance_history = deque(maxlen=120)
        self.logs = deque(maxlen=50)

        # Last known values
        self.last_price = 0.0
        self.last_update = None

state = BotState()


def add_log(level: str, message: str):
    """Add log entry to buffer"""
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    state.logs.append({
        "time": now.strftime("%H:%M:%S"),
        "level": level,
        "message": message
    })


@app.on_event("startup")
async def startup():
    """Initialize bot components on startup"""
    add_log("info", "Starting API server...")

    try:
        state.config = TradingConfig()
        state.mt5 = MT5Connector(
            login=state.config.mt5_login,
            password=state.config.mt5_password,
            server=state.config.mt5_server,
            path=state.config.mt5_path,
        )

        if state.mt5.connect():
            state.connected = True
            add_log("info", "MT5 connected successfully")

            # Initialize components
            state.smc = SMCAnalyzer()
            state.ml = TradingModel(model_path="models/xgboost_model")
            state.ml.load()
            state.hmm = MarketRegimeDetector(model_path="models/hmm_regime")
            state.hmm.load()
            state.session = SessionFilter()
            state.feature_eng = FeatureEngineer()

            add_log("info", f"ML Model loaded ({len(state.ml.feature_names)} features)")
        else:
            add_log("error", "Failed to connect to MT5")

    except Exception as e:
        add_log("error", f"Startup error: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if state.mt5:
        state.mt5.disconnect()
    add_log("info", "API server stopped")


@app.get("/api/status")
async def get_status():
    """Get current trading status"""
    wib = ZoneInfo("Asia/Jakarta")
    now = datetime.now(wib)

    result = {
        "timestamp": now.strftime("%H:%M:%S"),
        "connected": state.connected,
        "price": 0.0,
        "spread": 0.0,
        "priceChange": 0.0,
        "priceHistory": list(state.price_history),
        "balance": 0.0,
        "equity": 0.0,
        "profit": 0.0,
        "equityHistory": list(state.equity_history),
        "balanceHistory": list(state.balance_history),
        "session": "Unknown",
        "isGoldenTime": 19 <= now.hour < 23,
        "canTrade": False,
        "dailyLoss": 0.0,
        "dailyProfit": 0.0,
        "consecutiveLosses": 0,
        "riskPercent": 0.0,
        "smc": {"signal": "", "confidence": 0.0, "reason": ""},
        "ml": {"signal": "", "confidence": 0.0, "buyProb": 0.0, "sellProb": 0.0},
        "regime": {"name": "", "volatility": 0.0, "confidence": 0.0},
        "positions": [],
        "logs": list(state.logs),
    }

    if not state.connected or not state.mt5:
        return result

    try:
        # Price
        tick = state.mt5.get_tick(state.config.symbol)
        if tick:
            price = (tick.bid + tick.ask) / 2
            spread = (tick.ask - tick.bid) * 100

            # Calculate change
            price_change = price - state.last_price if state.last_price > 0 else 0
            state.last_price = price

            # Update history
            state.price_history.append(price)

            result["price"] = price
            result["spread"] = spread
            result["priceChange"] = price_change
            result["priceHistory"] = list(state.price_history)

        # Account
        balance = state.mt5.account_balance or 0
        equity = state.mt5.account_equity or 0
        profit = equity - balance

        state.equity_history.append(equity)
        state.balance_history.append(balance)

        result["balance"] = balance
        result["equity"] = equity
        result["profit"] = profit
        result["equityHistory"] = list(state.equity_history)
        result["balanceHistory"] = list(state.balance_history)

        # Session
        if state.session:
            session_info = state.session.get_status_report()
            if session_info:
                result["session"] = session_info.get('current_session', 'Unknown')
            can_trade, _, _ = state.session.can_trade()
            result["canTrade"] = can_trade

        # Risk state from file
        risk_file = Path("data/risk_state.txt")
        if risk_file.exists():
            content = risk_file.read_text()
            for line in content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key == 'daily_loss':
                        result["dailyLoss"] = float(value)
                    elif key == 'daily_profit':
                        result["dailyProfit"] = float(value)
                    elif key == 'consecutive_losses':
                        result["consecutiveLosses"] = int(value)

            # Calculate risk percent
            max_loss = state.config.capital * (state.config.risk.max_daily_loss / 100)
            if max_loss > 0:
                result["riskPercent"] = (result["dailyLoss"] / max_loss) * 100

        # Signals
        df = state.mt5.get_market_data(state.config.symbol, state.config.execution_timeframe, 200)
        if df is not None and len(df) > 50:
            # Feature engineering
            df = state.feature_eng.calculate_all(df, include_ml_features=True)
            df = state.smc.calculate_all(df)

            # Regime
            if state.hmm:
                df = state.hmm.predict(df)
                regime = state.hmm.get_current_state(df)
                if regime:
                    result["regime"] = {
                        "name": regime.regime.value.replace('_', ' ').title(),
                        "volatility": regime.volatility,
                        "confidence": regime.confidence,
                    }

            # SMC Signal
            smc_signal = state.smc.generate_signal(df)
            if smc_signal:
                result["smc"] = {
                    "signal": smc_signal.signal_type,
                    "confidence": smc_signal.confidence,
                    "reason": smc_signal.reason or "",
                }

            # ML Prediction
            if state.ml and state.ml.fitted:
                available_features = [f for f in state.ml.feature_names if f in df.columns]
                ml_pred = state.ml.predict(df, available_features)
                if ml_pred:
                    result["ml"] = {
                        "signal": ml_pred.signal,
                        "confidence": ml_pred.confidence,
                        "buyProb": ml_pred.probability,
                        "sellProb": 1.0 - ml_pred.probability,
                    }

        # Positions
        positions = state.mt5.get_open_positions(state.config.symbol)
        if positions is not None and not positions.is_empty():
            pos_list = []
            for row in positions.iter_rows(named=True):
                pos_list.append({
                    "ticket": row.get('ticket', 0),
                    "type": "BUY" if row.get('type', 0) == 0 else "SELL",
                    "volume": row.get('volume', 0),
                    "priceOpen": row.get('price_open', 0),
                    "profit": row.get('profit', 0),
                })
            result["positions"] = pos_list

        state.last_update = now

    except Exception as e:
        add_log("error", f"Status error: {str(e)[:50]}")

    return result


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "connected": state.connected}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
