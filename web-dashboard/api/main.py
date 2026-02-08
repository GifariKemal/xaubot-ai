"""
FastAPI Backend for Web Dashboard (Docker-compatible)
=====================================================
Serves trading bot status data to the web frontend.

Reads from data/bot_status.json which is written by main_live.py.
This allows the API to run in Docker without needing MT5 (Windows-only).
"""

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Trading Bot API", version="2.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Status file path (mounted as volume in Docker)
STATUS_FILE = Path("/app/data/bot_status.json")

# Default empty response
DEFAULT_STATUS = {
    "timestamp": "00:00:00",
    "connected": False,
    "price": 0.0,
    "spread": 0.0,
    "priceChange": 0.0,
    "priceHistory": [],
    "balance": 0.0,
    "equity": 0.0,
    "profit": 0.0,
    "equityHistory": [],
    "balanceHistory": [],
    "session": "Unknown",
    "isGoldenTime": False,
    "canTrade": False,
    "dailyLoss": 0.0,
    "dailyProfit": 0.0,
    "consecutiveLosses": 0,
    "riskPercent": 0.0,
    "smc": {"signal": "", "confidence": 0.0, "reason": ""},
    "ml": {"signal": "", "confidence": 0.0, "buyProb": 0.0, "sellProb": 0.0},
    "regime": {"name": "", "volatility": 0.0, "confidence": 0.0},
    "positions": [],
    "logs": [],
}


@app.get("/api/status")
async def get_status():
    """Get current trading status from bot's status file."""
    # Try local path first (non-Docker), then Docker path
    for path in [STATUS_FILE, Path("data/bot_status.json")]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data
            except (json.JSONDecodeError, OSError):
                continue

    # No status file — bot not running
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    result = DEFAULT_STATUS.copy()
    result["timestamp"] = now.strftime("%H:%M:%S")
    result["logs"] = [
        {
            "time": now.strftime("%H:%M:%S"),
            "level": "warning",
            "message": "Bot is not running — waiting for bot_status.json",
        }
    ]
    return result


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    bot_running = STATUS_FILE.exists() or Path("data/bot_status.json").exists()
    return {"status": "ok", "bot_running": bot_running}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
