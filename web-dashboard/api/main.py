"""
FastAPI Backend for Web Dashboard (Docker-compatible)
=====================================================
Serves trading bot status data to the web frontend.
Reads from data/bot_status.json (written by main_live.py)
and from PostgreSQL database for trade history, signals, model data.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import db

logger = logging.getLogger(__name__)

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
MODEL_METRICS_FILE = Path("/app/data/model_metrics.json")

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
    "entryFilters": [],
    "riskMode": {"mode": "unknown", "reason": "", "recommendedLot": 0, "maxAllowedLot": 0, "totalLoss": 0, "maxTotalLoss": 0, "remainingDailyRisk": 0},
    "cooldown": {"active": False, "secondsRemaining": 0, "totalSeconds": 150},
    "timeFilter": {"wibHour": 0, "isBlocked": False, "blockedHours": [9, 21]},
    "sessionMultiplier": 1.0,
    "positionDetails": [],
    "autoTrainer": {"lastRetrain": None, "currentAuc": None, "minAucThreshold": 0.65, "hoursSinceRetrain": 0, "nextRetrainHour": 5, "modelsFitted": False},
    "performance": {"loopCount": 0, "avgExecutionMs": 0, "uptimeHours": 0, "totalSessionTrades": 0, "totalSessionProfit": 0},
    "marketClose": {"hoursToDailyClose": 0, "hoursToWeekendClose": 0, "nearWeekend": False, "marketOpen": False},
    "h1BiasDetails": {"bias": "NEUTRAL", "ema20": 0, "price": 0},
}


# ─── Startup / Shutdown ───

@app.on_event("startup")
async def startup():
    try:
        db.init_pool()
        logger.info("DB pool ready")
    except Exception as e:
        logger.warning("DB not available: %s (trade history features disabled)", e)


@app.on_event("shutdown")
async def shutdown():
    db.close_pool()


# ─── Status Endpoints ───

@app.get("/api/status")
async def get_status():
    """Get current trading status from bot's status file."""
    for path in [STATUS_FILE, Path("data/bot_status.json")]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data
            except (json.JSONDecodeError, OSError):
                continue

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
    return {"status": "ok", "bot_running": bot_running, "db_available": db.is_available()}


# ─── Trade History Endpoints ───

def _date_filter(field: str, start_date: Optional[str], end_date: Optional[str]):
    """Build date filter SQL clauses."""
    clauses = []
    params = []
    if start_date:
        clauses.append(f"{field} >= %s")
        params.append(start_date)
    if end_date:
        clauses.append(f"{field} <= %s")
        params.append(end_date + " 23:59:59")
    return clauses, params


@app.get("/api/trades")
async def get_trades(
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=100),
    direction: str = Query("ALL"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get paginated trade history."""
    if not db.is_available():
        return {"trades": [], "total": 0, "page": page, "limit": limit}

    where = ["closed_at IS NOT NULL"]
    params = []

    if direction and direction != "ALL":
        where.append("direction = %s")
        params.append(direction.upper())

    date_clauses, date_params = _date_filter("closed_at", start_date, end_date)
    where.extend(date_clauses)
    params.extend(date_params)

    where_sql = " AND ".join(where)
    offset = (page - 1) * limit

    count_row = db.query(f"SELECT COUNT(*) as cnt FROM trades WHERE {where_sql}", tuple(params), one=True)
    total = count_row["cnt"] if count_row else 0

    params_with_pagination = params + [limit, offset]
    trades = db.query(
        f"""SELECT id, ticket, direction, entry_price, exit_price, lot_size,
                   profit_usd, profit_pips, sl_price, tp_price,
                   opened_at, closed_at, exit_reason, confidence,
                   regime, session, duration_minutes
            FROM trades
            WHERE {where_sql}
            ORDER BY closed_at DESC
            LIMIT %s OFFSET %s""",
        tuple(params_with_pagination),
    )

    for t in trades:
        for k in ("opened_at", "closed_at"):
            if t.get(k) and hasattr(t[k], "isoformat"):
                t[k] = t[k].isoformat()

    return {"trades": trades, "total": total, "page": page, "limit": limit}


@app.get("/api/trades/stats")
async def get_trade_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get aggregate trade statistics."""
    if not db.is_available():
        return {"totalTrades": 0, "winRate": 0, "netProfit": 0, "profitFactor": 0, "avgWin": 0, "avgLoss": 0, "bestTrade": 0, "worstTrade": 0}

    where = ["closed_at IS NOT NULL"]
    params = []
    date_clauses, date_params = _date_filter("closed_at", start_date, end_date)
    where.extend(date_clauses)
    params.extend(date_params)
    where_sql = " AND ".join(where)

    row = db.query(
        f"""SELECT
                COUNT(*) as total_trades,
                COUNT(*) FILTER (WHERE profit_usd > 0) as wins,
                COALESCE(SUM(profit_usd), 0) as net_profit,
                COALESCE(SUM(profit_usd) FILTER (WHERE profit_usd > 0), 0) as gross_profit,
                COALESCE(ABS(SUM(profit_usd) FILTER (WHERE profit_usd < 0)), 0.01) as gross_loss,
                COALESCE(AVG(profit_usd) FILTER (WHERE profit_usd > 0), 0) as avg_win,
                COALESCE(AVG(profit_usd) FILTER (WHERE profit_usd < 0), 0) as avg_loss,
                COALESCE(MAX(profit_usd), 0) as best_trade,
                COALESCE(MIN(profit_usd), 0) as worst_trade
            FROM trades WHERE {where_sql}""",
        tuple(params),
        one=True,
    )

    if not row or row["total_trades"] == 0:
        return {"totalTrades": 0, "winRate": 0, "netProfit": 0, "profitFactor": 0, "avgWin": 0, "avgLoss": 0, "bestTrade": 0, "worstTrade": 0}

    return {
        "totalTrades": row["total_trades"],
        "winRate": round(row["wins"] / row["total_trades"] * 100, 1) if row["total_trades"] > 0 else 0,
        "netProfit": round(float(row["net_profit"]), 2),
        "profitFactor": round(float(row["gross_profit"]) / float(row["gross_loss"]), 2),
        "avgWin": round(float(row["avg_win"]), 2),
        "avgLoss": round(float(row["avg_loss"]), 2),
        "bestTrade": round(float(row["best_trade"]), 2),
        "worstTrade": round(float(row["worst_trade"]), 2),
    }


@app.get("/api/trades/equity-curve")
async def get_equity_curve(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get cumulative equity curve from closed trades."""
    if not db.is_available():
        return {"points": []}

    where = ["closed_at IS NOT NULL"]
    params = []
    date_clauses, date_params = _date_filter("closed_at", start_date, end_date)
    where.extend(date_clauses)
    params.extend(date_params)
    where_sql = " AND ".join(where)

    rows = db.query(
        f"""SELECT closed_at, profit_usd,
                   SUM(profit_usd) OVER (ORDER BY closed_at) as cumulative
            FROM trades WHERE {where_sql}
            ORDER BY closed_at ASC""",
        tuple(params),
    )

    points = []
    for r in rows:
        dt = r["closed_at"].isoformat() if hasattr(r["closed_at"], "isoformat") else str(r["closed_at"])
        points.append({
            "time": dt,
            "profit": round(float(r["profit_usd"]), 2),
            "cumulative": round(float(r["cumulative"]), 2),
        })

    return {"points": points}


# ─── Model Insights Endpoints ───

@app.get("/api/model/metrics")
async def get_model_metrics():
    """Read model metrics from JSON file (written by bot on startup/retrain)."""
    for path in [MODEL_METRICS_FILE, Path("data/model_metrics.json")]:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
    return {"featureImportance": [], "trainAuc": 0, "testAuc": 0, "sampleCount": 0, "updatedAt": None}


@app.get("/api/model/training-history")
async def get_training_history():
    """Get model training run history."""
    if not db.is_available():
        return {"runs": []}

    rows = db.query(
        """SELECT id, started_at, completed_at, train_auc, test_auc,
                  sample_count, features_used, trigger_reason
           FROM training_runs
           ORDER BY started_at DESC
           LIMIT 20"""
    )
    for r in rows:
        for k in ("started_at", "completed_at"):
            if r.get(k) and hasattr(r[k], "isoformat"):
                r[k] = r[k].isoformat()
    return {"runs": rows}


@app.get("/api/model/regime-distribution")
async def get_regime_distribution():
    """Get regime distribution from recent market snapshots."""
    if not db.is_available():
        return {"distribution": []}

    rows = db.query(
        """SELECT regime, COUNT(*) as count
           FROM market_snapshots
           WHERE snapshot_time > NOW() - INTERVAL '7 days'
           GROUP BY regime
           ORDER BY count DESC"""
    )
    return {"distribution": rows}


# ─── Signal / Alert Endpoints ───

@app.get("/api/signals")
async def get_signals(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    type: str = Query("ALL"),
    executed: str = Query("all"),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Get paginated signal/alert history."""
    if not db.is_available():
        return {"signals": [], "total": 0, "page": page, "limit": limit}

    where = ["1=1"]
    params = []

    if type and type != "ALL":
        where.append("signal_type = %s")
        params.append(type.upper())

    if executed == "yes":
        where.append("executed = TRUE")
    elif executed == "no":
        where.append("executed = FALSE")

    date_clauses, date_params = _date_filter("signal_time", start_date, end_date)
    where.extend(date_clauses)
    params.extend(date_params)

    where_sql = " AND ".join(where)
    offset = (page - 1) * limit

    count_row = db.query(f"SELECT COUNT(*) as cnt FROM signals WHERE {where_sql}", tuple(params), one=True)
    total = count_row["cnt"] if count_row else 0

    params_with_pagination = params + [limit, offset]
    signals = db.query(
        f"""SELECT id, signal_time, signal_type, confidence, executed,
                   execution_reason, regime, session, smc_signal, ml_signal,
                   entry_price, sl_price, tp_price
            FROM signals
            WHERE {where_sql}
            ORDER BY signal_time DESC
            LIMIT %s OFFSET %s""",
        tuple(params_with_pagination),
    )

    for s in signals:
        if s.get("signal_time") and hasattr(s["signal_time"], "isoformat"):
            s["signal_time"] = s["signal_time"].isoformat()

    return {"signals": signals, "total": total, "page": page, "limit": limit}


@app.get("/api/signals/stats")
async def get_signal_stats(hours: int = Query(24, ge=1, le=168)):
    """Get signal statistics for the last N hours."""
    if not db.is_available():
        return {"total": 0, "executed": 0, "executionRate": 0, "avgConfidence": 0, "byType": {}}

    row = db.query(
        """SELECT
               COUNT(*) as total,
               COUNT(*) FILTER (WHERE executed = TRUE) as executed,
               COALESCE(AVG(confidence), 0) as avg_confidence
           FROM signals
           WHERE signal_time > NOW() - MAKE_INTERVAL(hours => %s)""",
        (hours,),
        one=True,
    )

    if not row or row["total"] == 0:
        return {"total": 0, "executed": 0, "executionRate": 0, "avgConfidence": 0, "byType": {}}

    by_type = db.query(
        """SELECT signal_type, COUNT(*) as count
           FROM signals
           WHERE signal_time > NOW() - MAKE_INTERVAL(hours => %s)
           GROUP BY signal_type""",
        (hours,),
    )

    return {
        "total": row["total"],
        "executed": row["executed"],
        "executionRate": round(row["executed"] / row["total"] * 100, 1) if row["total"] > 0 else 0,
        "avgConfidence": round(float(row["avg_confidence"]), 1),
        "byType": {r["signal_type"]: r["count"] for r in by_type},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
