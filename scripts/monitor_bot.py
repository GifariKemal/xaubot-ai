#!/usr/bin/env python3
"""
Bot Health Monitor & Trade Analyzer
Runs every 1 hour to check bot status and analyze trades
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import polars as pl
from dotenv import load_dotenv
import os

load_dotenv()

def check_bot_health():
    """Check if bot is running and healthy"""
    print("\n" + "="*60)
    print(f"BOT HEALTH CHECK - {datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M:%S WIB')}")
    print("="*60)

    # Check lock file
    lock_file = Path("data/bot.lock")
    if lock_file.exists():
        with open(lock_file) as f:
            content = f.read().strip()
            print(f"[OK] Bot lock exists: {content}")
    else:
        print("[ERROR] WARNING: No bot lock file found - bot may not be running!")
        return False

    # Check bot_status.json
    status_file = Path("data/bot_status.json")
    if status_file.exists():
        import json
        with open(status_file) as f:
            status = json.load(f)
        print(f"[OK] Bot connected: {status['connected']}")
        print(f"  Price: ${status['price']:.2f}")
        print(f"  Balance: ${status['balance']:.2f}")
        print(f"  Equity: ${status['equity']:.2f}")
        print(f"  Open positions: {len(status['positions'])}")
        print(f"  Daily P/L: ${status['dailyProfit']:.2f} / -${status['dailyLoss']:.2f}")

        # Check last update time
        try:
            last_update = datetime.strptime(status['timestamp'], "%H:%M:%S").replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day
            )
            time_since_update = (datetime.now() - last_update).total_seconds()
            if time_since_update > 300:  # 5 minutes
                print(f"[WARN] WARNING: Status last updated {time_since_update/60:.1f} minutes ago!")
            else:
                print(f"[OK] Status fresh ({time_since_update:.0f}s ago)")
        except:
            pass
    else:
        print("[ERROR] WARNING: No bot status file found!")
        return False

    return True

def analyze_todays_trades():
    """Analyze all trades from today"""
    print("\n" + "="*60)
    print("TODAY'S TRADE ANALYSIS")
    print("="*60)

    # Connect to MT5
    if not mt5.initialize():
        print("[X] Failed to connect to MT5")
        return

    try:
        # Get today's trades
        today_start = datetime.now(ZoneInfo("Asia/Jakarta")).replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_utc = today_start.astimezone(ZoneInfo("UTC"))

        deals = mt5.history_deals_get(today_start_utc, datetime.now())
        if not deals:
            print("No trades today")
            return

        # Convert to DataFrame
        deals_dict = [deal._asdict() for deal in deals]
        df = pl.DataFrame(deals_dict)

        # Filter closed positions (deals with profit/loss)
        closed_trades = df.filter(
            (pl.col("entry") == 1) & (pl.col("profit") != 0)
        )

        if len(closed_trades) == 0:
            print("No closed trades today")
            return

        # Calculate statistics
        total_trades = len(closed_trades)
        wins = closed_trades.filter(pl.col("profit") > 0)
        losses = closed_trades.filter(pl.col("profit") < 0)

        total_profit = closed_trades["profit"].sum()
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        avg_win = wins["profit"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["profit"].mean()) if len(losses) > 0 else 0

        print(f"Total Trades: {total_trades}")
        print(f"Wins: {win_count} | Losses: {loss_count}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P/L: ${total_profit:.2f}")
        print(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")

        # Identify issues
        print("\n" + "-"*60)
        print("ISSUE DETECTION")
        print("-"*60)

        issues = []

        # Check win rate
        if win_rate < 45:
            issues.append(f"[WARN]  CRITICAL: Win rate too low ({win_rate:.1f}% < 45%)")
        elif win_rate < 50:
            issues.append(f"[WARN]  WARNING: Win rate below target ({win_rate:.1f}% < 50%)")

        # Check average loss vs win
        if avg_loss > avg_win * 1.5:
            issues.append(f"[WARN]  WARNING: Average loss (${avg_loss:.2f}) > 1.5x average win (${avg_win:.2f})")

        # Check for large losses
        if len(losses) > 0:
            max_loss = abs(losses["profit"].min())
            if max_loss > 20:
                issues.append(f"[WARN]  CRITICAL: Large loss detected: ${max_loss:.2f}")

        # Check consecutive losses
        closed_sorted = closed_trades.sort("time")
        consecutive_losses = 0
        max_consecutive_losses = 0
        for profit in closed_sorted["profit"].to_list():
            if profit < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        if max_consecutive_losses >= 3:
            issues.append(f"[WARN]  WARNING: {max_consecutive_losses} consecutive losses detected")

        # Check BUY vs SELL performance
        buy_trades = closed_trades.filter(pl.col("type") == 0)
        sell_trades = closed_trades.filter(pl.col("type") == 1)

        if len(buy_trades) > 0:
            buy_winrate = len(buy_trades.filter(pl.col("profit") > 0)) / len(buy_trades) * 100
            print(f"\nBUY Trades: {len(buy_trades)} | Win Rate: {buy_winrate:.1f}%")

        if len(sell_trades) > 0:
            sell_winrate = len(sell_trades.filter(pl.col("profit") > 0)) / len(sell_trades) * 100
            print(f"SELL Trades: {len(sell_trades)} | Win Rate: {sell_winrate:.1f}%")

            if sell_winrate < 45 and len(sell_trades) >= 5:
                issues.append(f"[WARN]  CRITICAL: SELL win rate very low ({sell_winrate:.1f}%)")

        # Print issues
        if issues:
            print("\n" + "="*60)
            print("DETECTED ISSUES:")
            for issue in issues:
                print(issue)
            print("="*60)
        else:
            print("\n[OK] No critical issues detected")

        # Recent trades detail
        print("\n" + "-"*60)
        print("LAST 5 TRADES")
        print("-"*60)

        recent = closed_sorted.tail(5)
        for row in recent.iter_rows(named=True):
            trade_time = datetime.fromtimestamp(row['time'], ZoneInfo("Asia/Jakarta"))
            trade_type = "BUY" if row['type'] == 0 else "SELL"
            profit = row['profit']
            emoji = "[OK]" if profit > 0 else "[X]"
            print(f"{emoji} #{row['ticket']} {trade_type} ${profit:+.2f} @ {trade_time.strftime('%H:%M:%S')}")

    finally:
        mt5.shutdown()

def check_open_positions():
    """Check current open positions"""
    print("\n" + "="*60)
    print("OPEN POSITIONS")
    print("="*60)

    if not mt5.initialize():
        print("[X] Failed to connect to MT5")
        return

    try:
        positions = mt5.positions_get(symbol="XAUUSD")
        if not positions or len(positions) == 0:
            print("No open positions")
            return

        print(f"Open Positions: {len(positions)}\n")

        total_profit = 0
        for pos in positions:
            pos_type = "BUY" if pos.type == 0 else "SELL"
            duration = (datetime.now().timestamp() - pos.time) / 60  # minutes
            emoji = "[+]" if pos.profit > 0 else "[-]"
            print(f"{emoji} #{pos.ticket} {pos_type} {pos.volume} lots")
            print(f"   Entry: ${pos.price_open:.2f} | Current: ${pos.price_current:.2f}")
            print(f"   Profit: ${pos.profit:.2f} | Duration: {duration:.1f}m")
            print(f"   SL: ${pos.sl:.2f} | TP: ${pos.tp:.2f}")
            print()
            total_profit += pos.profit

        print(f"Total Floating P/L: ${total_profit:.2f}")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    try:
        # Run all checks
        bot_healthy = check_bot_health()
        analyze_todays_trades()
        check_open_positions()

        print("\n" + "="*60)
        print(f"Monitor completed at {datetime.now(ZoneInfo('Asia/Jakarta')).strftime('%H:%M:%S WIB')}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[X] ERROR during monitoring: {e}")
        import traceback
        traceback.print_exc()
