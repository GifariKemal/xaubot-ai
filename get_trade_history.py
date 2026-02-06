"""Get real trading history from MT5."""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5

if not mt5.initialize():
    print('MT5 init failed')
    exit()

if not mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER')):
    print('MT5 login failed')
    exit()

# Get account info
account = mt5.account_info()
print(f'Account: {account.login}')
print(f'Balance: ${account.balance:,.2f}')
print(f'Equity: ${account.equity:,.2f}')
print()

# Get trade history (last 14 days)
from_date = datetime.now() - timedelta(days=14)
to_date = datetime.now() + timedelta(days=1)

deals = mt5.history_deals_get(from_date, to_date)
print(f'Total deals in last 14 days: {len(deals) if deals else 0}')
print()

if deals:
    # Group by position to calculate trade results
    trades = {}
    for deal in deals:
        if deal.position_id > 0:
            if deal.position_id not in trades:
                trades[deal.position_id] = []
            trades[deal.position_id].append(deal)

    print('=' * 70)
    print('REAL TRADING HISTORY (Last 14 days)')
    print('=' * 70)

    total_profit = 0
    wins = 0
    losses = 0
    trade_list = []

    for pos_id, pos_deals in trades.items():
        if len(pos_deals) >= 2:
            # Has entry and exit
            entry = next((d for d in pos_deals if d.entry == 0), None)  # DEAL_ENTRY_IN
            exit_deal = next((d for d in pos_deals if d.entry == 1), None)  # DEAL_ENTRY_OUT

            if entry and exit_deal:
                profit = exit_deal.profit
                direction = 'BUY' if entry.type == 0 else 'SELL'
                entry_time = datetime.fromtimestamp(entry.time)
                exit_time = datetime.fromtimestamp(exit_deal.time)

                result = 'WIN' if profit > 0 else 'LOSS'
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                total_profit += profit

                trade_list.append({
                    'time': entry_time,
                    'direction': direction,
                    'lot': entry.volume,
                    'profit': profit,
                    'result': result
                })

    # Sort by time and print
    trade_list.sort(key=lambda x: x['time'])
    for t in trade_list[-50:]:  # Last 50 trades
        print(f"  {t['time']} | {t['direction']} | Lot: {t['lot']} | ${t['profit']:+.2f} [{t['result']}]")

    print()
    print('=' * 70)
    print('REAL TRADING SUMMARY')
    print('=' * 70)
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0

    print(f'  Total Trades    : {total_trades}')
    print(f'  Winning Trades  : {wins}')
    print(f'  Losing Trades   : {losses}')
    print(f'  Win Rate        : {win_rate:.1f}%')
    print(f'  Total P/L       : ${total_profit:+,.2f}')
    print(f'  Average/Trade   : ${avg_profit:+.2f}')

mt5.shutdown()
