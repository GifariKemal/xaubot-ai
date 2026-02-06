"""Quick status check script."""
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

mt5.initialize()
mt5.login(
    int(os.getenv('MT5_LOGIN')),
    os.getenv('MT5_PASSWORD'),
    os.getenv('MT5_SERVER')
)

# Account info
info = mt5.account_info()
print('='*50)
print('ACCOUNT STATUS')
print('='*50)
print(f'Balance: ${info.balance:,.2f}')
print(f'Equity: ${info.equity:,.2f}')
print(f'Profit: ${info.profit:,.2f}')
print(f'Margin: ${info.margin:,.2f}')

# Open positions
positions = mt5.positions_get(symbol='XAUUSD')
print(f'\nOpen Positions: {len(positions) if positions else 0}')
if positions:
    total_profit = 0
    for pos in positions:
        total_profit += pos.profit
        ptype = "BUY" if pos.type==0 else "SELL"
        print(f'  #{pos.ticket}: {ptype} {pos.volume} @ {pos.price_open:.2f} | P/L: ${pos.profit:.2f}')
    print(f'  Total Floating: ${total_profit:.2f}')

# Recent closed trades
history = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
if history:
    closed_trades = [d for d in history if d.profit != 0]
    print(f'\nClosed Trades (24h): {len(closed_trades)}')
    total_closed = 0
    for deal in closed_trades[-10:]:
        total_closed += deal.profit
        result = "WIN" if deal.profit > 0 else "LOSS"
        print(f'  #{deal.ticket}: {deal.symbol} ${deal.profit:+.2f} [{result}]')
    print(f'  Total Closed P/L: ${total_closed:+.2f}')

mt5.shutdown()
