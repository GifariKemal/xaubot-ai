"""Check open positions and account status."""
import os
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5

# Connect
mt5.initialize(
    login=int(os.getenv("MT5_LOGIN")),
    password=os.getenv("MT5_PASSWORD"),
    server=os.getenv("MT5_SERVER"),
    path=os.getenv("MT5_PATH"),
)

# Account info
account = mt5.account_info()
print("=" * 50)
print("ACCOUNT STATUS")
print("=" * 50)
print(f"Balance: ${account.balance:,.2f}")
print(f"Equity: ${account.equity:,.2f}")
print(f"Margin: ${account.margin:,.2f}")
print(f"Free Margin: ${account.margin_free:,.2f}")
print(f"Profit: ${account.profit:,.2f}")
print(f"Leverage: 1:{account.leverage}")

# Open positions
print("\n" + "=" * 50)
print("OPEN POSITIONS")
print("=" * 50)
positions = mt5.positions_get()
if positions:
    for pos in positions:
        print(f"#{pos.ticket} | {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} {pos.symbol}")
        print(f"   Open: {pos.price_open:.2f} | Current: {pos.price_current:.2f}")
        print(f"   SL: {pos.sl:.2f} | TP: {pos.tp:.2f}")
        print(f"   Profit: ${pos.profit:,.2f}")
        print()
else:
    print("No open positions")

# Recent history
print("=" * 50)
print("RECENT DEALS (Last 10)")
print("=" * 50)
from datetime import datetime, timedelta
deals = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
if deals:
    for deal in deals[-10:]:
        deal_type = "BUY" if deal.type == 0 else "SELL" if deal.type == 1 else "OTHER"
        print(f"#{deal.ticket} | {deal_type} {deal.volume} @ {deal.price:.2f} | Profit: ${deal.profit:,.2f}")
else:
    print("No recent deals")

mt5.shutdown()
