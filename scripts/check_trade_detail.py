"""Quick script to check trade #158666685 details."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
mt5.initialize(os.getenv('MT5_PATH'))
mt5.login(int(os.getenv('MT5_LOGIN')), os.getenv('MT5_PASSWORD'), os.getenv('MT5_SERVER'))

from_date = datetime(2026, 2, 7)
to_date = datetime(2026, 2, 10)

deals = mt5.history_deals_get(from_date, to_date)
if deals:
    for d in deals:
        if d.position_id == 158666685:
            deal_type = 'BUY' if d.type == 0 else 'SELL'
            entry = 'IN' if d.entry == 0 else 'OUT'
            t = datetime.fromtimestamp(d.time)
            print(f"Ticket: {d.ticket} | Position: {d.position_id}")
            print(f"  Time: {t} (WIB: {t.hour+7 if t.hour+7<24 else t.hour+7-24}:{t.minute:02d})")
            print(f"  Type: {deal_type} | Entry: {entry}")
            print(f"  Price: {d.price} | Volume: {d.volume}")
            print(f"  Profit: ${d.profit:.2f} | Commission: ${d.commission:.2f}")
            print(f"  Comment: {d.comment}")
            print()

# Also check recent orders for context
orders = mt5.history_orders_get(from_date, to_date)
if orders:
    for o in orders:
        if o.position_id == 158666685:
            t = datetime.fromtimestamp(o.time_setup)
            order_type = 'BUY' if o.type == 0 else 'SELL'
            print(f"Order: {o.ticket} | Position: {o.position_id}")
            print(f"  Setup: {t}")
            print(f"  Type: {order_type} | Price: {o.price_open}")
            print(f"  SL: {o.sl} | TP: {o.tp}")
            print(f"  Comment: {o.comment}")
            print()

mt5.shutdown()
