"""Modify TP of open positions to closer targets."""
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

# Get current tick
tick = mt5.symbol_info_tick("XAUUSD")
current_price = tick.bid
print(f"Current price: {current_price:.2f}")

# Get open positions
positions = mt5.positions_get()
if positions:
    for pos in positions:
        print(f"\n#{pos.ticket} | {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} {pos.symbol}")
        print(f"   Open: {pos.price_open:.2f} | Current: {pos.price_current:.2f}")
        print(f"   Current SL: {pos.sl:.2f} | Current TP: {pos.tp:.2f}")
        print(f"   Profit: ${pos.profit:,.2f}")

        # Set TP 5 points above current to lock in profits
        if pos.type == 0:  # BUY
            new_tp = current_price + 5  # 5 points above current for quick TP
            new_sl = pos.price_open - 10   # Tighter stop loss (protect profit)
        else:  # SELL
            new_tp = current_price - 5
            new_sl = pos.price_open + 10

        print(f"   New SL: {new_sl:.2f} | New TP: {new_tp:.2f}")

        # Modify position
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   MODIFIED successfully!")
        else:
            print(f"   Failed to modify: {result.comment} (code: {result.retcode})")
else:
    print("No open positions")

mt5.shutdown()
