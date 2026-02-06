"""Close all open positions."""
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

# Get open positions
positions = mt5.positions_get()
if positions:
    for pos in positions:
        print(f"\nClosing #{pos.ticket} | {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} {pos.symbol}")
        print(f"   Open: {pos.price_open:.2f} | Current: {pos.price_current:.2f}")
        print(f"   Profit: ${pos.profit:,.2f}")

        # Close position
        tick = mt5.symbol_info_tick(pos.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Manual close",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   CLOSED successfully! Profit: ${pos.profit:,.2f}")
        else:
            print(f"   Failed to close: {result.comment} (code: {result.retcode})")
else:
    print("No open positions")

# Check final balance
account = mt5.account_info()
print(f"\n{'='*50}")
print(f"Final Balance: ${account.balance:,.2f}")
print(f"Final Equity: ${account.equity:,.2f}")

mt5.shutdown()
