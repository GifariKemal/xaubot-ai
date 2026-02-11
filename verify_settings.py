#!/usr/bin/env python
"""Quick verification of v0.1.1 settings."""
from src.smart_risk_manager import create_smart_risk_manager

m = create_smart_risk_manager(5000)
print(f"Capital: ${m.capital:.2f}")
print(f"Max Loss Per Trade: ${m.max_loss_per_trade:.2f} ({m.max_loss_per_trade_percent}%)")
print(f"Daily Loss Limit: ${m.max_daily_loss_usd:.2f} ({m.max_daily_loss_percent}%)")
print("\n✅ FIX 5 Status: Max loss is", "${:.2f}".format(m.max_loss_per_trade), "- PASS!" if m.max_loss_per_trade == 25.0 else "- CHECK!")
