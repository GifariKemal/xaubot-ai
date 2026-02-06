"""Test new risk settings."""
from src.smart_risk_manager import create_smart_risk_manager

# Test dengan modal $50
print('=' * 50)
print('PENGATURAN RISK MANAGEMENT BARU')
print('=' * 50)
manager = create_smart_risk_manager(50)

print()
print('Dengan modal $50:')
print(f'  Daily Loss Limit (5%): ${manager.max_daily_loss_usd:.2f}')
print(f'  Total Loss Limit (10%): ${manager.max_total_loss_usd:.2f}')
print(f'  S/L Per Trade (1%): ${manager.max_loss_per_trade:.2f}')
print()
print(manager.get_risk_summary())
