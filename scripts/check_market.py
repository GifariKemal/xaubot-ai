"""Quick market analysis script"""
# Run from project root: python scripts/check_market.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.mt5_connector import MT5Connector
from src.smc_polars import SMCAnalyzer
from src.config import TradingConfig

config = TradingConfig()
mt5 = MT5Connector(config.mt5_login, config.mt5_password, config.mt5_server, config.mt5_path)
mt5.connect()

# Get data
df = mt5.get_market_data('XAUUSD', 'M15', 500)
print('=== MARKET DATA ===')
print(f'Candles: {len(df)}')
print(f'Last close: {df["close"].tail(1).item():.2f}')

# Current price
tick = mt5.get_tick('XAUUSD')
print(f'Bid: {tick.bid:.2f}, Ask: {tick.ask:.2f}')
print(f'Spread: {(tick.ask - tick.bid):.2f}')

# SMC Analysis
smc = SMCAnalyzer()
df_smc = smc.calculate_all(df)

# Check last 20 candles for SMC patterns
print('')
print('=== SMC PATTERNS (Last 20 candles) ===')
last_20 = df_smc.tail(20).select(['time', 'close', 'bos', 'choch', 'is_fvg_bull', 'is_fvg_bear', 'ob', 'fvg_signal', 'market_structure']).to_dicts()

pattern_found = False
for i, row in enumerate(last_20):
    markers = []
    if row.get('bos', 0) != 0:
        markers.append(f'BOS={row["bos"]}')
    if row.get('choch', 0) != 0:
        markers.append(f'CHoCH={row["choch"]}')
    if row.get('is_fvg_bull'):
        markers.append('FVG_BULL')
    if row.get('is_fvg_bear'):
        markers.append('FVG_BEAR')
    if row.get('ob', 0) > 0:
        markers.append('OB_BULL')
    if row.get('ob', 0) < 0:
        markers.append('OB_BEAR')

    if markers:
        pattern_found = True
        print(f'  [{i}] {row["close"]:.2f} | {" | ".join(markers)}')

if not pattern_found:
    print('  No patterns in last 20 candles!')

# Generate signal
signal = smc.generate_signal(df_smc)
print('')
print('=== SMC SIGNAL RESULT ===')
if signal:
    print(f'Signal: {signal.signal_type}')
    print(f'Entry: {signal.entry_price:.2f}')
    print(f'SL: {signal.stop_loss:.2f}')
    print(f'TP: {signal.take_profit:.2f}')
    print(f'Confidence: {signal.confidence:.0%}')
    print(f'Reason: {signal.reason}')
else:
    print('Signal: NONE - No valid setup')

    # Check last 5 candles
    print('')
    print('Last 5 candles detail:')
    last_5 = df_smc.tail(5).to_dicts()
    for i, row in enumerate(last_5):
        print(f'  [{i}] Close={row["close"]:.2f}, BOS={row.get("bos",0)}, CHoCH={row.get("choch",0)}, FVG_B={row.get("is_fvg_bull",False)}, FVG_S={row.get("is_fvg_bear",False)}, OB={row.get("ob",0)}')

# Check overall SMC stats
print('')
print('=== SMC STATISTICS (All 500 candles) ===')
bos_bull = df_smc.filter(df_smc['bos'] > 0).height
bos_bear = df_smc.filter(df_smc['bos'] < 0).height
choch_bull = df_smc.filter(df_smc['choch'] > 0).height
choch_bear = df_smc.filter(df_smc['choch'] < 0).height
fvg_bull = df_smc.filter(df_smc['is_fvg_bull'] == True).height
fvg_bear = df_smc.filter(df_smc['is_fvg_bear'] == True).height
ob_bull = df_smc.filter(df_smc['ob'] > 0).height
ob_bear = df_smc.filter(df_smc['ob'] < 0).height

print(f'BOS Bullish: {bos_bull}, BOS Bearish: {bos_bear}')
print(f'CHoCH Bullish: {choch_bull}, CHoCH Bearish: {choch_bear}')
print(f'FVG Bullish: {fvg_bull}, FVG Bearish: {fvg_bear}')
print(f'OB Bullish: {ob_bull}, OB Bearish: {ob_bear}')

# Check when was the last BOS/CHoCH
print('')
print('=== LAST STRUCTURE BREAKS ===')
bos_indices = df_smc.with_row_index().filter(df_smc['bos'] != 0).select(['index', 'time', 'close', 'bos']).tail(3).to_dicts()
choch_indices = df_smc.with_row_index().filter(df_smc['choch'] != 0).select(['index', 'time', 'close', 'choch']).tail(3).to_dicts()

print('Last 3 BOS:')
for row in bos_indices:
    candles_ago = 499 - row['index']
    print(f'  {row["time"]} | Close={row["close"]:.2f} | BOS={row["bos"]} | {candles_ago} candles ago')

print('Last 3 CHoCH:')
for row in choch_indices:
    candles_ago = 499 - row['index']
    print(f'  {row["time"]} | Close={row["close"]:.2f} | CHoCH={row["choch"]} | {candles_ago} candles ago')

mt5.disconnect()

# === MACRO CONTEXT (Phase 9 Enhancement) ===
print('')
print('=== MACRO-ECONOMIC CONTEXT FOR GOLD ===')
print('(Fetching macro data...)')

import asyncio
from src.macro_connector import MacroDataConnector

async def get_macro_context():
    """Fetch and display macro context."""
    try:
        connector = MacroDataConnector()
        summary = await connector.get_macro_context()
        print(summary)

        # Additional insights
        macro_score, components = await connector.calculate_macro_score()

        print('')
        print('=== MACRO SCORE BREAKDOWN ===')
        print(f'Overall Score: {macro_score:.2f} (0=Bearish, 0.5=Neutral, 1=Bullish)')
        print('')

        if components.get('dxy_score') is not None:
            print(f'  DXY Contribution: {components["dxy_score"]:.2f} (weight: 35%)')
        if components.get('vix_score') is not None:
            print(f'  VIX Contribution: {components["vix_score"]:.2f} (weight: 25%)')
        if components.get('yields_score') is not None:
            print(f'  Yields Contribution: {components["yields_score"]:.2f} (weight: 30%)')
        if components.get('fed_score') is not None:
            print(f'  Fed Rate Contribution: {components["fed_score"]:.2f} (weight: 10%)')

        print('')
        print('=== TRADING IMPLICATIONS ===')
        if macro_score < 0.3:
            print('  Macro environment is BEARISH for gold')
            print('  Consider: Reduce position sizes, avoid aggressive longs')
        elif macro_score < 0.7:
            print('  Macro environment is NEUTRAL for gold')
            print('  Consider: Trade technically, normal position sizing')
        else:
            print('  Macro environment is BULLISH for gold')
            print('  Consider: Favor long bias, can increase position sizes')

    except Exception as e:
        print(f'  Error fetching macro data: {e}')
        print('  Note: Set FRED_API_KEY env var for real yields & fed funds data')
        print('  (DXY and VIX work without API key)')

try:
    asyncio.run(get_macro_context())
except Exception as e:
    print(f'  Could not fetch macro data: {e}')
    print('  This is optional - SMC analysis above is still valid')
