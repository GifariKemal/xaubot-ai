#!/usr/bin/env python3
"""Quick performance analysis"""

# Last 10 trades
wins = [15.64, 14.58, 0.93, 0.53, 0.01, 0.03, 0.28, 0.58]
losses = [7.12, 34.70]

print("=" * 60)
print("XAUBOT AI v0.6.0 - PERFORMANCE ANALYSIS")
print("=" * 60)
print()
print(f"PROFIT METRICS:")
print(f"  Avg Win:  ${sum(wins)/len(wins):.2f}")
print(f"  Avg Loss: ${sum(losses)/len(losses):.2f}")
print(f"  Max Win:  ${max(wins):.2f}")
print(f"  Max Loss: ${max(losses):.2f}")
print(f"  Loss/Win Ratio: {sum(losses)/sum(wins):.2f}x")
print()
print("WIN DISTRIBUTION:")
micro = len([w for w in wins if w < 1])
small = len([w for w in wins if 1 <= w < 5])
good = len([w for w in wins if 5 <= w < 15])
excellent = len([w for w in wins if w >= 15])
total = len(wins)

print(f"  Micro (<$1):     {micro} trades ({micro/total*100:.0f}%)")
print(f"  Small ($1-5):    {small} trades ({small/total*100:.0f}%)")
print(f"  Good ($5-15):    {good} trades ({good/total*100:.0f}%)")
print(f"  Excellent (>$15): {excellent} trades ({excellent/total*100:.0f}%)")
print()
print("=" * 60)
