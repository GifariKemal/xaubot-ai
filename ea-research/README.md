# EA Research — Gold Expert Advisors Analysis

**Purpose:** Deep analysis of commercial Gold EAs to extract strategies, patterns, and improvement ideas for XAUBot AI.

**Date:** 2026-02-09

---

## Folder Structure

```
ea-research/
├── README.md                    # This file
├── gold-1-minute/              # Gold 1 Minute EA (FREE, M1, Price Action)
│   ├── ea-file.mq5             # EA source/compiled (if available)
│   ├── ANALYSIS.md             # Deep analysis
│   ├── strategy.md             # Strategy breakdown
│   └── screenshots/            # Performance screenshots
├── gold-1-minute-grid/         # Gold 1 Minute Grid ($200, M1, Grid+Trend)
│   ├── ANALYSIS.md
│   ├── strategy.md
│   └── research-notes.md
├── ai-gold-sniper/             # AI Gold Sniper ($499, H1, GPT-4o+CNN/RNN)
│   ├── ANALYSIS.md
│   ├── strategy.md
│   └── ml-approach.md
└── analysis/                   # Comparative analysis
    ├── COMPARISON.md           # Side-by-side comparison
    ├── strategy-patterns.md    # Common patterns across EAs
    └── improvement-ideas.md    # Ideas for XAUBot enhancement
```

---

## EAs Under Analysis

### 1. Gold 1 Minute (FREE)
- **Update:** 3 Feb 2026 (v10.6)
- **Price:** FREE (until v10.7 → $50)
- **Timeframe:** M1
- **Strategy:** Price Action (Engulfing, Breakout-Retest) + HTF Trend Filter
- **Link:** https://www.mql5.com/en/market/product/152875
- **Status:** 🔄 Downloading & Analyzing

### 2. Gold 1 Minute Grid ($200)
- **Update:** 8 Feb 2026 (v9.5) — LATEST
- **Price:** $200 USD (rental $100/3mo)
- **Timeframe:** M1
- **Strategy:** Grid + Protect Layers + Trend Filter
- **Link:** https://www.mql5.com/en/market/product/156724
- **Status:** 🔄 Analyzing (Commercial, no source)

### 3. AI Gold Sniper MT5 ($499)
- **Update:** 8 Feb 2026 (v4.3) — LATEST
- **Price:** $499 USD
- **Timeframe:** H1
- **Strategy:** GPT-4o + CNN/RNN + Deep RL + NLP News
- **Link:** https://www.mql5.com/en/market/product/133197
- **Status:** 🔄 Analyzing (Commercial, no source)

---

## Analysis Goals

1. ✅ **Strategy Extraction** — Understand core logic, entry/exit rules
2. ✅ **Risk Management** — How do they handle SL, TP, drawdown?
3. ✅ **Time Filtering** — Session/hour filters, news avoidance
4. ✅ **Position Management** — Single vs basket, trailing, breakeven
5. ✅ **ML Approach** (AI Gold Sniper) — How GPT-4o integrated? Feature engineering?
6. ✅ **Grid Strategy** (Gold Grid) — Safe grid vs risky grid, how to adapt?
7. ✅ **Comparison with XAUBot** — What can we learn? What's better in XAUBot?
8. ✅ **Improvement Ideas** — Concrete enhancements for XAUBot AI

---

## Research Methodology

### For FREE EAs (Gold 1 Minute):
1. Download EA from MQL5
2. Decompile if needed (for educational purposes only)
3. Extract strategy logic
4. Backtest on our data
5. Compare performance with XAUBot

### For Commercial EAs (Grid, AI Sniper):
1. Deep dive into product page descriptions
2. Analyze user reviews for strategy hints
3. Study screenshots and performance charts
4. Extract algorithmic patterns from behavior
5. Read developer comments/documentation
6. Reverse-engineer logic from signals (if demo available)

---

## Key Questions to Answer

### Strategy Questions:
- What timeframe is optimal for Gold? (M1 vs M15 vs H1)
- How effective is pure Price Action vs ML?
- Grid strategy: When is it safe? How to protect?
- Is GPT-4o/LLM useful for trading? How?

### Technical Questions:
- Feature engineering: What features do they use?
- Regime detection: Do any use HMM or similar?
- Risk management: Fixed lot vs dynamic sizing?
- Position management: Basket vs individual?

### Comparative Questions:
- XAUBot unique advantages?
- XAUBot weaknesses vs commercial EAs?
- Low-hanging fruit improvements?
- Long-term enhancement roadmap?

---

## Next Steps

1. ⏳ Download Gold 1 Minute EA (FREE)
2. ⏳ Deep analysis of each EA (create ANALYSIS.md in each folder)
3. ⏳ Extract strategy patterns (create strategy-patterns.md)
4. ⏳ Generate improvement ideas (create improvement-ideas.md)
5. ⏳ Create comprehensive comparison (create COMPARISON.md)
6. ⏳ Present findings and recommendations to user

---

## Notes

- **Legal:** All analysis for educational purposes only
- **Ethics:** No code theft; learn patterns, not copy implementations
- **Goal:** Improve XAUBot AI with battle-tested strategies from commercial EAs
- **Respect:** Give credit to EA developers for their innovations

---

**Status:** 🔄 In Progress
**Last Updated:** 2026-02-09 11:00 WIB
