//+------------------------------------------------------------------+
//| XAUBot_Pro_V3.mq5                                                |
//| Advanced M15 Gold Trading EA with Multi-Layer Quality Filtering |
//| Design: Capital Preservation Through Extreme Selectivity         |
//| Brand: suriota                                                   |
//+------------------------------------------------------------------+
#property copyright "XAUBot Pro - suriota"
#property version   "3.00"
#property description "4-Layer Quality + H1 Bias + ATR Adaptive + Patient Exits"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//===========================================
// SECTION 1: INPUTS (1-150)
//===========================================

input group "=== Risk Management ==="
input double   RiskPercent = 1.0;              // Base risk per trade (%)
input double   MinRiskPercent = 0.5;           // Minimum risk after losses
input double   MaxLot = 0.02;                  // Maximum lot size (safety cap)
input double   MinLot = 0.01;                  // Minimum lot size
input double   ATR_SL_Multiplier = 1.0;        // ATR multiplier for SL
input double   DailyLossLimit = 5.0;           // Daily loss limit (%)
input double   MonthlyLossLimit = 10.0;        // Monthly loss limit (%)
input int      MaxConsecutiveLosses = 3;       // Max consecutive losses before halt

input group "=== Entry Filters ==="
input int      EMA_Fast_M15 = 50;              // M15 Fast EMA period
input int      EMA_Slow_M15 = 200;             // M15 Slow EMA period
input int      EMA_Fast_H1 = 50;               // H1 Fast EMA period
input int      EMA_Slow_H1 = 200;              // H1 Slow EMA period
input int      ADX_Period = 14;                // ADX period
input double   ADX_Threshold = 25.0;           // Minimum ADX for entry
input int      RSI_Period = 14;                // RSI period
input double   MaxSpread = 20.0;               // Maximum spread (points)
input int      CooldownMinutes = 15;           // Cooldown between trades
input int      MaxPositions = 2;               // Maximum concurrent positions
input int      MaxTradesPerDay = 10;           // Maximum trades per day
input double   MinQualityScore = 60.0;         // Minimum quality score (0-100)

input group "=== Exit Management ==="
input double   TP_Hard_ATR = 2.0;              // Hard TP (ATR multiplier)
input double   BE_Trigger_ATR = 0.5;           // Breakeven trigger (ATR)
input double   BE_Lock_USD = 2.0;              // Breakeven lock profit ($)
input double   Trail_Trigger_ATR = 0.6;        // Trailing stop trigger (ATR)
input double   Trail_Distance_ATR = 0.3;       // Trailing distance (ATR)
input double   Hard_Stop_ATR = 0.6;            // Hard stop loss (ATR)
input int      MinTradeAgeMinutes = 5;         // Minimum trade age for stops
input int      TimeExit_Hour = 3;              // Time exit if not profitable (hours)
input int      AbsoluteExit_Hour = 5;          // Absolute exit time (hours)

input group "=== Panel & Logging ==="
input bool     ShowPanel = true;               // Show info panel
input ENUM_BASE_CORNER PanelCorner = CORNER_LEFT_UPPER;
input int      PanelOffsetX = 380;
input int      PanelOffsetY = 10;
input bool     EnableFileLog = true;           // Enable file logging
input bool     LogFilterRejects = false;       // Log filter rejections
input int      Magic = 202603;                 // Magic number

//===========================================
// SECTION 2: GLOBAL VARIABLES (151-250)
//===========================================

// Trading objects
CTrade trade;
CPositionInfo position;
CSymbolInfo symbolInfo;

// M15 Indicators
int handleEMAFast_M15, handleEMASlow_M15, handleADX_M15, handleRSI_M15, handleMACD_M15, handleATR_M15;
double emaFast_M15, emaSlow_M15, adxValue_M15, rsiValue_M15, macdMain_M15, macdSignal_M15, atrValue_M15;

// H1 Indicators
int handleEMAFast_H1, handleEMASlow_H1, handleRSI_H1, handleMACD_H1;
double emaFast_H1, emaSlow_H1, rsiValue_H1, macdMain_H1, macdSignal_H1;

// H1 Bias
int h1_bias = 0;  // -1=bearish, 0=neutral, +1=bullish
string h1_bias_str = "NEUTRAL";
int h1_bull_count = 0;
int h1_bear_count = 0;

// Risk state
double currentRisk = 1.0;
int consecutiveWins = 0;
int consecutiveLosses = 0;
double dailyProfit = 0;
double dailyLoss = 0;
int dailyTrades = 0;
double monthlyProfit = 0;
double monthlyLoss = 0;
bool canTrade = true;
string stopReason = "";

// Position tracking
datetime lastTradeTime = 0;
datetime lastBarTime = 0;
double peakProfit = 0;
datetime positionOpenTime = 0;
bool hasRecovered = false;

// Quality scoring
double technicalQuality = 0;
double monthlyRiskMult = 1.0;
double intraRiskMult = 1.0;
int patternWinRate = 50;

// Logging
int logFileHandle = INVALID_HANDLE;
string currentLogFile = "";
datetime lastLogDate = 0;
int currentDay = 0;
int currentMonth = 0;

// Circuit breakers
bool dailyLimitReached = false;
bool monthlyLimitReached = false;
bool consecutiveLossHalt = false;

//===========================================
// SECTION 3: STRUCTS (251-400)
//===========================================

struct SessionInfo
{
   bool isSydney;
   bool isLondon;
   bool isNewYork;
   double riskMultiplier;
   string name;
};

struct QualityScore
{
   double atrStability;      // 0-20
   double priceEfficiency;   // 0-20
   double trendStrength;     // 0-20
   double spreadQuality;     // 0-20
   double h1Alignment;       // 0-20
   double total;             // 0-100
   bool passed;
};

//===========================================
// SECTION 4: INITIALIZATION (401-550)
//===========================================

int OnInit()
{
   Print("╔════════════════════════════════════════╗");
   Print("║  XAUBot Pro V3 - suriota              ║");
   Print("║  Advanced Multi-Layer Quality System  ║");
   Print("╚════════════════════════════════════════╝");

   // Check timeframe
   if(Period() != PERIOD_M15)
   {
      Alert("⚠️ WARNING: EA designed for M15 timeframe! Current: ", EnumToString(Period()));
   }

   // Initialize symbol
   if(!symbolInfo.Name(_Symbol))
   {
      Print("ERROR: Failed to set symbol");
      return INIT_FAILED;
   }

   trade.SetExpertMagicNumber(Magic);

   // Create M15 indicators
   handleEMAFast_M15 = iMA(_Symbol, PERIOD_M15, EMA_Fast_M15, 0, MODE_EMA, PRICE_CLOSE);
   handleEMASlow_M15 = iMA(_Symbol, PERIOD_M15, EMA_Slow_M15, 0, MODE_EMA, PRICE_CLOSE);
   handleADX_M15 = iADX(_Symbol, PERIOD_M15, ADX_Period);
   handleRSI_M15 = iRSI(_Symbol, PERIOD_M15, RSI_Period, PRICE_CLOSE);
   handleMACD_M15 = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
   handleATR_M15 = iATR(_Symbol, PERIOD_M15, 14);

   // Create H1 indicators
   handleEMAFast_H1 = iMA(_Symbol, PERIOD_H1, EMA_Fast_H1, 0, MODE_EMA, PRICE_CLOSE);
   handleEMASlow_H1 = iMA(_Symbol, PERIOD_H1, EMA_Slow_H1, 0, MODE_EMA, PRICE_CLOSE);
   handleRSI_H1 = iRSI(_Symbol, PERIOD_H1, RSI_Period, PRICE_CLOSE);
   handleMACD_H1 = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);

   // Check handles
   if(handleEMAFast_M15 == INVALID_HANDLE || handleEMASlow_M15 == INVALID_HANDLE ||
      handleADX_M15 == INVALID_HANDLE || handleRSI_M15 == INVALID_HANDLE ||
      handleMACD_M15 == INVALID_HANDLE || handleATR_M15 == INVALID_HANDLE ||
      handleEMAFast_H1 == INVALID_HANDLE || handleEMASlow_H1 == INVALID_HANDLE ||
      handleRSI_H1 == INVALID_HANDLE || handleMACD_H1 == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators");
      return INIT_FAILED;
   }

   // Initialize risk
   currentRisk = RiskPercent;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   currentDay = dt.day;
   currentMonth = dt.mon;

   // Create panel
   if(ShowPanel)
      CreatePanel();

   // Open log file
   if(EnableFileLog)
      OpenLogFile();

   WriteLog("XAUBot Pro V3 - Initialization Complete");
   WriteLog(StringFormat("Config: Risk=%.1f%% | ADX≥%.1f | Quality≥%.0f | MaxLot=%.2f",
      RiskPercent, ADX_Threshold, MinQualityScore, MaxLot));

   Print("✓ XAUBot Pro V3 initialized successfully");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicators
   IndicatorRelease(handleEMAFast_M15);
   IndicatorRelease(handleEMASlow_M15);
   IndicatorRelease(handleADX_M15);
   IndicatorRelease(handleRSI_M15);
   IndicatorRelease(handleMACD_M15);
   IndicatorRelease(handleATR_M15);
   IndicatorRelease(handleEMAFast_H1);
   IndicatorRelease(handleEMASlow_H1);
   IndicatorRelease(handleRSI_H1);
   IndicatorRelease(handleMACD_H1);

   if(ShowPanel)
      DeletePanel();

   if(EnableFileLog)
      CloseLogFile();

   Comment("");
   Print("XAUBot Pro V3 stopped. Reason: ", reason);
}

//===========================================
// SECTION 5: MAIN TICK HANDLER (551-650)
//===========================================

void OnTick()
{
   // New bar detection
   datetime currentBarTime = iTime(_Symbol, PERIOD_M15, 0);
   bool newBar = (currentBarTime != lastBarTime);

   // Always manage positions
   ManagePosition();

   // Update panel every 5 seconds (not every tick)
   static datetime lastPanelUpdate = 0;
   if(ShowPanel && TimeCurrent() - lastPanelUpdate >= 5)
   {
      UpdatePanel();
      lastPanelUpdate = TimeCurrent();
   }

   if(!newBar)
      return;

   lastBarTime = currentBarTime;

   // Check day/month rollover
   CheckDayRollover();

   // Update all data
   if(!UpdateAllData())
      return;

   // Calculate H1 bias
   CalculateH1Bias();

   // Calculate quality score
   CalculateQualityScore();

   // Check entry if no position
   if(CountOpenPositions() < MaxPositions)
      CheckEntry();
}

//===========================================
// SECTION 6: H1 BIAS CALCULATION (651-800)
//===========================================

void CalculateH1Bias()
{
   // Reset counters
   h1_bull_count = 0;
   h1_bear_count = 0;

   // Indicator 1: EMA Trend (50 > 200 = bull)
   if(emaFast_H1 > emaSlow_H1)
      h1_bull_count++;
   else if(emaFast_H1 < emaSlow_H1)
      h1_bear_count++;

   // Indicator 2: Price position relative to EMAs
   double close_H1 = iClose(_Symbol, PERIOD_H1, 1);
   if(close_H1 > emaFast_H1 && close_H1 > emaSlow_H1)
      h1_bull_count++;
   else if(close_H1 < emaFast_H1 && close_H1 < emaSlow_H1)
      h1_bear_count++;

   // Indicator 3: RSI
   if(rsiValue_H1 > 55.0)
      h1_bull_count++;
   else if(rsiValue_H1 < 45.0)
      h1_bear_count++;

   // Indicator 4: MACD
   if(macdMain_H1 > macdSignal_H1 && macdMain_H1 > 0)
      h1_bull_count++;
   else if(macdMain_H1 < macdSignal_H1 && macdMain_H1 < 0)
      h1_bear_count++;

   // Indicator 5: Candle structure (last 3 H1 candles)
   int bullCandles = 0;
   int bearCandles = 0;
   for(int i = 1; i <= 3; i++)
   {
      double open = iOpen(_Symbol, PERIOD_H1, i);
      double close = iClose(_Symbol, PERIOD_H1, i);
      if(close > open)
         bullCandles++;
      else if(close < open)
         bearCandles++;
   }
   if(bullCandles >= 2)
      h1_bull_count++;
   else if(bearCandles >= 2)
      h1_bear_count++;

   // Determine bias (need 3+ indicators)
   if(h1_bull_count >= 3)
   {
      h1_bias = 1;
      h1_bias_str = "▲ BULL";
   }
   else if(h1_bear_count >= 3)
   {
      h1_bias = -1;
      h1_bias_str = "▼ BEAR";
   }
   else
   {
      h1_bias = 0;
      h1_bias_str = "━ NEUTRAL";
   }
}

//===========================================
// SECTION 7: M15 SIGNAL DETECTION (801-950)
//===========================================

bool CheckM15BuySignal()
{
   // H1 must be bullish or neutral
   if(h1_bias < 0)
      return false;

   // M15 EMA trend must be bullish
   if(emaFast_M15 <= emaSlow_M15)
      return false;

   // Price near EMA50 (pullback)
   double close = iClose(_Symbol, PERIOD_M15, 1);
   double distanceToEMA = MathAbs(close - emaFast_M15) / atrValue_M15;
   if(distanceToEMA > 0.3)  // Too far from EMA
      return false;

   // RSI in acceptable range
   if(rsiValue_M15 < 40.0 || rsiValue_M15 > 70.0)
      return false;

   // ADX shows trend
   if(adxValue_M15 < ADX_Threshold)
      return false;

   // MACD bullish
   if(macdMain_M15 <= macdSignal_M15)
      return false;

   return true;
}

bool CheckM15SellSignal()
{
   // H1 must be bearish or neutral
   if(h1_bias > 0)
      return false;

   // M15 EMA trend must be bearish
   if(emaFast_M15 >= emaSlow_M15)
      return false;

   // Price near EMA50 (pullback)
   double close = iClose(_Symbol, PERIOD_M15, 1);
   double distanceToEMA = MathAbs(close - emaFast_M15) / atrValue_M15;
   if(distanceToEMA > 0.3)
      return false;

   // RSI in acceptable range
   if(rsiValue_M15 > 60.0 || rsiValue_M15 < 30.0)
      return false;

   // ADX shows trend
   if(adxValue_M15 < ADX_Threshold)
      return false;

   // MACD bearish
   if(macdMain_M15 >= macdSignal_M15)
      return false;

   return true;
}

//===========================================
// SECTION 8: QUALITY SCORING (951-1150)
//===========================================

double GetMonthlyRiskMultiplier()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int month = dt.mon;

   // Historical volatility patterns
   if(month == 2 || month == 10)      // Feb, Oct - risk-off
      return 0.6;
   else if(month == 9)                 // Sep - high activity
      return 1.1;
   else if(month == 3 || month == 5 || month == 7 || month == 11)  // Normal
      return 1.0;
   else
      return 0.8;  // Other months - cautious
}

void CalculateQualityScore()
{
   QualityScore qs;

   // Layer 1: Monthly Risk Multiplier
   monthlyRiskMult = GetMonthlyRiskMultiplier();

   // Layer 2: Technical Quality (0-100)

   // 1. ATR Stability (20 pts)
   double atr_24h_arr[];
   ArraySetAsSeries(atr_24h_arr, true);
   CopyBuffer(handleATR_M15, 0, 0, 96, atr_24h_arr);  // 96 bars = 24h
   double atr_avg = 0;
   for(int i = 0; i < 96; i++)
      atr_avg += atr_24h_arr[i];
   atr_avg /= 96;

   double atr_deviation = MathAbs(atrValue_M15 - atr_avg) / atr_avg;
   if(atr_deviation < 0.1)
      qs.atrStability = 20;
   else if(atr_deviation < 0.2)
      qs.atrStability = 15;
   else if(atr_deviation < 0.3)
      qs.atrStability = 10;
   else
      qs.atrStability = 5;

   // 2. Price Efficiency (20 pts) - EMA separation in ATR units
   double ema_separation = MathAbs(emaFast_M15 - emaSlow_M15) / atrValue_M15;
   if(ema_separation > 2.0)
      qs.priceEfficiency = 20;
   else if(ema_separation > 1.5)
      qs.priceEfficiency = 15;
   else if(ema_separation > 1.0)
      qs.priceEfficiency = 10;
   else
      qs.priceEfficiency = 5;

   // 3. Trend Strength ADX (20 pts)
   if(adxValue_M15 >= 40.0)
      qs.trendStrength = 20;
   else if(adxValue_M15 >= 30.0)
      qs.trendStrength = 15;
   else if(adxValue_M15 >= 25.0)
      qs.trendStrength = 10;
   else
      qs.trendStrength = 0;

   // 4. Spread Quality (20 pts)
   double spread = symbolInfo.Spread();
   if(spread <= 10.0)
      qs.spreadQuality = 20;
   else if(spread <= 15.0)
      qs.spreadQuality = 15;
   else if(spread <= 20.0)
      qs.spreadQuality = 10;
   else
      qs.spreadQuality = 0;

   // 5. H1-M15 Alignment (20 pts)
   bool m15_bull = (emaFast_M15 > emaSlow_M15);
   bool m15_bear = (emaFast_M15 < emaSlow_M15);

   if((h1_bias == 1 && m15_bull) || (h1_bias == -1 && m15_bear))
      qs.h1Alignment = 20;  // Perfect alignment
   else if(h1_bias == 0)
      qs.h1Alignment = 10;  // Neutral H1
   else
      qs.h1Alignment = 0;   // Conflict

   qs.total = qs.atrStability + qs.priceEfficiency + qs.trendStrength + qs.spreadQuality + qs.h1Alignment;
   qs.passed = (qs.total >= MinQualityScore);

   technicalQuality = qs.total;

   // Layer 3: Intra-Period Risk Manager
   intraRiskMult = 1.0;
   if(consecutiveLosses >= 2)
      intraRiskMult = 0.5;
   else if(consecutiveLosses == 1)
      intraRiskMult = 0.75;

   // Layer 4: Pattern Filter (rolling win rate)
   // Simplified - use consecutive wins/losses as proxy
   if(consecutiveWins >= 2)
      patternWinRate = 70;
   else if(consecutiveWins == 1)
      patternWinRate = 60;
   else if(consecutiveLosses == 0)
      patternWinRate = 50;
   else if(consecutiveLosses == 1)
      patternWinRate = 40;
   else
      patternWinRate = 30;
}

//===========================================
// SECTION 9: ENTRY FILTERS (1151-1300)
//===========================================

bool CheckAllEntryFilters()
{
   // Filter 1: Quality Check
   if(technicalQuality < MinQualityScore)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Low quality score (%.0f < %.0f)", technicalQuality, MinQualityScore), "FILTER");
      return false;
   }

   // Filter 2: Circuit Breakers
   if(!canTrade)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Circuit breaker active (%s)", stopReason), "FILTER");
      return false;
   }

   // Filter 3: Spread
   if(symbolInfo.Spread() > MaxSpread)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Spread too high (%.0f > %.0f)", symbolInfo.Spread(), MaxSpread), "FILTER");
      return false;
   }

   // Filter 4: ADX
   if(adxValue_M15 < ADX_Threshold)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Weak trend (ADX %.1f < %.1f)", adxValue_M15, ADX_Threshold), "FILTER");
      return false;
   }

   // Filter 5: Session (preferably London/NY)
   SessionInfo session = GetCurrentSession();
   // Allow all sessions but with different risk multipliers (already factored into lot calculation)

   // Filter 6: Cooldown
   if(TimeCurrent() - lastTradeTime < CooldownMinutes * 60)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Cooldown period (%d min)", CooldownMinutes), "FILTER");
      return false;
   }

   // Filter 7: Max Positions
   if(CountOpenPositions() >= MaxPositions)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Max positions reached (%d)", MaxPositions), "FILTER");
      return false;
   }

   // Filter 8: ATR Volatility Range
   if(atrValue_M15 < 5.0 || atrValue_M15 > 25.0)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: ATR out of range (%.2f not in 5-25)", atrValue_M15), "FILTER");
      return false;
   }

   // Filter 9: Time-of-Hour (skip 30 min before H1 close)
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.min >= 30)  // Between :30 and :59
   {
      if(LogFilterRejects)
         WriteLog("SKIP: Near H1 close (avoiding instability)", "FILTER");
      return false;
   }

   return true;
}

void CheckEntry()
{
   if(!CheckAllEntryFilters())
      return;

   // Check signals
   bool buySignal = CheckM15BuySignal();
   bool sellSignal = CheckM15SellSignal();

   if(buySignal)
   {
      WriteLog(StringFormat("SIGNAL: BUY | H1:%s(%d/%d) | Q:%.0f | ADX:%.1f | RSI:%.1f",
         h1_bias_str, h1_bull_count, h1_bear_count, technicalQuality, adxValue_M15, rsiValue_M15), "SIGNAL");
      OpenTrade(ORDER_TYPE_BUY);
   }
   else if(sellSignal)
   {
      WriteLog(StringFormat("SIGNAL: SELL | H1:%s(%d/%d) | Q:%.0f | ADX:%.1f | RSI:%.1f",
         h1_bias_str, h1_bull_count, h1_bear_count, technicalQuality, adxValue_M15, rsiValue_M15), "SIGNAL");
      OpenTrade(ORDER_TYPE_SELL);
   }
}

//===========================================
// SECTION 10: POSITION MANAGEMENT (1301-1500)
//===========================================

void ManagePosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!position.SelectByIndex(i))
         continue;

      if(position.Symbol() != _Symbol || position.Magic() != Magic)
         continue;

      double currentPrice = (position.Type() == POSITION_TYPE_BUY) ? symbolInfo.Bid() : symbolInfo.Ask();
      double openPrice = position.PriceOpen();
      double currentProfit = position.Profit();
      double profitDistance = (position.Type() == POSITION_TYPE_BUY) ? (currentPrice - openPrice) : (openPrice - currentPrice);
      double profitInATR = profitDistance / atrValue_M15;

      // Update peak profit
      if(currentProfit > peakProfit)
         peakProfit = currentProfit;

      // Calculate trade age
      datetime tradeAge = TimeCurrent() - positionOpenTime;
      int ageMinutes = (int)(tradeAge / 60);

      // PRIORITY 1: Hard Take Profit
      if(profitInATR >= TP_Hard_ATR)
      {
         WriteLog(StringFormat("EXIT: Hard TP reached (%.2f ATR)", profitInATR), "EXIT");
         ClosePosition(position.Ticket(), "Hard TP");
         continue;
      }

      // PRIORITY 2: Breakeven Shield
      if(peakProfit >= BE_Trigger_ATR * atrValue_M15)
      {
         if(currentProfit < BE_Lock_USD)
         {
            WriteLog(StringFormat("EXIT: Breakeven shield (peak $%.2f, now $%.2f)", peakProfit, currentProfit), "EXIT");
            ClosePosition(position.Ticket(), "BE Shield");
            continue;
         }
      }

      // PRIORITY 3: ATR Trailing Stop
      if(peakProfit >= Trail_Trigger_ATR * atrValue_M15 && ageMinutes >= MinTradeAgeMinutes)
      {
         double trailFloor = peakProfit - (Trail_Distance_ATR * atrValue_M15);
         if(currentProfit < trailFloor)
         {
            WriteLog(StringFormat("EXIT: ATR trailing (peak $%.2f, floor $%.2f)", peakProfit, trailFloor), "EXIT");
            ClosePosition(position.Ticket(), "ATR Trail");
            continue;
         }
      }

      // PRIORITY 4: ATR Hard Stop
      if(profitInATR <= -Hard_Stop_ATR && ageMinutes >= MinTradeAgeMinutes)
      {
         WriteLog(StringFormat("EXIT: ATR hard stop (%.2f ATR loss)", profitInATR), "EXIT");
         ClosePosition(position.Ticket(), "ATR Stop");
         continue;
      }

      // PRIORITY 5: Momentum Reversal
      bool ema_cross_against = false;
      if(position.Type() == POSITION_TYPE_BUY && emaFast_M15 < emaSlow_M15)
         ema_cross_against = true;
      else if(position.Type() == POSITION_TYPE_SELL && emaFast_M15 > emaSlow_M15)
         ema_cross_against = true;

      if(ema_cross_against && profitInATR < 0.3)
      {
         WriteLog(StringFormat("EXIT: EMA reversal (profit %.2f ATR < 0.3)", profitInATR), "EXIT");
         ClosePosition(position.Ticket(), "Momentum Reversal");
         continue;
      }

      // PRIORITY 6: Time-Based Exit
      if(ageMinutes >= TimeExit_Hour * 60 && currentProfit <= 0)
      {
         WriteLog(StringFormat("EXIT: Time exit (%d min, not profitable)", ageMinutes), "EXIT");
         ClosePosition(position.Ticket(), "Time Exit");
         continue;
      }

      if(ageMinutes >= AbsoluteExit_Hour * 60)
      {
         WriteLog(StringFormat("EXIT: Absolute time limit (%d min)", ageMinutes), "EXIT");
         ClosePosition(position.Ticket(), "Absolute Exit");
         continue;
      }

      // PRIORITY 7: Weekend Close
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if((dt.day_of_week == 5 && dt.hour >= 22) || (dt.day_of_week == 6 && dt.hour < 5))
      {
         if(currentProfit > 0 || profitInATR > -0.3)
         {
            WriteLog(StringFormat("EXIT: Weekend close (profit $%.2f)", currentProfit), "EXIT");
            ClosePosition(position.Ticket(), "Weekend");
            continue;
         }
      }
   }
}

void ClosePosition(ulong ticket, string reason)
{
   if(trade.PositionClose(ticket))
   {
      WriteLog(StringFormat("POSITION CLOSED: %s", reason), "TRADE");
      peakProfit = 0;
      hasRecovered = false;
   }
   else
   {
      WriteLog(StringFormat("CLOSE FAILED: %s | Error: %s", reason, trade.ResultRetcodeDescription()), "ERROR");
   }
}

//===========================================
// SECTION 11: RISK CALCULATIONS (1501-1650)
//===========================================

void OpenTrade(ENUM_ORDER_TYPE orderType)
{
   // Check daily trade limit
   if(dailyTrades >= MaxTradesPerDay)
   {
      WriteLog(StringFormat("SKIP: Daily trade limit reached (%d)", MaxTradesPerDay), "FILTER");
      return;
   }

   double price = (orderType == ORDER_TYPE_BUY) ? symbolInfo.Ask() : symbolInfo.Bid();

   // Calculate SL/TP
   double slDistance = atrValue_M15 * ATR_SL_Multiplier;
   double tpDistance = atrValue_M15 * TP_Hard_ATR;

   double sl = NormalizeDouble((orderType == ORDER_TYPE_BUY) ? (price - slDistance) : (price + slDistance), _Digits);
   double tp = NormalizeDouble((orderType == ORDER_TYPE_BUY) ? (price + tpDistance) : (price - tpDistance), _Digits);

   // Calculate lot size with all multipliers
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   SessionInfo session = GetCurrentSession();

   double effectiveRisk = currentRisk * monthlyRiskMult * intraRiskMult * session.riskMultiplier;
   double riskMoney = balance * (effectiveRisk / 100.0);

   double tickValue = symbolInfo.TickValue();
   double tickSize = symbolInfo.TickSize();
   double slInTicks = MathAbs(price - sl) / tickSize;
   double lotSize = riskMoney / (slInTicks * tickValue);

   lotSize = NormalizeDouble(lotSize, 2);
   lotSize = MathMax(MinLot, MathMin(MaxLot, lotSize));

   // Open position
   if(trade.PositionOpen(_Symbol, orderType, lotSize, price, sl, tp, "XAUBot-V3"))
   {
      string tradeType = (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL");
      Print("✓ ", tradeType, " opened: Lot=", lotSize, " Price=", price);
      WriteLog(StringFormat("TRADE OPEN: %s | Lot:%.2f | Price:%.5f | SL:%.5f | TP:%.5f | ATR:%.2f | Risk:%.2f%% | Q:%.0f",
         tradeType, lotSize, price, sl, tp, atrValue_M15, effectiveRisk, technicalQuality), "TRADE");

      lastTradeTime = TimeCurrent();
      positionOpenTime = TimeCurrent();
      peakProfit = 0;
      hasRecovered = false;
      dailyTrades++;
   }
   else
   {
      WriteLog(StringFormat("TRADE FAILED: %s | Error: %s", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), trade.ResultRetcodeDescription()), "ERROR");
   }
}

SessionInfo GetCurrentSession()
{
   SessionInfo session;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;

   // Sydney: 22:00-06:00 GMT
   session.isSydney = (hour >= 22 || hour < 6);
   // London: 07:00-16:00 GMT
   session.isLondon = (hour >= 7 && hour < 16);
   // New York: 13:00-22:00 GMT
   session.isNewYork = (hour >= 13 && hour < 22);

   if(session.isSydney)
   {
      session.name = "SYDNEY";
      session.riskMultiplier = 0.5;  // Lower liquidity
   }
   else if(session.isLondon || session.isNewYork)
   {
      session.name = session.isLondon ? "LONDON" : "NEW YORK";
      session.riskMultiplier = 1.0;  // Normal
   }
   else
   {
      session.name = "OFF-HOURS";
      session.riskMultiplier = 0.7;
   }

   return session;
}

int CountOpenPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(position.SelectByIndex(i))
      {
         if(position.Symbol() == _Symbol && position.Magic() == Magic)
            count++;
      }
   }
   return count;
}

//===========================================
// SECTION 12: PANEL UI (1651-1800)
//===========================================

void CreatePanel()
{
   string prefix = "XAU_V3_";
   color bgColor = C'20,20,30';

   // Background (larger for more info)
   ObjectCreate(0, prefix+"BG", OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_XDISTANCE, PanelOffsetX);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_YDISTANCE, PanelOffsetY);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_XSIZE, 280);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_YSIZE, 260);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_BGCOLOR, bgColor);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_COLOR, C'40,40,50');
   ObjectSetInteger(0, prefix+"BG", OBJPROP_SELECTABLE, false);

   // Text labels (24 lines)
   string labels[] = {
      "Title", "Balance", "Equity", "Profit", "Sep1",
      "Status", "H1Bias", "M15Trend", "Session", "Sep2",
      "Position", "PosProfit", "PosPeak", "Sep3",
      "Risk", "Daily", "Monthly", "Spread", "Sep4",
      "Circuit1", "Circuit2", "Circuit3", "Sep5",
      "Layers"
   };

   for(int i=0; i<ArraySize(labels); i++)
   {
      string objName = prefix + labels[i];
      ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, objName, OBJPROP_CORNER, PanelCorner);
      ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, PanelOffsetX + 5);
      ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, PanelOffsetY + 5 + (i * 11));
      ObjectSetInteger(0, objName, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, 8);
      ObjectSetString(0, objName, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false);
   }
}

void UpdatePanel()
{
   if(!ShowPanel) return;

   string prefix = "XAU_V3_";

   // Title with branding
   ObjectSetString(0, prefix+"Title", OBJPROP_TEXT, "══ XAUBot Pro V3 - suriota ══");
   ObjectSetInteger(0, prefix+"Title", OBJPROP_COLOR, clrGold);

   // Account info
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double profit = AccountInfoDouble(ACCOUNT_PROFIT);

   ObjectSetString(0, prefix+"Balance", OBJPROP_TEXT, "Balance: $"+DoubleToString(balance,2));
   ObjectSetString(0, prefix+"Equity", OBJPROP_TEXT, "Equity:  $"+DoubleToString(equity,2));

   color profitColor = (profit>=0) ? clrLimeGreen : clrRed;
   string profitSign = (profit>=0) ? "+" : "";
   ObjectSetString(0, prefix+"Profit", OBJPROP_TEXT, "Profit:  "+profitSign+"$"+DoubleToString(profit,2));
   ObjectSetInteger(0, prefix+"Profit", OBJPROP_COLOR, profitColor);

   ObjectSetString(0, prefix+"Sep1", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix+"Sep1", OBJPROP_COLOR, C'60,60,80');

   // Trading status
   string statusText = canTrade ? StringFormat("Status: ✓ READY (Q: %.0f/100)", technicalQuality) : StringFormat("Status: ⏸ %s", stopReason);
   color statusColor = canTrade ? clrLimeGreen : clrRed;
   ObjectSetString(0, prefix+"Status", OBJPROP_TEXT, statusText);
   ObjectSetInteger(0, prefix+"Status", OBJPROP_COLOR, statusColor);

   // H1 Bias
   color biasColor = (h1_bias > 0) ? clrLimeGreen : (h1_bias < 0) ? clrRed : clrGray;
   ObjectSetString(0, prefix+"H1Bias", OBJPROP_TEXT, StringFormat("H1 Bias: %s (%d/%d)", h1_bias_str, h1_bull_count, h1_bear_count));
   ObjectSetInteger(0, prefix+"H1Bias", OBJPROP_COLOR, biasColor);

   // M15 Trend
   string m15_dir = (emaFast_M15 > emaSlow_M15) ? "▲ BULL" : "▼ BEAR";
   color m15Color = (emaFast_M15 > emaSlow_M15) ? clrLimeGreen : clrRed;
   ObjectSetString(0, prefix+"M15Trend", OBJPROP_TEXT, StringFormat("M15: %s | ADX: %.1f", m15_dir, adxValue_M15));
   ObjectSetInteger(0, prefix+"M15Trend", OBJPROP_COLOR, m15Color);

   // Session
   SessionInfo session = GetCurrentSession();
   ObjectSetString(0, prefix+"Session", OBJPROP_TEXT, StringFormat("Session: %s (%.1fx)", session.name, session.riskMultiplier));

   ObjectSetString(0, prefix+"Sep2", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix+"Sep2", OBJPROP_COLOR, C'60,60,80');

   // Position info
   if(position.Select(_Symbol))
   {
      string posType = (position.Type()==POSITION_TYPE_BUY) ? "BUY" : "SELL";
      color posColor = (position.Type()==POSITION_TYPE_BUY) ? clrDodgerBlue : clrOrangeRed;

      ObjectSetString(0, prefix+"Position", OBJPROP_TEXT, StringFormat("● %s | %.2f lot", posType, position.Volume()));
      ObjectSetInteger(0, prefix+"Position", OBJPROP_COLOR, posColor);

      double posProfit = position.Profit();
      int ageMin = (int)((TimeCurrent() - positionOpenTime) / 60);
      color pColor = (posProfit>=0) ? clrLimeGreen : clrRed;
      ObjectSetString(0, prefix+"PosProfit", OBJPROP_TEXT, StringFormat("P/L: $%.2f | Age: %dmin", posProfit, ageMin));
      ObjectSetInteger(0, prefix+"PosProfit", OBJPROP_COLOR, pColor);

      ObjectSetString(0, prefix+"PosPeak", OBJPROP_TEXT, StringFormat("Peak: $%.2f | ATR: $%.2f", peakProfit, atrValue_M15));
   }
   else
   {
      ObjectSetString(0, prefix+"Position", OBJPROP_TEXT, "● No Position");
      ObjectSetInteger(0, prefix+"Position", OBJPROP_COLOR, clrGray);
      ObjectSetString(0, prefix+"PosProfit", OBJPROP_TEXT, "");
      ObjectSetString(0, prefix+"PosPeak", OBJPROP_TEXT, "");
   }

   ObjectSetString(0, prefix+"Sep3", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix+"Sep3", OBJPROP_COLOR, C'60,60,80');

   // Risk info
   double effectiveRisk = currentRisk * monthlyRiskMult * intraRiskMult;
   string riskMode = (consecutiveLosses > 0) ? "Recovery" : "Normal";
   ObjectSetString(0, prefix+"Risk", OBJPROP_TEXT, StringFormat("Risk: %.1f%% (%s)", effectiveRisk, riskMode));

   // Daily stats
   double dailyLimit = balance * (DailyLossLimit / 100.0);
   ObjectSetString(0, prefix+"Daily", OBJPROP_TEXT, StringFormat("Daily: $%.0f / -$%.0f (%.0f%%)", dailyProfit-dailyLoss, dailyLimit, DailyLossLimit));

   // Monthly stats
   double monthlyLimit = balance * (MonthlyLossLimit / 100.0);
   ObjectSetString(0, prefix+"Monthly", OBJPROP_TEXT, StringFormat("Month: $%.0f / -$%.0f (%.0f%%)", monthlyProfit-monthlyLoss, monthlyLimit, MonthlyLossLimit));

   // Spread & Trades
   ObjectSetString(0, prefix+"Spread", OBJPROP_TEXT, StringFormat("Spread: %.0f/%.0f | Trades: %d/%d", symbolInfo.Spread(), MaxSpread, dailyTrades, MaxTradesPerDay));

   ObjectSetString(0, prefix+"Sep4", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix+"Sep4", OBJPROP_COLOR, C'60,60,80');

   // Circuit breakers
   string cb1 = dailyLimitReached ? "[HALT]" : "[  OK  ]";
   string cb2 = monthlyLimitReached ? "[HALT]" : "[  OK  ]";
   string cb3 = consecutiveLossHalt ? "[HALT]" : "[  OK  ]";
   color cb1_col = dailyLimitReached ? clrRed : clrLimeGreen;
   color cb2_col = monthlyLimitReached ? clrRed : clrLimeGreen;
   color cb3_col = consecutiveLossHalt ? clrRed : clrLimeGreen;

   ObjectSetString(0, prefix+"Circuit1", OBJPROP_TEXT, StringFormat("Daily: %s", cb1));
   ObjectSetInteger(0, prefix+"Circuit1", OBJPROP_COLOR, cb1_col);

   ObjectSetString(0, prefix+"Circuit2", OBJPROP_TEXT, StringFormat("Month: %s", cb2));
   ObjectSetInteger(0, prefix+"Circuit2", OBJPROP_COLOR, cb2_col);

   ObjectSetString(0, prefix+"Circuit3", OBJPROP_TEXT, StringFormat("Losses: %s (C:%d)", cb3, consecutiveLosses));
   ObjectSetInteger(0, prefix+"Circuit3", OBJPROP_COLOR, cb3_col);

   ObjectSetString(0, prefix+"Sep5", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix+"Sep5", OBJPROP_COLOR, C'60,60,80');

   // Layer summary
   ObjectSetString(0, prefix+"Layers", OBJPROP_TEXT, StringFormat("L1:%.1f L2:%.0f L3:%.1f L4:%d%%", monthlyRiskMult, technicalQuality, intraRiskMult, patternWinRate));
}

void DeletePanel()
{
   string prefix = "XAU_V3_";
   ObjectDelete(0, prefix+"BG");
   string labels[] = {
      "Title", "Balance", "Equity", "Profit", "Sep1",
      "Status", "H1Bias", "M15Trend", "Session", "Sep2",
      "Position", "PosProfit", "PosPeak", "Sep3",
      "Risk", "Daily", "Monthly", "Spread", "Sep4",
      "Circuit1", "Circuit2", "Circuit3", "Sep5",
      "Layers"
   };
   for(int i=0; i<ArraySize(labels); i++)
      ObjectDelete(0, prefix+labels[i]);
}

//===========================================
// SECTION 13: UTILITIES (1801-1900)
//===========================================

bool UpdateAllData()
{
   // Update M15 indicators
   double arr[];
   ArraySetAsSeries(arr, true);

   if(CopyBuffer(handleEMAFast_M15, 0, 0, 2, arr) <= 0) return false;
   emaFast_M15 = arr[1];

   if(CopyBuffer(handleEMASlow_M15, 0, 0, 2, arr) <= 0) return false;
   emaSlow_M15 = arr[1];

   if(CopyBuffer(handleADX_M15, 0, 0, 2, arr) <= 0) return false;
   adxValue_M15 = arr[1];

   if(CopyBuffer(handleRSI_M15, 0, 0, 2, arr) <= 0) return false;
   rsiValue_M15 = arr[1];

   if(CopyBuffer(handleMACD_M15, 0, 0, 2, arr) <= 0) return false;
   macdMain_M15 = arr[1];

   if(CopyBuffer(handleMACD_M15, 1, 0, 2, arr) <= 0) return false;
   macdSignal_M15 = arr[1];

   if(CopyBuffer(handleATR_M15, 0, 0, 2, arr) <= 0) return false;
   atrValue_M15 = arr[1];

   // Update H1 indicators
   if(CopyBuffer(handleEMAFast_H1, 0, 0, 2, arr) <= 0) return false;
   emaFast_H1 = arr[1];

   if(CopyBuffer(handleEMASlow_H1, 0, 0, 2, arr) <= 0) return false;
   emaSlow_H1 = arr[1];

   if(CopyBuffer(handleRSI_H1, 0, 0, 2, arr) <= 0) return false;
   rsiValue_H1 = arr[1];

   if(CopyBuffer(handleMACD_H1, 0, 0, 2, arr) <= 0) return false;
   macdMain_H1 = arr[1];

   if(CopyBuffer(handleMACD_H1, 1, 0, 2, arr) <= 0) return false;
   macdSignal_H1 = arr[1];

   return true;
}

void CheckDayRollover()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   // Day rollover
   if(dt.day != currentDay)
   {
      WriteLog(StringFormat("DAY ROLLOVER | Daily: $%.2f | Trades: %d", dailyProfit-dailyLoss, dailyTrades), "SYSTEM");
      dailyProfit = 0;
      dailyLoss = 0;
      dailyTrades = 0;
      dailyLimitReached = false;
      currentDay = dt.day;

      // Reset log file
      if(EnableFileLog)
      {
         CloseLogFile();
         OpenLogFile();
      }
   }

   // Month rollover
   if(dt.mon != currentMonth)
   {
      WriteLog(StringFormat("MONTH ROLLOVER | Monthly: $%.2f", monthlyProfit-monthlyLoss), "SYSTEM");
      monthlyProfit = 0;
      monthlyLoss = 0;
      monthlyLimitReached = false;
      currentMonth = dt.mon;
   }
}

void OnTradeTransaction(const MqlTradeTransaction& trans, const MqlTradeRequest& request, const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong dealTicket = trans.deal;
      if(dealTicket > 0 && HistoryDealSelect(dealTicket))
      {
         long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
         if(dealMagic == Magic)
         {
            double dealProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
            long dealEntry = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);

            if(dealEntry == DEAL_ENTRY_OUT)
            {
               bool isWin = (dealProfit > 0);

               if(isWin)
               {
                  consecutiveWins++;
                  consecutiveLosses = 0;
                  consecutiveLossHalt = false;
                  dailyProfit += dealProfit;
                  monthlyProfit += dealProfit;

                  if(consecutiveWins >= 2)
                     currentRisk = RiskPercent;

                  WriteLog(StringFormat("WIN | Profit: $%.2f | Consecutive: %d | Risk: %.1f%%", dealProfit, consecutiveWins, currentRisk), "WIN");
               }
               else
               {
                  consecutiveLosses++;
                  consecutiveWins = 0;
                  dailyLoss += MathAbs(dealProfit);
                  monthlyLoss += MathAbs(dealProfit);
                  currentRisk = MinRiskPercent;

                  // Check circuit breakers
                  double balance = AccountInfoDouble(ACCOUNT_BALANCE);
                  if(dailyLoss >= balance * (DailyLossLimit / 100.0))
                  {
                     dailyLimitReached = true;
                     canTrade = false;
                     stopReason = "Daily Loss Limit";
                     WriteLog(StringFormat("CIRCUIT BREAKER: Daily loss limit reached ($%.2f)", dailyLoss), "ALERT");
                  }

                  if(monthlyLoss >= balance * (MonthlyLossLimit / 100.0))
                  {
                     monthlyLimitReached = true;
                     canTrade = false;
                     stopReason = "Monthly Loss Limit";
                     WriteLog(StringFormat("CIRCUIT BREAKER: Monthly loss limit reached ($%.2f)", monthlyLoss), "ALERT");
                  }

                  if(consecutiveLosses >= MaxConsecutiveLosses)
                  {
                     consecutiveLossHalt = true;
                     canTrade = false;
                     stopReason = StringFormat("%d Consecutive Losses", MaxConsecutiveLosses);
                     WriteLog(StringFormat("CIRCUIT BREAKER: %d consecutive losses - trading halted", MaxConsecutiveLosses), "ALERT");
                  }

                  WriteLog(StringFormat("LOSS | Loss: $%.2f | Consecutive: %d | Risk: %.1f%% | Daily: $%.2f",
                     MathAbs(dealProfit), consecutiveLosses, currentRisk, dailyLoss), "LOSS");
               }

               // Re-enable trading if recovered from consecutive losses (after 1 win)
               if(isWin && consecutiveLossHalt)
               {
                  consecutiveLossHalt = false;
                  canTrade = true;
                  WriteLog("RECOVERY: Consecutive loss halt cleared after win", "SYSTEM");
               }
            }
         }
      }
   }
}

// Logging functions
bool OpenLogFile()
{
   if(!EnableFileLog) return true;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   string filename = StringFormat("XAUBot_V3_%04d-%02d-%02d.log", dt.year, dt.mon, dt.day);
   currentLogFile = filename;
   lastLogDate = TimeCurrent();

   logFileHandle = FileOpen(filename, FILE_WRITE|FILE_READ|FILE_TXT|FILE_ANSI);
   if(logFileHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to open log file: ", filename);
      return false;
   }

   FileSeek(logFileHandle, 0, SEEK_END);
   string marker = StringFormat("\n╔════════════════════════════════════════╗\n║  XAUBot Pro V3 - SESSION START        ║\n║  %s  ║\n╚════════════════════════════════════════╝\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
   FileWriteString(logFileHandle, marker);
   FileFlush(logFileHandle);

   return true;
}

void WriteLog(string message, string level="INFO")
{
   if(!EnableFileLog || logFileHandle == INVALID_HANDLE) return;

   string logLine = StringFormat("[%s] [%s] %s\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), level, message);
   FileWriteString(logFileHandle, logLine);
   FileFlush(logFileHandle);
}

void CloseLogFile()
{
   if(logFileHandle != INVALID_HANDLE)
   {
      string marker = StringFormat("\n[%s] ══════════ SESSION END ══════════\n\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
      FileWriteString(logFileHandle, marker);
      FileFlush(logFileHandle);
      FileClose(logFileHandle);
      logFileHandle = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
