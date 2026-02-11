//+------------------------------------------------------------------+
//| XAUBot_Pro_Lite.mq5                                              |
//| Optimized for M15 Gold Trading - High Win Rate Focus            |
//| Based on XAUBot AI Python + TOL LANGIT best practices           |
//+------------------------------------------------------------------+
#property copyright "XAUBot Pro - Gifari Kemal"
#property link      "https://github.com/GifariKemal/xaubot-ai"
#property version   "1.00"
#property description "Conservative M15 Gold EA - 70%+ Win Rate Target"
#property description "NO Martingale | Smart Filtering | Adaptive Risk"

//--- Include files
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input Parameters
//=== RISK SETTINGS ===
input group "=== Risk Management ==="
input double   RiskPercent = 1.0;           // Risk % per trade (base)
input double   MinRiskPercent = 0.5;        // Min risk after loss
input double   MaxLot = 0.2;                // Max lot size (for $500 account)
input double   MinLot = 0.01;               // Min lot size
input double   ATR_SL_Multiplier = 1.0;     // ATR multiplier for SL
input double   ATR_TP_Multiplier = 1.5;     // ATR multiplier for TP (1:1.5 RR)

//=== ENTRY FILTERS ===
input group "=== Entry Filters ==="
input int      EMA_Fast = 50;               // Fast EMA period
input int      EMA_Slow = 200;              // Slow EMA period
input int      ADX_Period = 14;             // ADX period for trend strength
input double   ADX_Threshold = 25.0;        // Min ADX for strong trend
input int      RSI_Period = 14;             // RSI period
input double   RSI_OB = 70.0;               // RSI overbought level
input double   RSI_OS = 30.0;               // RSI oversold level
input double   MaxSpread = 20.0;            // Max spread in points (2 pips)
input double   MaxATRMultiple = 2.0;        // Max ATR spike (vs 20-period avg)

//=== EXIT SETTINGS ===
input group "=== Exit Management ==="
input bool     UseBreakeven = true;         // Enable breakeven
input double   BE_Trigger_ATR = 0.5;        // Breakeven trigger (ATR multiple)
input double   BE_Lock_Pips = 5.0;          // Pips to lock at breakeven
input bool     UsePartialClose = true;      // Enable partial close
input double   Partial_Close_ATR = 1.0;     // Partial close at X ATR profit
input double   Partial_Close_Percent = 50.0;// % to close (50% = half position)
input bool     UseTrailing = true;          // Enable trailing stop
input double   Trail_Start_ATR = 0.8;       // Start trailing at X ATR profit
input double   Trail_Distance_ATR = 0.3;    // Trail distance (ATR multiple)
input int      MaxHoldBars = 16;            // Max hold time (bars) - 4h on M15

//=== TIME FILTERS ===
input group "=== Time & Session Filters ==="
input bool     UseTradingHours = true;      // Enable time filter
input int      StartHour = 8;               // Start trading hour (GMT)
input int      EndHour = 20;                // End trading hour (GMT)
input bool     AvoidMondayOpen = true;      // Skip Monday 00:00-06:00
input bool     AvoidFridayClose = true;     // Skip Friday after 18:00
input bool     TradeAsianSession = false;   // Trade Asian session (23:00-08:00)
input bool     TradeLondonSession = true;   // Trade London session (08:00-16:00)
input bool     TradeNYSession = true;       // Trade NY session (13:00-22:00)

//=== OTHER SETTINGS ===
input group "=== Other Settings ==="
input int      Magic = 202602;              // Magic number
input string   TradeComment = "XAUBot_Pro"; // Trade comment
input bool     ShowPanel = true;            // Show info panel on chart
input ENUM_BASE_CORNER PanelCorner = CORNER_RIGHT_LOWER; // Panel position
input int      PanelOffsetX = 10;          // Panel X offset from corner
input int      PanelOffsetY = 10;          // Panel Y offset from corner
input bool     EnableDetailedLogs = true;   // Enable detailed logs in Experts tab
input bool     EnableFileLogging = true;    // Save logs to file
input bool     DebugMode = false;           // Print debug info

//--- Global Variables
CTrade trade;
CPositionInfo position;
CSymbolInfo symbolInfo;

// Indicator handles
int handleEMAFast, handleEMASlow, handleADX, handleRSI, handleMACD, handleATR;
int handleATRLong; // For volatility spike detection

// Trading state
double currentRisk = RiskPercent;
int consecutiveWins = 0;
int consecutiveLosses = 0;
datetime lastTradeTime = 0;
datetime lastBarTime = 0;

// Market data
double emaFast, emaSlow, adxValue, rsiValue, macdMain, macdSignal, atrValue, atrAvg;
double currentSpread;

// Position tracking
bool isBreakevenSet = false;
bool isPartialClosed = false;
datetime positionOpenTime = 0;
double positionOpenPrice = 0;

// File logging
int logFileHandle = INVALID_HANDLE;
string currentLogFile = "";
datetime lastLogDate = 0;

//+------------------------------------------------------------------+
//| Get bar shift by time (replacement for MQL4's iBarShift)        |
//+------------------------------------------------------------------+
int GetBarShift(string symbol, ENUM_TIMEFRAMES timeframe, datetime time)
{
   if(time < 0) return -1;

   datetime timeArray[];
   ArraySetAsSeries(timeArray, true);

   int copied = CopyTime(symbol, timeframe, 0, Bars(symbol, timeframe), timeArray);
   if(copied <= 0) return -1;

   // Find the bar with this time
   for(int i = 0; i < copied; i++)
   {
      if(timeArray[i] <= time)
         return i;
   }

   return -1;
}

//+------------------------------------------------------------------+
//| Open log file for writing                                        |
//+------------------------------------------------------------------+
bool OpenLogFile()
{
   if(!EnableFileLogging) return true;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   // Create filename with date: XAUBot_YYYY-MM-DD.log
   string filename = StringFormat("XAUBot_%04d-%02d-%02d.log", dt.year, dt.mon, dt.day);

   currentLogFile = filename;
   lastLogDate = TimeCurrent();

   // Open or create log file (append mode)
   logFileHandle = FileOpen(filename, FILE_WRITE|FILE_READ|FILE_TXT|FILE_ANSI);

   if(logFileHandle == INVALID_HANDLE)
   {
      Print("❌ ERROR: Failed to open log file: ", filename, " Error: ", GetLastError());
      return false;
   }

   // Move to end of file for appending
   FileSeek(logFileHandle, 0, SEEK_END);

   // Write session start marker
   string startMarker = "\n" + StringFormat("╔═══════════════════════════════════════════════════════════════╗\n");
   startMarker += StringFormat("║  XAUBot Pro Lite v1.00 - Session Started                    ║\n");
   startMarker += StringFormat("║  DateTime: %-50s║\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
   startMarker += StringFormat("║  Account: %-51I64d║\n", AccountInfoInteger(ACCOUNT_LOGIN));
   startMarker += StringFormat("║  Symbol: %-52s║\n", _Symbol);
   startMarker += StringFormat("╚═══════════════════════════════════════════════════════════════╝\n");

   FileWriteString(logFileHandle, startMarker);
   FileFlush(logFileHandle);

   Print("✓ Log file opened: ", filename);
   return true;
}

//+------------------------------------------------------------------+
//| Close log file                                                   |
//+------------------------------------------------------------------+
void CloseLogFile()
{
   if(logFileHandle != INVALID_HANDLE)
   {
      // Write session end marker
      string endMarker = StringFormat("\n[%s] ═══ Session Ended ═══\n\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
      FileWriteString(logFileHandle, endMarker);
      FileFlush(logFileHandle);
      FileClose(logFileHandle);
      logFileHandle = INVALID_HANDLE;

      Print("✓ Log file closed: ", currentLogFile);
   }
}

//+------------------------------------------------------------------+
//| Write to log file                                                |
//+------------------------------------------------------------------+
void WriteLog(string message, string level = "INFO")
{
   if(!EnableFileLogging || logFileHandle == INVALID_HANDLE) return;

   // Check if we need to rotate log (new day)
   MqlDateTime currentDT, lastDT;
   TimeToStruct(TimeCurrent(), currentDT);
   TimeToStruct(lastLogDate, lastDT);

   if(currentDT.day != lastDT.day)
   {
      CloseLogFile();
      OpenLogFile();
   }

   // Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] Message
   string logLine = StringFormat("[%s] [%-5s] %s\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), level, message);

   FileWriteString(logFileHandle, logLine);
   FileFlush(logFileHandle); // Force write to disk
}

//+------------------------------------------------------------------+
//| Write trade event to log with details                           |
//+------------------------------------------------------------------+
void WriteTradeLog(string event, ENUM_ORDER_TYPE orderType, double lot, double price, double sl, double tp, double profit = 0)
{
   if(!EnableFileLogging) return;

   string typeStr = (orderType == ORDER_TYPE_BUY) ? "BUY" : "SELL";
   string message = StringFormat("%s | Type: %s | Lot: %.2f | Price: %." + IntegerToString(_Digits) + "f | SL: %." + IntegerToString(_Digits) + "f | TP: %." + IntegerToString(_Digits) + "f", event, typeStr, lot, price, sl, tp);

   if(profit != 0)
      message += StringFormat(" | Profit: $%.2f", profit);

   WriteLog(message, "TRADE");
}

//+------------------------------------------------------------------+
//| Write filter rejection to log                                   |
//+------------------------------------------------------------------+
void WriteFilterLog(string filterName, string reason)
{
   if(!EnableFileLogging || !DebugMode) return;

   string message = StringFormat("Filter Rejected: %s | Reason: %s", filterName, reason);
   WriteLog(message, "FILTER");
}

//+------------------------------------------------------------------+
//| Create graphical panel on chart                                  |
//+------------------------------------------------------------------+
void CreatePanel()
{
   string prefix = "XAUBot_";
   int fontSize = 8;
   string fontName = "Consolas";
   color bgColor = C'20,20,30';      // Dark background
   color textColor = clrWhite;

   // Create background rectangle
   string bgName = prefix + "BG";
   ObjectCreate(0, bgName, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, bgName, OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, bgName, OBJPROP_XDISTANCE, PanelOffsetX);
   ObjectSetInteger(0, bgName, OBJPROP_YDISTANCE, PanelOffsetY);
   ObjectSetInteger(0, bgName, OBJPROP_XSIZE, 280);
   ObjectSetInteger(0, bgName, OBJPROP_YSIZE, 200);
   ObjectSetInteger(0, bgName, OBJPROP_BGCOLOR, bgColor);
   ObjectSetInteger(0, bgName, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, bgName, OBJPROP_COLOR, C'40,40,50');
   ObjectSetInteger(0, bgName, OBJPROP_BACK, false);
   ObjectSetInteger(0, bgName, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, bgName, OBJPROP_HIDDEN, true);

   // Create text labels
   string labels[] = {
      "Title", "Balance", "Equity", "Profit", "Separator1",
      "Status", "Trend", "ADX", "RSI", "Separator2",
      "Position", "PosDtl1", "PosDtl2", "PosDtl3", "Separator3",
      "Risk", "Spread", "ATR", "WinLoss"
   };

   for(int i = 0; i < ArraySize(labels); i++)
   {
      string objName = prefix + labels[i];
      ObjectCreate(0, objName, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, objName, OBJPROP_CORNER, PanelCorner);
      ObjectSetInteger(0, objName, OBJPROP_XDISTANCE, PanelOffsetX + 5);
      ObjectSetInteger(0, objName, OBJPROP_YDISTANCE, PanelOffsetY + 5 + (i * 10));
      ObjectSetInteger(0, objName, OBJPROP_COLOR, textColor);
      ObjectSetInteger(0, objName, OBJPROP_FONTSIZE, fontSize);
      ObjectSetString(0, objName, OBJPROP_FONT, fontName);
      ObjectSetInteger(0, objName, OBJPROP_ANCHOR, ANCHOR_LEFT_UPPER);
      ObjectSetInteger(0, objName, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, objName, OBJPROP_HIDDEN, true);
   }

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Delete panel objects                                             |
//+------------------------------------------------------------------+
void DeletePanel()
{
   string prefix = "XAUBot_";
   ObjectDelete(0, prefix + "BG");

   string labels[] = {
      "Title", "Balance", "Equity", "Profit", "Separator1",
      "Status", "Trend", "ADX", "RSI", "Separator2",
      "Position", "PosDtl1", "PosDtl2", "PosDtl3", "Separator3",
      "Risk", "Spread", "ATR", "WinLoss"
   };

   for(int i = 0; i < ArraySize(labels); i++)
      ObjectDelete(0, prefix + labels[i]);

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Open log file first
   if(!OpenLogFile())
   {
      Print("⚠️ WARNING: Failed to open log file, continuing without file logging");
   }

   // Detailed startup logs
   if(EnableDetailedLogs)
   {
      Print("╔═══════════════════════════════════════════════════╗");
      Print("║     XAUBot Pro Lite v1.00 - Initialization      ║");
      Print("╚═══════════════════════════════════════════════════╝");
      Print("📅 Startup Time: ", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
      Print("💰 Account: ", AccountInfoInteger(ACCOUNT_LOGIN), " | Server: ", AccountInfoString(ACCOUNT_SERVER));
      Print("📊 Symbol: ", _Symbol, " | Timeframe: M15");
      Print("───────────────────────────────────────────────────");

      WriteLog("═══ XAUBot Pro Lite Initialization Started ═══");
      WriteLog(StringFormat("Account: %I64d | Server: %s | Symbol: %s | TF: M15", AccountInfoInteger(ACCOUNT_LOGIN), AccountInfoString(ACCOUNT_SERVER), _Symbol));
   }

   // Set symbol
   if(!symbolInfo.Name(_Symbol))
   {
      Print("❌ ERROR: Failed to set symbol info");
      return INIT_FAILED;
   }

   if(EnableDetailedLogs)
   {
      Print("✓ Symbol Info:");
      Print("  - Digits: ", _Digits);
      Print("  - Point: ", _Point);
      Print("  - Spread: ", symbolInfo.Spread(), " points");
      Print("  - Min Lot: ", symbolInfo.LotsMin());
      Print("  - Max Lot: ", symbolInfo.LotsMax());
      Print("  - Lot Step: ", symbolInfo.LotsStep());
   }

   // Set magic number
   trade.SetExpertMagicNumber(Magic);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);

   if(EnableDetailedLogs)
   {
      Print("✓ Trade Settings:");
      Print("  - Magic Number: ", Magic);
      Print("  - Max Deviation: 10 points");
      Print("  - Fill Type: FOK (Fill or Kill)");
   }

   // Initialize indicators
   handleEMAFast = iMA(_Symbol, PERIOD_CURRENT, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handleEMASlow = iMA(_Symbol, PERIOD_CURRENT, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   handleADX = iADX(_Symbol, PERIOD_CURRENT, ADX_Period);
   handleRSI = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   handleMACD = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   handleATR = iATR(_Symbol, PERIOD_CURRENT, 14);
   handleATRLong = iATR(_Symbol, PERIOD_CURRENT, 20);

   // Check handles
   if(handleEMAFast == INVALID_HANDLE || handleEMASlow == INVALID_HANDLE ||
      handleADX == INVALID_HANDLE || handleRSI == INVALID_HANDLE ||
      handleMACD == INVALID_HANDLE || handleATR == INVALID_HANDLE ||
      handleATRLong == INVALID_HANDLE)
   {
      Print("❌ ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   if(EnableDetailedLogs)
   {
      Print("✓ Indicators Loaded:");
      Print("  - EMA Fast: ", EMA_Fast, " | EMA Slow: ", EMA_Slow);
      Print("  - ADX: ", ADX_Period, " (threshold: ", ADX_Threshold, ")");
      Print("  - RSI: ", RSI_Period, " (range: ", RSI_OS, "-", RSI_OB, ")");
      Print("  - MACD: 12/26/9");
      Print("  - ATR: 14 (SL: ", ATR_SL_Multiplier, "x, TP: ", ATR_TP_Multiplier, "x)");
      Print("───────────────────────────────────────────────────");
      Print("✓ Risk Management:");
      Print("  - Base Risk: ", RiskPercent, "% per trade");
      Print("  - Min Risk (after loss): ", MinRiskPercent, "%");
      Print("  - Max Lot: ", MaxLot, " | Min Lot: ", MinLot);
      Print("  - Risk:Reward Ratio: 1:", ATR_TP_Multiplier / ATR_SL_Multiplier);
      Print("───────────────────────────────────────────────────");
      Print("✓ Entry Filters:");
      Print("  - Max Spread: ", MaxSpread, " points (", MaxSpread/10, " pips)");
      Print("  - ADX Threshold: ", ADX_Threshold, " (strong trend)");
      Print("  - ATR Spike Limit: ", MaxATRMultiple, "x average");
      Print("  - Cooldown: 15 minutes between trades");
      Print("───────────────────────────────────────────────────");
      Print("✓ Exit Management:");
      if(UseBreakeven) Print("  - Breakeven: ", BE_Trigger_ATR, " ATR (lock: ", BE_Lock_Pips, " pips)");
      if(UsePartialClose) Print("  - Partial Close: ", Partial_Close_Percent, "% at ", Partial_Close_ATR, " ATR");
      if(UseTrailing) Print("  - Trailing: Start at ", Trail_Start_ATR, " ATR, distance ", Trail_Distance_ATR, " ATR");
      Print("  - Max Hold Time: ", MaxHoldBars, " bars (", MaxHoldBars * 15, " minutes)");
      Print("───────────────────────────────────────────────────");
      Print("✓ Time Filters:");
      if(UseTradingHours) Print("  - Trading Hours: ", StartHour, ":00 - ", EndHour, ":00 GMT");
      if(TradeLondonSession) Print("  - London Session: ENABLED");
      if(TradeNYSession) Print("  - NY Session: ENABLED");
      if(!TradeAsianSession) Print("  - Asian Session: DISABLED");
      if(AvoidMondayOpen) Print("  - Avoid Monday 00:00-06:00: YES");
      if(AvoidFridayClose) Print("  - Avoid Friday 18:00+: YES");
      Print("═══════════════════════════════════════════════════");
      Print("🎯 TARGET: 70%+ Win Rate | Conservative Entry");
      Print("🚀 STATUS: READY TO TRADE");
      Print("═══════════════════════════════════════════════════");
   }

   // Create graphical panel
   if(ShowPanel)
      CreatePanel();

   // Log final initialization status
   WriteLog("✓ Initialization completed successfully", "INFO");
   WriteLog(StringFormat("Configuration: Risk=%.1f%% | TP=%.1fx ATR | SL=%.1fx ATR | MaxHold=%d bars", RiskPercent, ATR_TP_Multiplier, ATR_SL_Multiplier, MaxHoldBars));
   WriteLog(StringFormat("Filters: Spread<=%.0f | ADX>=%.0f | ATRSpike<=%.1fx | Cooldown=15min", MaxSpread, ADX_Threshold, MaxATRMultiple));

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(EnableDetailedLogs)
   {
      Print("═══════════════════════════════════════════════════");
      Print("🛑 XAUBot Pro Lite Stopped");
      Print("Reason: ", GetUninitReasonText(reason));
      Print("Final Balance: $", AccountInfoDouble(ACCOUNT_BALANCE));
      Print("Final Equity: $", AccountInfoDouble(ACCOUNT_EQUITY));
      Print("Consecutive Wins: ", consecutiveWins, " | Losses: ", consecutiveLosses);
      Print("═══════════════════════════════════════════════════");

      WriteLog("═══ XAUBot Pro Lite Shutdown ═══", "INFO");
      WriteLog(StringFormat("Reason: %s", GetUninitReasonText(reason)));
      WriteLog(StringFormat("Final Stats - Balance: $%.2f | Equity: $%.2f | Profit: $%.2f", AccountInfoDouble(ACCOUNT_BALANCE), AccountInfoDouble(ACCOUNT_EQUITY), AccountInfoDouble(ACCOUNT_PROFIT)));
      WriteLog(StringFormat("Performance - Consecutive Wins: %d | Losses: %d", consecutiveWins, consecutiveLosses));
   }

   // Release indicator handles
   IndicatorRelease(handleEMAFast);
   IndicatorRelease(handleEMASlow);
   IndicatorRelease(handleADX);
   IndicatorRelease(handleRSI);
   IndicatorRelease(handleMACD);
   IndicatorRelease(handleATR);
   IndicatorRelease(handleATRLong);

   // Delete graphical panel
   if(ShowPanel)
      DeletePanel();

   // Close log file
   CloseLogFile();

   Comment("");
}

//+------------------------------------------------------------------+
//| Get readable uninit reason                                       |
//+------------------------------------------------------------------+
string GetUninitReasonText(int reason)
{
   switch(reason)
   {
      case REASON_PROGRAM: return "Program terminated by user";
      case REASON_REMOVE: return "EA removed from chart";
      case REASON_RECOMPILE: return "EA recompiled";
      case REASON_CHARTCHANGE: return "Symbol/timeframe changed";
      case REASON_CHARTCLOSE: return "Chart closed";
      case REASON_PARAMETERS: return "Input parameters changed";
      case REASON_ACCOUNT: return "Account changed";
      case REASON_TEMPLATE: return "Template changed";
      case REASON_INITFAILED: return "Initialization failed";
      case REASON_CLOSE: return "Terminal closed";
      default: return "Unknown reason (" + IntegerToString(reason) + ")";
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar (M15 strategy)
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   bool newBar = (currentBarTime != lastBarTime);

   if(!newBar)
   {
      // Still manage open positions on every tick
      ManageOpenPosition();
      return;
   }

   lastBarTime = currentBarTime;

   // Update market data
   if(!UpdateMarketData())
   {
      if(DebugMode) Print("Failed to update market data");
      return;
   }

   // Manage existing positions
   ManageOpenPosition();

   // Check if we can open new trade
   if(position.Select(_Symbol))
   {
      // Already have position, skip entry logic
      return;
   }

   // Entry logic - only on new bar
   CheckForEntry();

   // Update info panel
   if(ShowPanel) UpdateInfoPanel();
}

//+------------------------------------------------------------------+
//| Update market data from indicators                               |
//+------------------------------------------------------------------+
bool UpdateMarketData()
{
   double emaFastArr[], emaSlowArr[], adxArr[], rsiArr[], macdMainArr[], macdSignalArr[];
   double atrArr[], atrLongArr[];

   ArraySetAsSeries(emaFastArr, true);
   ArraySetAsSeries(emaSlowArr, true);
   ArraySetAsSeries(adxArr, true);
   ArraySetAsSeries(rsiArr, true);
   ArraySetAsSeries(macdMainArr, true);
   ArraySetAsSeries(macdSignalArr, true);
   ArraySetAsSeries(atrArr, true);
   ArraySetAsSeries(atrLongArr, true);

   // Copy indicator buffers
   if(CopyBuffer(handleEMAFast, 0, 0, 2, emaFastArr) <= 0) return false;
   if(CopyBuffer(handleEMASlow, 0, 0, 2, emaSlowArr) <= 0) return false;
   if(CopyBuffer(handleADX, 0, 0, 2, adxArr) <= 0) return false;
   if(CopyBuffer(handleRSI, 0, 0, 2, rsiArr) <= 0) return false;
   if(CopyBuffer(handleMACD, 0, 0, 2, macdMainArr) <= 0) return false;
   if(CopyBuffer(handleMACD, 1, 0, 2, macdSignalArr) <= 0) return false;
   if(CopyBuffer(handleATR, 0, 0, 2, atrArr) <= 0) return false;
   if(CopyBuffer(handleATRLong, 0, 0, 20, atrLongArr) <= 0) return false;

   // Store values
   emaFast = emaFastArr[0];
   emaSlow = emaSlowArr[0];
   adxValue = adxArr[0];
   rsiValue = rsiArr[0];
   macdMain = macdMainArr[0];
   macdSignal = macdSignalArr[0];
   atrValue = atrArr[0];

   // Calculate ATR average for spike detection
   atrAvg = 0;
   for(int i = 0; i < 20; i++)
      atrAvg += atrLongArr[i];
   atrAvg /= 20;

   // Get current spread
   currentSpread = symbolInfo.Spread();

   return true;
}

//+------------------------------------------------------------------+
//| Check for entry conditions                                       |
//+------------------------------------------------------------------+
void CheckForEntry()
{
   // === FILTER 1: Time Filter ===
   if(!IsValidTradingTime())
   {
      if(DebugMode)
      {
         Print("⏰ Skip: Outside trading hours");
         WriteFilterLog("Time Filter", "Outside trading hours");
      }
      return;
   }

   // === FILTER 2: Spread Filter ===
   if(currentSpread > MaxSpread)
   {
      if(DebugMode)
      {
         Print("📊 Skip: Spread too high (", currentSpread, " > ", MaxSpread, ")");
         WriteFilterLog("Spread Filter", StringFormat("Spread %.0f > Max %.0f", currentSpread, MaxSpread));
      }
      return;
   }

   // === FILTER 3: Volatility Spike Filter ===
   if(atrValue > atrAvg * MaxATRMultiple)
   {
      if(DebugMode)
      {
         Print("⚡ Skip: ATR spike detected (", atrValue, " > ", atrAvg * MaxATRMultiple, ")");
         WriteFilterLog("Volatility Filter", StringFormat("ATR spike %.5f > %.5f", atrValue, atrAvg * MaxATRMultiple));
      }
      return;
   }

   // === FILTER 4: Trend Strength (ADX) ===
   if(adxValue < ADX_Threshold)
   {
      if(DebugMode)
      {
         Print("📉 Skip: Weak trend (ADX ", adxValue, " < ", ADX_Threshold, ")");
         WriteFilterLog("ADX Filter", StringFormat("ADX %.1f < Threshold %.1f", adxValue, ADX_Threshold));
      }
      return;
   }

   // === DETERMINE TREND DIRECTION ===
   bool isBullishTrend = (emaFast > emaSlow);
   bool isBearishTrend = (emaFast < emaSlow);

   // === CHECK BUY CONDITIONS ===
   if(isBullishTrend)
   {
      bool buyCondition = CheckBuySignal();
      if(buyCondition)
      {
         if(DebugMode) Print("🟢 BUY Signal detected!");
         OpenTrade(ORDER_TYPE_BUY);
         return;
      }
   }

   // === CHECK SELL CONDITIONS ===
   if(isBearishTrend)
   {
      bool sellCondition = CheckSellSignal();
      if(sellCondition)
      {
         if(DebugMode) Print("🔴 SELL Signal detected!");
         OpenTrade(ORDER_TYPE_SELL);
         return;
      }
   }
}

//+------------------------------------------------------------------+
//| Check BUY signal conditions                                      |
//+------------------------------------------------------------------+
bool CheckBuySignal()
{
   // Condition 1: RSI not overbought (avoid chasing)
   if(rsiValue > RSI_OB)
   {
      if(DebugMode) Print("⚠️ RSI overbought: ", rsiValue);
      return false;
   }

   // Condition 2: RSI in favorable zone (40-70)
   if(rsiValue < 40.0)
   {
      if(DebugMode) Print("⚠️ RSI too low: ", rsiValue);
      return false;
   }

   // Condition 3: MACD bullish
   if(macdMain <= macdSignal)
   {
      if(DebugMode) Print("⚠️ MACD not bullish");
      return false;
   }

   // Condition 4: Price above both EMAs (strong uptrend)
   double currentPrice = symbolInfo.Ask();
   if(currentPrice < emaFast || currentPrice < emaSlow)
   {
      if(DebugMode) Print("⚠️ Price not above EMAs");
      return false;
   }

   // Condition 5: Cooldown period (avoid overtrading)
   if(TimeCurrent() - lastTradeTime < 900) // 15 minutes = 1 bar
   {
      if(DebugMode) Print("⚠️ Cooldown period active");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Check SELL signal conditions                                     |
//+------------------------------------------------------------------+
bool CheckSellSignal()
{
   // Condition 1: RSI not oversold (avoid chasing)
   if(rsiValue < RSI_OS)
   {
      if(DebugMode) Print("⚠️ RSI oversold: ", rsiValue);
      return false;
   }

   // Condition 2: RSI in favorable zone (30-60)
   if(rsiValue > 60.0)
   {
      if(DebugMode) Print("⚠️ RSI too high: ", rsiValue);
      return false;
   }

   // Condition 3: MACD bearish
   if(macdMain >= macdSignal)
   {
      if(DebugMode) Print("⚠️ MACD not bearish");
      return false;
   }

   // Condition 4: Price below both EMAs (strong downtrend)
   double currentPrice = symbolInfo.Bid();
   if(currentPrice > emaFast || currentPrice > emaSlow)
   {
      if(DebugMode) Print("⚠️ Price not below EMAs");
      return false;
   }

   // Condition 5: Cooldown period
   if(TimeCurrent() - lastTradeTime < 900) // 15 minutes
   {
      if(DebugMode) Print("⚠️ Cooldown period active");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Open trade with risk management                                  |
//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE orderType)
{
   double price, sl, tp, lotSize;

   // Get entry price
   if(orderType == ORDER_TYPE_BUY)
      price = symbolInfo.Ask();
   else
      price = symbolInfo.Bid();

   // Calculate SL & TP based on ATR
   double slDistance = atrValue * ATR_SL_Multiplier;
   double tpDistance = atrValue * ATR_TP_Multiplier;

   if(orderType == ORDER_TYPE_BUY)
   {
      sl = NormalizeDouble(price - slDistance, _Digits);
      tp = NormalizeDouble(price + tpDistance, _Digits);
   }
   else
   {
      sl = NormalizeDouble(price + slDistance, _Digits);
      tp = NormalizeDouble(price - tpDistance, _Digits);
   }

   // Calculate lot size based on risk
   lotSize = CalculateLotSize(MathAbs(price - sl));

   // Validate lot size
   double minVol = symbolInfo.LotsMin();
   double maxVol = symbolInfo.LotsMax();
   double volStep = symbolInfo.LotsStep();

   lotSize = MathMax(minVol, MathMin(maxVol, lotSize));
   lotSize = NormalizeDouble(lotSize / volStep, 0) * volStep;

   // Final checks
   if(lotSize < MinLot)
   {
      Print("❌ Lot size too small: ", lotSize);
      return;
   }

   if(lotSize > MaxLot)
   {
      Print("⚠️ Lot size capped at MaxLot: ", MaxLot);
      lotSize = MaxLot;
   }

   // Send order
   bool result = trade.PositionOpen(_Symbol, orderType, lotSize, price, sl, tp, TradeComment);

   if(result)
   {
      Print("✅ ", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), " opened: Lot=", lotSize,
            " Price=", price, " SL=", sl, " TP=", tp, " Risk=", currentRisk, "%");

      // Log trade details
      WriteTradeLog("TRADE OPENED", orderType, lotSize, price, sl, tp);

      // Log market conditions at entry
      WriteLog(StringFormat("Entry Conditions - EMA: %.5f/%.5f | ADX: %.1f | RSI: %.1f | MACD: %.5f/%.5f | ATR: %.5f", emaFast, emaSlow, adxValue, rsiValue, macdMain, macdSignal, atrValue));

      lastTradeTime = TimeCurrent();
      positionOpenTime = TimeCurrent();
      positionOpenPrice = price;
      isBreakevenSet = false;
      isPartialClosed = false;
   }
   else
   {
      Print("❌ Order failed: ", trade.ResultRetcodeDescription());
      WriteLog(StringFormat("TRADE FAILED - Type: %s | Error: %s | Code: %d", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), trade.ResultRetcodeDescription(), trade.ResultRetcode()), "ERROR");
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk percentage                      |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance)
{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = accountBalance * (currentRisk / 100.0);

   double tickValue = symbolInfo.TickValue();
   double tickSize = symbolInfo.TickSize();

   double slInTicks = slDistance / tickSize;
   double lotSize = riskMoney / (slInTicks * tickValue);

   return lotSize;
}

//+------------------------------------------------------------------+
//| Manage open position (BE, trailing, partial close, time exit)   |
//+------------------------------------------------------------------+
void ManageOpenPosition()
{
   if(!position.Select(_Symbol))
      return; // No position open

   double currentPrice = (position.Type() == POSITION_TYPE_BUY) ? symbolInfo.Bid() : symbolInfo.Ask();
   double openPrice = position.PriceOpen();
   double currentSL = position.StopLoss();
   double currentTP = position.TakeProfit();
   ulong ticket = position.Ticket();

   // Calculate profit in ATR multiples
   double profitDistance = (position.Type() == POSITION_TYPE_BUY) ?
                           (currentPrice - openPrice) : (openPrice - currentPrice);
   double profitInATR = profitDistance / atrValue;

   // === EXIT 1: Time Exit ===
   int barsOpen = GetBarShift(_Symbol, PERIOD_CURRENT, positionOpenTime);
   if(barsOpen >= MaxHoldBars)
   {
      double closeProfit = position.Profit();
      Print("⏰ Time exit: Position held for ", barsOpen, " bars (max ", MaxHoldBars, ")");
      WriteLog(StringFormat("TIME EXIT - Held %d/%d bars | Profit: $%.2f", barsOpen, MaxHoldBars, closeProfit));

      trade.PositionClose(ticket);
      UpdateTradingState(false); // Consider as loss for risk adjustment
      return;
   }

   // === EXIT 2: Partial Close ===
   if(UsePartialClose && !isPartialClosed && profitInATR >= Partial_Close_ATR)
   {
      double closeVolume = position.Volume() * (Partial_Close_Percent / 100.0);
      double minVol = symbolInfo.LotsMin();

      if(closeVolume >= minVol)
      {
         Print("💰 Partial close: ", Partial_Close_Percent, "% at ", profitInATR, " ATR profit");
         WriteLog(StringFormat("PARTIAL CLOSE - %.0f%% at %.2f ATR profit | Volume: %.2f", Partial_Close_Percent, profitInATR, closeVolume));

         trade.PositionClosePartial(ticket, closeVolume);
         isPartialClosed = true;
      }
   }

   // === EXIT 3: Breakeven ===
   if(UseBreakeven && !isBreakevenSet && profitInATR >= BE_Trigger_ATR)
   {
      double newSL = NormalizeDouble(openPrice + (position.Type() == POSITION_TYPE_BUY ?
                                     BE_Lock_Pips * _Point : -BE_Lock_Pips * _Point), _Digits);

      bool slImproved = (position.Type() == POSITION_TYPE_BUY) ? (newSL > currentSL || currentSL == 0) :
                                                                   (newSL < currentSL || currentSL == 0);

      if(slImproved)
      {
         Print("🛡️ Breakeven set at ", newSL, " (profit: ", profitInATR, " ATR)");
         WriteLog(StringFormat("BREAKEVEN SET - New SL: %.5f | Profit: %.2f ATR | Lock: %.1f pips", newSL, profitInATR, BE_Lock_Pips));

         trade.PositionModify(ticket, newSL, currentTP);
         isBreakevenSet = true;
      }
   }

   // === EXIT 4: Trailing Stop ===
   if(UseTrailing && profitInATR >= Trail_Start_ATR)
   {
      double trailDistance = atrValue * Trail_Distance_ATR;
      double newSL = NormalizeDouble((position.Type() == POSITION_TYPE_BUY) ?
                                     (currentPrice - trailDistance) : (currentPrice + trailDistance), _Digits);

      bool slImproved = (position.Type() == POSITION_TYPE_BUY) ? (newSL > currentSL) : (newSL < currentSL);

      if(slImproved)
      {
         Print("🔄 Trailing SL updated: ", newSL, " (profit: ", profitInATR, " ATR)");
         WriteLog(StringFormat("TRAILING UPDATE - New SL: %.5f | Profit: %.2f ATR | Distance: %.2f ATR", newSL, profitInATR, Trail_Distance_ATR));

         trade.PositionModify(ticket, newSL, currentTP);
      }
   }
}

//+------------------------------------------------------------------+
//| Update trading state after trade close                          |
//+------------------------------------------------------------------+
void UpdateTradingState(bool isWin)
{
   if(isWin)
   {
      consecutiveWins++;
      consecutiveLosses = 0;

      // Restore risk after 2 consecutive wins
      if(consecutiveWins >= 2)
      {
         currentRisk = RiskPercent;
         WriteLog(StringFormat("Risk restored to %.1f%% after %d wins", currentRisk, consecutiveWins));
      }

      Print("✅ WIN | Consecutive wins: ", consecutiveWins);
      WriteLog(StringFormat("WIN RECORDED - Consecutive: %d | Current Risk: %.1f%%", consecutiveWins, currentRisk), "WIN");
   }
   else
   {
      consecutiveLosses++;
      consecutiveWins = 0;

      // Reduce risk after loss
      double oldRisk = currentRisk;
      currentRisk = MinRiskPercent;

      Print("❌ LOSS | Risk reduced to ", currentRisk, "%");
      WriteLog(StringFormat("LOSS RECORDED - Consecutive: %d | Risk: %.1f%% → %.1f%%", consecutiveLosses, oldRisk, currentRisk), "LOSS");
   }
}

//+------------------------------------------------------------------+
//| Check if current time is valid for trading                      |
//+------------------------------------------------------------------+
bool IsValidTradingTime()
{
   if(!UseTradingHours)
      return true;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   int currentHour = dt.hour;
   int dayOfWeek = dt.day_of_week;

   // Avoid Monday open
   if(AvoidMondayOpen && dayOfWeek == 1 && currentHour < 6)
      return false;

   // Avoid Friday close
   if(AvoidFridayClose && dayOfWeek == 5 && currentHour >= 18)
      return false;

   // Check trading hours
   if(currentHour < StartHour || currentHour >= EndHour)
      return false;

   // Session filters
   bool inAsianSession = (currentHour >= 23 || currentHour < 8);
   bool inLondonSession = (currentHour >= 8 && currentHour < 16);
   bool inNYSession = (currentHour >= 13 && currentHour < 22);

   if(inAsianSession && !TradeAsianSession) return false;
   if(inLondonSession && !TradeLondonSession) return false;
   if(inNYSession && !TradeNYSession) return false;

   return true;
}

//+------------------------------------------------------------------+
//| Update info panel on chart                                       |
//+------------------------------------------------------------------+
void UpdateInfoPanel()
{
   if(!ShowPanel) return;

   string prefix = "XAUBot_";

   // Title
   ObjectSetString(0, prefix + "Title", OBJPROP_TEXT, "═══ XAUBot Pro v1.00 ═══");
   ObjectSetInteger(0, prefix + "Title", OBJPROP_COLOR, clrGold);

   // Account info
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double profit = AccountInfoDouble(ACCOUNT_PROFIT);

   ObjectSetString(0, prefix + "Balance", OBJPROP_TEXT, "Balance: $" + DoubleToString(balance, 2));
   ObjectSetString(0, prefix + "Equity", OBJPROP_TEXT, "Equity:  $" + DoubleToString(equity, 2));

   color profitColor = (profit >= 0) ? clrLimeGreen : clrRed;
   string profitSign = (profit >= 0) ? "+" : "";
   ObjectSetString(0, prefix + "Profit", OBJPROP_TEXT, "Profit:  " + profitSign + "$" + DoubleToString(profit, 2));
   ObjectSetInteger(0, prefix + "Profit", OBJPROP_COLOR, profitColor);

   ObjectSetString(0, prefix + "Separator1", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix + "Separator1", OBJPROP_COLOR, C'60,60,80');

   // Trading status
   bool canTrade = IsValidTradingTime() && (currentSpread <= MaxSpread) && (atrValue <= atrAvg * MaxATRMultiple);
   string statusText = canTrade ? "Status: ✓ READY" : "Status: ⏸ WAITING";
   color statusColor = canTrade ? clrLimeGreen : clrOrange;
   ObjectSetString(0, prefix + "Status", OBJPROP_TEXT, statusText);
   ObjectSetInteger(0, prefix + "Status", OBJPROP_COLOR, statusColor);

   // Trend info
   string trendDir = (emaFast > emaSlow) ? "▲ BULL" : "▼ BEAR";
   string trendStrength = (adxValue >= ADX_Threshold) ? "STRONG" : "WEAK";
   color trendColor = (emaFast > emaSlow) ? clrLimeGreen : clrRed;

   ObjectSetString(0, prefix + "Trend", OBJPROP_TEXT, "Trend: " + trendDir + " (" + trendStrength + ")");
   ObjectSetInteger(0, prefix + "Trend", OBJPROP_COLOR, trendColor);

   ObjectSetString(0, prefix + "ADX", OBJPROP_TEXT, "ADX: " + DoubleToString(adxValue, 1) + " (min " + DoubleToString(ADX_Threshold, 0) + ")");
   ObjectSetString(0, prefix + "RSI", OBJPROP_TEXT, "RSI: " + DoubleToString(rsiValue, 1));

   ObjectSetString(0, prefix + "Separator2", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix + "Separator2", OBJPROP_COLOR, C'60,60,80');

   // Position info
   if(position.Select(_Symbol))
   {
      string posType = (position.Type() == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      color posColor = (position.Type() == POSITION_TYPE_BUY) ? clrDodgerBlue : clrOrangeRed;

      ObjectSetString(0, prefix + "Position", OBJPROP_TEXT, "● " + posType + " | Lot: " + DoubleToString(position.Volume(), 2));
      ObjectSetInteger(0, prefix + "Position", OBJPROP_COLOR, posColor);

      double profitDistance = (position.Type() == POSITION_TYPE_BUY) ?
                              (symbolInfo.Bid() - position.PriceOpen()) :
                              (position.PriceOpen() - symbolInfo.Ask());
      double profitInATR = profitDistance / atrValue;
      double profitMoney = position.Profit();

      color profitClr = (profitMoney >= 0) ? clrLimeGreen : clrRed;
      string profitSgn = (profitMoney >= 0) ? "+" : "";

      ObjectSetString(0, prefix + "PosDtl1", OBJPROP_TEXT, "P/L: " + profitSgn + "$" + DoubleToString(profitMoney, 2) + " (" + DoubleToString(profitInATR, 2) + " ATR)");
      ObjectSetInteger(0, prefix + "PosDtl1", OBJPROP_COLOR, profitClr);

      int barsOpen = GetBarShift(_Symbol, PERIOD_CURRENT, positionOpenTime);
      string beStatus = isBreakevenSet ? "✓ BE" : "";
      string partialStatus = isPartialClosed ? "✓ Part" : "";

      ObjectSetString(0, prefix + "PosDtl2", OBJPROP_TEXT, "Age: " + IntegerToString(barsOpen) + "/" + IntegerToString(MaxHoldBars) + " bars " + beStatus + " " + partialStatus);

      string stateText = "";
      if(isBreakevenSet && isPartialClosed) stateText = "BE+Partial";
      else if(isBreakevenSet) stateText = "Breakeven";
      else if(isPartialClosed) stateText = "Partial Closed";
      else stateText = "Active";

      ObjectSetString(0, prefix + "PosDtl3", OBJPROP_TEXT, "State: " + stateText);
   }
   else
   {
      ObjectSetString(0, prefix + "Position", OBJPROP_TEXT, "● No Position");
      ObjectSetInteger(0, prefix + "Position", OBJPROP_COLOR, clrGray);
      ObjectSetString(0, prefix + "PosDtl1", OBJPROP_TEXT, "");
      ObjectSetString(0, prefix + "PosDtl2", OBJPROP_TEXT, "");
      ObjectSetString(0, prefix + "PosDtl3", OBJPROP_TEXT, "");
   }

   ObjectSetString(0, prefix + "Separator3", OBJPROP_TEXT, "─────────────────────────");
   ObjectSetInteger(0, prefix + "Separator3", OBJPROP_COLOR, C'60,60,80');

   // Risk & market info
   string riskText = "Risk: " + DoubleToString(currentRisk, 1) + "%";
   if(currentRisk < RiskPercent) riskText += " (Recovery)";

   ObjectSetString(0, prefix + "Risk", OBJPROP_TEXT, riskText);
   ObjectSetInteger(0, prefix + "Risk", OBJPROP_COLOR, (currentRisk < RiskPercent) ? clrYellow : clrWhite);

   color spreadColor = (currentSpread <= MaxSpread) ? clrLimeGreen : clrRed;
   ObjectSetString(0, prefix + "Spread", OBJPROP_TEXT, "Spread: " + DoubleToString(currentSpread, 0) + "/" + DoubleToString(MaxSpread, 0) + " pts");
   ObjectSetInteger(0, prefix + "Spread", OBJPROP_COLOR, spreadColor);

   ObjectSetString(0, prefix + "ATR", OBJPROP_TEXT, "ATR: " + DoubleToString(atrValue, _Digits));

   string winLossText = "W:" + IntegerToString(consecutiveWins) + " | L:" + IntegerToString(consecutiveLosses);
   ObjectSetString(0, prefix + "WinLoss", OBJPROP_TEXT, winLossText);

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Trade event handler (for win/loss tracking)                     |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   // Detect position close
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong dealTicket = trans.deal;
      if(dealTicket > 0)
      {
         if(HistoryDealSelect(dealTicket))
         {
            long dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
            if(dealMagic == Magic)
            {
               double dealProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
               long dealEntry = HistoryDealGetInteger(dealTicket, DEAL_ENTRY);
               long dealType = HistoryDealGetInteger(dealTicket, DEAL_TYPE);
               double dealVolume = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
               double dealPrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);

               // Check if it's an exit deal
               if(dealEntry == DEAL_ENTRY_OUT)
               {
                  bool isWin = (dealProfit > 0);
                  string typeStr = (dealType == DEAL_TYPE_BUY) ? "SELL (close)" : "BUY (close)";

                  // Log trade close
                  WriteTradeLog("TRADE CLOSED", (ENUM_ORDER_TYPE)dealType, dealVolume, dealPrice, 0, 0, dealProfit);

                  string exitReason = "";
                  if(isBreakevenSet && isPartialClosed) exitReason = "BE + Partial";
                  else if(isBreakevenSet) exitReason = "Breakeven Hit";
                  else if(isPartialClosed) exitReason = "After Partial";
                  else exitReason = "SL/TP";

                  WriteLog(StringFormat("Exit Reason: %s | Duration: %d bars | Final P/L: $%.2f", exitReason, GetBarShift(_Symbol, PERIOD_CURRENT, positionOpenTime), dealProfit));

                  UpdateTradingState(isWin);
               }
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
