//+------------------------------------------------------------------+
//| XAUBot_Pro_Lite_v2.mq5                                           |
//| Clean rebuild - M15 Gold Trading EA                              |
//+------------------------------------------------------------------+
#property copyright "XAUBot Pro"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//=== INPUT PARAMETERS ===
input group "Risk Management"
input double   RiskPercent = 1.0;
input double   MinRiskPercent = 0.5;
input double   MaxLot = 0.2;
input double   MinLot = 0.01;
input double   ATR_SL_Multiplier = 1.0;
input double   ATR_TP_Multiplier = 1.5;

input group "Entry Filters"
input int      EMA_Fast = 50;
input int      EMA_Slow = 200;
input int      ADX_Period = 14;
input double   ADX_Threshold = 25.0;
input int      RSI_Period = 14;
input double   RSI_OB = 70.0;
input double   RSI_OS = 30.0;
input double   MaxSpread = 20.0;

input group "Exit Management"
input bool     UseBreakeven = true;
input double   BE_Trigger_ATR = 0.5;
input double   BE_Lock_Pips = 5.0;
input int      MaxHoldBars = 16;

input group "Other"
input int      Magic = 202602;
input bool     ShowPanel = true;
input ENUM_BASE_CORNER PanelCorner = CORNER_LEFT_UPPER;
input int      PanelOffsetX = 400;
input int      PanelOffsetY = 10;
input bool     EnableFileLog = true;
input bool     LogFilterRejects = true;

//=== GLOBAL VARIABLES ===
CTrade trade;
CPositionInfo position;
CSymbolInfo symbolInfo;

int handleEMAFast, handleEMASlow, handleADX, handleRSI, handleMACD, handleATR;
double emaFast, emaSlow, adxValue, rsiValue, macdMain, macdSignal, atrValue;
double currentRisk = 1.0;
int consecutiveWins = 0;
int consecutiveLosses = 0;
datetime lastTradeTime = 0;
datetime lastBarTime = 0;
bool isBreakevenSet = false;
datetime positionOpenTime = 0;
int logFileHandle = INVALID_HANDLE;
string currentLogFile = "";
datetime lastLogDate = 0;

//+------------------------------------------------------------------+
//| Open log file                                                    |
//+------------------------------------------------------------------+
bool OpenLogFile()
{
   if(!EnableFileLog) return true;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   string filename = StringFormat("XAUBot_%04d-%02d-%02d.log", dt.year, dt.mon, dt.day);
   currentLogFile = filename;
   lastLogDate = TimeCurrent();

   logFileHandle = FileOpen(filename, FILE_WRITE|FILE_READ|FILE_TXT|FILE_ANSI);
   if(logFileHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to open log file: ", filename);
      return false;
   }

   FileSeek(logFileHandle, 0, SEEK_END);

   string marker = StringFormat("\n========== SESSION START: %s ==========\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
   FileWriteString(logFileHandle, marker);
   FileFlush(logFileHandle);

   return true;
}

//+------------------------------------------------------------------+
//| Write to log file                                                |
//+------------------------------------------------------------------+
void WriteLog(string message, string level="INFO")
{
   if(!EnableFileLog || logFileHandle == INVALID_HANDLE) return;

   MqlDateTime currentDT, lastDT;
   TimeToStruct(TimeCurrent(), currentDT);
   TimeToStruct(lastLogDate, lastDT);

   if(currentDT.day != lastDT.day)
   {
      CloseLogFile();
      OpenLogFile();
   }

   string logLine = StringFormat("[%s] [%s] %s\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), level, message);
   FileWriteString(logFileHandle, logLine);
   FileFlush(logFileHandle);
}

//+------------------------------------------------------------------+
//| Close log file                                                   |
//+------------------------------------------------------------------+
void CloseLogFile()
{
   if(logFileHandle != INVALID_HANDLE)
   {
      string marker = StringFormat("[%s] ========== SESSION END ==========\n\n", TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
      FileWriteString(logFileHandle, marker);
      FileFlush(logFileHandle);
      FileClose(logFileHandle);
      logFileHandle = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
//| Create graphical panel                                           |
//+------------------------------------------------------------------+
void CreatePanel()
{
   string prefix = "XAU_";
   color bgColor = C'20,20,30';

   // Background
   ObjectCreate(0, prefix+"BG", OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_CORNER, PanelCorner);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_XDISTANCE, PanelOffsetX);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_YDISTANCE, PanelOffsetY);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_XSIZE, 250);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_YSIZE, 180);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_BGCOLOR, bgColor);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, prefix+"BG", OBJPROP_COLOR, C'40,40,50');
   ObjectSetInteger(0, prefix+"BG", OBJPROP_SELECTABLE, false);

   // Text labels
   string labels[] = {"Title", "Balance", "Equity", "Profit", "Sep1", "Status", "Trend", "ADX", "RSI", "Sep2", "Position", "PosDetail", "Sep3", "Risk", "Spread", "Stats"};

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

//+------------------------------------------------------------------+
//| Update panel info                                                |
//+------------------------------------------------------------------+
void UpdatePanel()
{
   if(!ShowPanel) return;

   string prefix = "XAU_";

   // Title
   ObjectSetString(0, prefix+"Title", OBJPROP_TEXT, "═══ XAUBot v2 ═══");
   ObjectSetInteger(0, prefix+"Title", OBJPROP_COLOR, clrGold);

   // Account
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double profit = AccountInfoDouble(ACCOUNT_PROFIT);

   ObjectSetString(0, prefix+"Balance", OBJPROP_TEXT, "Balance: $"+DoubleToString(balance,2));
   ObjectSetString(0, prefix+"Equity", OBJPROP_TEXT, "Equity:  $"+DoubleToString(equity,2));

   color profitColor = (profit>=0) ? clrLimeGreen : clrRed;
   string profitSign = (profit>=0) ? "+" : "";
   ObjectSetString(0, prefix+"Profit", OBJPROP_TEXT, "Profit:  "+profitSign+"$"+DoubleToString(profit,2));
   ObjectSetInteger(0, prefix+"Profit", OBJPROP_COLOR, profitColor);

   ObjectSetString(0, prefix+"Sep1", OBJPROP_TEXT, "─────────────────────");
   ObjectSetInteger(0, prefix+"Sep1", OBJPROP_COLOR, C'60,60,80');

   // Trading status
   bool canTrade = (symbolInfo.Spread() <= MaxSpread) && (adxValue >= ADX_Threshold);
   string statusText = canTrade ? "Status: ✓ READY" : "Status: ⏸ WAIT";
   color statusColor = canTrade ? clrLimeGreen : clrOrange;
   ObjectSetString(0, prefix+"Status", OBJPROP_TEXT, statusText);
   ObjectSetInteger(0, prefix+"Status", OBJPROP_COLOR, statusColor);

   // Trend
   string trendDir = (emaFast > emaSlow) ? "▲ BULL" : "▼ BEAR";
   string trendStrength = (adxValue >= ADX_Threshold) ? "STRONG" : "WEAK";
   color trendColor = (emaFast > emaSlow) ? clrLimeGreen : clrRed;
   ObjectSetString(0, prefix+"Trend", OBJPROP_TEXT, "Trend: "+trendDir+" ("+trendStrength+")");
   ObjectSetInteger(0, prefix+"Trend", OBJPROP_COLOR, trendColor);

   ObjectSetString(0, prefix+"ADX", OBJPROP_TEXT, "ADX: "+DoubleToString(adxValue,1)+" (min 25)");
   ObjectSetString(0, prefix+"RSI", OBJPROP_TEXT, "RSI: "+DoubleToString(rsiValue,1));

   ObjectSetString(0, prefix+"Sep2", OBJPROP_TEXT, "─────────────────────");
   ObjectSetInteger(0, prefix+"Sep2", OBJPROP_COLOR, C'60,60,80');

   // Position
   if(position.Select(_Symbol))
   {
      string posType = (position.Type()==POSITION_TYPE_BUY) ? "BUY" : "SELL";
      color posColor = (position.Type()==POSITION_TYPE_BUY) ? clrDodgerBlue : clrOrangeRed;
      double posProfit = position.Profit();

      ObjectSetString(0, prefix+"Position", OBJPROP_TEXT, "● "+posType+" | Lot: "+DoubleToString(position.Volume(),2));
      ObjectSetInteger(0, prefix+"Position", OBJPROP_COLOR, posColor);

      color pColor = (posProfit>=0) ? clrLimeGreen : clrRed;
      string pSign = (posProfit>=0) ? "+" : "";
      ObjectSetString(0, prefix+"PosDetail", OBJPROP_TEXT, "P/L: "+pSign+"$"+DoubleToString(posProfit,2));
      ObjectSetInteger(0, prefix+"PosDetail", OBJPROP_COLOR, pColor);
   }
   else
   {
      ObjectSetString(0, prefix+"Position", OBJPROP_TEXT, "● No Position");
      ObjectSetInteger(0, prefix+"Position", OBJPROP_COLOR, clrGray);
      ObjectSetString(0, prefix+"PosDetail", OBJPROP_TEXT, "");
   }

   ObjectSetString(0, prefix+"Sep3", OBJPROP_TEXT, "─────────────────────");
   ObjectSetInteger(0, prefix+"Sep3", OBJPROP_COLOR, C'60,60,80');

   // Risk & Info
   string riskText = "Risk: "+DoubleToString(currentRisk,1)+"%";
   if(currentRisk < RiskPercent) riskText += " (Recovery)";
   ObjectSetString(0, prefix+"Risk", OBJPROP_TEXT, riskText);
   ObjectSetInteger(0, prefix+"Risk", OBJPROP_COLOR, (currentRisk<RiskPercent) ? clrYellow : clrWhite);

   double spread = symbolInfo.Spread();
   color spreadColor = (spread <= MaxSpread) ? clrLimeGreen : clrRed;
   ObjectSetString(0, prefix+"Spread", OBJPROP_TEXT, "Spread: "+DoubleToString(spread,0)+"/"+DoubleToString(MaxSpread,0));
   ObjectSetInteger(0, prefix+"Spread", OBJPROP_COLOR, spreadColor);

   ObjectSetString(0, prefix+"Stats", OBJPROP_TEXT, "W:"+IntegerToString(consecutiveWins)+" | L:"+IntegerToString(consecutiveLosses));
}

//+------------------------------------------------------------------+
//| Delete panel                                                     |
//+------------------------------------------------------------------+
void DeletePanel()
{
   string prefix = "XAU_";
   ObjectDelete(0, prefix+"BG");
   string labels[] = {"Title", "Balance", "Equity", "Profit", "Sep1", "Status", "Trend", "ADX", "RSI", "Sep2", "Position", "PosDetail", "Sep3", "Risk", "Spread", "Stats"};
   for(int i=0; i<ArraySize(labels); i++)
      ObjectDelete(0, prefix+labels[i]);
}

//+------------------------------------------------------------------+
int OnInit()
{
   Print("XAUBot Pro Lite v2 - Initialization Started");

   // Check timeframe
   if(Period() != PERIOD_M15)
   {
      Alert("⚠️ WARNING: EA designed for M15 timeframe! Current: ", EnumToString(Period()));
      Print("⚠️ WARNING: Please attach EA to M15 chart for optimal performance");
   }

   if(!symbolInfo.Name(_Symbol))
   {
      Print("ERROR: Failed to set symbol");
      return INIT_FAILED;
   }

   trade.SetExpertMagicNumber(Magic);

   handleEMAFast = iMA(_Symbol, PERIOD_CURRENT, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   handleEMASlow = iMA(_Symbol, PERIOD_CURRENT, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   handleADX = iADX(_Symbol, PERIOD_CURRENT, ADX_Period);
   handleRSI = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   handleMACD = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
   handleATR = iATR(_Symbol, PERIOD_CURRENT, 14);

   if(handleEMAFast == INVALID_HANDLE || handleEMASlow == INVALID_HANDLE ||
      handleADX == INVALID_HANDLE || handleRSI == INVALID_HANDLE ||
      handleMACD == INVALID_HANDLE || handleATR == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators");
      return INIT_FAILED;
   }

   currentRisk = RiskPercent;

   if(ShowPanel)
      CreatePanel();

   if(EnableFileLog)
      OpenLogFile();

   WriteLog("XAUBot Pro Lite v2 - Initialization Complete");
   WriteLog(StringFormat("Config: Risk=%.1f%% | TP=%.1fx ATR | SL=%.1fx ATR | M15 timeframe", RiskPercent, ATR_TP_Multiplier, ATR_SL_Multiplier));

   Print("XAUBot Pro Lite v2 - Initialization Complete");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(handleEMAFast);
   IndicatorRelease(handleEMASlow);
   IndicatorRelease(handleADX);
   IndicatorRelease(handleRSI);
   IndicatorRelease(handleMACD);
   IndicatorRelease(handleATR);

   if(ShowPanel)
      DeletePanel();

   if(EnableFileLog)
      CloseLogFile();

   Comment("");
   Print("XAUBot stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   bool newBar = (currentBarTime != lastBarTime);

   if(!newBar)
   {
      ManagePosition();
      return;
   }

   lastBarTime = currentBarTime;

   if(!UpdateData()) return;

   ManagePosition();

   if(!position.Select(_Symbol))
      CheckEntry();

   if(ShowPanel)
      UpdatePanel();
}

//+------------------------------------------------------------------+
bool UpdateData()
{
   double emaFastArr[], emaSlowArr[], adxArr[], rsiArr[], macdMainArr[], macdSignalArr[], atrArr[];

   ArraySetAsSeries(emaFastArr, true);
   ArraySetAsSeries(emaSlowArr, true);
   ArraySetAsSeries(adxArr, true);
   ArraySetAsSeries(rsiArr, true);
   ArraySetAsSeries(macdMainArr, true);
   ArraySetAsSeries(macdSignalArr, true);
   ArraySetAsSeries(atrArr, true);

   if(CopyBuffer(handleEMAFast, 0, 0, 2, emaFastArr) <= 0) return false;
   if(CopyBuffer(handleEMASlow, 0, 0, 2, emaSlowArr) <= 0) return false;
   if(CopyBuffer(handleADX, 0, 0, 2, adxArr) <= 0) return false;
   if(CopyBuffer(handleRSI, 0, 0, 2, rsiArr) <= 0) return false;
   if(CopyBuffer(handleMACD, 0, 0, 2, macdMainArr) <= 0) return false;
   if(CopyBuffer(handleMACD, 1, 0, 2, macdSignalArr) <= 0) return false;
   if(CopyBuffer(handleATR, 0, 0, 2, atrArr) <= 0) return false;

   emaFast = emaFastArr[0];
   emaSlow = emaSlowArr[0];
   adxValue = adxArr[0];
   rsiValue = rsiArr[0];
   macdMain = macdMainArr[0];
   macdSignal = macdSignalArr[0];
   atrValue = atrArr[0];

   return true;
}

//+------------------------------------------------------------------+
void CheckEntry()
{
   double spread = symbolInfo.Spread();

   // Filter 1: Spread
   if(spread > MaxSpread)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Spread too high (%.0f > %.0f)", spread, MaxSpread), "FILTER");
      return;
   }

   // Filter 2: ADX
   if(adxValue < ADX_Threshold)
   {
      if(LogFilterRejects)
         WriteLog(StringFormat("SKIP: Weak trend (ADX %.1f < %.1f)", adxValue, ADX_Threshold), "FILTER");
      return;
   }

   // Filter 3: Cooldown
   if(TimeCurrent() - lastTradeTime < 900)
   {
      if(LogFilterRejects)
         WriteLog("SKIP: Cooldown period (15 min)", "FILTER");
      return;
   }

   bool isBullish = (emaFast > emaSlow);
   bool isBearish = (emaFast < emaSlow);

   // BUY Signal
   if(isBullish)
   {
      if(rsiValue < 40.0)
      {
         if(LogFilterRejects)
            WriteLog(StringFormat("SKIP BUY: RSI too low (%.1f < 40)", rsiValue), "FILTER");
         return;
      }
      if(rsiValue > RSI_OB)
      {
         if(LogFilterRejects)
            WriteLog(StringFormat("SKIP BUY: RSI overbought (%.1f > %.1f)", rsiValue, RSI_OB), "FILTER");
         return;
      }
      if(macdMain > macdSignal)
      {
         WriteLog(StringFormat("SIGNAL: BUY | EMA: %.5f>%.5f | ADX: %.1f | RSI: %.1f | MACD: %.5f>%.5f", emaFast, emaSlow, adxValue, rsiValue, macdMain, macdSignal), "SIGNAL");
         OpenTrade(ORDER_TYPE_BUY);
      }
   }
   // SELL Signal
   else if(isBearish)
   {
      if(rsiValue > 60.0)
      {
         if(LogFilterRejects)
            WriteLog(StringFormat("SKIP SELL: RSI too high (%.1f > 60)", rsiValue), "FILTER");
         return;
      }
      if(rsiValue < RSI_OS)
      {
         if(LogFilterRejects)
            WriteLog(StringFormat("SKIP SELL: RSI oversold (%.1f < %.1f)", rsiValue, RSI_OS), "FILTER");
         return;
      }
      if(macdMain < macdSignal)
      {
         WriteLog(StringFormat("SIGNAL: SELL | EMA: %.5f<%.5f | ADX: %.1f | RSI: %.1f | MACD: %.5f<%.5f", emaFast, emaSlow, adxValue, rsiValue, macdMain, macdSignal), "SIGNAL");
         OpenTrade(ORDER_TYPE_SELL);
      }
   }
}

//+------------------------------------------------------------------+
void OpenTrade(ENUM_ORDER_TYPE orderType)
{
   double price = (orderType == ORDER_TYPE_BUY) ? symbolInfo.Ask() : symbolInfo.Bid();

   double slDistance = atrValue * ATR_SL_Multiplier;
   double tpDistance = atrValue * ATR_TP_Multiplier;

   double sl = NormalizeDouble((orderType == ORDER_TYPE_BUY) ? (price - slDistance) : (price + slDistance), _Digits);
   double tp = NormalizeDouble((orderType == ORDER_TYPE_BUY) ? (price + tpDistance) : (price - tpDistance), _Digits);

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * (currentRisk / 100.0);
   double tickValue = symbolInfo.TickValue();
   double tickSize = symbolInfo.TickSize();
   double slInTicks = MathAbs(price - sl) / tickSize;
   double lotSize = riskMoney / (slInTicks * tickValue);

   lotSize = NormalizeDouble(lotSize, 2);
   lotSize = MathMax(MinLot, MathMin(MaxLot, lotSize));

   if(trade.PositionOpen(_Symbol, orderType, lotSize, price, sl, tp, "XAUBot"))
   {
      string tradeType = (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL");
      Print(tradeType, " opened: Lot=", lotSize, " Price=", price);
      WriteLog(StringFormat("TRADE OPEN: %s | Lot: %.2f | Price: %.5f | SL: %.5f | TP: %.5f | ATR: %.5f", tradeType, lotSize, price, sl, tp, atrValue), "TRADE");

      lastTradeTime = TimeCurrent();
      positionOpenTime = TimeCurrent();
      isBreakevenSet = false;
   }
   else
   {
      WriteLog(StringFormat("TRADE FAILED: %s | Error: %s", (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), trade.ResultRetcodeDescription()), "ERROR");
   }
}

//+------------------------------------------------------------------+
void ManagePosition()
{
   if(!position.Select(_Symbol)) return;

   double currentPrice = (position.Type() == POSITION_TYPE_BUY) ? symbolInfo.Bid() : symbolInfo.Ask();
   double openPrice = position.PriceOpen();
   double profitDistance = (position.Type() == POSITION_TYPE_BUY) ? (currentPrice - openPrice) : (openPrice - currentPrice);
   double profitInATR = profitDistance / atrValue;

   // Breakeven
   if(UseBreakeven && !isBreakevenSet && profitInATR >= BE_Trigger_ATR)
   {
      double newSL = NormalizeDouble(openPrice + ((position.Type() == POSITION_TYPE_BUY) ? BE_Lock_Pips * _Point : -BE_Lock_Pips * _Point), _Digits);

      if(trade.PositionModify(position.Ticket(), newSL, position.TakeProfit()))
      {
         Print("Breakeven set at ", newSL);
         WriteLog(StringFormat("BREAKEVEN: SL moved to %.5f | Profit: %.2f ATR", newSL, profitInATR), "EXIT");
         isBreakevenSet = true;
      }
   }
}

//+------------------------------------------------------------------+
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
                  if(consecutiveWins >= 2) currentRisk = RiskPercent;
                  Print("WIN | Consecutive: ", consecutiveWins);
                  WriteLog(StringFormat("TRADE CLOSE: WIN | Profit: $%.2f | Consecutive: %d | Risk: %.1f%%", dealProfit, consecutiveWins, currentRisk), "WIN");
               }
               else
               {
                  consecutiveLosses++;
                  consecutiveWins = 0;
                  currentRisk = MinRiskPercent;
                  Print("LOSS | Risk reduced to ", currentRisk, "%");
                  WriteLog(StringFormat("TRADE CLOSE: LOSS | Loss: $%.2f | Consecutive: %d | Risk reduced to %.1f%%", dealProfit, consecutiveLosses, currentRisk), "LOSS");
               }
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
