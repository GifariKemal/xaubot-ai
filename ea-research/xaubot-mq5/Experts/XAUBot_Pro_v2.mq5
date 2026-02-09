//+------------------------------------------------------------------+
//|                                              XAUBot_Pro_v2.mq5   |
//|                  XAUBot AI - MQ5 Edition v2.0 (Phase 2 Complete) |
//|   Phase 1 + Phase 2: SMC + Basket + Protect + Macro Features    |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI - Gifari Kemal"
#property link      "https://github.com/GifariKemal/xaubot-ai"
#property version   "2.00"
#property description "XAUBot Pro MQ5 v2.0 - Full AI Trading System"
#property description "Phase 1: Long-term trend + Directional bias + H4 emergency"
#property description "Phase 2: SMC + Basket + Protect + Macro features"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

#include "../Include/XAUBot_Config.mqh"
#include "../Include/XAUBot_TrendFilter.mqh"
#include "../Include/XAUBot_EmergencyStop.mqh"
#include "../Include/XAUBot_SMC.mqh"
#include "../Include/XAUBot_PositionManager.mqh"
#include "../Include/XAUBot_MacroFeatures.mqh"

//--- Input Parameters
input group "========== Capital & Risk =========="
input ENUM_CAPITAL_MODE InpCapitalMode = CAPITAL_SMALL;
input double InpRiskPercent = 1.5;
input double InpMaxDailyLoss = 8.0;

input group "========== Phase 1 Enhancements =========="
input bool InpUseLongTermTrend = true;
input bool InpApplyDirectionalBias = true;
input bool InpUseH4EmergencyStop = true;
input bool InpUseMacroFeatures = true;

input group "========== Phase 2 Features =========="
input bool InpUseSMC = true;                      // Use Smart Money Concepts
input bool InpUseBasketManagement = true;         // Use Basket Position Management
input double InpBasketTP_USD = 50.0;              // Basket Take Profit ($)
input bool InpUseProtectLogic = true;             // Use Protect Positions
input double InpProtectTriggerPips = 30.0;        // Protect Trigger (pips loss)

input group "========== Entry Filters =========="
input double InpConfidenceThreshold = 0.60;       // Min Confidence (Phase 2: 60%)
input bool InpUseSessionFilter = true;
input bool InpUseSpreadFilter = true;
input double InpMaxSpreadPips = 0.5;
input int InpCooldownBars = 3;

input group "========== Stop Loss & Take Profit =========="
input double InpSL_ATR_Multiplier = 1.5;
input double InpTP_RiskReward = 1.5;
input bool InpUseSmartBreakeven = true;
input int InpBreakevenTriggerPips = 20;
input int InpBreakevenLockPips = 5;

input group "========== Position Management =========="
input int InpMaxPositions = 3;
input int InpMagicNumber = 20260209;

input group "========== Time Filters =========="
input string InpSkipHours = "9,21";

//--- Global Objects
CTrade            g_trade;
CPositionInfo     g_position;
CAccountInfo      g_account;
CTrendFilter      g_trend_filter;
CEmergencyStop    g_emergency_stop;
CSMCAnalyzer      g_smc;                // Phase 2
CPositionManager  g_position_manager;   // Phase 2
CMacroFeatures    g_macro;              // Phase 2

//--- Global Variables
datetime g_last_trade_time = 0;
int      g_atr_handle = INVALID_HANDLE;
double   g_daily_starting_balance = 0;
datetime g_last_daily_reset = 0;
datetime g_last_smc_update = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("  XAUBot Pro MQ5 v2.0 - Phase 2");
   Print("========================================");

   InitConfig();
   ApplyInputParameters();

   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   g_trade.LogLevel(LOG_LEVEL_ERRORS);

   // Phase 1 Initialization
   if(!g_trend_filter.Init(_Symbol))
   {
      Print("ERROR: Failed to initialize Trend Filter");
      return INIT_FAILED;
   }

   if(!g_emergency_stop.Init(_Symbol))
   {
      Print("ERROR: Failed to initialize Emergency Stop");
      return INIT_FAILED;
   }

   // Phase 2 Initialization
   if(InpUseSMC)
   {
      if(!g_smc.Init(_Symbol, PERIOD_M15, 200))
      {
         Print("ERROR: Failed to initialize SMC Analyzer");
         return INIT_FAILED;
      }
      Print("✅ SMC Analyzer Initialized");
   }

   if(InpUseBasketManagement || InpUseProtectLogic)
   {
      if(!g_position_manager.Init(InpMagicNumber))
      {
         Print("ERROR: Failed to initialize Position Manager");
         return INIT_FAILED;
      }
      Print("✅ Position Manager Initialized");
      Print("  Basket Management: ", (InpUseBasketManagement ? "ENABLED" : "DISABLED"));
      Print("  Protect Logic: ", (InpUseProtectLogic ? "ENABLED" : "DISABLED"));
   }

   if(InpUseMacroFeatures)
   {
      if(!g_macro.Init(_Symbol))
      {
         Print("ERROR: Failed to initialize Macro Features");
         return INIT_FAILED;
      }
      Print("✅ Macro Features Initialized");
   }

   g_atr_handle = iATR(_Symbol, PERIOD_M15, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create ATR indicator");
      return INIT_FAILED;
   }

   g_daily_starting_balance = g_account.Balance();
   g_last_daily_reset = TimeCurrent();

   Print("✅ XAUBot Pro v2.0 Initialized Successfully!");
   Print("  Phase 1: ✅ Long-term trend + Bias + H4 emergency");
   Print("  Phase 2: ✅ SMC + Basket + Protect + Macro");
   Print("========================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("XAUBot Pro v2.0 Shutting Down...");

   g_trend_filter.Deinit();
   g_emergency_stop.Deinit();
   g_smc.Deinit();
   g_position_manager.Deinit();
   g_macro.Deinit();

   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);

   Print("XAUBot Pro v2.0 Deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   static datetime last_bar_time = 0;
   datetime current_bar_time = iTime(_Symbol, PERIOD_M15, 0);

   if(current_bar_time == last_bar_time)
      return;

   last_bar_time = current_bar_time;

   // === MAIN TRADING LOGIC ===

   CheckDailyReset();

   // Phase 1: Emergency stop
   if(g_emergency_stop.CheckH4EmergencyReversal())
   {
      CloseAllPositions("H4 Emergency Reversal");
      return;
   }

   if(g_emergency_stop.IsLocked())
   {
      UpdateComment();
      return;
   }

   if(!CheckDailyDrawdownLimit())
   {
      UpdateComment();
      return;
   }

   // Phase 2: Update SMC structures (every 4 hours)
   if(InpUseSMC && (TimeCurrent() - g_last_smc_update >= 14400))
   {
      g_smc.UpdateOrderBlocks();
      g_smc.UpdateFairValueGaps();
      g_smc.UpdateBreakOfStructure();
      g_last_smc_update = TimeCurrent();
   }

   // Phase 2: Manage positions (basket + protect)
   ManagePositions();

   // Phase 2: Check basket TP
   if(InpUseBasketManagement)
   {
      g_position_manager.CheckBasketTP(InpMagicNumber);
   }

   // Check if can open new
   if(!CanOpenNewPosition())
      return;

   // Generate signal
   ENUM_TRADE_SIGNAL signal = GenerateTradingSignal();

   if(signal == SIGNAL_NONE || signal == SIGNAL_HOLD)
      return;

   // Execute
   ExecuteTrade(signal);

   UpdateComment();
}

//+------------------------------------------------------------------+
//| Apply Input Parameters                                            |
//+------------------------------------------------------------------+
void ApplyInputParameters()
{
   g_config.capital_mode = InpCapitalMode;
   g_config.risk_percent = InpRiskPercent;
   g_config.max_daily_loss_percent = InpMaxDailyLoss;

   // Phase 1
   g_config.use_long_term_trend = InpUseLongTermTrend;
   g_config.apply_directional_bias = InpApplyDirectionalBias;
   g_config.enable_h4_reversal_lock = InpUseH4EmergencyStop;
   g_config.use_macro_features = InpUseMacroFeatures;

   // Phase 2
   g_config.use_basket_management = InpUseBasketManagement;
   g_config.basket_tp_usd = InpBasketTP_USD;

   g_config.confidence_threshold = InpConfidenceThreshold;
   g_config.use_session_filter = InpUseSessionFilter;
   g_config.use_spread_filter = InpUseSpreadFilter;
   g_config.max_spread_pips = InpMaxSpreadPips;
   g_config.cooldown_bars = InpCooldownBars;

   g_config.sl_atr_multiplier = InpSL_ATR_Multiplier;
   g_config.tp_risk_reward = InpTP_RiskReward;
   g_config.use_smart_breakeven = InpUseSmartBreakeven;
   g_config.breakeven_trigger_pips = InpBreakevenTriggerPips;
   g_config.breakeven_lock_pips = InpBreakevenLockPips;

   g_config.max_positions = InpMaxPositions;

   ParseSkipHours(InpSkipHours);
}

//+------------------------------------------------------------------+
//| Parse Skip Hours                                                  |
//+------------------------------------------------------------------+
void ParseSkipHours(string hours_str)
{
   string hours[];
   int count = StringSplit(hours_str, ',', hours);

   ArrayResize(g_config.skip_hours, count);

   for(int i = 0; i < count; i++)
   {
      g_config.skip_hours[i] = (int)StringToInteger(hours[i]);
   }
}

//+------------------------------------------------------------------+
//| Check Daily Reset                                                 |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
   MqlDateTime dt_current, dt_last;
   TimeToStruct(TimeCurrent(), dt_current);
   TimeToStruct(g_last_daily_reset, dt_last);

   if(dt_current.day != dt_last.day)
   {
      g_daily_starting_balance = g_account.Balance();
      g_last_daily_reset = TimeCurrent();
      g_emergency_stop.ClearLockout();

      Print("📅 NEW DAY RESET: Balance=$", g_daily_starting_balance);
   }
}

//+------------------------------------------------------------------+
//| Check Daily Drawdown Limit                                        |
//+------------------------------------------------------------------+
bool CheckDailyDrawdownLimit()
{
   if(!g_config.enable_daily_limit)
      return true;

   double current_balance = g_account.Balance();
   double daily_loss = g_daily_starting_balance - current_balance;
   double max_loss = g_daily_starting_balance * (g_config.max_daily_loss_percent / 100.0);

   if(daily_loss >= max_loss)
   {
      Print("⛔ DAILY DRAWDOWN LIMIT: Loss=$", daily_loss);
      CloseAllPositions("Daily Limit");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Can Open New Position                                             |
//+------------------------------------------------------------------+
bool CanOpenNewPosition()
{
   int open_positions = CountOpenPositions();
   if(open_positions >= g_config.max_positions)
      return false;

   if(g_config.use_cooldown)
   {
      datetime cooldown_time = g_last_trade_time + g_config.cooldown_bars * PeriodSeconds(PERIOD_M15);
      if(TimeCurrent() < cooldown_time)
         return false;
   }

   if(g_config.use_spread_filter)
   {
      double spread_pips = GetSpreadPips();
      if(spread_pips > g_config.max_spread_pips)
         return false;
   }

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   for(int i = 0; i < ArraySize(g_config.skip_hours); i++)
   {
      if(dt.hour == g_config.skip_hours[i])
         return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Generate Trading Signal (Phase 2: SMC + Macro)                   |
//+------------------------------------------------------------------+
ENUM_TRADE_SIGNAL GenerateTradingSignal()
{
   double confidence = 0.65;  // Base confidence
   ENUM_TRADE_SIGNAL signal = SIGNAL_NONE;

   // === SIGNAL GENERATION ===

   // Step 1: Trend direction
   signal = DetermineTrendDirection();
   if(signal == SIGNAL_NONE)
      return SIGNAL_NONE;

   // Step 2: Phase 1 - Long-term trend filter
   if(!g_trend_filter.CheckLongTermTrend(signal))
      return SIGNAL_NONE;

   // Step 3: Phase 2 - SMC confirmation
   if(InpUseSMC)
   {
      double ob_level = 0;
      double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

      if(signal == SIGNAL_BUY)
      {
         // Check bullish OB touch
         if(!g_smc.CheckBullishOBTouch(current_price, ob_level))
         {
            Print("No bullish OB touch at current price");
            return SIGNAL_NONE;
         }
         confidence += 0.10;  // +10% for OB confluence
      }
      else if(signal == SIGNAL_SELL)
      {
         if(!g_smc.CheckBearishOBTouch(current_price, ob_level))
         {
            Print("No bearish OB touch at current price");
            return SIGNAL_NONE;
         }
         confidence += 0.10;
      }
   }

   // Step 4: Phase 2 - Macro confirmation
   if(InpUseMacroFeatures)
   {
      if(!g_macro.CheckMacroConfirmation(signal))
         return SIGNAL_NONE;

      confidence += 0.05;  // +5% for macro confluence
   }

   // Step 5: Phase 1 - Apply directional bias
   confidence = ApplyDirectionalBias(confidence, signal);

   // Step 6: Check confidence threshold
   if(confidence < g_config.confidence_threshold)
      return SIGNAL_NONE;

   Print("✅ SIGNAL GENERATED: ", EnumToString(signal), " | Confidence: ", confidence);

   return signal;
}

//+------------------------------------------------------------------+
//| Determine Trend Direction                                         |
//+------------------------------------------------------------------+
ENUM_TRADE_SIGNAL DetermineTrendDirection()
{
   double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ema20_h1 = g_trend_filter.GetEMA20_H1();

   if(ema20_h1 == 0)
      return SIGNAL_NONE;

   if(current_price > ema20_h1)
      return SIGNAL_BUY;
   else if(current_price < ema20_h1)
      return SIGNAL_SELL;

   return SIGNAL_NONE;
}

//+------------------------------------------------------------------+
//| Execute Trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_TRADE_SIGNAL signal)
{
   double atr = GetATR();
   if(atr == 0)
      return;

   double sl_pips = atr * g_config.sl_atr_multiplier * 10000;
   double tp_pips = sl_pips * g_config.tp_risk_reward;

   double lot = CalculateLotSize(sl_pips);

   double entry_price = (signal == SIGNAL_BUY) ?
                        SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                        SymbolInfoDouble(_Symbol, SYMBOL_BID);

   double sl_price, tp_price;
   if(signal == SIGNAL_BUY)
   {
      sl_price = entry_price - sl_pips * _Point;
      tp_price = entry_price + tp_pips * _Point;
   }
   else
   {
      sl_price = entry_price + sl_pips * _Point;
      tp_price = entry_price - tp_pips * _Point;
   }

   sl_price = NormalizeDouble(sl_price, _Digits);
   tp_price = NormalizeDouble(tp_price, _Digits);

   bool result = false;
   if(signal == SIGNAL_BUY)
      result = g_trade.Buy(lot, _Symbol, entry_price, sl_price, tp_price, "XAUBot v2 BUY");
   else
      result = g_trade.Sell(lot, _Symbol, entry_price, sl_price, tp_price, "XAUBot v2 SELL");

   if(result)
   {
      g_last_trade_time = TimeCurrent();
      Print("✅ Trade Executed: ", EnumToString(signal), " | Lot: ", lot, " | SL: ", sl_pips, " | TP: ", tp_pips);
   }
   else
   {
      Print("❌ Trade Failed: ", g_trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Calculate Lot Size                                                |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_pips)
{
   double risk_amount = g_account.Balance() * (g_config.risk_percent / 100.0);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double lot = risk_amount / (sl_pips * tick_value);

   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathFloor(lot / lot_step) * lot_step;
   lot = MathMax(lot, min_lot);
   lot = MathMin(lot, max_lot);

   return lot;
}

//+------------------------------------------------------------------+
//| Manage Positions (Phase 2: Breakeven + Protect)                  |
//+------------------------------------------------------------------+
void ManagePositions()
{
   // Smart breakeven
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!g_position.SelectByIndex(i))
         continue;

      if(g_position.Symbol() != _Symbol || g_position.Magic() != InpMagicNumber)
         continue;

      if(g_config.use_smart_breakeven)
         CheckSmartBreakeven(g_position.Ticket());
   }

   // Phase 2: Protect positions
   if(InpUseProtectLogic)
   {
      g_position_manager.CheckProtectTriggers(InpMagicNumber);
   }
}

//+------------------------------------------------------------------+
//| Check Smart Breakeven                                             |
//+------------------------------------------------------------------+
void CheckSmartBreakeven(ulong ticket)
{
   if(!g_position.SelectByTicket(ticket))
      return;

   double open_price = g_position.PriceOpen();
   double current_price = g_position.PriceCurrent();
   double sl = g_position.StopLoss();

   double profit_pips = 0;
   if(g_position.PositionType() == POSITION_TYPE_BUY)
      profit_pips = (current_price - open_price) / _Point;
   else
      profit_pips = (open_price - current_price) / _Point;

   if(profit_pips >= g_config.breakeven_trigger_pips)
   {
      double breakeven_price = open_price + g_config.breakeven_lock_pips * _Point *
                               (g_position.PositionType() == POSITION_TYPE_BUY ? 1 : -1);

      if(MathAbs(sl - breakeven_price) > _Point)
      {
         g_trade.PositionModify(ticket, breakeven_price, g_position.TakeProfit());
         Print("🔒 Breakeven SET for ticket ", ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Close All Positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason)
{
   Print("🚨 CLOSING ALL POSITIONS: ", reason);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(g_position.SelectByIndex(i))
      {
         if(g_position.Symbol() == _Symbol && g_position.Magic() == InpMagicNumber)
            g_trade.PositionClose(g_position.Ticket());
      }
   }
}

//+------------------------------------------------------------------+
//| Update Comment                                                    |
//+------------------------------------------------------------------+
void UpdateComment()
{
   string comment = "XAUBot Pro v2.0 (Phase 2)\n";
   comment += "━━━━━━━━━━━━━━━━━━━\n";
   comment += "Emergency: " + g_emergency_stop.GetStatus() + "\n";
   comment += "Positions: " + IntegerToString(CountOpenPositions()) + "/" + IntegerToString(g_config.max_positions) + "\n";

   if(InpUseBasketManagement)
   {
      g_position_manager.UpdateBaskets(InpMagicNumber);
      comment += "Baskets: " + IntegerToString(g_position_manager.GetBasketCount()) + "\n";
   }

   if(InpUseProtectLogic)
      comment += "Protects: " + IntegerToString(g_position_manager.GetProtectCount()) + "\n";

   if(InpUseMacroFeatures)
      comment += g_macro.GetMacroSummary() + "\n";

   if(InpUseSMC)
      comment += "OBs: " + IntegerToString(g_smc.GetActiveOBCount()) + "\n";

   Comment(comment);
}

//+------------------------------------------------------------------+
//| Helper Functions                                                  |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(g_position.SelectByIndex(i))
      {
         if(g_position.Symbol() == _Symbol && g_position.Magic() == InpMagicNumber)
            count++;
      }
   }
   return count;
}

double GetATR()
{
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   if(CopyBuffer(g_atr_handle, 0, 0, 1, atr_buffer) <= 0)
      return 0;
   return atr_buffer[0];
}

double GetSpreadPips()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return (ask - bid) / _Point / 10;
}

//+------------------------------------------------------------------+
