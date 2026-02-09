//+------------------------------------------------------------------+
//|                                                   XAUBot_Pro.mq5 |
//|                           XAUBot AI - MQ5 Edition v1.0           |
//|      Based on: Python XAUBot + Research (3 Commercial EAs)       |
//|      Phase 1: Long-term trend + Directional bias + H4 emergency  |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI - Gifari Kemal"
#property link      "https://github.com/GifariKemal/xaubot-ai"
#property version   "1.00"
#property description "XAUBot Pro MQ5 - Hybrid AI Trading System"
#property description "Features: SMC + ML-inspired rules + Phase 1 enhancements"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

#include "../Include/XAUBot_Config.mqh"
#include "../Include/XAUBot_TrendFilter.mqh"
#include "../Include/XAUBot_EmergencyStop.mqh"

//--- Input Parameters
input group "========== Capital & Risk =========="
input ENUM_CAPITAL_MODE InpCapitalMode = CAPITAL_SMALL;  // Capital Mode
input double InpRiskPercent = 1.5;                       // Risk Per Trade (%)
input double InpMaxDailyLoss = 8.0;                      // Max Daily Loss (%)

input group "========== Phase 1 Enhancements =========="
input bool InpUseLongTermTrend = true;                   // Use 200 EMA H1/H4 Filter
input bool InpApplyDirectionalBias = true;               // Apply Gold BUY Bias (10%)
input bool InpUseH4EmergencyStop = true;                 // Use H4 Emergency Reversal Stop
input bool InpUseMacroFeatures = true;                   // Check DXY/Oil Correlation

input group "========== Entry Filters =========="
input double InpConfidenceThreshold = 0.55;              // Min Confidence (0-1)
input bool InpUseSessionFilter = true;                   // Filter by Session
input bool InpUseSpreadFilter = true;                    // Filter by Spread
input double InpMaxSpreadPips = 0.5;                     // Max Spread (pips)
input int InpCooldownBars = 3;                           // Cooldown Between Trades (bars)

input group "========== Stop Loss & Take Profit =========="
input double InpSL_ATR_Multiplier = 1.5;                 // SL = ATR × Multiplier
input double InpTP_RiskReward = 1.5;                     // TP = SL × Risk:Reward
input bool InpUseSmartBreakeven = true;                  // Use Smart Breakeven
input int InpBreakevenTriggerPips = 20;                  // Breakeven Trigger (pips)
input int InpBreakevenLockPips = 5;                      // Breakeven Lock (pips)

input group "========== Position Management =========="
input int InpMaxPositions = 3;                           // Max Concurrent Positions
input int InpMagicNumber = 20260209;                     // Magic Number

input group "========== Time Filters =========="
input string InpSkipHours = "9,21";                      // Skip Hours (WIB, comma-separated)

//--- Global Objects
CTrade            g_trade;
CPositionInfo     g_position;
CAccountInfo      g_account;
CTrendFilter      g_trend_filter;
CEmergencyStop    g_emergency_stop;

//--- Global Variables
datetime g_last_trade_time = 0;
int      g_atr_handle = INVALID_HANDLE;
double   g_daily_starting_balance = 0;
datetime g_last_daily_reset = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("  XAUBot Pro MQ5 - Initializing...");
   Print("========================================");

   // Initialize config
   InitConfig();
   ApplyInputParameters();

   // Initialize trade object
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(10);
   g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   g_trade.LogLevel(LOG_LEVEL_ERRORS);

   // Initialize trend filter
   if(!g_trend_filter.Init(_Symbol))
   {
      Print("ERROR: Failed to initialize Trend Filter");
      return INIT_FAILED;
   }

   // Initialize emergency stop
   if(!g_emergency_stop.Init(_Symbol))
   {
      Print("ERROR: Failed to initialize Emergency Stop");
      return INIT_FAILED;
   }

   // Initialize ATR indicator
   g_atr_handle = iATR(_Symbol, PERIOD_M15, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create ATR indicator");
      return INIT_FAILED;
   }

   // Set daily starting balance
   g_daily_starting_balance = g_account.Balance();
   g_last_daily_reset = TimeCurrent();

   Print("✅ XAUBot Pro Initialized Successfully!");
   Print("  Symbol: ", _Symbol);
   Print("  Capital Mode: ", EnumToString(g_config.capital_mode));
   Print("  Risk Per Trade: ", g_config.risk_percent, "%");
   Print("  Phase 1 Features: ENABLED");
   Print("========================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("XAUBot Pro Shutting Down... Reason: ", reason);

   g_trend_filter.Deinit();
   g_emergency_stop.Deinit();

   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);

   Print("XAUBot Pro Deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if new bar formed (M15)
   static datetime last_bar_time = 0;
   datetime current_bar_time = iTime(_Symbol, PERIOD_M15, 0);

   if(current_bar_time == last_bar_time)
      return;  // Wait for new bar

   last_bar_time = current_bar_time;

   // === Main Trading Logic ===

   // 1. Check daily reset
   CheckDailyReset();

   // 2. Check emergency stop
   if(g_emergency_stop.CheckH4EmergencyReversal())
   {
      CloseAllPositions("H4 Emergency Reversal");
      return;
   }

   // 3. Check if locked
   if(g_emergency_stop.IsLocked())
   {
      Comment("🚨 EMERGENCY LOCKOUT: ", g_emergency_stop.GetStatus());
      return;
   }

   // 4. Check daily drawdown limit
   if(!CheckDailyDrawdownLimit())
   {
      Comment("⛔ DAILY DRAWDOWN LIMIT REACHED");
      return;
   }

   // 5. Manage existing positions
   ManagePositions();

   // 6. Check if can open new position
   if(!CanOpenNewPosition())
      return;

   // 7. Generate trading signal
   ENUM_TRADE_SIGNAL signal = GenerateTradingSignal();

   if(signal == SIGNAL_NONE || signal == SIGNAL_HOLD)
      return;

   // 8. Execute trade
   ExecuteTrade(signal);
}

//+------------------------------------------------------------------+
//| Apply Input Parameters to Config                                 |
//+------------------------------------------------------------------+
void ApplyInputParameters()
{
   g_config.capital_mode = InpCapitalMode;
   g_config.risk_percent = InpRiskPercent;
   g_config.max_daily_loss_percent = InpMaxDailyLoss;

   g_config.use_long_term_trend = InpUseLongTermTrend;
   g_config.apply_directional_bias = InpApplyDirectionalBias;
   g_config.enable_h4_reversal_lock = InpUseH4EmergencyStop;
   g_config.use_macro_features = InpUseMacroFeatures;

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

   // Parse skip hours
   ParseSkipHours(InpSkipHours);
}

//+------------------------------------------------------------------+
//| Parse Skip Hours from String                                      |
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

   // Reset if new day
   if(dt_current.day != dt_last.day)
   {
      g_daily_starting_balance = g_account.Balance();
      g_last_daily_reset = TimeCurrent();
      g_emergency_stop.ClearLockout();

      Print("📅 NEW DAY RESET: Starting Balance = $", g_daily_starting_balance);
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
      Print("⛔ DAILY DRAWDOWN LIMIT REACHED: Loss=$", daily_loss, " Max=$", max_loss);
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
   // Check max positions
   int open_positions = CountOpenPositions();
   if(open_positions >= g_config.max_positions)
      return false;

   // Check cooldown
   if(g_config.use_cooldown)
   {
      datetime cooldown_time = g_last_trade_time + g_config.cooldown_bars * PeriodSeconds(PERIOD_M15);
      if(TimeCurrent() < cooldown_time)
         return false;
   }

   // Check spread
   if(g_config.use_spread_filter)
   {
      double spread_pips = GetSpreadPips();
      if(spread_pips > g_config.max_spread_pips)
      {
         Print("Spread too wide: ", spread_pips, " pips");
         return false;
      }
   }

   // Check skip hours
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   for(int i = 0; i < ArraySize(g_config.skip_hours); i++)
   {
      if(dt.hour == g_config.skip_hours[i])
      {
         Print("Skip hour: ", dt.hour, ":00 WIB");
         return false;
      }
   }

   return true;
}

//+------------------------------------------------------------------+
//| Generate Trading Signal                                           |
//+------------------------------------------------------------------+
ENUM_TRADE_SIGNAL GenerateTradingSignal()
{
   // Simplified signal generation (replace with full SMC + ML logic)

   double confidence = 0.65;  // Placeholder - would come from ML model

   // Check trend filters
   ENUM_TRADE_SIGNAL signal = DetermineTrendDirection();

   if(signal == SIGNAL_NONE)
      return SIGNAL_NONE;

   // Phase 1: Check long-term trend filter
   if(!g_trend_filter.CheckLongTermTrend(signal))
      return SIGNAL_NONE;

   // Phase 1: Apply directional bias
   confidence = ApplyDirectionalBias(confidence, signal);

   // Check confidence threshold
   if(confidence < g_config.confidence_threshold)
      return SIGNAL_NONE;

   return signal;
}

//+------------------------------------------------------------------+
//| Determine Trend Direction (Simplified)                           |
//+------------------------------------------------------------------+
ENUM_TRADE_SIGNAL DetermineTrendDirection()
{
   // Check short-term trend (EMA20 H1)
   double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ema20_h1 = g_trend_filter.GetEMA20_H1();

   if(ema20_h1 == 0)
      return SIGNAL_NONE;

   // Simple logic: Above EMA20 = BUY, Below EMA20 = SELL
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

   // Calculate SL/TP
   double sl_pips = atr * g_config.sl_atr_multiplier * 10000;  // Convert to pips
   double tp_pips = sl_pips * g_config.tp_risk_reward;

   // Calculate lot size
   double lot = CalculateLotSize(sl_pips);

   // Get entry price
   double entry_price = (signal == SIGNAL_BUY) ?
                        SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                        SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // Calculate SL/TP prices
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

   // Normalize prices
   sl_price = NormalizeDouble(sl_price, _Digits);
   tp_price = NormalizeDouble(tp_price, _Digits);

   // Execute order
   bool result = false;
   if(signal == SIGNAL_BUY)
      result = g_trade.Buy(lot, _Symbol, entry_price, sl_price, tp_price, "XAUBot BUY");
   else
      result = g_trade.Sell(lot, _Symbol, entry_price, sl_price, tp_price, "XAUBot SELL");

   if(result)
   {
      g_last_trade_time = TimeCurrent();
      Print("✅ Trade Executed: ", EnumToString(signal), " | Lot: ", lot, " | SL: ", sl_pips, " pips | TP: ", tp_pips, " pips");
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

   // Normalize to broker's lot step
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathFloor(lot / lot_step) * lot_step;
   lot = MathMax(lot, min_lot);
   lot = MathMin(lot, max_lot);

   return lot;
}

//+------------------------------------------------------------------+
//| Manage Existing Positions                                         |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!g_position.SelectByIndex(i))
         continue;

      if(g_position.Symbol() != _Symbol)
         continue;

      if(g_position.Magic() != InpMagicNumber)
         continue;

      // Smart Breakeven
      if(g_config.use_smart_breakeven)
      {
         CheckSmartBreakeven(g_position.Ticket());
      }
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

   // Check if profit reached trigger
   if(profit_pips >= g_config.breakeven_trigger_pips)
   {
      // Check if SL not already at breakeven
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
      if(!g_position.SelectByIndex(i))
         continue;

      if(g_position.Symbol() != _Symbol)
         continue;

      if(g_position.Magic() != InpMagicNumber)
         continue;

      g_trade.PositionClose(g_position.Ticket());
   }
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
   return (ask - bid) / _Point / 10;  // Convert to pips
}

//+------------------------------------------------------------------+
