//+------------------------------------------------------------------+
//|                                              XAUBot_Config.mqh   |
//|                                  XAUBot AI - MQ5 Edition v1.0    |
//|                     Based on research: 3 Commercial EAs + Python |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI"
#property link      "https://github.com/GifariKemal/xaubot-ai"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| Enums                                                             |
//+------------------------------------------------------------------+
enum ENUM_CAPITAL_MODE
{
   CAPITAL_MICRO,    // Micro: <$500 (2% risk)
   CAPITAL_SMALL,    // Small: $500-$10k (1.5% risk)
   CAPITAL_MEDIUM,   // Medium: $10k-$100k (0.5% risk)
   CAPITAL_LARGE     // Large: >$100k (0.25% risk)
};

enum ENUM_REGIME_STATE
{
   REGIME_LOW_VOL,      // Low Volatility (Safe)
   REGIME_MEDIUM_VOL,   // Medium Volatility (Normal)
   REGIME_HIGH_VOL,     // High Volatility (Reduce)
   REGIME_CRISIS        // Crisis (Sleep)
};

enum ENUM_TRADE_SIGNAL
{
   SIGNAL_NONE,   // No signal
   SIGNAL_BUY,    // Buy signal
   SIGNAL_SELL,   // Sell signal
   SIGNAL_HOLD    // Hold (no action)
};

enum ENUM_SESSION
{
   SESSION_SYDNEY,   // Sydney: 21:00-06:00 GMT
   SESSION_TOKYO,    // Tokyo: 00:00-09:00 GMT
   SESSION_LONDON,   // London: 08:00-17:00 GMT
   SESSION_NEWYORK,  // New York: 13:00-22:00 GMT
   SESSION_NONE      // Outside sessions
};

//+------------------------------------------------------------------+
//| Configuration Struct                                              |
//+------------------------------------------------------------------+
struct TradingConfig
{
   // Capital & Risk
   ENUM_CAPITAL_MODE  capital_mode;
   double             risk_percent;
   double             max_daily_loss_percent;
   double             initial_balance;

   // Timeframe
   ENUM_TIMEFRAMES    trading_tf;        // M15
   ENUM_TIMEFRAMES    trend_tf_h1;       // H1
   ENUM_TIMEFRAMES    trend_tf_h4;       // H4

   // ML/Signal Settings
   double             confidence_threshold;
   double             buy_bias_multiplier;    // Phase 1: 1.1 (10% boost)
   double             sell_bias_multiplier;   // Phase 1: 0.95 (5% penalty)

   // Entry Filters
   bool               use_regime_filter;
   bool               use_session_filter;
   bool               use_spread_filter;
   double             max_spread_pips;
   bool               use_cooldown;
   int                cooldown_bars;

   // Phase 1 Enhancements
   bool               use_long_term_trend;    // 200 EMA H1/H4 filter
   bool               use_h4_emergency_stop;  // H4 reversal detection
   bool               use_macro_features;     // DXY, Oil correlation
   bool               apply_directional_bias; // Gold BUY bias

   // Stop Loss & Take Profit
   double             sl_atr_multiplier;
   double             tp_risk_reward;
   bool               use_smart_breakeven;
   int                breakeven_trigger_pips;
   int                breakeven_lock_pips;

   // Position Management
   int                max_positions;
   double             max_exposure_percent;
   bool               use_basket_management;
   double             basket_tp_usd;

   // Emergency Stops
   bool               enable_daily_limit;
   bool               enable_h4_reversal_lock;

   // News Filter
   bool               filter_high_impact_news;
   int                skip_hours[];           // WIB hours to skip
};

//+------------------------------------------------------------------+
//| Global Configuration                                              |
//+------------------------------------------------------------------+
TradingConfig g_config;

//+------------------------------------------------------------------+
//| Initialize Configuration with Defaults                            |
//+------------------------------------------------------------------+
void InitConfig()
{
   // Detect capital mode based on balance
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(balance < 500)
      g_config.capital_mode = CAPITAL_MICRO;
   else if(balance < 10000)
      g_config.capital_mode = CAPITAL_SMALL;
   else if(balance < 100000)
      g_config.capital_mode = CAPITAL_MEDIUM;
   else
      g_config.capital_mode = CAPITAL_LARGE;

   // Set risk based on capital mode
   switch(g_config.capital_mode)
   {
      case CAPITAL_MICRO:
         g_config.risk_percent = 2.0;
         g_config.max_daily_loss_percent = 10.0;
         break;
      case CAPITAL_SMALL:
         g_config.risk_percent = 1.5;
         g_config.max_daily_loss_percent = 8.0;
         break;
      case CAPITAL_MEDIUM:
         g_config.risk_percent = 0.5;
         g_config.max_daily_loss_percent = 5.0;
         break;
      case CAPITAL_LARGE:
         g_config.risk_percent = 0.25;
         g_config.max_daily_loss_percent = 3.0;
         break;
   }

   g_config.initial_balance = balance;

   // Timeframes
   g_config.trading_tf = PERIOD_M15;
   g_config.trend_tf_h1 = PERIOD_H1;
   g_config.trend_tf_h4 = PERIOD_H4;

   // ML/Signal
   g_config.confidence_threshold = 0.55;  // 55% minimum confidence
   g_config.buy_bias_multiplier = 1.1;    // Phase 1: 10% BUY boost
   g_config.sell_bias_multiplier = 0.95;  // Phase 1: 5% SELL penalty

   // Entry Filters
   g_config.use_regime_filter = true;
   g_config.use_session_filter = true;
   g_config.use_spread_filter = true;
   g_config.max_spread_pips = 0.5;
   g_config.use_cooldown = true;
   g_config.cooldown_bars = 3;  // 3 bars (45 min on M15)

   // Phase 1 Enhancements
   g_config.use_long_term_trend = true;       // NEW: 200 EMA filter
   g_config.use_h4_emergency_stop = true;     // NEW: H4 reversal lock
   g_config.use_macro_features = true;        // NEW: DXY/Oil check
   g_config.apply_directional_bias = true;    // NEW: Gold BUY bias

   // SL/TP
   g_config.sl_atr_multiplier = 1.5;
   g_config.tp_risk_reward = 1.5;
   g_config.use_smart_breakeven = true;
   g_config.breakeven_trigger_pips = 20;
   g_config.breakeven_lock_pips = 5;

   // Position Management
   g_config.max_positions = 3;
   g_config.max_exposure_percent = 6.0;  // 3 positions × 2% risk
   g_config.use_basket_management = false;  // Phase 2 feature
   g_config.basket_tp_usd = 50.0;

   // Emergency
   g_config.enable_daily_limit = true;
   g_config.enable_h4_reversal_lock = true;

   // News Filter
   g_config.filter_high_impact_news = true;
   ArrayResize(g_config.skip_hours, 2);
   g_config.skip_hours[0] = 9;   // 09:00 WIB (skip)
   g_config.skip_hours[1] = 21;  // 21:00 WIB (skip)

   Print("XAUBot Config Initialized:");
   Print("  Capital Mode: ", EnumToString(g_config.capital_mode));
   Print("  Risk Per Trade: ", g_config.risk_percent, "%");
   Print("  Max Daily Loss: ", g_config.max_daily_loss_percent, "%");
   Print("  Phase 1 Features: ENABLED");
}

//+------------------------------------------------------------------+
//| Get Risk Percent Based on Regime                                 |
//+------------------------------------------------------------------+
double GetRiskPercent(ENUM_REGIME_STATE regime)
{
   double base_risk = g_config.risk_percent;

   switch(regime)
   {
      case REGIME_LOW_VOL:
         return base_risk * 1.0;  // Full risk
      case REGIME_MEDIUM_VOL:
         return base_risk * 1.0;  // Full risk
      case REGIME_HIGH_VOL:
         return base_risk * 0.5;  // Half risk
      case REGIME_CRISIS:
         return 0.0;              // No trading
   }

   return base_risk;
}

//+------------------------------------------------------------------+
//| Apply Directional Bias (Phase 1 Enhancement)                     |
//+------------------------------------------------------------------+
double ApplyDirectionalBias(double confidence, ENUM_TRADE_SIGNAL signal)
{
   if(!g_config.apply_directional_bias)
      return confidence;

   // Gold has long-term BUY bias (20-year uptrend)
   if(signal == SIGNAL_BUY)
      return MathMin(confidence * g_config.buy_bias_multiplier, 1.0);
   else if(signal == SIGNAL_SELL)
      return confidence * g_config.sell_bias_multiplier;

   return confidence;
}

//+------------------------------------------------------------------+
//| Color Definitions                                                 |
//+------------------------------------------------------------------+
#define CLR_BUY    clrLimeGreen
#define CLR_SELL   clrRed
#define CLR_OB     clrDodgerBlue
#define CLR_FVG    clrGold
#define CLR_BOS    clrMagenta

//+------------------------------------------------------------------+
//| Magic Numbers                                                     |
//+------------------------------------------------------------------+
#define MAGIC_XAUBOT_BUY   20260209   // Date-based magic
#define MAGIC_XAUBOT_SELL  20260210

//+------------------------------------------------------------------+
