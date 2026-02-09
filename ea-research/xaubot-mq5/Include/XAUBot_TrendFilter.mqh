//+------------------------------------------------------------------+
//|                                          XAUBot_TrendFilter.mqh  |
//|                       Phase 1 Enhancement: Long-Term Trend Filter |
//|                  Inspired by: Gold 1 Minute EA (200 EMA 3 TFs)   |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI"
#property version   "1.00"
#property strict

#include "XAUBot_Config.mqh"

//+------------------------------------------------------------------+
//| Trend Filter Class                                                |
//+------------------------------------------------------------------+
class CTrendFilter
{
private:
   int      m_ema20_h1_handle;
   int      m_ema200_h1_handle;
   int      m_ema200_h4_handle;

   double   m_ema20_h1_buffer[];
   double   m_ema200_h1_buffer[];
   double   m_ema200_h4_buffer[];

   string   m_symbol;

public:
   CTrendFilter();
   ~CTrendFilter();

   bool     Init(string symbol);
   void     Deinit();

   // Main filter functions
   bool     CheckLongTermTrend(ENUM_TRADE_SIGNAL signal);
   bool     CheckShortTermTrend(ENUM_TRADE_SIGNAL signal);

   // Helper functions
   double   GetEMA20_H1();
   double   GetEMA200_H1();
   double   GetEMA200_H4();

   // Trend strength
   double   GetTrendStrength();
   bool     IsTrendingMarket();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CTrendFilter::CTrendFilter()
{
   m_ema20_h1_handle = INVALID_HANDLE;
   m_ema200_h1_handle = INVALID_HANDLE;
   m_ema200_h4_handle = INVALID_HANDLE;

   ArraySetAsSeries(m_ema20_h1_buffer, true);
   ArraySetAsSeries(m_ema200_h1_buffer, true);
   ArraySetAsSeries(m_ema200_h4_buffer, true);
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CTrendFilter::~CTrendFilter()
{
   Deinit();
}

//+------------------------------------------------------------------+
//| Initialize Indicators                                             |
//+------------------------------------------------------------------+
bool CTrendFilter::Init(string symbol)
{
   m_symbol = symbol;

   // EMA 20 on H1 (short-term trend from XAUBot)
   m_ema20_h1_handle = iMA(m_symbol, PERIOD_H1, 20, 0, MODE_EMA, PRICE_CLOSE);
   if(m_ema20_h1_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create EMA20 H1 indicator");
      return false;
   }

   // EMA 200 on H1 (long-term trend - Phase 1)
   m_ema200_h1_handle = iMA(m_symbol, PERIOD_H1, 200, 0, MODE_EMA, PRICE_CLOSE);
   if(m_ema200_h1_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create EMA200 H1 indicator");
      return false;
   }

   // EMA 200 on H4 (very long-term trend - Phase 1)
   m_ema200_h4_handle = iMA(m_symbol, PERIOD_H4, 200, 0, MODE_EMA, PRICE_CLOSE);
   if(m_ema200_h4_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create EMA200 H4 indicator");
      return false;
   }

   Print("Trend Filter Initialized: EMA20(H1), EMA200(H1), EMA200(H4)");
   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CTrendFilter::Deinit()
{
   if(m_ema20_h1_handle != INVALID_HANDLE)
      IndicatorRelease(m_ema20_h1_handle);
   if(m_ema200_h1_handle != INVALID_HANDLE)
      IndicatorRelease(m_ema200_h1_handle);
   if(m_ema200_h4_handle != INVALID_HANDLE)
      IndicatorRelease(m_ema200_h4_handle);
}

//+------------------------------------------------------------------+
//| Check Long-Term Trend (Phase 1 Enhancement)                      |
//| Logic: Price must be above/below 200 EMA on both H1 and H4       |
//| Inspired by: Gold 1 Minute EA (3-timeframe filter)               |
//+------------------------------------------------------------------+
bool CTrendFilter::CheckLongTermTrend(ENUM_TRADE_SIGNAL signal)
{
   if(!g_config.use_long_term_trend)
      return true;  // Filter disabled, pass

   // Get current price
   double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);

   // Get EMA values
   double ema200_h1 = GetEMA200_H1();
   double ema200_h4 = GetEMA200_H4();

   if(ema200_h1 == 0 || ema200_h4 == 0)
      return false;  // Data not ready

   // BUY: Price must be above both 200 EMAs (bullish trend)
   if(signal == SIGNAL_BUY)
   {
      bool above_h1 = current_price > ema200_h1;
      bool above_h4 = current_price > ema200_h4;

      if(!above_h1 || !above_h4)
      {
         Print("Long-term trend filter BLOCKED BUY: Price=", current_price,
               " EMA200_H1=", ema200_h1, " EMA200_H4=", ema200_h4);
         return false;
      }

      return true;
   }

   // SELL: Price must be below both 200 EMAs (bearish trend)
   else if(signal == SIGNAL_SELL)
   {
      bool below_h1 = current_price < ema200_h1;
      bool below_h4 = current_price < ema200_h4;

      if(!below_h1 || !below_h4)
      {
         Print("Long-term trend filter BLOCKED SELL: Price=", current_price,
               " EMA200_H1=", ema200_h1, " EMA200_H4=", ema200_h4);
         return false;
      }

      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Check Short-Term Trend (XAUBot Original)                         |
//| Logic: Price must be above/below 20 EMA on H1                    |
//+------------------------------------------------------------------+
bool CTrendFilter::CheckShortTermTrend(ENUM_TRADE_SIGNAL signal)
{
   double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   double ema20_h1 = GetEMA20_H1();

   if(ema20_h1 == 0)
      return false;

   if(signal == SIGNAL_BUY)
      return current_price > ema20_h1;
   else if(signal == SIGNAL_SELL)
      return current_price < ema20_h1;

   return false;
}

//+------------------------------------------------------------------+
//| Get EMA 20 H1                                                     |
//+------------------------------------------------------------------+
double CTrendFilter::GetEMA20_H1()
{
   if(CopyBuffer(m_ema20_h1_handle, 0, 0, 1, m_ema20_h1_buffer) <= 0)
      return 0;
   return m_ema20_h1_buffer[0];
}

//+------------------------------------------------------------------+
//| Get EMA 200 H1                                                    |
//+------------------------------------------------------------------+
double CTrendFilter::GetEMA200_H1()
{
   if(CopyBuffer(m_ema200_h1_handle, 0, 0, 1, m_ema200_h1_buffer) <= 0)
      return 0;
   return m_ema200_h1_buffer[0];
}

//+------------------------------------------------------------------+
//| Get EMA 200 H4                                                    |
//+------------------------------------------------------------------+
double CTrendFilter::GetEMA200_H4()
{
   if(CopyBuffer(m_ema200_h4_handle, 0, 0, 1, m_ema200_h4_buffer) <= 0)
      return 0;
   return m_ema200_h4_buffer[0];
}

//+------------------------------------------------------------------+
//| Get Trend Strength                                                |
//| Returns: 0-100 (0=ranging, 100=strong trend)                     |
//+------------------------------------------------------------------+
double CTrendFilter::GetTrendStrength()
{
   double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
   double ema20 = GetEMA20_H1();
   double ema200_h1 = GetEMA200_H1();

   if(ema20 == 0 || ema200_h1 == 0)
      return 0;

   // Calculate ATR for normalization
   int atr_handle = iATR(m_symbol, PERIOD_H1, 14);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);

   if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0)
   {
      IndicatorRelease(atr_handle);
      return 0;
   }

   double atr = atr_buffer[0];
   IndicatorRelease(atr_handle);

   if(atr == 0)
      return 0;

   // Trend strength = Distance from EMA20 to EMA200 / ATR
   double distance = MathAbs(ema20 - ema200_h1);
   double strength = (distance / atr) * 10.0;  // Scale to 0-100

   return MathMin(strength, 100.0);
}

//+------------------------------------------------------------------+
//| Is Trending Market                                                |
//| Returns true if market is in strong trend (not ranging)          |
//+------------------------------------------------------------------+
bool CTrendFilter::IsTrendingMarket()
{
   double strength = GetTrendStrength();
   return strength > 30.0;  // Threshold: 30% trend strength
}

//+------------------------------------------------------------------+
