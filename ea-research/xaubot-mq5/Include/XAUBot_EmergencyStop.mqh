//+------------------------------------------------------------------+
//|                                      XAUBot_EmergencyStop.mqh    |
//|                   Phase 1 Enhancement: H4 Emergency Reversal Stop |
//|                     Inspired by: Gold 1 Minute Grid EA           |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI"
#property version   "1.00"
#property strict

#include "XAUBot_Config.mqh"

//+------------------------------------------------------------------+
//| Emergency Stop Class                                              |
//+------------------------------------------------------------------+
class CEmergencyStop
{
private:
   string   m_symbol;
   datetime m_lockout_until;
   bool     m_is_locked;

   // Detection methods
   bool     DetectH4BearishEngulfing();
   bool     DetectH4BullishEngulfing();
   bool     DetectH4PinBar(bool &is_bearish);
   bool     DetectH4EMADeathCross();

public:
   CEmergencyStop();
   ~CEmergencyStop();

   bool     Init(string symbol);
   void     Deinit();

   // Main functions
   bool     CheckH4EmergencyReversal();
   bool     IsLocked();
   void     SetLockout(int hours);
   void     ClearLockout();

   // Status
   string   GetStatus();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CEmergencyStop::CEmergencyStop()
{
   m_lockout_until = 0;
   m_is_locked = false;
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CEmergencyStop::~CEmergencyStop()
{
   Deinit();
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CEmergencyStop::Init(string symbol)
{
   m_symbol = symbol;
   m_lockout_until = 0;
   m_is_locked = false;

   Print("Emergency Stop System Initialized");
   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CEmergencyStop::Deinit()
{
   // Nothing to cleanup
}

//+------------------------------------------------------------------+
//| Check H4 Emergency Reversal (Phase 1 Enhancement)                |
//| Detects major reversal patterns on H4 → Emergency exit           |
//| Inspired by: Gold Grid EA's H4 reversal safety lock              |
//+------------------------------------------------------------------+
bool CEmergencyStop::CheckH4EmergencyReversal()
{
   if(!g_config.enable_h4_reversal_lock)
      return false;

   // Check if already in lockout
   if(IsLocked())
      return false;

   // Pattern 1: Bearish Engulfing
   if(DetectH4BearishEngulfing())
   {
      Print("🚨 H4 BEARISH ENGULFING DETECTED — EMERGENCY EXIT!");
      SetLockout(4);  // Lock trading for 4 hours (1 H4 candle)
      return true;
   }

   // Pattern 2: Bullish Engulfing
   if(DetectH4BullishEngulfing())
   {
      Print("🚨 H4 BULLISH ENGULFING DETECTED — EMERGENCY EXIT!");
      SetLockout(4);
      return true;
   }

   // Pattern 3: Pin Bar (long wick reversal)
   bool is_bearish_pin;
   if(DetectH4PinBar(is_bearish_pin))
   {
      Print("🚨 H4 PIN BAR DETECTED (", (is_bearish_pin ? "BEARISH" : "BULLISH"), ") — EMERGENCY EXIT!");
      SetLockout(4);
      return true;
   }

   // Pattern 4: EMA Death Cross (EMA20 crosses EMA50)
   if(DetectH4EMADeathCross())
   {
      Print("🚨 H4 EMA DEATH CROSS — EMERGENCY EXIT!");
      SetLockout(8);  // Longer lockout (2 H4 candles)
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Detect H4 Bearish Engulfing                                       |
//+------------------------------------------------------------------+
bool CEmergencyStop::DetectH4BearishEngulfing()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(m_symbol, PERIOD_H4, 0, 3, rates) < 3)
      return false;

   // Current candle (index 0)
   double open0  = rates[0].open;
   double close0 = rates[0].close;
   double high0  = rates[0].high;
   double low0   = rates[0].low;

   // Previous candle (index 1)
   double open1  = rates[1].open;
   double close1 = rates[1].close;
   double high1  = rates[1].high;
   double low1   = rates[1].low;

   // Bearish engulfing conditions:
   // 1. Previous candle is bullish (close > open)
   // 2. Current candle is bearish (close < open)
   // 3. Current opens above previous close
   // 4. Current closes below previous open
   // 5. Current body engulfs previous body

   bool prev_bullish = close1 > open1;
   bool curr_bearish = close0 < open0;
   bool opens_above = open0 > close1;
   bool closes_below = close0 < open1;

   double prev_body = MathAbs(close1 - open1);
   double curr_body = MathAbs(close0 - open0);
   bool engulfs = curr_body > prev_body * 1.2;  // Current 20% larger

   return (prev_bullish && curr_bearish && opens_above && closes_below && engulfs);
}

//+------------------------------------------------------------------+
//| Detect H4 Bullish Engulfing                                       |
//+------------------------------------------------------------------+
bool CEmergencyStop::DetectH4BullishEngulfing()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(m_symbol, PERIOD_H4, 0, 3, rates) < 3)
      return false;

   double open0  = rates[0].open;
   double close0 = rates[0].close;
   double open1  = rates[1].open;
   double close1 = rates[1].close;

   bool prev_bearish = close1 < open1;
   bool curr_bullish = close0 > open0;
   bool opens_below = open0 < close1;
   bool closes_above = close0 > open1;

   double prev_body = MathAbs(close1 - open1);
   double curr_body = MathAbs(close0 - open0);
   bool engulfs = curr_body > prev_body * 1.2;

   return (prev_bearish && curr_bullish && opens_below && closes_above && engulfs);
}

//+------------------------------------------------------------------+
//| Detect H4 Pin Bar                                                 |
//+------------------------------------------------------------------+
bool CEmergencyStop::DetectH4PinBar(bool &is_bearish)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(m_symbol, PERIOD_H4, 0, 2, rates) < 2)
      return false;

   double open  = rates[0].open;
   double close = rates[0].close;
   double high  = rates[0].high;
   double low   = rates[0].low;

   double body = MathAbs(close - open);
   double upper_wick = high - MathMax(open, close);
   double lower_wick = MathMin(open, close) - low;
   double total_range = high - low;

   if(total_range == 0)
      return false;

   // Bearish pin bar: Long upper wick (rejection from top)
   bool bearish_pin = (upper_wick > body * 3.0) && (upper_wick > total_range * 0.6);

   // Bullish pin bar: Long lower wick (rejection from bottom)
   bool bullish_pin = (lower_wick > body * 3.0) && (lower_wick > total_range * 0.6);

   if(bearish_pin)
   {
      is_bearish = true;
      return true;
   }

   if(bullish_pin)
   {
      is_bearish = false;
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Detect H4 EMA Death Cross                                         |
//+------------------------------------------------------------------+
bool CEmergencyStop::DetectH4EMADeathCross()
{
   int ema20_handle = iMA(m_symbol, PERIOD_H4, 20, 0, MODE_EMA, PRICE_CLOSE);
   int ema50_handle = iMA(m_symbol, PERIOD_H4, 50, 0, MODE_EMA, PRICE_CLOSE);

   double ema20_buffer[], ema50_buffer[];
   ArraySetAsSeries(ema20_buffer, true);
   ArraySetAsSeries(ema50_buffer, true);

   if(CopyBuffer(ema20_handle, 0, 0, 2, ema20_buffer) < 2 ||
      CopyBuffer(ema50_handle, 0, 0, 2, ema50_buffer) < 2)
   {
      IndicatorRelease(ema20_handle);
      IndicatorRelease(ema50_handle);
      return false;
   }

   // Death cross: EMA20 crosses below EMA50
   bool was_above = ema20_buffer[1] > ema50_buffer[1];
   bool now_below = ema20_buffer[0] < ema50_buffer[0];

   IndicatorRelease(ema20_handle);
   IndicatorRelease(ema50_handle);

   return (was_above && now_below);
}

//+------------------------------------------------------------------+
//| Is Currently Locked                                               |
//+------------------------------------------------------------------+
bool CEmergencyStop::IsLocked()
{
   if(!m_is_locked)
      return false;

   // Check if lockout expired
   if(TimeCurrent() >= m_lockout_until)
   {
      ClearLockout();
      Print("Emergency lockout EXPIRED — Trading resumed");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Set Lockout Period                                                |
//+------------------------------------------------------------------+
void CEmergencyStop::SetLockout(int hours)
{
   m_lockout_until = TimeCurrent() + hours * 3600;
   m_is_locked = true;

   Print("Emergency lockout SET for ", hours, " hours until ", TimeToString(m_lockout_until));
}

//+------------------------------------------------------------------+
//| Clear Lockout                                                     |
//+------------------------------------------------------------------+
void CEmergencyStop::ClearLockout()
{
   m_lockout_until = 0;
   m_is_locked = false;
}

//+------------------------------------------------------------------+
//| Get Status                                                        |
//+------------------------------------------------------------------+
string CEmergencyStop::GetStatus()
{
   if(!m_is_locked)
      return "ACTIVE";

   int remaining_seconds = (int)(m_lockout_until - TimeCurrent());
   int remaining_hours = remaining_seconds / 3600;
   int remaining_mins = (remaining_seconds % 3600) / 60;

   return StringFormat("LOCKED (%dh %dm remaining)", remaining_hours, remaining_mins);
}

//+------------------------------------------------------------------+
