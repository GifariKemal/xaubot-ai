//+------------------------------------------------------------------+
//|                                                 XAUBot_SMC.mqh   |
//|                      Smart Money Concepts Implementation         |
//|          Order Blocks, Fair Value Gaps, BOS, CHoCH Detection     |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI"
#property version   "1.00"
#property strict

#include "XAUBot_Config.mqh"

//+------------------------------------------------------------------+
//| SMC Structure Definitions                                         |
//+------------------------------------------------------------------+
struct OrderBlock
{
   datetime time;
   double   high;
   double   low;
   bool     is_bullish;
   int      strength;     // 1-5 (5=strongest)
   bool     mitigated;
};

struct FairValueGap
{
   datetime time;
   double   high;
   double   low;
   bool     is_bullish;
   bool     filled;
};

struct BreakOfStructure
{
   datetime time;
   double   price;
   bool     is_bullish;  // True=BOS up, False=BOS down
};

//+------------------------------------------------------------------+
//| SMC Analyzer Class                                                |
//+------------------------------------------------------------------+
class CSMCAnalyzer
{
private:
   string   m_symbol;
   ENUM_TIMEFRAMES m_timeframe;

   OrderBlock      m_order_blocks[];
   FairValueGap    m_fvgs[];
   BreakOfStructure m_bos_history[];

   // Detection parameters
   int      m_lookback_bars;
   double   m_ob_min_body_percent;
   double   m_fvg_min_size_atr;

   // Helper functions
   bool     IsSwingHigh(int index, int left_bars, int right_bars);
   bool     IsSwingLow(int index, int left_bars, int right_bars);
   double   GetBodySize(const MqlRates &rate);
   double   GetCandleRange(const MqlRates &rate);
   int      CalculateOBStrength(const MqlRates &rates[], int ob_index);

public:
   CSMCAnalyzer();
   ~CSMCAnalyzer();

   bool     Init(string symbol, ENUM_TIMEFRAMES tf, int lookback=200);
   void     Deinit();

   // Main detection functions
   void     UpdateOrderBlocks();
   void     UpdateFairValueGaps();
   void     UpdateBreakOfStructure();

   // Signal generation
   bool     CheckBullishOBTouch(double current_price, double &ob_level);
   bool     CheckBearishOBTouch(double current_price, double &ob_level);
   bool     CheckBullishFVG(double current_price);
   bool     CheckBearishFVG(double current_price);
   bool     IsBullishBOS();
   bool     IsBearishBOS();

   // Getters
   int      GetActiveOBCount();
   int      GetActiveFVGCount();
   OrderBlock GetNearestBullishOB(double current_price);
   OrderBlock GetNearestBearishOB(double current_price);
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CSMCAnalyzer::CSMCAnalyzer()
{
   m_lookback_bars = 200;
   m_ob_min_body_percent = 0.6;  // Min 60% body size
   m_fvg_min_size_atr = 0.3;     // Min 30% of ATR
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CSMCAnalyzer::~CSMCAnalyzer()
{
   Deinit();
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CSMCAnalyzer::Init(string symbol, ENUM_TIMEFRAMES tf, int lookback=200)
{
   m_symbol = symbol;
   m_timeframe = tf;
   m_lookback_bars = lookback;

   ArrayResize(m_order_blocks, 0);
   ArrayResize(m_fvgs, 0);
   ArrayResize(m_bos_history, 0);

   Print("SMC Analyzer Initialized: ", symbol, " ", EnumToString(tf));
   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CSMCAnalyzer::Deinit()
{
   ArrayFree(m_order_blocks);
   ArrayFree(m_fvgs);
   ArrayFree(m_bos_history);
}

//+------------------------------------------------------------------+
//| Update Order Blocks                                               |
//+------------------------------------------------------------------+
void CSMCAnalyzer::UpdateOrderBlocks()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(m_symbol, m_timeframe, 0, m_lookback_bars, rates);
   if(copied < m_lookback_bars)
      return;

   // Clear old OBs
   ArrayResize(m_order_blocks, 0);

   // Scan for Order Blocks
   for(int i = 10; i < m_lookback_bars - 10; i++)
   {
      // Bullish Order Block: Strong down candle before up move
      if(rates[i].close < rates[i].open)  // Bearish candle
      {
         double body_size = GetBodySize(rates[i]);
         double candle_range = GetCandleRange(rates[i]);

         if(body_size > candle_range * m_ob_min_body_percent)  // Strong body
         {
            // Check if followed by bullish move
            bool has_bullish_move = false;
            for(int j = i - 1; j >= MathMax(0, i - 5); j--)
            {
               if(rates[j].close > rates[i].high)
               {
                  has_bullish_move = true;
                  break;
               }
            }

            if(has_bullish_move)
            {
               OrderBlock ob;
               ob.time = rates[i].time;
               ob.high = rates[i].high;
               ob.low = rates[i].low;
               ob.is_bullish = true;
               ob.strength = CalculateOBStrength(rates, i);
               ob.mitigated = false;

               // Check if price touched this OB
               double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
               if(current_price < ob.low)
                  ob.mitigated = true;

               int size = ArraySize(m_order_blocks);
               ArrayResize(m_order_blocks, size + 1);
               m_order_blocks[size] = ob;
            }
         }
      }

      // Bearish Order Block: Strong up candle before down move
      if(rates[i].close > rates[i].open)  // Bullish candle
      {
         double body_size = GetBodySize(rates[i]);
         double candle_range = GetCandleRange(rates[i]);

         if(body_size > candle_range * m_ob_min_body_percent)
         {
            bool has_bearish_move = false;
            for(int j = i - 1; j >= MathMax(0, i - 5); j--)
            {
               if(rates[j].close < rates[i].low)
               {
                  has_bearish_move = true;
                  break;
               }
            }

            if(has_bearish_move)
            {
               OrderBlock ob;
               ob.time = rates[i].time;
               ob.high = rates[i].high;
               ob.low = rates[i].low;
               ob.is_bullish = false;
               ob.strength = CalculateOBStrength(rates, i);
               ob.mitigated = false;

               double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
               if(current_price > ob.high)
                  ob.mitigated = true;

               int size = ArraySize(m_order_blocks);
               ArrayResize(m_order_blocks, size + 1);
               m_order_blocks[size] = ob;
            }
         }
      }
   }

   Print("SMC: Found ", ArraySize(m_order_blocks), " Order Blocks");
}

//+------------------------------------------------------------------+
//| Update Fair Value Gaps                                           |
//+------------------------------------------------------------------+
void CSMCAnalyzer::UpdateFairValueGaps()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(m_symbol, m_timeframe, 0, m_lookback_bars, rates);
   if(copied < m_lookback_bars)
      return;

   // Get ATR for minimum gap size
   int atr_handle = iATR(m_symbol, m_timeframe, 14);
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   double atr = atr_buffer[0];
   IndicatorRelease(atr_handle);

   ArrayResize(m_fvgs, 0);

   // Scan for FVGs (3-candle pattern)
   for(int i = 2; i < m_lookback_bars - 2; i++)
   {
      // Bullish FVG: Gap between candle[i+1].high and candle[i-1].low
      double gap_bullish = rates[i - 1].low - rates[i + 1].high;

      if(gap_bullish > atr * m_fvg_min_size_atr)
      {
         FairValueGap fvg;
         fvg.time = rates[i].time;
         fvg.high = rates[i - 1].low;
         fvg.low = rates[i + 1].high;
         fvg.is_bullish = true;
         fvg.filled = false;

         // Check if filled
         double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
         if(current_price >= fvg.low && current_price <= fvg.high)
            fvg.filled = true;

         int size = ArraySize(m_fvgs);
         ArrayResize(m_fvgs, size + 1);
         m_fvgs[size] = fvg;
      }

      // Bearish FVG: Gap between candle[i+1].low and candle[i-1].high
      double gap_bearish = rates[i + 1].low - rates[i - 1].high;

      if(gap_bearish > atr * m_fvg_min_size_atr)
      {
         FairValueGap fvg;
         fvg.time = rates[i].time;
         fvg.high = rates[i + 1].low;
         fvg.low = rates[i - 1].high;
         fvg.is_bullish = false;
         fvg.filled = false;

         double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
         if(current_price >= fvg.low && current_price <= fvg.high)
            fvg.filled = true;

         int size = ArraySize(m_fvgs);
         ArrayResize(m_fvgs, size + 1);
         m_fvgs[size] = fvg;
      }
   }

   Print("SMC: Found ", ArraySize(m_fvgs), " Fair Value Gaps");
}

//+------------------------------------------------------------------+
//| Update Break of Structure                                         |
//+------------------------------------------------------------------+
void CSMCAnalyzer::UpdateBreakOfStructure()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(m_symbol, m_timeframe, 0, 100, rates);
   if(copied < 100)
      return;

   // Find recent swing highs and lows
   double last_swing_high = 0;
   double last_swing_low = 0;

   for(int i = 10; i < 50; i++)
   {
      if(IsSwingHigh(i, 5, 5))
      {
         last_swing_high = rates[i].high;
         break;
      }
   }

   for(int i = 10; i < 50; i++)
   {
      if(IsSwingLow(i, 5, 5))
      {
         last_swing_low = rates[i].low;
         break;
      }
   }

   if(last_swing_high == 0 || last_swing_low == 0)
      return;

   double current_price = SymbolInfoDouble(m_symbol, SYMBOL_BID);

   // Bullish BOS: Price breaks above recent swing high
   if(current_price > last_swing_high)
   {
      BreakOfStructure bos;
      bos.time = TimeCurrent();
      bos.price = last_swing_high;
      bos.is_bullish = true;

      int size = ArraySize(m_bos_history);
      ArrayResize(m_bos_history, size + 1);
      m_bos_history[size] = bos;
   }

   // Bearish BOS: Price breaks below recent swing low
   if(current_price < last_swing_low)
   {
      BreakOfStructure bos;
      bos.time = TimeCurrent();
      bos.price = last_swing_low;
      bos.is_bullish = false;

      int size = ArraySize(m_bos_history);
      ArrayResize(m_bos_history, size + 1);
      m_bos_history[size] = bos;
   }
}

//+------------------------------------------------------------------+
//| Check Bullish OB Touch                                           |
//+------------------------------------------------------------------+
bool CSMCAnalyzer::CheckBullishOBTouch(double current_price, double &ob_level)
{
   for(int i = 0; i < ArraySize(m_order_blocks); i++)
   {
      if(!m_order_blocks[i].is_bullish || m_order_blocks[i].mitigated)
         continue;

      // Check if price is within OB zone
      if(current_price >= m_order_blocks[i].low && current_price <= m_order_blocks[i].high)
      {
         ob_level = m_order_blocks[i].low;
         return true;
      }
   }

   return false;
}

//+------------------------------------------------------------------+
//| Check Bearish OB Touch                                           |
//+------------------------------------------------------------------+
bool CSMCAnalyzer::CheckBearishOBTouch(double current_price, double &ob_level)
{
   for(int i = 0; i < ArraySize(m_order_blocks); i++)
   {
      if(m_order_blocks[i].is_bullish || m_order_blocks[i].mitigated)
         continue;

      if(current_price >= m_order_blocks[i].low && current_price <= m_order_blocks[i].high)
      {
         ob_level = m_order_blocks[i].high;
         return true;
      }
   }

   return false;
}

//+------------------------------------------------------------------+
//| Helper Functions                                                  |
//+------------------------------------------------------------------+
bool CSMCAnalyzer::IsSwingHigh(int index, int left_bars, int right_bars)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(m_symbol, m_timeframe, 0, index + left_bars + 1, rates) < index + left_bars + 1)
      return false;

   double pivot_high = rates[index].high;

   for(int i = 1; i <= left_bars; i++)
      if(rates[index + i].high >= pivot_high)
         return false;

   for(int i = 1; i <= right_bars; i++)
      if(rates[index - i].high >= pivot_high)
         return false;

   return true;
}

bool CSMCAnalyzer::IsSwingLow(int index, int left_bars, int right_bars)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(m_symbol, m_timeframe, 0, index + left_bars + 1, rates) < index + left_bars + 1)
      return false;

   double pivot_low = rates[index].low;

   for(int i = 1; i <= left_bars; i++)
      if(rates[index + i].low <= pivot_low)
         return false;

   for(int i = 1; i <= right_bars; i++)
      if(rates[index - i].low <= pivot_low)
         return false;

   return true;
}

double CSMCAnalyzer::GetBodySize(const MqlRates &rate)
{
   return MathAbs(rate.close - rate.open);
}

double CSMCAnalyzer::GetCandleRange(const MqlRates &rate)
{
   return rate.high - rate.low;
}

int CSMCAnalyzer::CalculateOBStrength(const MqlRates &rates[], int ob_index)
{
   double body_size = GetBodySize(rates[ob_index]);
   double candle_range = GetCandleRange(rates[ob_index]);

   if(candle_range == 0)
      return 1;

   double body_percent = body_size / candle_range;

   if(body_percent > 0.9) return 5;
   if(body_percent > 0.8) return 4;
   if(body_percent > 0.7) return 3;
   if(body_percent > 0.6) return 2;
   return 1;
}

bool CSMCAnalyzer::IsBullishBOS()
{
   int size = ArraySize(m_bos_history);
   if(size == 0)
      return false;

   return m_bos_history[size - 1].is_bullish;
}

bool CSMCAnalyzer::IsBearishBOS()
{
   int size = ArraySize(m_bos_history);
   if(size == 0)
      return false;

   return !m_bos_history[size - 1].is_bullish;
}

int CSMCAnalyzer::GetActiveOBCount()
{
   int count = 0;
   for(int i = 0; i < ArraySize(m_order_blocks); i++)
   {
      if(!m_order_blocks[i].mitigated)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
