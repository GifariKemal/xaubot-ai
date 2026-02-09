//+------------------------------------------------------------------+
//|                                      XAUBot_MacroFeatures.mqh    |
//|                   Phase 2: Macro Correlation Features            |
//|                   Inspired by: AI Gold Sniper EA                  |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI"
#property version   "1.00"
#property strict

#include "XAUBot_Config.mqh"

//+------------------------------------------------------------------+
//| Macro Features Class                                              |
//+------------------------------------------------------------------+
class CMacroFeatures
{
private:
   // Symbols
   string   m_dxy_symbol;      // USD Index
   string   m_oil_symbol;      // Crude Oil (WTI)
   string   m_gold_symbol;     // Gold (XAUUSD)

   // Handles
   int      m_dxy_rsi_handle;
   int      m_oil_rsi_handle;

   // Correlation thresholds
   double   m_dxy_inverse_threshold;    // DXY up → Gold down (inverse)
   double   m_oil_positive_threshold;   // Oil up → Gold up (positive)

   // Helper functions
   double   GetSymbolReturn(string symbol, ENUM_TIMEFRAMES tf, int bars);
   double   GetSymbolRSI(string symbol, ENUM_TIMEFRAMES tf);
   bool     IsSymbolAvailable(string symbol);

public:
   CMacroFeatures();
   ~CMacroFeatures();

   bool     Init(string gold_symbol);
   void     Deinit();

   // Main check functions
   bool     CheckMacroConfirmation(ENUM_TRADE_SIGNAL signal);
   double   GetDXYInfluence();
   double   GetOilInfluence();

   // Detailed checks
   bool     IsDXYSupportingBuy();
   bool     IsDXYSupportingSell();
   bool     IsOilSupportingBuy();
   bool     IsOilSupportingSell();

   // Getters
   double   GetDXYReturn();
   double   GetOilReturn();
   string   GetMacroSummary();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CMacroFeatures::CMacroFeatures()
{
   // Symbol names (may vary by broker)
   m_dxy_symbol = "USDX";        // Try: USDX, DXY, US30, USIDX
   m_oil_symbol = "WTIUSD";      // Try: WTIUSD, CL, USO, OIL

   m_dxy_rsi_handle = INVALID_HANDLE;
   m_oil_rsi_handle = INVALID_HANDLE;

   m_dxy_inverse_threshold = 0.5;    // 0.5% DXY move
   m_oil_positive_threshold = 1.0;   // 1.0% Oil move
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CMacroFeatures::~CMacroFeatures()
{
   Deinit();
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CMacroFeatures::Init(string gold_symbol)
{
   m_gold_symbol = gold_symbol;

   // Try alternative symbol names if not available
   string dxy_alternatives[] = {"USDX", "DXY", "US30", "USIDX", "DXYUSD"};
   string oil_alternatives[] = {"WTIUSD", "CL", "XTIUSD", "OIL", "USOIL"};

   // Find available DXY symbol
   bool dxy_found = false;
   for(int i = 0; i < ArraySize(dxy_alternatives); i++)
   {
      if(IsSymbolAvailable(dxy_alternatives[i]))
      {
         m_dxy_symbol = dxy_alternatives[i];
         dxy_found = true;
         Print("✅ DXY Symbol Found: ", m_dxy_symbol);
         break;
      }
   }

   if(!dxy_found)
   {
      Print("⚠️ DXY symbol not available on broker - Macro features limited");
   }
   else
   {
      // Initialize DXY RSI
      m_dxy_rsi_handle = iRSI(m_dxy_symbol, PERIOD_H1, 14, PRICE_CLOSE);
      if(m_dxy_rsi_handle == INVALID_HANDLE)
         Print("⚠️ Failed to create DXY RSI indicator");
   }

   // Find available Oil symbol
   bool oil_found = false;
   for(int i = 0; i < ArraySize(oil_alternatives); i++)
   {
      if(IsSymbolAvailable(oil_alternatives[i]))
      {
         m_oil_symbol = oil_alternatives[i];
         oil_found = true;
         Print("✅ Oil Symbol Found: ", m_oil_symbol);
         break;
      }
   }

   if(!oil_found)
   {
      Print("⚠️ Oil symbol not available on broker - Macro features limited");
   }
   else
   {
      // Initialize Oil RSI
      m_oil_rsi_handle = iRSI(m_oil_symbol, PERIOD_H1, 14, PRICE_CLOSE);
      if(m_oil_rsi_handle == INVALID_HANDLE)
         Print("⚠️ Failed to create Oil RSI indicator");
   }

   Print("Macro Features Initialized");
   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CMacroFeatures::Deinit()
{
   if(m_dxy_rsi_handle != INVALID_HANDLE)
      IndicatorRelease(m_dxy_rsi_handle);
   if(m_oil_rsi_handle != INVALID_HANDLE)
      IndicatorRelease(m_oil_rsi_handle);
}

//+------------------------------------------------------------------+
//| Check Macro Confirmation                                          |
//+------------------------------------------------------------------+
bool CMacroFeatures::CheckMacroConfirmation(ENUM_TRADE_SIGNAL signal)
{
   if(!g_config.use_macro_features)
      return true;  // Feature disabled, pass

   bool dxy_ok = true;
   bool oil_ok = true;

   // Check DXY (inverse correlation)
   if(IsSymbolAvailable(m_dxy_symbol))
   {
      if(signal == SIGNAL_BUY)
         dxy_ok = IsDXYSupportingBuy();  // DXY should be weak
      else if(signal == SIGNAL_SELL)
         dxy_ok = IsDXYSupportingSell(); // DXY should be strong
   }

   // Check Oil (positive correlation)
   if(IsSymbolAvailable(m_oil_symbol))
   {
      if(signal == SIGNAL_BUY)
         oil_ok = IsOilSupportingBuy();  // Oil should be rising
      else if(signal == SIGNAL_SELL)
         oil_ok = IsOilSupportingSell(); // Oil should be falling
   }

   // Both should confirm (or be neutral)
   bool confirmed = dxy_ok && oil_ok;

   if(!confirmed)
   {
      Print("❌ Macro features NOT confirming ", EnumToString(signal));
      Print("  DXY: ", (dxy_ok ? "✅" : "❌"), " | Oil: ", (oil_ok ? "✅" : "❌"));
   }

   return confirmed;
}

//+------------------------------------------------------------------+
//| Is DXY Supporting Buy                                             |
//+------------------------------------------------------------------+
bool CMacroFeatures::IsDXYSupportingBuy()
{
   // Gold BUY → DXY should be falling or weak
   double dxy_return = GetDXYReturn();

   // Strong DXY rise blocks Gold BUY
   if(dxy_return > m_dxy_inverse_threshold)
   {
      Print("DXY too strong for Gold BUY: +", dxy_return, "%");
      return false;
   }

   return true;  // DXY falling or neutral = good for Gold BUY
}

//+------------------------------------------------------------------+
//| Is DXY Supporting Sell                                            |
//+------------------------------------------------------------------+
bool CMacroFeatures::IsDXYSupportingSell()
{
   // Gold SELL → DXY should be rising or strong
   double dxy_return = GetDXYReturn();

   // Strong DXY fall blocks Gold SELL
   if(dxy_return < -m_dxy_inverse_threshold)
   {
      Print("DXY too weak for Gold SELL: ", dxy_return, "%");
      return false;
   }

   return true;  // DXY rising or neutral = good for Gold SELL
}

//+------------------------------------------------------------------+
//| Is Oil Supporting Buy                                             |
//+------------------------------------------------------------------+
bool CMacroFeatures::IsOilSupportingBuy()
{
   // Gold BUY → Oil should be rising (risk-on)
   double oil_return = GetOilReturn();

   // Strong Oil fall blocks Gold BUY
   if(oil_return < -m_oil_positive_threshold)
   {
      Print("Oil too weak for Gold BUY: ", oil_return, "%");
      return false;
   }

   return true;  // Oil rising or neutral = good for Gold BUY
}

//+------------------------------------------------------------------+
//| Is Oil Supporting Sell                                            |
//+------------------------------------------------------------------+
bool CMacroFeatures::IsOilSupportingSell()
{
   // Gold SELL → Oil should be falling (risk-off)
   double oil_return = GetOilReturn();

   // Strong Oil rise blocks Gold SELL
   if(oil_return > m_oil_positive_threshold)
   {
      Print("Oil too strong for Gold SELL: +", oil_return, "%");
      return false;
   }

   return true;  // Oil falling or neutral = good for Gold SELL
}

//+------------------------------------------------------------------+
//| Get DXY Return                                                    |
//+------------------------------------------------------------------+
double CMacroFeatures::GetDXYReturn()
{
   if(!IsSymbolAvailable(m_dxy_symbol))
      return 0;

   return GetSymbolReturn(m_dxy_symbol, PERIOD_H1, 10);  // 10-bar (10h) return
}

//+------------------------------------------------------------------+
//| Get Oil Return                                                    |
//+------------------------------------------------------------------+
double CMacroFeatures::GetOilReturn()
{
   if(!IsSymbolAvailable(m_oil_symbol))
      return 0;

   return GetSymbolReturn(m_oil_symbol, PERIOD_H1, 10);
}

//+------------------------------------------------------------------+
//| Get Symbol Return                                                 |
//+------------------------------------------------------------------+
double CMacroFeatures::GetSymbolReturn(string symbol, ENUM_TIMEFRAMES tf, int bars)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   if(CopyRates(symbol, tf, 0, bars + 1, rates) < bars + 1)
      return 0;

   double price_now = rates[0].close;
   double price_before = rates[bars].close;

   if(price_before == 0)
      return 0;

   return ((price_now / price_before) - 1.0) * 100.0;  // Return in %
}

//+------------------------------------------------------------------+
//| Get Symbol RSI                                                    |
//+------------------------------------------------------------------+
double CMacroFeatures::GetSymbolRSI(string symbol, ENUM_TIMEFRAMES tf)
{
   int rsi_handle = iRSI(symbol, tf, 14, PRICE_CLOSE);
   if(rsi_handle == INVALID_HANDLE)
      return 50.0;

   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer, true);

   if(CopyBuffer(rsi_handle, 0, 0, 1, rsi_buffer) <= 0)
   {
      IndicatorRelease(rsi_handle);
      return 50.0;
   }

   double rsi = rsi_buffer[0];
   IndicatorRelease(rsi_handle);

   return rsi;
}

//+------------------------------------------------------------------+
//| Is Symbol Available                                               |
//+------------------------------------------------------------------+
bool CMacroFeatures::IsSymbolAvailable(string symbol)
{
   return SymbolSelect(symbol, true);
}

//+------------------------------------------------------------------+
//| Get DXY Influence                                                 |
//+------------------------------------------------------------------+
double CMacroFeatures::GetDXYInfluence()
{
   // Returns -1 (bearish for Gold) to +1 (bullish for Gold)
   double dxy_return = GetDXYReturn();

   // Inverse correlation: DXY up = Gold down
   return -dxy_return / 2.0;  // Normalize to [-1, +1]
}

//+------------------------------------------------------------------+
//| Get Oil Influence                                                 |
//+------------------------------------------------------------------+
double CMacroFeatures::GetOilInfluence()
{
   // Returns -1 (bearish for Gold) to +1 (bullish for Gold)
   double oil_return = GetOilReturn();

   // Positive correlation: Oil up = Gold up (risk-on)
   return oil_return / 2.0;  // Normalize to [-1, +1]
}

//+------------------------------------------------------------------+
//| Get Macro Summary                                                 |
//+------------------------------------------------------------------+
string CMacroFeatures::GetMacroSummary()
{
   string summary = "Macro: ";

   if(IsSymbolAvailable(m_dxy_symbol))
   {
      double dxy_ret = GetDXYReturn();
      summary += "DXY " + DoubleToString(dxy_ret, 2) + "% ";
   }
   else
   {
      summary += "DXY N/A ";
   }

   if(IsSymbolAvailable(m_oil_symbol))
   {
      double oil_ret = GetOilReturn();
      summary += "Oil " + DoubleToString(oil_ret, 2) + "%";
   }
   else
   {
      summary += "Oil N/A";
   }

   return summary;
}

//+------------------------------------------------------------------+
