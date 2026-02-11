//+------------------------------------------------------------------+
//| XAUBot_Test_Simple.mq5                                           |
//| Simple version to test compilation                                |
//+------------------------------------------------------------------+
#property copyright "XAUBot Pro"
#property version   "1.00"

#include <Trade\Trade.mqh>

input double RiskPercent = 1.0;

CTrade trade;

int OnInit()
{
   Print("XAUBot Test Simple - Initialized");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   Print("XAUBot Test Simple - Stopped");
}

void OnTick()
{
   // Simple test - just print on every 100th tick
   static int tickCount = 0;
   tickCount++;

   if(tickCount % 100 == 0)
   {
      Print("Tick ", tickCount, " | Bid: ", SymbolInfoDouble(_Symbol, SYMBOL_BID));
   }
}
//+------------------------------------------------------------------+
