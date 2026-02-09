//+------------------------------------------------------------------+
//|                                    XAUBot_PositionManager.mqh    |
//|                Phase 2: Basket Management + Protect Positions     |
//|              Inspired by: Gold 1 Minute Grid EA                   |
//+------------------------------------------------------------------+
#property copyright "XAUBot AI"
#property version   "1.00"
#property strict

#include "XAUBot_Config.mqh"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//+------------------------------------------------------------------+
//| Position Basket Structure                                         |
//+------------------------------------------------------------------+
struct PositionBasket
{
   ulong    tickets[];
   datetime first_open_time;
   datetime last_open_time;
   double   total_profit_usd;
   double   total_lots;
   double   avg_entry_price;
   int      position_count;
   bool     has_protect;
   ulong    protect_ticket;
};

//+------------------------------------------------------------------+
//| Protect Position Structure                                        |
//+------------------------------------------------------------------+
struct ProtectPosition
{
   ulong    parent_ticket;
   ulong    protect_ticket;
   datetime created_time;
   double   protect_lot;
   int      layer;  // 1, 2, or 3
};

//+------------------------------------------------------------------+
//| Position Manager Class                                            |
//+------------------------------------------------------------------+
class CPositionManager
{
private:
   CTrade            m_trade;
   CPositionInfo     m_position;

   PositionBasket    m_baskets[];
   ProtectPosition   m_protects[];

   // Basket parameters
   int      m_basket_window_minutes;
   double   m_basket_tp_usd;

   // Protect parameters
   bool     m_enable_protect;
   double   m_protect_trigger_pips;
   double   m_protect_lot_multiplier;
   int      m_max_protect_layers;

   // Helper functions
   void     GroupPositionsIntoBaskets(int magic_number);
   double   CalculateBasketProfit(const PositionBasket &basket);
   bool     ShouldOpenProtect(ulong ticket, double &loss_pips);
   void     UpdateProtectPositions();

public:
   CPositionManager();
   ~CPositionManager();

   bool     Init(int magic_number);
   void     Deinit();

   // Basket management
   void     UpdateBaskets(int magic_number);
   bool     CheckBasketTP(int magic_number);
   void     CloseBasket(const PositionBasket &basket);

   // Protect logic
   void     CheckProtectTriggers(int magic_number);
   bool     OpenProtectPosition(ulong parent_ticket, double loss_pips);

   // Getters
   int      GetBasketCount() { return ArraySize(m_baskets); }
   int      GetProtectCount() { return ArraySize(m_protects); }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CPositionManager::CPositionManager()
{
   m_basket_window_minutes = 60;  // Group positions within 1 hour
   m_basket_tp_usd = 50.0;        // Close basket when total profit >= $50

   m_enable_protect = true;
   m_protect_trigger_pips = 30.0;      // Open protect after -30 pips loss
   m_protect_lot_multiplier = 0.5;     // Protect size = 50% of original
   m_max_protect_layers = 1;           // Max 1 protect per position (safe)
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CPositionManager::~CPositionManager()
{
   Deinit();
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CPositionManager::Init(int magic_number)
{
   m_trade.SetExpertMagicNumber(magic_number);
   m_trade.SetDeviationInPoints(10);
   m_trade.SetTypeFilling(ORDER_FILLING_FOK);

   ArrayResize(m_baskets, 0);
   ArrayResize(m_protects, 0);

   Print("Position Manager Initialized");
   Print("  Basket Window: ", m_basket_window_minutes, " minutes");
   Print("  Basket TP: $", m_basket_tp_usd);
   Print("  Protect Logic: ", (m_enable_protect ? "ENABLED" : "DISABLED"));

   return true;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CPositionManager::Deinit()
{
   ArrayFree(m_baskets);
   ArrayFree(m_protects);
}

//+------------------------------------------------------------------+
//| Update Baskets                                                    |
//+------------------------------------------------------------------+
void CPositionManager::UpdateBaskets(int magic_number)
{
   GroupPositionsIntoBaskets(magic_number);
}

//+------------------------------------------------------------------+
//| Group Positions into Baskets                                      |
//+------------------------------------------------------------------+
void CPositionManager::GroupPositionsIntoBaskets(int magic_number)
{
   ArrayResize(m_baskets, 0);

   ulong all_tickets[];
   datetime all_times[];
   int count = 0;

   // Collect all positions
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Magic() == magic_number)
         {
            ArrayResize(all_tickets, count + 1);
            ArrayResize(all_times, count + 1);

            all_tickets[count] = m_position.Ticket();
            all_times[count] = m_position.Time();
            count++;
         }
      }
   }

   if(count == 0)
      return;

   // Sort by time
   for(int i = 0; i < count - 1; i++)
   {
      for(int j = i + 1; j < count; j++)
      {
         if(all_times[i] > all_times[j])
         {
            datetime temp_time = all_times[i];
            all_times[i] = all_times[j];
            all_times[j] = temp_time;

            ulong temp_ticket = all_tickets[i];
            all_tickets[i] = all_tickets[j];
            all_tickets[j] = temp_ticket;
         }
      }
   }

   // Group into baskets (positions within time window)
   int basket_count = 0;
   PositionBasket current_basket;
   ArrayResize(current_basket.tickets, 0);
   current_basket.first_open_time = 0;
   current_basket.has_protect = false;

   for(int i = 0; i < count; i++)
   {
      if(ArraySize(current_basket.tickets) == 0)
      {
         // Start new basket
         ArrayResize(current_basket.tickets, 1);
         current_basket.tickets[0] = all_tickets[i];
         current_basket.first_open_time = all_times[i];
         current_basket.last_open_time = all_times[i];
      }
      else
      {
         // Check if within time window
         int time_diff_minutes = (int)((all_times[i] - current_basket.first_open_time) / 60);

         if(time_diff_minutes <= m_basket_window_minutes)
         {
            // Add to current basket
            int size = ArraySize(current_basket.tickets);
            ArrayResize(current_basket.tickets, size + 1);
            current_basket.tickets[size] = all_tickets[i];
            current_basket.last_open_time = all_times[i];
         }
         else
         {
            // Save current basket and start new one
            ArrayResize(m_baskets, basket_count + 1);
            m_baskets[basket_count] = current_basket;
            basket_count++;

            // Start new basket
            ArrayResize(current_basket.tickets, 1);
            current_basket.tickets[0] = all_tickets[i];
            current_basket.first_open_time = all_times[i];
            current_basket.last_open_time = all_times[i];
            current_basket.has_protect = false;
         }
      }
   }

   // Save last basket
   if(ArraySize(current_basket.tickets) > 0)
   {
      ArrayResize(m_baskets, basket_count + 1);
      m_baskets[basket_count] = current_basket;
   }
}

//+------------------------------------------------------------------+
//| Check Basket TP                                                   |
//+------------------------------------------------------------------+
bool CPositionManager::CheckBasketTP(int magic_number)
{
   if(!g_config.use_basket_management)
      return false;

   UpdateBaskets(magic_number);

   for(int i = 0; i < ArraySize(m_baskets); i++)
   {
      double basket_profit = CalculateBasketProfit(m_baskets[i]);

      if(basket_profit >= m_basket_tp_usd)
      {
         Print("📦 BASKET TP HIT: $", basket_profit, " (target: $", m_basket_tp_usd, ")");
         CloseBasket(m_baskets[i]);
         return true;
      }
   }

   return false;
}

//+------------------------------------------------------------------+
//| Calculate Basket Profit                                           |
//+------------------------------------------------------------------+
double CPositionManager::CalculateBasketProfit(const PositionBasket &basket)
{
   double total_profit = 0;

   for(int i = 0; i < ArraySize(basket.tickets); i++)
   {
      if(m_position.SelectByTicket(basket.tickets[i]))
      {
         total_profit += m_position.Profit() + m_position.Swap() + m_position.Commission();
      }
   }

   return total_profit;
}

//+------------------------------------------------------------------+
//| Close Basket                                                      |
//+------------------------------------------------------------------+
void CPositionManager::CloseBasket(const PositionBasket &basket)
{
   Print("Closing basket with ", ArraySize(basket.tickets), " positions...");

   for(int i = 0; i < ArraySize(basket.tickets); i++)
   {
      if(m_position.SelectByTicket(basket.tickets[i]))
      {
         m_trade.PositionClose(basket.tickets[i]);
         Print("  ✅ Closed ticket ", basket.tickets[i]);
      }
   }
}

//+------------------------------------------------------------------+
//| Check Protect Triggers                                            |
//+------------------------------------------------------------------+
void CPositionManager::CheckProtectTriggers(int magic_number)
{
   if(!m_enable_protect)
      return;

   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!m_position.SelectByIndex(i))
         continue;

      if(m_position.Magic() != magic_number)
         continue;

      // Check if already has protect
      bool has_protect = false;
      for(int j = 0; j < ArraySize(m_protects); j++)
      {
         if(m_protects[j].parent_ticket == m_position.Ticket())
         {
            has_protect = true;
            break;
         }
      }

      if(has_protect)
         continue;

      // Check if should open protect
      double loss_pips;
      if(ShouldOpenProtect(m_position.Ticket(), loss_pips))
      {
         OpenProtectPosition(m_position.Ticket(), loss_pips);
      }
   }

   UpdateProtectPositions();
}

//+------------------------------------------------------------------+
//| Should Open Protect                                               |
//+------------------------------------------------------------------+
bool CPositionManager::ShouldOpenProtect(ulong ticket, double &loss_pips)
{
   if(!m_position.SelectByTicket(ticket))
      return false;

   double open_price = m_position.PriceOpen();
   double current_price = m_position.PriceCurrent();
   double point = SymbolInfoDouble(m_position.Symbol(), SYMBOL_POINT);

   if(m_position.PositionType() == POSITION_TYPE_BUY)
      loss_pips = (open_price - current_price) / point / 10;  // Loss in pips
   else
      loss_pips = (current_price - open_price) / point / 10;

   if(loss_pips >= m_protect_trigger_pips)
   {
      Print("⚠️ Protect trigger for ticket ", ticket, ": Loss ", loss_pips, " pips");
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Open Protect Position                                             |
//+------------------------------------------------------------------+
bool CPositionManager::OpenProtectPosition(ulong parent_ticket, double loss_pips)
{
   if(!m_position.SelectByTicket(parent_ticket))
      return false;

   // Calculate protect lot size (50% of original)
   double original_lot = m_position.Volume();
   double protect_lot = original_lot * m_protect_lot_multiplier;

   // Normalize lot
   double min_lot = SymbolInfoDouble(m_position.Symbol(), SYMBOL_VOLUME_MIN);
   double lot_step = SymbolInfoDouble(m_position.Symbol(), SYMBOL_VOLUME_STEP);
   protect_lot = MathMax(protect_lot, min_lot);
   protect_lot = MathFloor(protect_lot / lot_step) * lot_step;

   // Get current price
   double entry_price;
   ENUM_POSITION_TYPE pos_type = m_position.PositionType();

   if(pos_type == POSITION_TYPE_BUY)
      entry_price = SymbolInfoDouble(m_position.Symbol(), SYMBOL_ASK);
   else
      entry_price = SymbolInfoDouble(m_position.Symbol(), SYMBOL_BID);

   // Calculate SL/TP (same as parent)
   double sl = m_position.StopLoss();
   double tp = m_position.TakeProfit();

   // Open protect position (same direction, better price)
   bool result = false;
   if(pos_type == POSITION_TYPE_BUY)
      result = m_trade.Buy(protect_lot, m_position.Symbol(), entry_price, sl, tp, "PROTECT_" + IntegerToString(parent_ticket));
   else
      result = m_trade.Sell(protect_lot, m_position.Symbol(), entry_price, sl, tp, "PROTECT_" + IntegerToString(parent_ticket));

   if(result)
   {
      ProtectPosition protect;
      protect.parent_ticket = parent_ticket;
      protect.protect_ticket = m_trade.ResultOrder();
      protect.created_time = TimeCurrent();
      protect.protect_lot = protect_lot;
      protect.layer = 1;

      int size = ArraySize(m_protects);
      ArrayResize(m_protects, size + 1);
      m_protects[size] = protect;

      Print("🛡️ PROTECT OPENED: Parent=", parent_ticket, " Protect=", protect.protect_ticket,
            " Lot=", protect_lot, " Loss=", loss_pips, " pips");

      return true;
   }

   Print("❌ Failed to open protect position: ", m_trade.ResultRetcodeDescription());
   return false;
}

//+------------------------------------------------------------------+
//| Update Protect Positions                                          |
//+------------------------------------------------------------------+
void CPositionManager::UpdateProtectPositions()
{
   // Remove protects if parent closed
   for(int i = ArraySize(m_protects) - 1; i >= 0; i--)
   {
      bool parent_exists = false;

      for(int j = 0; j < PositionsTotal(); j++)
      {
         if(m_position.SelectByIndex(j))
         {
            if(m_position.Ticket() == m_protects[i].parent_ticket)
            {
               parent_exists = true;
               break;
            }
         }
      }

      if(!parent_exists)
      {
         // Parent closed, remove protect from array
         Print("Parent ", m_protects[i].parent_ticket, " closed, removing protect tracking");

         // Shift array
         for(int k = i; k < ArraySize(m_protects) - 1; k++)
         {
            m_protects[k] = m_protects[k + 1];
         }
         ArrayResize(m_protects, ArraySize(m_protects) - 1);
      }
   }
}

//+------------------------------------------------------------------+
