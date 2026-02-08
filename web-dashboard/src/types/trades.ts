export interface Trade {
  id: number;
  ticket: number;
  direction: "BUY" | "SELL";
  entry_price: number;
  exit_price: number;
  lot_size: number;
  profit_usd: number;
  profit_pips: number;
  sl_price: number;
  tp_price: number;
  opened_at: string;
  closed_at: string;
  exit_reason: string;
  confidence: number;
  regime: string;
  session: string;
  duration_minutes: number;
}

export interface TradeStats {
  totalTrades: number;
  winRate: number;
  netProfit: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  bestTrade: number;
  worstTrade: number;
}

export interface EquityCurvePoint {
  time: string;
  profit: number;
  cumulative: number;
}

export interface TradesResponse {
  trades: Trade[];
  total: number;
  page: number;
  limit: number;
}
