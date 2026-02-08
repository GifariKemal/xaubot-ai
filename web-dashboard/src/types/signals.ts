export interface Signal {
  id: number;
  signal_time: string;
  signal_type: "BUY" | "SELL" | "HOLD";
  confidence: number;
  executed: boolean;
  execution_reason: string;
  regime: string;
  session: string;
  smc_signal: string;
  ml_signal: string;
  entry_price: number;
  sl_price: number;
  tp_price: number;
}

export interface SignalStats {
  total: number;
  executed: number;
  executionRate: number;
  avgConfidence: number;
  byType: Record<string, number>;
}

export interface SignalsResponse {
  signals: Signal[];
  total: number;
  page: number;
  limit: number;
}
