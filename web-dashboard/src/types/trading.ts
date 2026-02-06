// Trading data types

export interface TradingStatus {
  timestamp: string;
  connected: boolean;

  // Price
  price: number;
  spread: number;
  priceChange: number;
  priceHistory: number[];

  // Account
  balance: number;
  equity: number;
  profit: number;
  equityHistory: number[];
  balanceHistory: number[];

  // Session
  session: string;
  isGoldenTime: boolean;
  canTrade: boolean;

  // Risk
  dailyLoss: number;
  dailyProfit: number;
  consecutiveLosses: number;
  riskPercent: number;

  // Signals
  smc: {
    signal: string;
    confidence: number;
    reason: string;
  };
  ml: {
    signal: string;
    confidence: number;
    buyProb: number;
    sellProb: number;
  };
  regime: {
    name: string;
    volatility: number;
    confidence: number;
  };

  // Positions
  positions: Position[];

  // Log
  logs: LogEntry[];
}

export interface Position {
  ticket: number;
  type: 'BUY' | 'SELL';
  volume: number;
  priceOpen: number;
  profit: number;
}

export interface LogEntry {
  time: string;
  level: 'info' | 'warn' | 'error' | 'trade';
  message: string;
}
