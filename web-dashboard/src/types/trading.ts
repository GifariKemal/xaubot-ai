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
    updatedAt?: string;
  };
  ml: {
    signal: string;
    confidence: number;
    buyProb: number;
    sellProb: number;
    updatedAt?: string;
  };
  regime: {
    name: string;
    volatility: number;
    confidence: number;
    updatedAt?: string;
  };

  // Positions
  positions: Position[];

  // Log
  logs: LogEntry[];

  // Bot Settings
  settings?: BotSettings;

  // Entry Conditions
  h1Bias?: string;
  dynamicThreshold?: number;
  marketQuality?: string;
  marketScore?: number;
}

export interface BotSettings {
  capitalMode: string;
  capital: number;
  riskPerTrade: number;
  maxDailyLoss: number;
  maxPositions: number;
  maxLotSize: number;
  leverage: number;
  executionTF: string;
  trendTF: string;
  minRR: number;
  mlConfidence: number;
  cooldownSeconds: number;
  symbol: string;
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
