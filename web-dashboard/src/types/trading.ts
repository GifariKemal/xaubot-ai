// Trading data types

export interface EntryFilter {
  name: string;
  passed: boolean;
  detail: string;
}

export interface RiskMode {
  mode: string;
  reason: string;
  recommendedLot: number;
  maxAllowedLot: number;
  totalLoss: number;
  maxTotalLoss: number;
  remainingDailyRisk: number;
}

export interface CooldownStatus {
  active: boolean;
  secondsRemaining: number;
  totalSeconds: number;
}

export interface TimeFilter {
  wibHour: number;
  isBlocked: boolean;
  blockedHours: number[];
}

export interface PositionDetail {
  ticket: number;
  peakProfit: number;
  drawdownFromPeak: number;
  momentum: number;
  tpProbability: number;
  reversalWarnings: number;
  stalls: number;
  tradeHours: number;
}

export interface AutoTrainerStatus {
  lastRetrain: string | null;
  currentAuc: number | null;
  minAucThreshold: number;
  hoursSinceRetrain: number;
  nextRetrainHour: number;
  modelsFitted: boolean;
}

export interface PerformanceStatus {
  loopCount: number;
  avgExecutionMs: number;
  uptimeHours: number;
  totalSessionTrades: number;
  totalSessionProfit: number;
}

export interface MarketCloseStatus {
  hoursToDailyClose: number;
  hoursToWeekendClose: number;
  nearWeekend: boolean;
  marketOpen: boolean;
}

export interface H1BiasDetails {
  bias: string;
  ema20: number;
  price: number;
}

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

  // === NEW: Extended monitoring ===
  entryFilters?: EntryFilter[];
  riskMode?: RiskMode;
  cooldown?: CooldownStatus;
  timeFilter?: TimeFilter;
  sessionMultiplier?: number;
  positionDetails?: PositionDetail[];
  autoTrainer?: AutoTrainerStatus;
  performance?: PerformanceStatus;
  marketClose?: MarketCloseStatus;
  h1BiasDetails?: H1BiasDetails;
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
