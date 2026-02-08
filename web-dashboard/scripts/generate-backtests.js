/**
 * Script to generate src/data/backtests.ts from backtest result log files.
 * Scans backtests/*_results/ for the most recent .log file in each directory,
 * parses performance metrics and trade log, and outputs static TS data.
 *
 * Run: node scripts/generate-backtests.js
 */
const fs = require("fs");
const path = require("path");

const BACKTESTS_DIR = path.resolve(__dirname, "..", "..", "backtests");
const OUT = path.join(__dirname, "..", "src", "data", "backtests.ts");

function escapeForTemplate(str) {
  return str.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
}

function parseMetric(text, pattern) {
  const m = text.match(pattern);
  return m ? m[1].trim() : null;
}

function parseNumber(text, pattern) {
  const val = parseMetric(text, pattern);
  if (!val) return 0;
  return parseFloat(val.replace(/[,$]/g, "")) || 0;
}

function parsePercent(text, pattern) {
  const val = parseMetric(text, pattern);
  if (!val) return 0;
  return parseFloat(val.replace("%", "")) || 0;
}

function parseExitReasons(text) {
  const section = text.match(/--- EXIT REASON(?:S| BREAKDOWN) ---\n([\s\S]*?)(?=\n---|\n\n\n)/);
  if (!section) return [];
  const lines = section[1].trim().split("\n");
  return lines.map((line) => {
    const m = line.match(/^\s*(\S+)\s*:\s*(\d+)\s*\(\s*([\d.]+)%\)/);
    if (!m) return null;
    return { reason: m[1], count: parseInt(m[2]), pct: parseFloat(m[3]) };
  }).filter(Boolean);
}

function parseDirectionBreakdown(text) {
  const section = text.match(/--- DIRECTION(?:\s+BREAKDOWN)? ---\n([\s\S]*?)(?=\n---|\n\n\n)/);
  if (!section) return [];
  const lines = section[1].trim().split("\n");
  return lines.map((line) => {
    const m = line.match(/^\s*(BUY|SELL):\s*(\d+)\s*trades?,\s*([\d.]+)%\s*WR,\s*\$\s*([-\d,.]+)/);
    if (!m) return null;
    return { direction: m[1], trades: parseInt(m[2]), winRate: parseFloat(m[3]), pnl: parseFloat(m[4].replace(/,/g, "")) };
  }).filter(Boolean);
}

function parseSessionBreakdown(text) {
  const section = text.match(/--- SESSION BREAKDOWN ---\n([\s\S]*?)(?=\n---|\n\n\n)/);
  if (!section) return [];
  const lines = section[1].trim().split("\n");
  return lines.map((line) => {
    const m = line.match(/^\s*(.+?)\s*:\s*(\d+)\s*trades?,\s*([\d.]+)%\s*WR,\s*\$\s*([-\d,.]+)/);
    if (!m) return null;
    return { session: m[1].trim(), trades: parseInt(m[2]), winRate: parseFloat(m[3]), pnl: parseFloat(m[4].replace(/,/g, "")) };
  }).filter(Boolean);
}

function parseTrades(text) {
  const section = text.match(/--- TRADE LOG ---\n.*\n-+\n([\s\S]*?)$/);
  if (!section) return [];
  const lines = section[1].trim().split("\n");
  return lines.slice(0, 500).map((line) => {
    // Format: #  date time  DIR  entry  exit  P/L  result  exit_reason  conf  mode  session
    const m = line.match(
      /^\s*(\d+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+(BUY|SELL)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+(WIN|LOSS)\s+(\S+)\s+(\d+)%\s+(\S+)\s+(.+)$/
    );
    if (!m) return null;
    return {
      num: parseInt(m[1]),
      time: m[2],
      dir: m[3],
      entry: parseFloat(m[4]),
      exit: parseFloat(m[5]),
      pnl: parseFloat(m[6]),
      result: m[7],
      exitReason: m[8],
      conf: parseInt(m[9]),
      mode: m[10],
      session: m[11].trim(),
    };
  }).filter(Boolean);
}

function formatName(dirName) {
  // "01_smc_only_results" -> "SMC Only"
  return dirName
    .replace(/_results$/, "")
    .replace(/^\d+_/, "")
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// Find all backtest result directories
const resultDirs = fs
  .readdirSync(BACKTESTS_DIR)
  .filter((d) => d.endsWith("_results") && fs.statSync(path.join(BACKTESTS_DIR, d)).isDirectory())
  .sort();

console.log(`Found ${resultDirs.length} backtest result directories`);

const results = [];

for (const dir of resultDirs) {
  const dirPath = path.join(BACKTESTS_DIR, dir);
  const logFiles = fs
    .readdirSync(dirPath)
    .filter((f) => f.endsWith(".log"))
    .sort()
    .reverse(); // most recent first

  if (logFiles.length === 0) {
    console.warn(`  SKIP ${dir}: no .log files`);
    continue;
  }

  const logFile = logFiles[0];
  const logPath = path.join(dirPath, logFile);
  const text = fs.readFileSync(logPath, "utf-8");

  const idMatch = dir.match(/^(\d+)/);
  const id = idMatch ? parseInt(idMatch[1]) : results.length + 1;

  const result = {
    id,
    slug: dir.replace(/_results$/, ""),
    name: formatName(dir),
    logFile,
    generatedAt: parseMetric(text, /Generated:\s*(.+)/),
    period: parseMetric(text, /Period:\s*(.+)/),
    strategy: parseMetric(text, /Strategy:\s*(.+)/),
    // Support both verbose "Total Trades: 686" and compact "Trades: 683 | WR: 73.4%" formats
    totalTrades: parseNumber(text, /Total Trades:\s*([\d,]+)/) || parseNumber(text, /Trades:\s*([\d,]+)/),
    wins: parseNumber(text, /Wins:\s*([\d,]+)/),
    losses: parseNumber(text, /Losses:\s*([\d,]+)/),
    winRate: parsePercent(text, /Win Rate:\s*([\d.]+)%/) || parsePercent(text, /WR:\s*([\d.]+)%/),
    totalProfit: parseNumber(text, /Total Profit:\s*\$([\d,.]+)/),
    totalLoss: parseNumber(text, /Total Loss:\s*\$([\d,.]+)/),
    netPnl: parseNumber(text, /Net PnL:\s*\$\s*([-\d,.]+)/),
    profitFactor: parseNumber(text, /Profit Factor:\s*([\d.]+)/) || parseNumber(text, /PF:\s*([\d.]+)/),
    maxDrawdown: parsePercent(text, /Max Drawdown:\s*([\d.]+)%/) || parsePercent(text, /Max DD:\s*([\d.]+)%/),
    maxDrawdownUsd: parseNumber(text, /Max Drawdown:\s*[\d.]+%\s*\(\$([\d,.]+)\)/),
    avgWin: parseNumber(text, /Avg Win:\s*\$([\d,.]+)/),
    avgLoss: parseNumber(text, /Avg Loss:\s*\$([\d,.]+)/),
    expectancy: parseNumber(text, /Expectancy:\s*\$([-\d,.]+)/),
    sharpeRatio: parseNumber(text, /Sharpe Ratio:\s*([-\d.]+)/) || parseNumber(text, /Sharpe:\s*([-\d.]+)/),
    exitReasons: parseExitReasons(text),
    directionBreakdown: parseDirectionBreakdown(text),
    sessionBreakdown: parseSessionBreakdown(text),
    tradeCount: 0, // set below
  };

  // Parse trades (store just count for the static file - trades are big)
  const trades = parseTrades(text);
  result.tradeCount = trades.length;

  // Fix negative Net PnL (the regex may miss the sign)
  if (text.includes("Net PnL:") && text.match(/Net PnL:\s*-/)) {
    result.netPnl = -Math.abs(result.netPnl);
  }

  results.push(result);
  console.log(`  OK ${dir}: ${result.totalTrades} trades, ${result.winRate}% WR, $${result.netPnl} PnL`);
}

// Sort by id
results.sort((a, b) => a.id - b.id);

// Generate output
const outDir = path.dirname(OUT);
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

let output = `// AUTO-GENERATED — do not edit manually.
// Run: node scripts/generate-backtests.js

export interface ExitReason {
  reason: string;
  count: number;
  pct: number;
}

export interface DirectionBreakdown {
  direction: string;
  trades: number;
  winRate: number;
  pnl: number;
}

export interface SessionBreakdown {
  session: string;
  trades: number;
  winRate: number;
  pnl: number;
}

export interface BacktestResult {
  id: number;
  slug: string;
  name: string;
  logFile: string;
  generatedAt: string | null;
  period: string | null;
  strategy: string | null;
  totalTrades: number;
  wins: number;
  losses: number;
  winRate: number;
  totalProfit: number;
  totalLoss: number;
  netPnl: number;
  profitFactor: number;
  maxDrawdown: number;
  maxDrawdownUsd: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  sharpeRatio: number;
  exitReasons: ExitReason[];
  directionBreakdown: DirectionBreakdown[];
  sessionBreakdown: SessionBreakdown[];
  tradeCount: number;
}

export const backtestResults: BacktestResult[] = ${JSON.stringify(results, null, 2)};
`;

fs.writeFileSync(OUT, output, "utf-8");
console.log(`\nGenerated ${OUT} with ${results.length} backtest results.`);
