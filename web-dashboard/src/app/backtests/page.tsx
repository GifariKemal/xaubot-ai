"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  FlaskConical,
  Trophy,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Activity,
  Layers,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/theme-toggle";
import { backtestResults, type BacktestResult } from "@/data/backtests";
import { formatUSD } from "@/lib/utils";
import { cn } from "@/lib/utils";

function MetricsGrid({ bt }: { bt: BacktestResult }) {
  const metrics = [
    { label: "Total Trades", value: bt.totalTrades, fmt: (v: number) => String(v), color: "text-apple-blue" },
    { label: "Win Rate", value: bt.winRate, fmt: (v: number) => `${v.toFixed(1)}%`, color: "text-apple-green" },
    { label: "Net PnL", value: bt.netPnl, fmt: (v: number) => formatUSD(v), color: bt.netPnl >= 0 ? "text-success" : "text-danger" },
    { label: "Profit Factor", value: bt.profitFactor, fmt: (v: number) => v.toFixed(2), color: "text-apple-purple" },
    { label: "Max Drawdown", value: bt.maxDrawdown, fmt: (v: number) => `${v.toFixed(1)}%`, color: "text-apple-orange" },
    { label: "Sharpe Ratio", value: bt.sharpeRatio, fmt: (v: number) => v.toFixed(2), color: "text-apple-cyan" },
    { label: "Avg Win", value: bt.avgWin, fmt: (v: number) => formatUSD(v), color: "text-success" },
    { label: "Avg Loss", value: bt.avgLoss, fmt: (v: number) => formatUSD(v), color: "text-danger" },
  ];

  return (
    <div className="grid grid-cols-4 gap-2">
      {metrics.map((m) => (
        <div key={m.label} className="glass rounded-lg p-3">
          <p className="text-xs text-muted-foreground mb-1">{m.label}</p>
          <p className={`text-lg font-bold font-number ${m.color}`}>{m.fmt(m.value)}</p>
        </div>
      ))}
    </div>
  );
}

function ExitReasonsBar({ bt }: { bt: BacktestResult }) {
  if (bt.exitReasons.length === 0) return null;
  const max = Math.max(...bt.exitReasons.map((r) => r.count));
  const colors = [
    "bar-blue", "bar-green", "bar-orange", "bar-red", "bar-purple", "bar-cyan",
    "bar-blue", "bar-green", "bar-orange", "bar-red", "bar-purple",
  ];

  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold mb-3">Exit Reasons</h3>
      <div className="space-y-2">
        {bt.exitReasons.map((r, i) => (
          <div key={r.reason} className="flex items-center gap-2 text-xs">
            <span className="w-28 text-muted-foreground truncate">{r.reason}</span>
            <div className="flex-1 h-4 bg-surface-light rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${colors[i % colors.length]} bar-animate-in`}
                style={{ width: `${(r.count / max) * 100}%`, animationDelay: `${i * 50}ms` }}
              />
            </div>
            <span className="w-8 text-right font-number">{r.count}</span>
            <span className="w-12 text-right font-number text-muted-foreground">{r.pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SessionBars({ bt }: { bt: BacktestResult }) {
  if (bt.sessionBreakdown.length === 0) return null;

  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold mb-3">Session Performance</h3>
      <div className="space-y-3">
        {bt.sessionBreakdown.map((s) => (
          <div key={s.session} className="flex items-center gap-3 text-xs">
            <span className="w-40 text-muted-foreground truncate">{s.session}</span>
            <Badge variant={s.pnl >= 0 ? "success" : "danger"} className="text-xs">
              {s.winRate}% WR
            </Badge>
            <span className="font-number">{s.trades} trades</span>
            <span className={`ml-auto font-number font-semibold ${s.pnl >= 0 ? "text-success" : "text-danger"}`}>
              {formatUSD(s.pnl)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ComparisonTable({ results }: { results: BacktestResult[] }) {
  const sorted = [...results].filter((r) => r.totalTrades > 0).sort((a, b) => b.netPnl - a.netPnl);

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Layers className="h-4 w-4 text-apple-purple" />
          Perbandingan Strategi
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground">
              <th className="text-left px-4 py-2 font-medium">#</th>
              <th className="text-left px-3 py-2 font-medium">Strategy</th>
              <th className="text-right px-3 py-2 font-medium">Trades</th>
              <th className="text-right px-3 py-2 font-medium">Win Rate</th>
              <th className="text-right px-3 py-2 font-medium">Net PnL</th>
              <th className="text-right px-3 py-2 font-medium">PF</th>
              <th className="text-right px-3 py-2 font-medium">Max DD</th>
              <th className="text-right px-3 py-2 font-medium">Sharpe</th>
              <th className="text-right px-3 py-2 font-medium">Expectancy</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((bt, i) => (
              <tr key={bt.id} className="border-b border-border/50 row-hover">
                <td className="px-4 py-2 font-number text-muted-foreground">{i + 1}</td>
                <td className="px-3 py-2 font-medium">{bt.name}</td>
                <td className="px-3 py-2 text-right font-number">{bt.totalTrades}</td>
                <td className="px-3 py-2 text-right font-number">{bt.winRate.toFixed(1)}%</td>
                <td className={`px-3 py-2 text-right font-number font-semibold ${bt.netPnl >= 0 ? "text-success" : "text-danger"}`}>
                  {formatUSD(bt.netPnl)}
                </td>
                <td className="px-3 py-2 text-right font-number">{bt.profitFactor.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-number text-apple-orange">{bt.maxDrawdown.toFixed(1)}%</td>
                <td className="px-3 py-2 text-right font-number">{bt.sharpeRatio.toFixed(2)}</td>
                <td className={`px-3 py-2 text-right font-number ${bt.expectancy >= 0 ? "text-success" : "text-danger"}`}>
                  {formatUSD(bt.expectancy)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function BacktestsPage() {
  const [selectedId, setSelectedId] = useState(backtestResults[0]?.id ?? 1);
  const [tab, setTab] = useState<"detail" | "compare">("detail");

  const validResults = useMemo(
    () => backtestResults.filter((r) => r.totalTrades > 0),
    []
  );

  const selected = useMemo(
    () => validResults.find((r) => r.id === selectedId) ?? validResults[0],
    [selectedId, validResults]
  );

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* Header */}
      <header className="shrink-0 w-full border-b border-border bg-white/70 dark:bg-white/[0.03] backdrop-blur-2xl">
        <div className="flex h-11 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface border border-border hover:border-primary/20 transition-colors text-sm text-muted-foreground hover:text-primary"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Dashboard</span>
            </Link>
            <div className="w-px h-5 bg-border" />
            <FlaskConical className="h-4 w-4 text-apple-purple" />
            <h1 className="text-base font-bold">Backtest Viewer</h1>
          </div>
          <div className="flex items-center gap-3">
            {/* Tab switcher */}
            <div className="flex rounded-lg bg-surface-light border border-border p-0.5">
              <button
                onClick={() => setTab("detail")}
                className={cn(
                  "px-3 py-1 rounded-md text-xs font-medium transition-colors",
                  tab === "detail" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                )}
              >
                Detail
              </button>
              <button
                onClick={() => setTab("compare")}
                className={cn(
                  "px-3 py-1 rounded-md text-xs font-medium transition-colors",
                  tab === "compare" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                )}
              >
                Perbandingan
              </button>
            </div>
            <ThemeToggle />
            <span className="text-xs text-muted-foreground font-mono">XAUBOT AI</span>
          </div>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Sidebar — backtest list */}
        <aside className="w-72 shrink-0 border-r border-border bg-white/60 dark:bg-white/[0.02] backdrop-blur-sm overflow-y-auto">
          <div className="p-2.5 border-b border-border">
            <p className="text-xs text-muted-foreground font-medium">
              {validResults.length} backtests
            </p>
          </div>
          <div className="py-1">
            {validResults.map((bt) => (
              <button
                key={bt.id}
                onClick={() => { setSelectedId(bt.id); setTab("detail"); }}
                className={cn(
                  "w-full flex items-center justify-between px-3 py-2 text-sm transition-colors",
                  bt.id === selectedId
                    ? "bg-primary/10 text-primary font-medium border-r-2 border-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-surface-light"
                )}
              >
                <span className="truncate">
                  <span className="font-number text-xs opacity-50 mr-1.5">#{bt.id}</span>
                  {bt.name}
                </span>
                <div className="flex items-center gap-1.5 shrink-0 ml-2">
                  <Badge
                    variant={bt.netPnl >= 0 ? "success" : "danger"}
                    className="text-[10px] px-1.5 py-0"
                  >
                    {bt.netPnl >= 0 ? "+" : ""}{formatUSD(bt.netPnl)}
                  </Badge>
                </div>
              </button>
            ))}
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-4 space-y-4">
          {tab === "detail" && selected ? (
            <>
              {/* Title */}
              <div className="glass rounded-xl p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-bold flex items-center gap-2">
                      <Activity className="h-5 w-5 text-apple-blue" />
                      #{selected.id} {selected.name}
                    </h2>
                    {selected.strategy && (
                      <p className="text-xs text-muted-foreground mt-1">{selected.strategy}</p>
                    )}
                  </div>
                  <div className="text-right text-xs text-muted-foreground">
                    {selected.period && <p>{selected.period}</p>}
                    {selected.generatedAt && <p>{selected.generatedAt}</p>}
                  </div>
                </div>
              </div>

              <MetricsGrid bt={selected} />

              <div className="grid grid-cols-2 gap-4">
                <ExitReasonsBar bt={selected} />
                <SessionBars bt={selected} />
              </div>

              {/* Direction breakdown */}
              {selected.directionBreakdown.length > 0 && (
                <div className="glass rounded-xl p-4">
                  <h3 className="text-sm font-semibold mb-3">Direction Breakdown</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {selected.directionBreakdown.map((d) => (
                      <div key={d.direction} className="flex items-center gap-3">
                        <Badge variant={d.direction === "BUY" ? "success" : "danger"}>
                          {d.direction}
                        </Badge>
                        <span className="text-sm font-number">{d.trades} trades</span>
                        <span className="text-sm font-number">{d.winRate}% WR</span>
                        <span className={`ml-auto text-sm font-number font-semibold ${d.pnl >= 0 ? "text-success" : "text-danger"}`}>
                          {formatUSD(d.pnl)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <ComparisonTable results={validResults} />
          )}
        </main>
      </div>
    </div>
  );
}
