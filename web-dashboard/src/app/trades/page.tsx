"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  History,
  TrendingUp,
  TrendingDown,
  Trophy,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Filter,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/theme-toggle";
import { useTrades, useTradeStats, useEquityCurve } from "@/hooks/use-trades";
import { formatUSD } from "@/lib/utils";
import { format } from "date-fns";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

function StatsCards({ startDate, endDate }: { startDate: string; endDate: string }) {
  const { stats } = useTradeStats(startDate, endDate);

  const cards = [
    {
      label: "Total Trades",
      value: stats?.totalTrades ?? 0,
      format: (v: number) => String(v),
      icon: History,
      color: "text-apple-blue",
      accent: "accent-top-blue",
    },
    {
      label: "Win Rate",
      value: stats?.winRate ?? 0,
      format: (v: number) => `${v.toFixed(1)}%`,
      icon: Trophy,
      color: "text-apple-green",
      accent: "accent-top-green",
    },
    {
      label: "Net Profit",
      value: stats?.netProfit ?? 0,
      format: (v: number) => formatUSD(v),
      icon: TrendingUp,
      color: (stats?.netProfit ?? 0) >= 0 ? "text-success" : "text-danger",
      accent: (stats?.netProfit ?? 0) >= 0 ? "accent-top-green" : "accent-top-red",
    },
    {
      label: "Profit Factor",
      value: stats?.profitFactor ?? 0,
      format: (v: number) => v.toFixed(2),
      icon: BarChart3,
      color: "text-apple-purple",
      accent: "accent-top-purple",
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-3">
      {cards.map((c) => (
        <div key={c.label} className={`glass rounded-xl p-4 ${c.accent}`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground font-medium">{c.label}</span>
            <c.icon className={`h-4 w-4 ${c.color}`} />
          </div>
          <p className={`text-2xl font-bold font-number ${c.color}`}>
            {c.format(c.value)}
          </p>
        </div>
      ))}
    </div>
  );
}

function EquityCurveChart({ startDate, endDate }: { startDate: string; endDate: string }) {
  const { points, loading } = useEquityCurve(startDate, endDate);

  if (loading || points.length === 0) {
    return (
      <div className="glass rounded-xl p-4 h-64 flex items-center justify-center text-muted-foreground text-sm">
        {loading ? "Loading equity curve..." : "No trade data available"}
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-4">
      <h3 className="text-sm font-semibold mb-3">Equity Curve</h3>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points}>
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => {
                try { return format(new Date(v), "dd MMM"); } catch { return v; }
              }}
              stroke="var(--color-muted-foreground)"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              tickFormatter={(v) => `$${v}`}
              stroke="var(--color-muted-foreground)"
              tickLine={false}
              axisLine={false}
              width={60}
            />
            <Tooltip
              contentStyle={{
                background: "var(--color-popover)",
                border: "1px solid var(--color-border)",
                borderRadius: 10,
                fontSize: 12,
              }}
              formatter={(v: number) => [formatUSD(v), "Cumulative P/L"]}
              labelFormatter={(v) => {
                try { return format(new Date(v), "dd MMM yyyy HH:mm"); } catch { return v; }
              }}
            />
            <ReferenceLine y={0} stroke="var(--color-border)" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="cumulative"
              stroke="var(--apple-blue)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TradeTable({
  filters,
  setFilters,
}: {
  filters: { page: number; limit: number; direction: string; startDate: string; endDate: string };
  setFilters: React.Dispatch<React.SetStateAction<typeof filters>>;
}) {
  const { trades, total, loading } = useTrades(filters);
  const totalPages = Math.ceil(total / filters.limit) || 1;

  return (
    <div className="glass rounded-xl overflow-hidden">
      {/* Filter bar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
        <Filter className="h-4 w-4 text-muted-foreground" />
        <select
          value={filters.direction}
          onChange={(e) => setFilters((p) => ({ ...p, direction: e.target.value, page: 1 }))}
          className="text-sm px-2 py-1 rounded-md bg-surface border border-border"
        >
          <option value="ALL">All Directions</option>
          <option value="BUY">BUY Only</option>
          <option value="SELL">SELL Only</option>
        </select>
        <input
          type="date"
          value={filters.startDate}
          onChange={(e) => setFilters((p) => ({ ...p, startDate: e.target.value, page: 1 }))}
          className="text-sm px-2 py-1 rounded-md bg-surface border border-border"
        />
        <span className="text-xs text-muted-foreground">to</span>
        <input
          type="date"
          value={filters.endDate}
          onChange={(e) => setFilters((p) => ({ ...p, endDate: e.target.value, page: 1 }))}
          className="text-sm px-2 py-1 rounded-md bg-surface border border-border"
        />
        <span className="ml-auto text-xs text-muted-foreground font-number">
          {total} trades
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground">
              <th className="text-left px-4 py-2 font-medium">Time</th>
              <th className="text-left px-3 py-2 font-medium">Dir</th>
              <th className="text-right px-3 py-2 font-medium">Entry</th>
              <th className="text-right px-3 py-2 font-medium">Exit</th>
              <th className="text-right px-3 py-2 font-medium">Lot</th>
              <th className="text-right px-3 py-2 font-medium">P/L</th>
              <th className="text-left px-3 py-2 font-medium">Exit Reason</th>
              <th className="text-right px-3 py-2 font-medium">Conf</th>
              <th className="text-left px-3 py-2 font-medium">Regime</th>
              <th className="text-left px-3 py-2 font-medium">Session</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={10} className="text-center py-8 text-muted-foreground">Loading...</td>
              </tr>
            ) : trades.length === 0 ? (
              <tr>
                <td colSpan={10} className="text-center py-8 text-muted-foreground">No trades found</td>
              </tr>
            ) : (
              trades.map((t) => (
                <tr key={t.id} className="border-b border-border/50 row-hover">
                  <td className="px-4 py-2 font-number text-xs whitespace-nowrap">
                    {(() => {
                      try { return format(new Date(t.closed_at), "dd MMM HH:mm"); }
                      catch { return t.closed_at; }
                    })()}
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant={t.direction === "BUY" ? "success" : "danger"} className="text-xs">
                      {t.direction}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-right font-number">{t.entry_price?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-number">{t.exit_price?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-number">{t.lot_size?.toFixed(2)}</td>
                  <td className={`px-3 py-2 text-right font-number font-semibold ${t.profit_usd >= 0 ? "text-success" : "text-danger"}`}>
                    {formatUSD(t.profit_usd)}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">{t.exit_reason}</td>
                  <td className="px-3 py-2 text-right font-number">{(t.confidence * 100).toFixed(0)}%</td>
                  <td className="px-3 py-2 text-xs">{t.regime}</td>
                  <td className="px-3 py-2 text-xs">{t.session}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between px-4 py-3 border-t border-border">
        <span className="text-xs text-muted-foreground">
          Page {filters.page} of {totalPages}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFilters((p) => ({ ...p, page: Math.max(1, p.page - 1) }))}
            disabled={filters.page <= 1}
            className="p-1.5 rounded-md hover:bg-surface disabled:opacity-30"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            onClick={() => setFilters((p) => ({ ...p, page: Math.min(totalPages, p.page + 1) }))}
            disabled={filters.page >= totalPages}
            className="p-1.5 rounded-md hover:bg-surface disabled:opacity-30"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default function TradesPage() {
  const [filters, setFilters] = useState({
    page: 1,
    limit: 25,
    direction: "ALL",
    startDate: "",
    endDate: "",
  });

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
            <History className="h-4 w-4 text-apple-blue" />
            <h1 className="text-base font-bold">Trade History</h1>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <span className="text-xs text-muted-foreground font-mono">XAUBOT AI</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-y-auto p-4 space-y-4">
        <StatsCards startDate={filters.startDate} endDate={filters.endDate} />
        <EquityCurveChart startDate={filters.startDate} endDate={filters.endDate} />
        <TradeTable filters={filters} setFilters={setFilters} />
      </main>
    </div>
  );
}
