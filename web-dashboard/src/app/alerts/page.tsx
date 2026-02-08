"use client";

import { useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Bell,
  Activity,
  CheckCircle2,
  XCircle,
  Filter,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Gauge,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/theme-toggle";
import { useSignals, useSignalStats } from "@/hooks/use-signals";
import { formatUSD } from "@/lib/utils";
import { format } from "date-fns";

function StatsRow() {
  const { stats } = useSignalStats(24);

  const items = [
    {
      label: "Signals (24h)",
      value: stats?.total ?? 0,
      fmt: (v: number) => String(v),
      icon: Bell,
      color: "text-apple-blue",
      accent: "accent-top-blue",
    },
    {
      label: "Executed",
      value: stats?.executed ?? 0,
      fmt: (v: number) => String(v),
      icon: Zap,
      color: "text-apple-green",
      accent: "accent-top-green",
    },
    {
      label: "Execution Rate",
      value: stats?.executionRate ?? 0,
      fmt: (v: number) => `${v.toFixed(1)}%`,
      icon: Activity,
      color: "text-apple-purple",
      accent: "accent-top-purple",
    },
    {
      label: "Avg Confidence",
      value: stats?.avgConfidence ?? 0,
      fmt: (v: number) => `${v.toFixed(1)}%`,
      icon: Gauge,
      color: "text-apple-cyan",
      accent: "accent-top-cyan",
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-3">
      {items.map((item) => (
        <div key={item.label} className={`glass rounded-xl p-4 ${item.accent}`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground font-medium">{item.label}</span>
            <item.icon className={`h-4 w-4 ${item.color}`} />
          </div>
          <p className={`text-2xl font-bold font-number ${item.color}`}>
            {item.fmt(item.value)}
          </p>
        </div>
      ))}
    </div>
  );
}

const signalIcon = (type: string) => {
  switch (type) {
    case "BUY": return <TrendingUp className="h-3.5 w-3.5" />;
    case "SELL": return <TrendingDown className="h-3.5 w-3.5" />;
    default: return <Minus className="h-3.5 w-3.5" />;
  }
};

const signalBadgeVariant = (type: string) => {
  switch (type) {
    case "BUY": return "success" as const;
    case "SELL": return "danger" as const;
    default: return "warning" as const;
  }
};

function SignalTable({
  filters,
  setFilters,
}: {
  filters: { page: number; limit: number; type: string; executed: string; startDate: string; endDate: string };
  setFilters: React.Dispatch<React.SetStateAction<typeof filters>>;
}) {
  const { signals, total, loading } = useSignals(filters);
  const totalPages = Math.ceil(total / filters.limit) || 1;

  return (
    <div className="glass rounded-xl overflow-hidden">
      {/* Filter bar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
        <Filter className="h-4 w-4 text-muted-foreground" />
        <select
          value={filters.type}
          onChange={(e) => setFilters((p) => ({ ...p, type: e.target.value, page: 1 }))}
          className="text-sm px-2 py-1 rounded-md bg-surface border border-border"
        >
          <option value="ALL">All Types</option>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
          <option value="HOLD">HOLD</option>
        </select>
        <select
          value={filters.executed}
          onChange={(e) => setFilters((p) => ({ ...p, executed: e.target.value, page: 1 }))}
          className="text-sm px-2 py-1 rounded-md bg-surface border border-border"
        >
          <option value="all">All Status</option>
          <option value="yes">Executed</option>
          <option value="no">Not Executed</option>
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
          {total} signals
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground">
              <th className="text-left px-4 py-2 font-medium">Time</th>
              <th className="text-left px-3 py-2 font-medium">Signal</th>
              <th className="text-right px-3 py-2 font-medium">Confidence</th>
              <th className="text-center px-3 py-2 font-medium">Executed</th>
              <th className="text-left px-3 py-2 font-medium">Reason</th>
              <th className="text-left px-3 py-2 font-medium">SMC</th>
              <th className="text-left px-3 py-2 font-medium">ML</th>
              <th className="text-left px-3 py-2 font-medium">Regime</th>
              <th className="text-left px-3 py-2 font-medium">Session</th>
              <th className="text-right px-3 py-2 font-medium">Entry</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={10} className="text-center py-8 text-muted-foreground">Loading...</td>
              </tr>
            ) : signals.length === 0 ? (
              <tr>
                <td colSpan={10} className="text-center py-8 text-muted-foreground">No signals found</td>
              </tr>
            ) : (
              signals.map((s) => (
                <tr key={s.id} className="border-b border-border/50 row-hover">
                  <td className="px-4 py-2 font-number text-xs whitespace-nowrap">
                    {(() => {
                      try { return format(new Date(s.signal_time), "dd MMM HH:mm"); }
                      catch { return s.signal_time; }
                    })()}
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant={signalBadgeVariant(s.signal_type)} className="gap-1 text-xs">
                      {signalIcon(s.signal_type)}
                      {s.signal_type}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-right font-number">
                    {(s.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="px-3 py-2 text-center">
                    {s.executed ? (
                      <CheckCircle2 className="h-4 w-4 text-success inline" />
                    ) : (
                      <XCircle className="h-4 w-4 text-muted-foreground inline" />
                    )}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground max-w-[200px] truncate">
                    {s.execution_reason || "—"}
                  </td>
                  <td className="px-3 py-2 text-xs">{s.smc_signal || "—"}</td>
                  <td className="px-3 py-2 text-xs">{s.ml_signal || "—"}</td>
                  <td className="px-3 py-2 text-xs">{s.regime || "—"}</td>
                  <td className="px-3 py-2 text-xs">{s.session || "—"}</td>
                  <td className="px-3 py-2 text-right font-number">
                    {s.entry_price ? s.entry_price.toFixed(2) : "—"}
                  </td>
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

export default function AlertsPage() {
  const [filters, setFilters] = useState({
    page: 1,
    limit: 50,
    type: "ALL",
    executed: "all",
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
            <Bell className="h-4 w-4 text-apple-orange" />
            <h1 className="text-base font-bold">Alert / Signal Log</h1>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <span className="text-xs text-muted-foreground font-mono">XAUBOT AI</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-y-auto p-4 space-y-4">
        <StatsRow />
        <SignalTable filters={filters} setFilters={setFilters} />
      </main>
    </div>
  );
}
