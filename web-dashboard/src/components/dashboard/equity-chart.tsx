"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";
import { Wallet } from "lucide-react";

interface EquityChartProps {
  equityData: number[];
  balanceData: number[];
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    timeZone: "Asia/Jakarta",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function EquityChart({ equityData, balanceData }: EquityChartProps) {
  const chartData = useMemo(() => {
    if (equityData.length === 0) return [];
    const now = Date.now();
    const totalMs = 2 * 60 * 60 * 1000;
    const step = equityData.length > 1 ? totalMs / (equityData.length - 1) : 0;

    return equityData.map((equity, i) => {
      const ts = new Date(now - totalMs + i * step);
      return {
        time: formatTime(ts),
        timestamp: ts.getTime(),
        equity,
        balance: balanceData[i] || equity,
      };
    });
  }, [equityData, balanceData]);

  const tickIndices = useMemo(() => {
    if (chartData.length <= 6) return chartData.map((d) => d.timestamp);
    const step = Math.floor(chartData.length / 5);
    const ticks: number[] = [];
    for (let i = 0; i < chartData.length; i += step) {
      ticks.push(chartData[i].timestamp);
    }
    if (ticks[ticks.length - 1] !== chartData[chartData.length - 1].timestamp) {
      ticks.push(chartData[chartData.length - 1].timestamp);
    }
    return ticks;
  }, [chartData]);

  return (
    <Card className="glass h-full overflow-hidden flex flex-col accent-top-green">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-apple-green flex items-center gap-1.5 uppercase tracking-wider">
          <Wallet className="h-4 w-4" />
          Equity (2H)
          {equityData.length > 0 && (
            <span className="ml-auto text-base font-number text-apple-green font-bold">
              ${equityData[equityData.length - 1]?.toFixed(2)}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 pb-1">
        <div className="h-full w-full">
          {chartData.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.06)" />
                <XAxis
                  dataKey="timestamp"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  ticks={tickIndices}
                  tickFormatter={(ts: number) => formatTime(new Date(ts))}
                  tick={{ fontSize: 11, fill: "#86868b", fontFamily: "var(--font-mono)" }}
                  axisLine={{ stroke: "rgba(0,0,0,0.08)" }}
                  tickLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 11, fill: "#86868b", fontFamily: "var(--font-mono)" }}
                  axisLine={false}
                  tickLine={false}
                  width={58}
                  tickFormatter={(v: number) => `$${v.toFixed(0)}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(255, 255, 255, 0.85)",
                    border: "1px solid rgba(52, 199, 89, 0.2)",
                    borderRadius: "12px",
                    fontSize: "13px",
                    fontFamily: "var(--font-mono)",
                    color: "#1d1d1f",
                    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.08)",
                    backdropFilter: "blur(20px)",
                    padding: "6px 10px",
                  }}
                  labelFormatter={(ts: number) => formatTime(new Date(ts))}
                  formatter={(value: number, name: string) => [
                    `$${value.toFixed(2)}`,
                    name === "equity" ? "Equity" : "Balance",
                  ]}
                  cursor={{ stroke: "#34C759", strokeDasharray: "3 3" }}
                />
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34C759" stopOpacity={0.25} />
                    <stop offset="40%" stopColor="#34C759" stopOpacity={0.08} />
                    <stop offset="100%" stopColor="#34C759" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="balance" stroke="#AF52DE" strokeWidth={1.5} strokeDasharray="4 4" fill="none" dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="equity" stroke="#34C759" strokeWidth={2.5} fill="url(#equityGradient)" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground/50">
              <div className="text-center space-y-1">
                <Wallet className="h-6 w-6 mx-auto opacity-30" />
                <p className="text-base">Collecting data...</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
