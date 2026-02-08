"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";
import { TrendingUp } from "lucide-react";

interface PriceChartProps {
  data: number[];
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    timeZone: "Asia/Jakarta",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function PriceChart({ data }: PriceChartProps) {
  const chartData = useMemo(() => {
    if (data.length === 0) return [];
    const now = Date.now();
    // Spread data points evenly over 2 hours (7200s)
    const totalMs = 2 * 60 * 60 * 1000;
    const step = data.length > 1 ? totalMs / (data.length - 1) : 0;

    return data.map((price, i) => {
      const ts = new Date(now - totalMs + i * step);
      return {
        time: formatTime(ts),
        timestamp: ts.getTime(),
        price,
      };
    });
  }, [data]);

  // Show ~6 evenly spaced tick labels on X axis
  const tickIndices = useMemo(() => {
    if (chartData.length <= 6) return chartData.map((d) => d.timestamp);
    const step = Math.floor(chartData.length / 5);
    const ticks: number[] = [];
    for (let i = 0; i < chartData.length; i += step) {
      ticks.push(chartData[i].timestamp);
    }
    // Always include last
    if (ticks[ticks.length - 1] !== chartData[chartData.length - 1].timestamp) {
      ticks.push(chartData[chartData.length - 1].timestamp);
    }
    return ticks;
  }, [chartData]);

  return (
    <Card className="glass h-full overflow-hidden flex flex-col accent-top-blue">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-apple-blue flex items-center gap-1.5 uppercase tracking-wider">
          <TrendingUp className="h-4 w-4" />
          Price Chart (2H)
          {data.length > 0 && (
            <span className="ml-auto text-base font-number text-apple-blue font-bold">
              ${data[data.length - 1]?.toFixed(2)}
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
                  tickFormatter={(v: number) => v.toFixed(0)}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(255, 255, 255, 0.85)",
                    border: "1px solid rgba(0, 122, 255, 0.2)",
                    borderRadius: "12px",
                    fontSize: "13px",
                    fontFamily: "var(--font-mono)",
                    color: "#1d1d1f",
                    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.08)",
                    backdropFilter: "blur(20px)",
                    padding: "6px 10px",
                  }}
                  labelFormatter={(ts: number) => formatTime(new Date(ts))}
                  formatter={(value: number) => [`$${value.toFixed(2)}`, "Price"]}
                  cursor={{ stroke: "#007AFF", strokeDasharray: "3 3" }}
                />
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#007AFF" stopOpacity={0.25} />
                    <stop offset="40%" stopColor="#007AFF" stopOpacity={0.08} />
                    <stop offset="100%" stopColor="#007AFF" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="price" stroke="#007AFF" strokeWidth={2.5} fill="url(#priceGradient)" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground/50">
              <div className="text-center space-y-1">
                <TrendingUp className="h-6 w-6 mx-auto opacity-30" />
                <p className="text-base">Collecting data...</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
