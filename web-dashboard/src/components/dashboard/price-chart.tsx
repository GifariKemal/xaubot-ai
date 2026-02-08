"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { TrendingUp } from "lucide-react";

interface PriceChartProps {
  data: number[];
}

export function PriceChart({ data }: PriceChartProps) {
  const chartData = data.map((price, i) => ({ index: i, price }));

  return (
    <Card className="glass h-full flex flex-col">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <TrendingUp className="h-3.5 w-3.5" />
          Price Chart (2H)
          {data.length > 0 && (
            <span className="ml-auto text-xs font-number text-foreground">
              ${data[data.length - 1]?.toFixed(2)}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <div className="h-full w-full">
          {data.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <XAxis dataKey="index" hide />
                <YAxis domain={["auto", "auto"]} hide />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--color-card)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "6px",
                    fontSize: "11px",
                    fontFamily: "var(--font-mono)",
                  }}
                  labelStyle={{ display: "none" }}
                  formatter={(value: number) => [`$${value.toFixed(2)}`, "Price"]}
                />
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="price"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  fill="url(#priceGradient)"
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground/50">
              <div className="text-center space-y-1">
                <TrendingUp className="h-5 w-5 mx-auto opacity-30" />
                <p className="text-xs">Collecting data...</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
