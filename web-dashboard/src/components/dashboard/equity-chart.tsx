"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { Wallet } from "lucide-react";

interface EquityChartProps {
  equityData: number[];
  balanceData: number[];
}

export function EquityChart({ equityData, balanceData }: EquityChartProps) {
  const chartData = equityData.map((equity, i) => ({
    index: i,
    equity,
    balance: balanceData[i] || equity,
  }));

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Wallet className="h-3.5 w-3.5" />
          Equity vs Balance (2H)
          {equityData.length > 0 && (
            <span className="ml-auto text-xs font-number text-success">
              ${equityData[equityData.length - 1]?.toFixed(2)}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[100px] w-full">
          {equityData.length > 1 ? (
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
                  formatter={(value: number, name: string) => [
                    `$${value.toFixed(2)}`,
                    name === "equity" ? "Equity" : "Balance",
                  ]}
                />
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="balance"
                  stroke="#555"
                  strokeWidth={1}
                  strokeDasharray="4 4"
                  fill="none"
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#22c55e"
                  strokeWidth={1.5}
                  fill="url(#equityGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground/50">
              <div className="text-center space-y-1">
                <Wallet className="h-5 w-5 mx-auto opacity-30" />
                <p className="text-xs">Collecting data...</p>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
