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
    <Card className="bg-card/50 backdrop-blur col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Wallet className="h-4 w-4" />
          EQUITY vs BALANCE (2H)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[120px] w-full">
          {equityData.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <XAxis dataKey="index" hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ display: 'none' }}
                  formatter={(value: number, name: string) => [
                    `$${value.toFixed(2)}`,
                    name === 'equity' ? 'Equity' : 'Balance'
                  ]}
                />
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="balance"
                  stroke="#666"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  fill="none"
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="url(#equityGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-muted-foreground">
              Waiting for data...
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
