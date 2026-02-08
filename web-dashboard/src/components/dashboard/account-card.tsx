"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Wallet } from "lucide-react";
import { Sparkline } from "./sparkline";
import { cn, formatUSD, getValueColor } from "@/lib/utils";

interface AccountCardProps {
  balance: number;
  equity: number;
  profit: number;
  equityHistory?: number[];
}

export function AccountCard({ balance, equity, profit, equityHistory = [] }: AccountCardProps) {
  const isProfit = profit >= 0;

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Wallet className="h-3.5 w-3.5" />
          Account
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Balance</span>
          <span className="text-sm font-semibold font-number">{formatUSD(balance)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Equity</span>
          <span className="text-sm font-semibold font-number">{formatUSD(equity)}</span>
        </div>
        <div className="flex justify-between items-center pt-1 border-t border-border">
          <span className="text-[11px] text-muted-foreground">P/L</span>
          <span className={cn("text-base font-bold font-number", getValueColor(profit))}>
            {isProfit ? "+" : ""}{formatUSD(profit)}
          </span>
        </div>

        {equityHistory.length > 2 && (
          <div className="-mx-1">
            <Sparkline
              data={equityHistory.slice(-30)}
              color={isProfit ? "#22c55e" : "#ef4444"}
              height={20}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
