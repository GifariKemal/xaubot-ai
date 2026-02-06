"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Wallet } from "lucide-react";

interface AccountCardProps {
  balance: number;
  equity: number;
  profit: number;
}

export function AccountCard({ balance, equity, profit }: AccountCardProps) {
  const isProfit = profit >= 0;

  return (
    <Card className="bg-card/50 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Wallet className="h-4 w-4" />
          ACCOUNT
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Balance</span>
          <span className="font-semibold">${balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Equity</span>
          <span className="font-semibold">${equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
        </div>
        <div className="flex justify-between items-center pt-2 border-t">
          <span className="text-sm text-muted-foreground">P/L</span>
          <span className={`font-bold ${isProfit ? 'text-green-500' : 'text-red-500'}`}>
            {isProfit ? '+' : ''}${profit.toFixed(2)}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
