"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Wallet } from "lucide-react";
import { Sparkline } from "./sparkline";
import { cn, formatUSD } from "@/lib/utils";
import { useAnimatedValue } from "@/hooks/use-animated-value";

interface AccountCardProps {
  balance: number;
  equity: number;
  profit: number;
  equityHistory?: number[];
}

export function AccountCard({ balance, equity, profit, equityHistory = [] }: AccountCardProps) {
  const isProfit = profit >= 0;
  const animBalance = useAnimatedValue(balance);
  const animEquity = useAnimatedValue(equity);
  const animProfit = useAnimatedValue(profit);

  const margin = balance > 0 ? ((equity / balance) * 100) : 100;
  const drawdown = equityHistory.length > 0 ? Math.max(...equityHistory) - equity : 0;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className={cn("glass h-full overflow-hidden flex flex-col cursor-pointer", isProfit ? "accent-top-green glass-green" : "accent-top-red glass-red")}>
          <CardHeader>
            <CardTitle className={cn(
              "text-sm font-medium flex items-center gap-1.5 uppercase tracking-wider",
              isProfit ? "text-apple-green" : "text-apple-red"
            )}>
              <Wallet className="h-4 w-4" />
              Account
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 flex flex-col justify-between">
            <div className="space-y-1">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">Balance</span>
                    <span
                      key={animBalance.changeKey}
                      className={cn(
                        "text-base font-semibold font-number text-foreground",
                        animBalance.direction === "up" && "flash-up",
                        animBalance.direction === "down" && "flash-down"
                      )}
                    >
                      {formatUSD(animBalance.displayValue)}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent><p>Click card for detail</p></TooltipContent>
              </Tooltip>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Equity</span>
                <span
                  key={animEquity.changeKey}
                  className={cn(
                    "text-base font-semibold font-number text-apple-cyan",
                    animEquity.direction === "up" && "flash-up",
                    animEquity.direction === "down" && "flash-down"
                  )}
                >
                  {formatUSD(animEquity.displayValue)}
                </span>
              </div>
            </div>

            <div className="flex justify-between items-center pt-1.5 border-t border-border">
              <span className="text-sm text-muted-foreground">P/L</span>
              <span
                key={animProfit.changeKey}
                className={cn(
                  "text-xl font-bold font-number",
                  isProfit ? "text-success" : "text-danger",
                  animProfit.direction === "up" && "flash-up",
                  animProfit.direction === "down" && "flash-down"
                )}
              >
                {animProfit.displayValue >= 0 ? "+" : ""}{formatUSD(animProfit.displayValue)}
              </span>
            </div>

            {equityHistory.length > 2 && (
              <div className="-mx-1 mt-auto">
                <Sparkline data={equityHistory.slice(-30)} color={isProfit ? "#34C759" : "#FF3B30"} height={18} />
              </div>
            )}
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className={cn("flex items-center gap-2", isProfit ? "text-apple-green" : "text-apple-red")}>
            <Wallet className="h-5 w-5" />
            Account Detail
          </DialogTitle>
          <DialogDescription>Balance, equity, and performance metrics</DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Balance</span>
              <p className="text-2xl font-bold font-number">{formatUSD(balance)}</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Equity</span>
              <p className="text-2xl font-bold font-number text-apple-cyan">{formatUSD(equity)}</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Floating P/L</span>
              <p className={cn("text-xl font-bold font-number", isProfit ? "text-success" : "text-danger")}>
                {profit >= 0 ? "+" : ""}{formatUSD(profit)}
              </p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Margin Level</span>
              <p className={cn("text-xl font-bold font-number", margin >= 100 ? "text-apple-green" : "text-apple-red")}>
                {margin.toFixed(1)}%
              </p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Drawdown</span>
              <p className={cn("text-xl font-bold font-number", drawdown > 0 ? "text-apple-red" : "text-muted-foreground")}>
                {formatUSD(drawdown)}
              </p>
            </div>
          </div>

          {equityHistory.length > 2 && (
            <div>
              <span className="text-sm text-muted-foreground mb-1 block">Equity Curve ({equityHistory.length} points)</span>
              <Sparkline data={equityHistory} color={isProfit ? "#34C759" : "#FF3B30"} height={80} />
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
