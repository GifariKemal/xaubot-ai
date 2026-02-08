"use client";

import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ShieldAlert, AlertTriangle } from "lucide-react";
import { cn, formatUSD } from "@/lib/utils";
import { useAnimatedValue } from "@/hooks/use-animated-value";
import type { RiskMode } from "@/types/trading";

interface RiskCardProps {
  dailyLoss: number;
  dailyProfit: number;
  consecutiveLosses: number;
  riskPercent: number;
  riskMode?: RiskMode;
}

function getRiskModeVariant(mode: string): "success" | "warning" | "danger" | "secondary" {
  switch (mode) {
    case "normal": return "success";
    case "recovery": return "warning";
    case "protected": case "stopped": return "danger";
    default: return "secondary";
  }
}

export function RiskCard({ dailyLoss, dailyProfit, consecutiveLosses, riskPercent, riskMode }: RiskCardProps) {
  const isHigh = riskPercent >= 80;
  const isMedium = riskPercent >= 50;
  const mode = riskMode?.mode || "unknown";
  const firstMount = useRef(true);

  const animRisk = useAnimatedValue(riskPercent);
  const animLoss = useAnimatedValue(dailyLoss);
  const animProfit = useAnimatedValue(dailyProfit);

  const getRiskColor = () => isHigh ? "text-danger" : isMedium ? "text-warning" : "text-success";
  const getBarClass = () => isHigh ? "bar-red" : isMedium ? "bar-orange" : "bar-green";
  const getTopClass = () => isHigh ? "accent-top-red" : isMedium ? "accent-top-orange" : "accent-top-green";

  const shouldAnimate = firstMount.current;
  if (firstMount.current) firstMount.current = false;

  const netPL = dailyProfit - dailyLoss;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className={cn("glass h-full overflow-hidden flex flex-col cursor-pointer", getTopClass(), isHigh ? "glass-red" : isMedium ? "glass-orange" : "glass-green")}>
          <CardHeader>
            <CardTitle className={cn(
              "text-sm font-medium flex items-center gap-1.5 uppercase tracking-wider",
              isHigh ? "text-apple-red" : isMedium ? "text-apple-orange" : "text-apple-green"
            )}>
              <ShieldAlert className="h-4 w-4" />
              Risk
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant={getRiskModeVariant(mode)} className={cn("ml-auto text-xs h-5 px-1.5 uppercase", mode === "stopped" && "animate-pulse")}>
                    {mode}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{mode === "normal" ? "Full position sizing" : mode === "recovery" ? "Reduced lots after losses" : mode === "stopped" ? "Daily limit hit" : mode}</p>
                </TooltipContent>
              </Tooltip>
              {riskPercent >= 100 && (
                <span className="flex items-center gap-1 text-xs bg-danger text-white px-1.5 py-0.5 rounded-full animate-pulse">
                  <AlertTriangle className="h-3 w-3" /> BREACHED
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 flex flex-col justify-between">
            <div className="space-y-0.5">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Daily Loss</span>
                <span className="text-base font-semibold font-number text-apple-red">{formatUSD(animLoss.displayValue)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Daily Profit</span>
                <span className="text-base font-semibold font-number text-apple-green">{formatUSD(animProfit.displayValue)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Consec. Loss</span>
                <span className={cn("text-base font-semibold font-number", consecutiveLosses >= 3 ? "text-apple-orange" : "text-foreground")}>
                  {consecutiveLosses}
                </span>
              </div>
            </div>

            <div className="pt-1 border-t border-border">
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-muted-foreground">Risk Used</span>
                <span className={cn("text-lg font-bold font-number", getRiskColor())}>{animRisk.displayValue.toFixed(0)}%</span>
              </div>
              <div className="h-2.5 w-full bg-black/[0.04] rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    getBarClass(),
                    shouldAnimate && "bar-animate-in"
                  )}
                  style={{ width: `${Math.min(animRisk.displayValue, 100)}%` }}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className={cn("flex items-center gap-2", isHigh ? "text-apple-red" : isMedium ? "text-apple-orange" : "text-apple-green")}>
            <ShieldAlert className="h-5 w-5" />
            Risk Management Detail
          </DialogTitle>
          <DialogDescription>Daily risk metrics and mode status</DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="flex items-center gap-3">
            <Badge variant={getRiskModeVariant(mode)} className="text-sm px-3 py-1 uppercase">{mode}</Badge>
            <span className={cn("text-3xl font-bold font-number", getRiskColor())}>{riskPercent.toFixed(1)}%</span>
            <span className="text-muted-foreground">risk used</span>
          </div>

          <div className="h-3 w-full bg-black/[0.04] rounded-full overflow-hidden">
            <div className={cn("h-full rounded-full transition-all", getBarClass())} style={{ width: `${Math.min(riskPercent, 100)}%` }} />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Daily Loss</span>
              <p className="text-xl font-bold font-number text-apple-red">{formatUSD(dailyLoss)}</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Daily Profit</span>
              <p className="text-xl font-bold font-number text-apple-green">{formatUSD(dailyProfit)}</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Net P/L</span>
              <p className={cn("text-xl font-bold font-number", netPL >= 0 ? "text-apple-green" : "text-apple-red")}>
                {netPL >= 0 ? "+" : ""}{formatUSD(netPL)}
              </p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Consecutive Losses</span>
              <p className={cn("text-xl font-bold font-number", consecutiveLosses >= 3 ? "text-apple-orange" : "text-foreground")}>
                {consecutiveLosses}
              </p>
            </div>
          </div>

          {riskMode && (
            <div className="pt-3 border-t border-border space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Recommended Lot</span>
                <span className="font-semibold font-number text-apple-purple">{riskMode.recommendedLot}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Max Allowed Lot</span>
                <span className="font-semibold font-number">{riskMode.maxAllowedLot}</span>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
