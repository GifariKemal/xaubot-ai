"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ShieldAlert, AlertTriangle } from "lucide-react";
import { cn, formatUSD } from "@/lib/utils";

interface RiskCardProps {
  dailyLoss: number;
  dailyProfit: number;
  consecutiveLosses: number;
  riskPercent: number;
}

export function RiskCard({ dailyLoss, dailyProfit, consecutiveLosses, riskPercent }: RiskCardProps) {
  const isCritical = riskPercent >= 100;
  const isHigh = riskPercent >= 80;
  const isMedium = riskPercent >= 50;

  const getRiskColor = () => {
    if (isHigh) return "text-danger";
    if (isMedium) return "text-warning";
    return "text-success";
  };

  const getSegmentFill = () => {
    if (isHigh) return "bg-danger";
    if (isMedium) return "bg-warning";
    return "bg-success";
  };

  return (
    <Card className={cn(
      "glass",
      isCritical && "border-danger/50 ring-1 ring-danger/20",
      isHigh && !isCritical && "border-danger/30"
    )}>
      <CardHeader>
        <CardTitle className={cn(
          "text-[11px] font-medium flex items-center gap-1.5 uppercase tracking-wider",
          isHigh ? "text-danger" : "text-muted-foreground"
        )}>
          <ShieldAlert className="h-3.5 w-3.5" />
          Risk
          {isCritical && (
            <span className="ml-auto flex items-center gap-1 text-[10px] bg-danger text-white px-1.5 py-0.5 rounded-full animate-pulse">
              <AlertTriangle className="h-2.5 w-2.5" />
              BREACHED
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Daily Loss</span>
          <span className="text-xs font-semibold font-number text-danger">{formatUSD(dailyLoss)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Daily Profit</span>
          <span className="text-xs font-semibold font-number text-success">{formatUSD(dailyProfit)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Consec. Losses</span>
          <span className={cn(
            "text-xs font-semibold font-number",
            consecutiveLosses >= 3 ? "text-warning" : "text-foreground"
          )}>
            {consecutiveLosses}
          </span>
        </div>

        <div className="pt-1 border-t border-border">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[11px] text-muted-foreground">Risk Used</span>
            <span className={cn("text-sm font-bold font-number", getRiskColor())}>
              {riskPercent.toFixed(0)}%
            </span>
          </div>
          <div className="h-1.5 w-full bg-surface-light rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-500", getSegmentFill())}
              style={{ width: `${Math.min(riskPercent, 100)}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
