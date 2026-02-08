"use client";

import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { BarChart3, Target, Zap, TrendingUp } from "lucide-react";
import { cn, formatUSD } from "@/lib/utils";
import { useAnimatedValue } from "@/hooks/use-animated-value";
import type { PerformanceStatus, RiskMode } from "@/types/trading";

interface PerformanceCardProps {
  marketScore?: number;
  marketQuality?: string;
  dynamicThreshold?: number;
  performance?: PerformanceStatus;
  riskMode?: RiskMode;
}

function getQualityVariant(quality: string): "success" | "warning" | "danger" | "info" | "secondary" {
  switch (quality?.toUpperCase()) {
    case "EXCELLENT": return "success";
    case "GOOD": return "info";
    case "MODERATE": return "warning";
    case "POOR": case "AVOID": return "danger";
    default: return "secondary";
  }
}

function getScoreColor(score: number) {
  if (score >= 80) return "text-apple-green";
  if (score >= 65) return "text-apple-blue";
  if (score >= 50) return "text-apple-orange";
  return "text-apple-red";
}

function getScoreBarClass(score: number) {
  if (score >= 80) return "bar-green";
  if (score >= 65) return "bar-blue";
  if (score >= 50) return "bar-orange";
  return "bar-red";
}

export function PerformanceCard({ marketScore = 0, marketQuality = "unknown", dynamicThreshold = 0, performance, riskMode }: PerformanceCardProps) {
  const firstMount = useRef(true);
  const animScore = useAnimatedValue(marketScore);

  const shouldAnimate = firstMount.current;
  if (firstMount.current) firstMount.current = false;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className="glass h-full overflow-hidden flex flex-col accent-top-orange glass-orange cursor-pointer">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-apple-orange flex items-center gap-1.5 uppercase tracking-wider">
              <BarChart3 className="h-4 w-4" />
              Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 flex flex-col justify-between">
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="text-sm text-muted-foreground flex items-center gap-1 cursor-help">
                      <Target className="h-3.5 w-3.5 text-apple-orange" />
                      Score
                    </span>
                  </TooltipTrigger>
                  <TooltipContent><p>Click for detail</p></TooltipContent>
                </Tooltip>
                <div className="flex items-center gap-1">
                  <span
                    key={animScore.changeKey}
                    className={cn(
                      "text-xl font-bold font-number",
                      getScoreColor(marketScore),
                      animScore.direction === "up" && "flash-up",
                      animScore.direction === "down" && "flash-down"
                    )}
                  >
                    {animScore.displayValue.toFixed(0)}
                  </span>
                  <span className="text-xs text-muted-foreground">/100</span>
                </div>
              </div>
              <div className="h-2.5 w-full bg-black/[0.04] rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    getScoreBarClass(marketScore),
                    shouldAnimate && "bar-animate-in"
                  )}
                  style={{ width: `${Math.min(animScore.displayValue, 100)}%` }}
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Quality</span>
                <Badge variant={getQualityVariant(marketQuality)} className="text-xs h-5 px-1.5 uppercase">{marketQuality}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground flex items-center gap-1"><Zap className="h-3.5 w-3.5 text-apple-cyan" />Threshold</span>
                <span className="text-sm font-bold font-number text-apple-cyan">{(dynamicThreshold * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="pt-1 border-t border-border space-y-0.5">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground flex items-center gap-1"><TrendingUp className="h-3.5 w-3.5 text-apple-blue" />Trades</span>
                <span className="text-sm font-semibold font-number text-apple-blue">{performance?.totalSessionTrades ?? 0}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">P&L</span>
                <span className={cn("text-sm font-semibold font-number", (performance?.totalSessionProfit ?? 0) >= 0 ? "text-apple-green" : "text-apple-red")}>
                  {formatUSD(performance?.totalSessionProfit ?? 0)}
                </span>
              </div>
              {riskMode && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Lot</span>
                  <span className="text-sm font-semibold font-number text-apple-purple">{riskMode.recommendedLot}</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-apple-orange">
            <BarChart3 className="h-5 w-5" />
            Performance Detail
          </DialogTitle>
          <DialogDescription>Market quality scoring and session trading stats</DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="flex items-center gap-3">
            <span className={cn("text-4xl font-bold font-number", getScoreColor(marketScore))}>{marketScore}</span>
            <span className="text-muted-foreground text-lg">/100</span>
            <Badge variant={getQualityVariant(marketQuality)} className="text-sm px-3 py-1 uppercase">{marketQuality}</Badge>
          </div>

          <div className="h-3 w-full bg-black/[0.04] rounded-full overflow-hidden">
            <div className={cn("h-full rounded-full transition-all", getScoreBarClass(marketScore))} style={{ width: `${Math.min(marketScore, 100)}%` }} />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Dynamic Threshold</span>
              <p className="text-xl font-bold font-number text-apple-cyan">{(dynamicThreshold * 100).toFixed(1)}%</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Session Trades</span>
              <p className="text-xl font-bold font-number text-apple-blue">{performance?.totalSessionTrades ?? 0}</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Session P/L</span>
              <p className={cn("text-xl font-bold font-number", (performance?.totalSessionProfit ?? 0) >= 0 ? "text-apple-green" : "text-apple-red")}>
                {formatUSD(performance?.totalSessionProfit ?? 0)}
              </p>
            </div>
            {riskMode && (
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Recommended Lot</span>
                <p className="text-xl font-bold font-number text-apple-purple">{riskMode.recommendedLot}</p>
              </div>
            )}
          </div>

          {performance && (
            <div className="pt-3 border-t border-border space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Uptime</span>
                <span className="font-semibold font-number text-apple-cyan">{performance.uptimeHours}h</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Avg Execution</span>
                <span className={cn("font-semibold font-number", performance.avgExecutionMs > 50 ? "text-apple-orange" : "text-apple-green")}>
                  {performance.avgExecutionMs}ms
                </span>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
