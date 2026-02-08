"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Activity, Clock } from "lucide-react";
import { cn, getConfidenceColor } from "@/lib/utils";
import { useAnimatedValue } from "@/hooks/use-animated-value";

interface RegimeCardProps {
  name: string;
  volatility: number;
  confidence: number;
  updatedAt?: string;
  h1Bias?: string;
}

export function RegimeCard({ name, volatility, confidence, updatedAt, h1Bias }: RegimeCardProps) {
  const getRegimeBadgeVariant = (regime: string) => {
    const lower = regime.toLowerCase();
    if (lower.includes("high") || lower.includes("volatile") || lower.includes("crisis")) return "danger";
    if (lower.includes("low") || lower.includes("ranging")) return "success";
    if (lower.includes("trend")) return "info";
    return "warning";
  };

  const confidencePercent = confidence * 100;
  const animVol = useAnimatedValue(volatility);
  const animConf = useAnimatedValue(confidencePercent);

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className="glass h-full overflow-hidden flex flex-col accent-top-pink glass-pink cursor-pointer">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-apple-pink flex items-center gap-1.5 uppercase tracking-wider">
              <Activity className="h-4 w-4" />
              Regime
              {updatedAt && (
                <span className="ml-auto flex items-center gap-1 text-xs text-muted-foreground/60 font-number normal-case tracking-normal">
                  <Clock className="h-3 w-3" />
                  {updatedAt}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 flex flex-col justify-between">
            <Tooltip>
              <TooltipTrigger asChild>
                <div>
                  <Badge variant={getRegimeBadgeVariant(name) as any} className="text-base font-bold px-2 py-0.5">
                    {name || "Unknown"}
                  </Badge>
                </div>
              </TooltipTrigger>
              <TooltipContent><p>Click for detail</p></TooltipContent>
            </Tooltip>

            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Volatility</span>
                <span className="text-base font-semibold font-number text-apple-orange">{animVol.displayValue.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Confidence</span>
                <span className={cn("text-base font-semibold font-number", getConfidenceColor(animConf.displayValue))}>
                  {animConf.displayValue.toFixed(0)}%
                </span>
              </div>
              {h1Bias && (
                <div className="flex justify-between items-center pt-1 border-t border-border">
                  <span className="text-sm text-muted-foreground">H1 Bias</span>
                  <span className={cn(
                    "text-base font-bold",
                    h1Bias === "BULLISH" ? "text-apple-green" : h1Bias === "BEARISH" ? "text-apple-red" : "text-muted-foreground"
                  )}>
                    {h1Bias === "BULLISH" ? "^ " : h1Bias === "BEARISH" ? "v " : ""}{h1Bias}
                  </span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-apple-pink">
            <Activity className="h-5 w-5" />
            Market Regime Detail
          </DialogTitle>
          <DialogDescription>HMM-detected market regime and volatility analysis</DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="flex items-center gap-3">
            <Badge variant={getRegimeBadgeVariant(name) as any} className="text-lg font-bold px-3 py-1">
              {name || "Unknown"}
            </Badge>
            {updatedAt && <span className="text-sm text-muted-foreground font-number">{updatedAt}</span>}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Volatility</span>
              <p className="text-2xl font-bold font-number text-apple-orange">{volatility.toFixed(4)}</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Confidence</span>
              <p className={cn("text-2xl font-bold font-number", getConfidenceColor(confidencePercent))}>
                {confidencePercent.toFixed(1)}%
              </p>
              <div className="h-2 w-full bg-black/[0.04] rounded-full overflow-hidden">
                <div className={cn("h-full rounded-full", confidencePercent >= 70 ? "bar-green" : confidencePercent >= 50 ? "bar-orange" : "bar-red")} style={{ width: `${confidencePercent}%` }} />
              </div>
            </div>
          </div>

          {h1Bias && (
            <div className="pt-3 border-t border-border">
              <span className="text-sm text-muted-foreground">H1 Timeframe Bias</span>
              <p className={cn(
                "text-xl font-bold mt-1",
                h1Bias === "BULLISH" ? "text-apple-green" : h1Bias === "BEARISH" ? "text-apple-red" : "text-muted-foreground"
              )}>
                {h1Bias}
              </p>
            </div>
          )}

          <div className="pt-3 border-t border-border text-sm text-muted-foreground">
            <p>The Hidden Markov Model classifies market states based on volatility patterns. Regime transitions affect position sizing, signal thresholds, and exit management.</p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
