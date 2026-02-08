"use client";

import { useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Brain, BarChart3, Clock } from "lucide-react";
import { cn, getSignalColor, getConfidenceColor } from "@/lib/utils";
import { useAnimatedValue } from "@/hooks/use-animated-value";

interface SignalCardProps {
  title: string;
  icon: "smc" | "ml";
  signal: string;
  confidence: number;
  detail?: string;
  buyProb?: number;
  sellProb?: number;
  updatedAt?: string;
  threshold?: number;
  marketQuality?: string;
}

export function SignalCard({ title, icon, signal, confidence, detail, buyProb, sellProb, updatedAt, threshold, marketQuality }: SignalCardProps) {
  const confidencePercent = confidence * 100;
  const hasSignal = signal && signal.toUpperCase() !== "NO SIGNAL" && signal !== "";
  const normalized = (signal || "").toUpperCase();
  const firstMount = useRef(true);

  const animConf = useAnimatedValue(confidencePercent);

  const getBorderClass = () => {
    if (normalized === "BUY") return "signal-buy";
    if (normalized === "SELL") return "signal-sell";
    if (normalized === "HOLD") return "signal-hold";
    return "signal-none";
  };

  const getBarClass = () => {
    if (normalized === "BUY") return "bar-green";
    if (normalized === "SELL") return "bar-red";
    if (normalized === "HOLD") return "bar-orange";
    return "bg-muted";
  };

  const iconColor = icon === "smc" ? "text-apple-cyan" : "text-apple-purple";
  const topClass = icon === "smc" ? "accent-top-cyan" : "accent-top-purple";
  const hoverClass = icon === "smc" ? "glass-cyan" : "glass-purple";

  const shouldAnimate = firstMount.current;
  if (firstMount.current) firstMount.current = false;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className={cn("glass h-full overflow-hidden flex flex-col cursor-pointer", topClass, hoverClass, getBorderClass())}>
          <CardHeader>
            <CardTitle className={cn("text-sm font-medium flex items-center gap-1.5 uppercase tracking-wider", iconColor)}>
              {icon === "smc" ? <BarChart3 className="h-4 w-4" /> : <Brain className="h-4 w-4" />}
              {title}
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
                <span className={cn(
                  "text-2xl font-bold block",
                  hasSignal ? getSignalColor(signal) : "text-muted-foreground/60"
                )}>
                  {signal || "NO SIGNAL"}
                </span>
              </TooltipTrigger>
              <TooltipContent>
                <p>Click for detail</p>
              </TooltipContent>
            </Tooltip>

            <div>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-muted-foreground">Confidence</span>
                <span className={cn("text-sm font-semibold font-number", getConfidenceColor(animConf.displayValue))}>
                  {animConf.displayValue.toFixed(0)}%
                  {threshold !== undefined && (
                    <span className="text-muted-foreground font-normal"> /{(threshold * 100).toFixed(0)}%</span>
                  )}
                </span>
              </div>
              <div className="relative h-2 w-full bg-black/[0.04] rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-300",
                    getBarClass(),
                    shouldAnimate && "bar-animate-in"
                  )}
                  style={{ width: `${animConf.displayValue}%` }}
                />
                {threshold !== undefined && (
                  <div className="absolute top-0 h-full w-[2px] bg-foreground/30" style={{ left: `${threshold * 100}%` }} />
                )}
              </div>
            </div>

            {detail && <p className="text-sm text-muted-foreground line-clamp-1">{detail}</p>}

            {buyProb !== undefined && sellProb !== undefined && (
              <div className="flex justify-between gap-2 text-sm font-number">
                <span><span className="text-muted-foreground">Buy </span><span className="text-apple-green font-semibold">{(buyProb * 100).toFixed(0)}%</span></span>
                <span><span className="text-muted-foreground">Sell </span><span className="text-apple-red font-semibold">{(sellProb * 100).toFixed(0)}%</span></span>
              </div>
            )}
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className={cn("flex items-center gap-2", iconColor)}>
            {icon === "smc" ? <BarChart3 className="h-5 w-5" /> : <Brain className="h-5 w-5" />}
            {title} Detail
          </DialogTitle>
          <DialogDescription>
            {icon === "smc" ? "Smart Money Concepts analysis — Order Blocks, FVG, BOS/CHoCH" : "XGBoost ML model prediction probabilities"}
          </DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="flex items-center gap-3">
            <span className={cn("text-3xl font-bold", hasSignal ? getSignalColor(signal) : "text-muted-foreground/60")}>
              {signal || "NO SIGNAL"}
            </span>
            {updatedAt && <span className="text-sm text-muted-foreground font-number">{updatedAt}</span>}
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-muted-foreground">Confidence</span>
              <span className={cn("text-lg font-bold font-number", getConfidenceColor(confidencePercent))}>
                {confidencePercent.toFixed(1)}%
              </span>
            </div>
            <div className="relative h-3 w-full bg-black/[0.04] rounded-full overflow-hidden">
              <div className={cn("h-full rounded-full transition-all", getBarClass())} style={{ width: `${confidencePercent}%` }} />
              {threshold !== undefined && (
                <div className="absolute top-0 h-full w-[2px] bg-foreground/40" style={{ left: `${threshold * 100}%` }} />
              )}
            </div>
            {threshold !== undefined && (
              <p className="text-sm text-muted-foreground mt-1">Threshold: {(threshold * 100).toFixed(0)}% — {confidencePercent >= threshold * 100 ? "ABOVE" : "BELOW"}</p>
            )}
          </div>

          {buyProb !== undefined && sellProb !== undefined && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Buy Probability</span>
                <p className="text-xl font-bold font-number text-apple-green">{(buyProb * 100).toFixed(1)}%</p>
                <div className="h-2 w-full bg-black/[0.04] rounded-full overflow-hidden">
                  <div className="h-full rounded-full bar-green" style={{ width: `${buyProb * 100}%` }} />
                </div>
              </div>
              <div className="space-y-1">
                <span className="text-sm text-muted-foreground">Sell Probability</span>
                <p className="text-xl font-bold font-number text-apple-red">{(sellProb * 100).toFixed(1)}%</p>
                <div className="h-2 w-full bg-black/[0.04] rounded-full overflow-hidden">
                  <div className="h-full rounded-full bar-red" style={{ width: `${sellProb * 100}%` }} />
                </div>
              </div>
            </div>
          )}

          {detail && (
            <div className="pt-3 border-t border-border">
              <span className="text-sm text-muted-foreground">Detail</span>
              <p className="text-sm mt-1">{detail}</p>
            </div>
          )}

          {marketQuality && (
            <div className="flex justify-between text-sm pt-2 border-t border-border">
              <span className="text-muted-foreground">Market Quality</span>
              <span className="font-semibold uppercase">{marketQuality}</span>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
