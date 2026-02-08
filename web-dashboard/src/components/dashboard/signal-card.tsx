"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, BarChart3, Clock } from "lucide-react";
import { cn, getSignalColor, getConfidenceColor } from "@/lib/utils";

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

export function SignalCard({
  title,
  icon,
  signal,
  confidence,
  detail,
  buyProb,
  sellProb,
  updatedAt,
  threshold,
  marketQuality,
}: SignalCardProps) {
  const confidencePercent = confidence * 100;
  const hasSignal = signal && signal.toUpperCase() !== "NO SIGNAL" && signal !== "";
  const normalized = (signal || "").toUpperCase();

  const getBorderClass = () => {
    if (normalized === "BUY") return "signal-buy";
    if (normalized === "SELL") return "signal-sell";
    if (normalized === "HOLD") return "signal-hold";
    return "signal-none";
  };

  const getBarColor = () => {
    if (normalized === "BUY") return "bg-success";
    if (normalized === "SELL") return "bg-danger";
    if (normalized === "HOLD") return "bg-warning";
    return "bg-muted";
  };

  return (
    <Card className={cn("glass", getBorderClass())}>
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          {icon === "smc" ? <BarChart3 className="h-3.5 w-3.5" /> : <Brain className="h-3.5 w-3.5" />}
          {title}
          {updatedAt && (
            <span className="ml-auto flex items-center gap-1 text-[10px] text-muted-foreground/60 font-number normal-case tracking-normal">
              <Clock className="h-2.5 w-2.5" />
              {updatedAt}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <span className={cn(
          "text-xl font-bold block",
          hasSignal ? getSignalColor(signal) : "text-muted-foreground/60"
        )}>
          {signal || "NO SIGNAL"}
        </span>

        {/* Confidence bar */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-[11px] text-muted-foreground">Confidence</span>
            <span className={cn(
              "text-[11px] font-semibold font-number",
              getConfidenceColor(confidencePercent)
            )}>
              {confidencePercent.toFixed(0)}%
              {threshold !== undefined && (
                <span className="text-muted-foreground font-normal">
                  /{(threshold * 100).toFixed(0)}%
                </span>
              )}
            </span>
          </div>
          <div className="relative h-1.5 w-full bg-surface-light rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-300", getBarColor())}
              style={{ width: `${confidencePercent}%` }}
            />
            {threshold !== undefined && (
              <div
                className="absolute top-0 h-full w-[2px] bg-foreground/50"
                style={{ left: `${threshold * 100}%` }}
              />
            )}
          </div>
          {threshold !== undefined && (
            <div className="flex justify-between items-center mt-0.5">
              <span className="text-[10px] text-muted-foreground/60">
                {confidencePercent >= threshold * 100 ? "✓ Above" : "✗ Below"} threshold
              </span>
              {marketQuality && (
                <span className="text-[10px] text-muted-foreground/60 font-number">
                  Mkt: {marketQuality}
                </span>
              )}
            </div>
          )}
        </div>

        {detail && (
          <p className="text-[11px] text-muted-foreground line-clamp-1">{detail}</p>
        )}

        {buyProb !== undefined && sellProb !== undefined && (
          <div className="flex justify-between gap-2 text-[11px] font-number">
            <span>
              <span className="text-muted-foreground">Buy </span>
              <span className="text-success font-semibold">{(buyProb * 100).toFixed(0)}%</span>
            </span>
            <span>
              <span className="text-muted-foreground">Sell </span>
              <span className="text-danger font-semibold">{(sellProb * 100).toFixed(0)}%</span>
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
