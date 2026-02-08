"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Clock } from "lucide-react";
import { cn, getConfidenceColor } from "@/lib/utils";

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

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Activity className="h-3.5 w-3.5" />
          Market Regime
          {updatedAt && (
            <span className="ml-auto flex items-center gap-1 text-[10px] text-muted-foreground/60 font-number normal-case tracking-normal">
              <Clock className="h-2.5 w-2.5" />
              {updatedAt}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <Badge variant={getRegimeBadgeVariant(name) as any} className="text-xs font-bold">
          {name || "Unknown"}
        </Badge>

        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Volatility</span>
          <span className="text-sm font-semibold font-number">{volatility.toFixed(2)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[11px] text-muted-foreground">Confidence</span>
          <span className={cn(
            "text-sm font-semibold font-number",
            getConfidenceColor(confidencePercent)
          )}>
            {confidencePercent.toFixed(0)}%
          </span>
        </div>
        {h1Bias && (
          <div className="flex justify-between items-center pt-1 border-t border-border">
            <span className="text-[11px] text-muted-foreground">H1 Bias</span>
            <span className={cn(
              "text-xs font-bold",
              h1Bias === "BULLISH" ? "text-success" :
              h1Bias === "BEARISH" ? "text-danger" :
              "text-muted-foreground"
            )}>
              {h1Bias === "BULLISH" ? "↑ " : h1Bias === "BEARISH" ? "↓ " : ""}{h1Bias}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
