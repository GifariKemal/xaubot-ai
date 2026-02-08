"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock, Sparkles, CheckCircle2, XCircle, Ban } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TimeFilter } from "@/types/trading";

interface SessionCardProps {
  session: string;
  isGoldenTime: boolean;
  canTrade: boolean;
  sessionMultiplier?: number;
  timeFilter?: TimeFilter;
}

export function SessionCard({ session, isGoldenTime, canTrade, sessionMultiplier, timeFilter }: SessionCardProps) {
  const getSessionColor = (s: string) => {
    const lower = s.toLowerCase();
    if (lower.includes("london")) return "text-info";
    if (lower.includes("new york") || lower.includes("ny")) return "text-success";
    if (lower.includes("sydney") || lower.includes("asian")) return "text-accent";
    return "text-warning";
  };

  const mult = sessionMultiplier ?? 1.0;
  const multLabel = `${mult}x`;
  const multVariant = mult < 1 ? "warning" : mult > 1 ? "success" : "secondary";

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Clock className="h-3.5 w-3.5" />
          Session
          {sessionMultiplier != null && (
            <Badge variant={multVariant as "warning" | "success" | "secondary"} className="ml-auto text-[10px] h-4 px-1">
              {multLabel}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        <span className={cn("text-lg font-bold block", getSessionColor(session))}>
          {session || "Closed"}
        </span>

        <div className="flex items-center gap-1.5">
          <Sparkles className={cn(
            "h-3 w-3",
            isGoldenTime ? "text-warning" : "text-muted-foreground/40"
          )} />
          <span className={cn(
            "text-[11px]",
            isGoldenTime ? "text-warning font-semibold" : "text-muted-foreground"
          )}>
            {isGoldenTime ? "Golden Hour" : "Standard Hours"}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {canTrade ? (
            <CheckCircle2 className="h-3 w-3 text-success" />
          ) : (
            <XCircle className="h-3 w-3 text-danger" />
          )}
          <Badge variant={canTrade ? "success" : "danger"} className="text-[10px] h-5">
            {canTrade ? "CAN TRADE" : "NO TRADE"}
          </Badge>
        </div>

        {/* Time Filter Status */}
        {timeFilter && (
          <div className="flex items-center gap-1.5 pt-0.5 border-t border-border">
            {timeFilter.isBlocked ? (
              <Ban className="h-3 w-3 text-danger" />
            ) : (
              <Clock className="h-3 w-3 text-muted-foreground/40" />
            )}
            <span className={cn(
              "text-[10px]",
              timeFilter.isBlocked ? "text-danger font-semibold" : "text-muted-foreground"
            )}>
              WIB {timeFilter.wibHour}:00{timeFilter.isBlocked ? " BLOCKED" : ""}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
