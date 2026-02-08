"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
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
    if (lower.includes("london")) return "text-apple-blue";
    if (lower.includes("new york") || lower.includes("ny")) return "text-apple-green";
    if (lower.includes("sydney") || lower.includes("asian")) return "text-apple-purple";
    return "text-apple-orange";
  };

  const mult = sessionMultiplier ?? 1.0;
  const isDisabledSession = session.toLowerCase().includes("tokyo-london") || mult === 0;

  return (
    <Card className="glass h-full overflow-hidden flex flex-col accent-top-purple glass-purple">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-apple-purple flex items-center gap-1.5 uppercase tracking-wider">
          <Clock className="h-4 w-4" />
          Session
          {sessionMultiplier != null && (
            <Badge variant={mult < 1 ? "warning" : mult > 1 ? "success" : "secondary"} className="ml-auto text-xs h-5 px-1.5">
              {mult}x
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 flex flex-col justify-between">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className={cn(
              "text-xl font-bold block",
              isDisabledSession ? "text-muted-foreground/50" : getSessionColor(session)
            )}>
              {session || "Closed"}
              {isDisabledSession && <span className="text-sm font-normal text-danger ml-2">OFF</span>}
            </span>
          </TooltipTrigger>
          <TooltipContent><p>Current forex trading session</p></TooltipContent>
        </Tooltip>

        <div className="space-y-1">
          <div className="flex items-center gap-1.5">
            <Sparkles className={cn("h-4 w-4", isGoldenTime ? "text-apple-orange" : "text-muted-foreground/40")} />
            <span className={cn("text-sm", isGoldenTime ? "text-apple-orange font-semibold" : "text-muted-foreground")}>
              {isGoldenTime ? "Golden Hour" : "Standard"}
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            {canTrade ? <CheckCircle2 className="h-4 w-4 text-success" /> : <XCircle className="h-4 w-4 text-danger" />}
            <Badge variant={canTrade ? "success" : "danger"} className="text-xs h-5">
              {canTrade ? "CAN TRADE" : "NO TRADE"}
            </Badge>
          </div>

          {timeFilter && (
            <div className="flex items-center gap-1.5 pt-1 border-t border-border">
              {timeFilter.isBlocked ? <Ban className="h-4 w-4 text-danger" /> : <Clock className="h-4 w-4 text-muted-foreground/40" />}
              <span className={cn("text-sm", timeFilter.isBlocked ? "text-danger font-semibold" : "text-muted-foreground")}>
                WIB {timeFilter.wibHour}:00{timeFilter.isBlocked ? " BLOCKED" : ""}
              </span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
