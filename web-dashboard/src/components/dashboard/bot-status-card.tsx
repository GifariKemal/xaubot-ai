"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Activity, Timer, Brain, Gauge, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import type { RiskMode, CooldownStatus, AutoTrainerStatus, PerformanceStatus, MarketCloseStatus } from "@/types/trading";

interface BotStatusCardProps {
  riskMode?: RiskMode;
  cooldown?: CooldownStatus;
  autoTrainer?: AutoTrainerStatus;
  performance?: PerformanceStatus;
  marketClose?: MarketCloseStatus;
}

function getRiskModeVariant(mode: string) {
  switch (mode) {
    case "normal": return "success";
    case "recovery": return "warning";
    case "protected": return "danger";
    case "stopped": return "danger";
    default: return "secondary";
  }
}

export function BotStatusCard({ riskMode, cooldown, autoTrainer, performance, marketClose }: BotStatusCardProps) {
  const mode = riskMode?.mode || "unknown";
  const aucColor = (autoTrainer?.currentAuc ?? 0) >= 0.7 ? "text-success" : (autoTrainer?.currentAuc ?? 0) >= 0.65 ? "text-warning" : "text-danger";

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Activity className="h-3.5 w-3.5" />
          Bot Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {/* Risk Mode */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground">Risk Mode</span>
          <Badge
            variant={getRiskModeVariant(mode) as "success" | "warning" | "danger" | "secondary"}
            className={cn("text-[10px] h-4 px-1.5 uppercase", mode === "stopped" && "animate-pulse")}
          >
            {mode}
          </Badge>
        </div>

        {/* Cooldown */}
        <div className="space-y-0.5">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground flex items-center gap-1">
              <Timer className="h-2.5 w-2.5" />
              Cooldown
            </span>
            <span className={cn("text-[10px] font-number", cooldown?.active ? "text-warning" : "text-muted-foreground/60")}>
              {cooldown?.active ? `${cooldown.secondsRemaining}s` : "Ready"}
            </span>
          </div>
          {cooldown?.active && (
            <div className="h-1 w-full bg-surface-light rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-warning transition-all duration-1000"
                style={{ width: `${cooldown.totalSeconds > 0 ? ((cooldown.totalSeconds - cooldown.secondsRemaining) / cooldown.totalSeconds) * 100 : 0}%` }}
              />
            </div>
          )}
        </div>

        {/* Auto Trainer */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground flex items-center gap-1">
            <Brain className="h-2.5 w-2.5" />
            Model AUC
          </span>
          <span className={cn("text-[10px] font-bold font-number", aucColor)}>
            {autoTrainer?.currentAuc != null ? autoTrainer.currentAuc.toFixed(3) : "N/A"}
          </span>
        </div>

        {/* Performance */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground flex items-center gap-1">
            <Gauge className="h-2.5 w-2.5" />
            Uptime
          </span>
          <span className="text-[10px] font-number text-foreground">
            {performance ? `${performance.uptimeHours}h | ${performance.loopCount} loops` : "—"}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground">Exec Speed</span>
          <span className={cn("text-[10px] font-number", (performance?.avgExecutionMs ?? 0) > 50 ? "text-warning" : "text-success")}>
            {performance ? `${performance.avgExecutionMs}ms` : "—"}
          </span>
        </div>

        {/* Market Close */}
        <div className="pt-0.5 border-t border-border flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground flex items-center gap-1">
            <Clock className="h-2.5 w-2.5" />
            Close
          </span>
          <span className={cn("text-[10px] font-number", marketClose?.nearWeekend ? "text-warning font-bold" : "text-muted-foreground")}>
            {marketClose ? `D:${marketClose.hoursToDailyClose}h W:${marketClose.hoursToWeekendClose}h` : "—"}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
