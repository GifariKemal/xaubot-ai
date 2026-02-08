"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Activity, Timer, Brain, Gauge, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import type { RiskMode, CooldownStatus, AutoTrainerStatus, PerformanceStatus, MarketCloseStatus, BotSettings } from "@/types/trading";

interface BotStatusCardProps {
  riskMode?: RiskMode;
  cooldown?: CooldownStatus;
  autoTrainer?: AutoTrainerStatus;
  performance?: PerformanceStatus;
  marketClose?: MarketCloseStatus;
  settings?: BotSettings;
}

function getRiskModeVariant(mode: string) {
  switch (mode) {
    case "normal": return "success";
    case "recovery": return "warning";
    case "protected": case "stopped": return "danger";
    default: return "secondary";
  }
}

export function BotStatusCard({ riskMode, cooldown, autoTrainer, performance, marketClose, settings }: BotStatusCardProps) {
  const mode = riskMode?.mode || "unknown";
  const aucColor = (autoTrainer?.currentAuc ?? 0) >= 0.7 ? "text-apple-green" : (autoTrainer?.currentAuc ?? 0) >= 0.65 ? "text-apple-orange" : "text-apple-red";

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className="glass h-full overflow-hidden flex flex-col accent-top-blue glass-blue cursor-pointer">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-apple-blue flex items-center gap-1.5 uppercase tracking-wider">
              <Activity className="h-4 w-4" />
              Bot Status
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 overflow-auto space-y-1">
            <div className="flex items-center justify-between rounded px-1 -mx-1 row-hover">
              <span className="text-sm text-muted-foreground">Mode</span>
              <Badge variant={getRiskModeVariant(mode) as any} className={cn("text-xs h-5 px-1.5 uppercase", mode === "stopped" && "animate-pulse")}>{mode}</Badge>
            </div>

            <div className="flex items-center justify-between rounded px-1 -mx-1 row-hover">
              <span className="text-sm text-muted-foreground flex items-center gap-1"><Timer className="h-3.5 w-3.5 text-apple-orange" />Cooldown</span>
              <span className={cn("text-sm font-number", cooldown?.active ? "text-apple-orange font-semibold" : "text-muted-foreground/60")}>
                {cooldown?.active ? `${cooldown.secondsRemaining}s` : "Ready"}
              </span>
            </div>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-between cursor-help rounded px-1 -mx-1 row-hover">
                  <span className="text-sm text-muted-foreground flex items-center gap-1"><Brain className="h-3.5 w-3.5 text-apple-purple" />AUC</span>
                  <span className={cn("text-sm font-bold font-number", aucColor)}>
                    {autoTrainer?.currentAuc != null ? autoTrainer.currentAuc.toFixed(3) : "N/A"}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent><p>Model accuracy. &gt;0.70 good</p></TooltipContent>
            </Tooltip>

            <div className="flex items-center justify-between rounded px-1 -mx-1 row-hover">
              <span className="text-sm text-muted-foreground flex items-center gap-1"><Gauge className="h-3.5 w-3.5 text-apple-cyan" />Uptime</span>
              <span className="text-sm font-number text-apple-cyan">{performance ? `${performance.uptimeHours}h` : "\u2014"}</span>
            </div>

            <div className="flex items-center justify-between rounded px-1 -mx-1 row-hover">
              <span className="text-sm text-muted-foreground">Speed</span>
              <span className={cn("text-sm font-number", (performance?.avgExecutionMs ?? 0) > 50 ? "text-apple-orange" : "text-apple-green")}>
                {performance ? `${performance.avgExecutionMs}ms` : "\u2014"}
              </span>
            </div>

            <div className="flex items-center justify-between rounded px-1 -mx-1 row-hover">
              <span className="text-sm text-muted-foreground flex items-center gap-1"><Clock className="h-3.5 w-3.5 text-apple-blue" />Close</span>
              <span className={cn("text-sm font-number", marketClose?.nearWeekend ? "text-apple-orange font-bold" : "text-muted-foreground")}>
                {marketClose ? `D:${marketClose.hoursToDailyClose}h W:${marketClose.hoursToWeekendClose}h` : "\u2014"}
              </span>
            </div>

            {settings && (
              <div className="pt-1 border-t border-border space-y-0.5">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Capital</span>
                  <span className="font-number font-semibold text-apple-green">${settings.capital.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Risk</span>
                  <span className="font-number text-apple-orange">{settings.riskPerTrade}% | R:R 1:{settings.minRR}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">TF</span>
                  <span className="font-number text-apple-blue">{settings.executionTF}/{settings.trendTF}</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-apple-blue">
            <Activity className="h-5 w-5" />
            Bot Status Detail
          </DialogTitle>
          <DialogDescription>Full bot operational status and configuration</DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="flex items-center gap-3">
            <Badge variant={getRiskModeVariant(mode) as any} className="text-sm px-3 py-1 uppercase">{mode}</Badge>
            {cooldown?.active && <Badge variant="warning" className="text-sm px-3 py-1">Cooldown {cooldown.secondsRemaining}s</Badge>}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Model AUC</span>
              <p className={cn("text-xl font-bold font-number", aucColor)}>
                {autoTrainer?.currentAuc != null ? autoTrainer.currentAuc.toFixed(4) : "N/A"}
              </p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Uptime</span>
              <p className="text-xl font-bold font-number text-apple-cyan">{performance?.uptimeHours ?? 0}h</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Avg Execution</span>
              <p className={cn("text-xl font-bold font-number", (performance?.avgExecutionMs ?? 0) > 50 ? "text-apple-orange" : "text-apple-green")}>
                {performance?.avgExecutionMs ?? 0}ms
              </p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Session Trades</span>
              <p className="text-xl font-bold font-number text-apple-blue">{performance?.totalSessionTrades ?? 0}</p>
            </div>
          </div>

          {marketClose && (
            <div className="pt-3 border-t border-border">
              <span className="text-sm text-muted-foreground mb-2 block">Market Close</span>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <span className="text-sm text-muted-foreground">Daily Close</span>
                  <p className="text-lg font-bold font-number">{marketClose.hoursToDailyClose}h</p>
                </div>
                <div className="space-y-1">
                  <span className="text-sm text-muted-foreground">Weekend Close</span>
                  <p className={cn("text-lg font-bold font-number", marketClose.nearWeekend ? "text-apple-orange" : "text-foreground")}>
                    {marketClose.hoursToWeekendClose}h
                    {marketClose.nearWeekend && " (NEAR)"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {settings && (
            <div className="pt-3 border-t border-border">
              <span className="text-sm text-muted-foreground mb-2 block">Configuration</span>
              <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Capital</span><span className="font-semibold font-number">${settings.capital.toLocaleString()}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Mode</span><span className="font-semibold uppercase">{settings.capitalMode}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Risk/Trade</span><span className="font-semibold font-number">{settings.riskPerTrade}%</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Max Daily Loss</span><span className="font-semibold font-number">{settings.maxDailyLoss}%</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Timeframes</span><span className="font-semibold font-number">{settings.executionTF}/{settings.trendTF}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Leverage</span><span className="font-semibold font-number">1:{settings.leverage}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Max Lot</span><span className="font-semibold font-number">{settings.maxLotSize}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Max Positions</span><span className="font-semibold font-number">{settings.maxPositions}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Min R:R</span><span className="font-semibold font-number">1:{settings.minRR}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">ML Confidence</span><span className="font-semibold font-number">{(settings.mlConfidence * 100).toFixed(0)}%</span></div>
              </div>
            </div>
          )}

          {autoTrainer && (
            <div className="pt-3 border-t border-border">
              <span className="text-sm text-muted-foreground mb-2 block">Auto-Trainer</span>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Current AUC</span><span className={cn("font-bold font-number", aucColor)}>{autoTrainer.currentAuc?.toFixed(4) ?? "N/A"}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Last Train</span><span className="font-number">{autoTrainer.lastRetrain ?? "Never"}</span></div>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
