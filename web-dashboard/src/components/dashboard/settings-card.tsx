"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings2 } from "lucide-react";
import type { BotSettings } from "@/types/trading";

interface SettingsCardProps {
  settings: BotSettings;
}

export function SettingsCard({ settings }: SettingsCardProps) {
  const rows: { label: string; value: string }[] = [
    { label: "Mode", value: settings.capitalMode.toUpperCase() },
    { label: "Capital", value: `$${settings.capital.toLocaleString()}` },
    { label: "TF", value: `${settings.executionTF}/${settings.trendTF}` },
    { label: "Risk", value: `${settings.riskPerTrade}%` },
    { label: "Max Loss", value: `${settings.maxDailyLoss}%` },
    { label: "Leverage", value: `1:${settings.leverage}` },
    { label: "Max Lot", value: `${settings.maxLotSize}` },
    { label: "Max Pos", value: `${settings.maxPositions}` },
    { label: "R:R", value: `1:${settings.minRR}` },
    { label: "ML Conf", value: `${(settings.mlConfidence * 100).toFixed(0)}%` },
    { label: "Cooldown", value: `${settings.cooldownSeconds}s` },
    { label: "Symbol", value: settings.symbol },
  ];

  return (
    <Card className="glass h-full overflow-hidden flex flex-col">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Settings2 className="h-4 w-4" />
          Settings
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-auto">
        <div className="grid grid-cols-3 gap-x-3 gap-y-1">
          {rows.map((row) => (
            <div key={row.label} className="flex justify-between items-center gap-1 rounded px-1 -mx-1 row-hover">
              <span className="text-sm text-muted-foreground truncate">{row.label}</span>
              <span className="text-sm font-semibold font-number text-foreground shrink-0">{row.value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
