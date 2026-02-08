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
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Settings2 className="h-3.5 w-3.5" />
          Bot Settings
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-x-3 gap-y-1">
          {rows.map((row) => (
            <div key={row.label} className="flex justify-between items-center gap-1">
              <span className="text-[10px] text-muted-foreground truncate">{row.label}</span>
              <span className="text-[10px] font-semibold font-number text-foreground shrink-0">{row.value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
