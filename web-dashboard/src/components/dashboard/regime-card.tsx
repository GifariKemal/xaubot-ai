"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity } from "lucide-react";

interface RegimeCardProps {
  name: string;
  volatility: number;
  confidence: number;
}

export function RegimeCard({ name, volatility, confidence }: RegimeCardProps) {
  const getRegimeColor = (regime: string) => {
    if (regime.toLowerCase().includes('high')) return 'text-red-500';
    if (regime.toLowerCase().includes('low')) return 'text-green-500';
    return 'text-amber-500';
  };

  return (
    <Card className="bg-card/50 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Activity className="h-4 w-4" />
          MARKET REGIME
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-center">
          <span className={`text-lg font-bold ${getRegimeColor(name)}`}>
            {name || '---'}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Volatility</span>
          <span className="font-semibold">{volatility.toFixed(2)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Confidence</span>
          <span className="font-semibold">{(confidence * 100).toFixed(0)}%</span>
        </div>
      </CardContent>
    </Card>
  );
}
