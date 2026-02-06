"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ShieldAlert } from "lucide-react";

interface RiskCardProps {
  dailyLoss: number;
  dailyProfit: number;
  consecutiveLosses: number;
  riskPercent: number;
}

export function RiskCard({ dailyLoss, dailyProfit, consecutiveLosses, riskPercent }: RiskCardProps) {
  const isHighRisk = riskPercent >= 80;
  const isMediumRisk = riskPercent >= 50;

  return (
    <Card className={`bg-card/50 backdrop-blur ${isHighRisk ? 'border-red-500 border-2 animate-pulse' : ''}`}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <ShieldAlert className={`h-4 w-4 ${isHighRisk ? 'text-red-500' : ''}`} />
          RISK STATUS
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Daily Loss</span>
          <span className="font-semibold text-red-500">${dailyLoss.toFixed(2)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Daily Profit</span>
          <span className="font-semibold text-green-500">${dailyProfit.toFixed(2)}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-sm text-muted-foreground">Consec. Losses</span>
          <span className="font-semibold">{consecutiveLosses}</span>
        </div>

        <div className="pt-2 border-t">
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm text-muted-foreground">Risk Used</span>
            <span className={`font-bold ${isHighRisk ? 'text-red-500' : isMediumRisk ? 'text-amber-500' : 'text-green-500'}`}>
              {riskPercent.toFixed(0)}%
            </span>
          </div>
          <Progress
            value={riskPercent}
            className={`h-2 ${isHighRisk ? '[&>div]:bg-red-500' : isMediumRisk ? '[&>div]:bg-amber-500' : '[&>div]:bg-green-500'}`}
          />
        </div>
      </CardContent>
    </Card>
  );
}
