"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Brain, BarChart3 } from "lucide-react";

interface SignalCardProps {
  title: string;
  icon: "smc" | "ml";
  signal: string;
  confidence: number;
  detail?: string;
  buyProb?: number;
  sellProb?: number;
}

export function SignalCard({ title, icon, signal, confidence, detail, buyProb, sellProb }: SignalCardProps) {
  const getSignalColor = (sig: string) => {
    if (sig === 'BUY') return 'text-green-500';
    if (sig === 'SELL') return 'text-red-500';
    if (sig === 'HOLD') return 'text-amber-500';
    return 'text-muted-foreground';
  };

  const getProgressColor = (sig: string) => {
    if (sig === 'BUY') return '[&>div]:bg-green-500';
    if (sig === 'SELL') return '[&>div]:bg-red-500';
    if (sig === 'HOLD') return '[&>div]:bg-amber-500';
    return '';
  };

  return (
    <Card className="bg-card/50 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          {icon === 'smc' ? <BarChart3 className="h-4 w-4" /> : <Brain className="h-4 w-4" />}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-center">
          <span className={`text-2xl font-bold ${getSignalColor(signal)}`}>
            {signal || 'NO SIGNAL'}
          </span>
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-muted-foreground">Confidence</span>
            <span className="text-xs font-semibold">{(confidence * 100).toFixed(0)}%</span>
          </div>
          <Progress value={confidence * 100} className={`h-1.5 ${getProgressColor(signal)}`} />
        </div>

        {detail && (
          <p className="text-xs text-muted-foreground line-clamp-2">{detail}</p>
        )}

        {buyProb !== undefined && sellProb !== undefined && (
          <div className="flex justify-between text-xs">
            <span className="text-green-500">Buy: {(buyProb * 100).toFixed(0)}%</span>
            <span className="text-red-500">Sell: {(sellProb * 100).toFixed(0)}%</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
