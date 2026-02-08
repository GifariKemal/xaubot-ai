"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, TrendingDown } from "lucide-react";
import { Sparkline } from "./sparkline";
import { cn, formatGoldPrice, getValueColor } from "@/lib/utils";

interface PriceCardProps {
  price: number;
  spread: number;
  priceChange: number;
  priceHistory?: number[];
}

export function PriceCard({ price, spread, priceChange, priceHistory = [] }: PriceCardProps) {
  const isUp = priceChange >= 0;

  return (
    <Card className="glass">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
          XAUUSD
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-1.5">
          <span className={cn("text-2xl font-bold font-number", getValueColor(priceChange))}>
            ${formatGoldPrice(price)}
          </span>
        </div>

        <div className="flex items-center justify-between mt-1">
          <div className="flex items-center gap-1">
            {isUp ? (
              <TrendingUp className="h-3 w-3 text-success" />
            ) : (
              <TrendingDown className="h-3 w-3 text-danger" />
            )}
            <span className={cn("text-xs font-medium font-number", getValueColor(priceChange))}>
              {isUp ? "+" : ""}{priceChange.toFixed(2)}
            </span>
          </div>
          <span className="text-[11px] text-muted-foreground font-number">
            {spread.toFixed(1)}p
          </span>
        </div>

        {priceHistory.length > 2 && (
          <div className="mt-1.5 -mx-1">
            <Sparkline
              data={priceHistory.slice(-30)}
              color={isUp ? "#22c55e" : "#ef4444"}
              height={24}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
