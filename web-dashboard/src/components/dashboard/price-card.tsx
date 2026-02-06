"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, TrendingDown } from "lucide-react";

interface PriceCardProps {
  price: number;
  spread: number;
  priceChange: number;
}

export function PriceCard({ price, spread, priceChange }: PriceCardProps) {
  const isUp = priceChange >= 0;

  return (
    <Card className="bg-card/50 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          PRICE
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2">
          <span className={`text-3xl font-bold ${isUp ? 'text-green-500' : 'text-red-500'}`}>
            {price.toFixed(2)}
          </span>
          <span className="text-xs text-muted-foreground">XAUUSD</span>
        </div>
        <div className="flex items-center gap-2 mt-2">
          {isUp ? (
            <TrendingUp className="h-4 w-4 text-green-500" />
          ) : (
            <TrendingDown className="h-4 w-4 text-red-500" />
          )}
          <span className={`text-sm ${isUp ? 'text-green-500' : 'text-red-500'}`}>
            {isUp ? '+' : ''}{priceChange.toFixed(2)}
          </span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Spread: {spread.toFixed(1)} pips
        </p>
      </CardContent>
    </Card>
  );
}
