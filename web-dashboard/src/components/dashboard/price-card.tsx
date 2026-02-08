"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TrendingUp, TrendingDown } from "lucide-react";
import { Sparkline } from "./sparkline";
import { cn, formatGoldPrice, getValueColor } from "@/lib/utils";
import { useAnimatedValue } from "@/hooks/use-animated-value";

interface PriceCardProps {
  price: number;
  spread: number;
  priceChange: number;
  priceHistory?: number[];
}

export function PriceCard({ price, spread, priceChange, priceHistory = [] }: PriceCardProps) {
  const isUp = priceChange >= 0;
  const animPrice = useAnimatedValue(price);
  const animChange = useAnimatedValue(priceChange);

  const high = priceHistory.length > 0 ? Math.max(...priceHistory) : price;
  const low = priceHistory.length > 0 ? Math.min(...priceHistory) : price;
  const range = high - low;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className={cn("glass h-full overflow-hidden flex flex-col accent-top-blue cursor-pointer", isUp ? "glass-green" : "glass-red")}>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-apple-blue flex items-center gap-1.5 uppercase tracking-wider">
              <TrendingUp className="h-4 w-4" />
              XAUUSD
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 min-h-0 flex flex-col justify-between">
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  key={animPrice.changeKey}
                  className={cn(
                    "text-3xl font-bold font-number leading-tight",
                    isUp ? "text-success" : "text-danger",
                    animPrice.direction === "up" && "flash-up",
                    animPrice.direction === "down" && "flash-down"
                  )}
                >
                  ${formatGoldPrice(animPrice.displayValue)}
                </span>
              </TooltipTrigger>
              <TooltipContent>
                <p>Current XAUUSD spot price — click for detail</p>
              </TooltipContent>
            </Tooltip>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1">
                {isUp ? <TrendingUp className="h-4 w-4 text-success" /> : <TrendingDown className="h-4 w-4 text-danger" />}
                <span
                  key={animChange.changeKey}
                  className={cn(
                    "text-base font-semibold font-number",
                    getValueColor(priceChange),
                    animChange.direction === "up" && "flash-up",
                    animChange.direction === "down" && "flash-down"
                  )}
                >
                  {animChange.displayValue >= 0 ? "+" : ""}{animChange.displayValue.toFixed(2)}
                </span>
              </div>
              <span className="text-sm text-muted-foreground font-number">{spread.toFixed(1)}p</span>
            </div>

            {priceHistory.length > 2 && (
              <div className="-mx-1 mt-auto">
                <Sparkline data={priceHistory.slice(-30)} color={isUp ? "#34C759" : "#FF3B30"} height={20} />
              </div>
            )}
          </CardContent>
        </Card>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-apple-blue">
            <TrendingUp className="h-5 w-5" />
            XAUUSD Price Detail
          </DialogTitle>
          <DialogDescription>Live gold price and recent history</DialogDescription>
        </DialogHeader>
        <div className="pt-4 space-y-4">
          <div className="flex items-baseline gap-3">
            <span className={cn("text-4xl font-bold font-number", isUp ? "text-success" : "text-danger")}>
              ${formatGoldPrice(price)}
            </span>
            <span className={cn("text-xl font-semibold font-number", getValueColor(priceChange))}>
              {priceChange >= 0 ? "+" : ""}{priceChange.toFixed(2)}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Spread</span>
              <p className="text-lg font-semibold font-number">{spread.toFixed(1)} pts</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Session High</span>
              <p className="text-lg font-semibold font-number text-apple-green">${formatGoldPrice(high)}</p>
            </div>
            <div className="space-y-1">
              <span className="text-sm text-muted-foreground">Session Low</span>
              <p className="text-lg font-semibold font-number text-apple-red">${formatGoldPrice(low)}</p>
            </div>
          </div>

          <div className="space-y-1">
            <span className="text-sm text-muted-foreground">Range: {range.toFixed(2)} pts</span>
            <div className="h-2 w-full bg-black/[0.04] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bar-blue"
                style={{ width: range > 0 ? `${Math.min(((price - low) / range) * 100, 100)}%` : "50%" }}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground font-number">
              <span>${formatGoldPrice(low)}</span>
              <span>${formatGoldPrice(high)}</span>
            </div>
          </div>

          {priceHistory.length > 2 && (
            <div>
              <span className="text-sm text-muted-foreground mb-1 block">Price History ({priceHistory.length} ticks)</span>
              <Sparkline data={priceHistory} color={isUp ? "#34C759" : "#FF3B30"} height={80} />
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
