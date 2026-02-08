"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, Inbox } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Position } from "@/types/trading";

interface PositionsCardProps {
  positions: Position[];
}

export function PositionsCard({ positions }: PositionsCardProps) {
  return (
    <Card className="glass h-full flex flex-col">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Layers className="h-3.5 w-3.5" />
          Positions
          {positions.length > 0 && (
            <Badge variant="secondary" className="ml-auto text-[10px] h-4 px-1.5">
              {positions.length}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-auto">
        {positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Inbox className="h-5 w-5 text-muted-foreground/30 mb-1" />
            <p className="text-[11px] text-muted-foreground/60">No open positions</p>
          </div>
        ) : (
          <div className="space-y-1">
            {positions.map((pos) => (
              <div
                key={pos.ticket}
                className={cn(
                  "flex items-center justify-between p-1.5 rounded-md bg-surface-light/50",
                  pos.type === "BUY" ? "border-l-2 border-l-success" : "border-l-2 border-l-danger"
                )}
              >
                <div className="flex items-center gap-1.5">
                  <Badge
                    variant={pos.type === "BUY" ? "success" : "danger"}
                    className="text-[10px] h-4 px-1"
                  >
                    {pos.type}
                  </Badge>
                  <span className="text-[11px] font-number">
                    {pos.volume} @ {pos.priceOpen.toFixed(2)}
                  </span>
                </div>
                <span className={cn(
                  "text-[11px] font-bold font-number",
                  pos.profit >= 0 ? "text-success" : "text-danger"
                )}>
                  {pos.profit >= 0 ? "+" : ""}${pos.profit.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
