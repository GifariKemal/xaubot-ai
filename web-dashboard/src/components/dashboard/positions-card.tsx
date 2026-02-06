"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Layers } from "lucide-react";
import type { Position } from "@/types/trading";

interface PositionsCardProps {
  positions: Position[];
}

export function PositionsCard({ positions }: PositionsCardProps) {
  return (
    <Card className="bg-card/50 backdrop-blur">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Layers className="h-4 w-4" />
          OPEN POSITIONS
          {positions.length > 0 && (
            <Badge variant="secondary" className="ml-auto">{positions.length}</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[100px]">
          {positions.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">
              No open positions
            </p>
          ) : (
            <div className="space-y-2">
              {positions.map((pos) => (
                <div
                  key={pos.ticket}
                  className="flex items-center justify-between p-2 rounded-md bg-muted/50"
                >
                  <div className="flex items-center gap-2">
                    <Badge variant={pos.type === 'BUY' ? 'default' : 'destructive'} className="text-xs">
                      {pos.type}
                    </Badge>
                    <span className="text-sm">{pos.volume} @ {pos.priceOpen.toFixed(2)}</span>
                  </div>
                  <span className={`font-semibold ${pos.profit >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {pos.profit >= 0 ? '+' : ''}${pos.profit.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
