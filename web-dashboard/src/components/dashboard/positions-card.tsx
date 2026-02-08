"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Layers, Inbox, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Position, PositionDetail } from "@/types/trading";

interface PositionsCardProps {
  positions: Position[];
  positionDetails?: PositionDetail[];
}

export function PositionsCard({ positions, positionDetails }: PositionsCardProps) {
  const [expandedTicket, setExpandedTicket] = useState<number | null>(null);
  const getDetail = (ticket: number) => positionDetails?.find((d) => d.ticket === ticket);

  return (
    <Card className="glass h-full overflow-hidden flex flex-col accent-top-cyan glass-cyan">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-apple-cyan flex items-center gap-1.5 uppercase tracking-wider">
          <Layers className="h-4 w-4" />
          Positions
          {positions.length > 0 && (
            <Badge variant="info" className="ml-auto text-xs h-5 px-1.5">{positions.length}</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-auto">
        {positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <Inbox className="h-5 w-5 text-muted-foreground/30 mb-1" />
            <p className="text-sm text-muted-foreground/60">No open positions</p>
          </div>
        ) : (
          <div className="space-y-1">
            {positions.map((pos) => {
              const detail = getDetail(pos.ticket);
              const isExpanded = expandedTicket === pos.ticket;

              return (
                <div key={pos.ticket}>
                  <div
                    className={cn(
                      "flex items-center justify-between p-1.5 rounded-md bg-black/[0.03] row-hover",
                      pos.type === "BUY" ? "border-l-2 border-l-apple-green" : "border-l-2 border-l-apple-red",
                      detail && "cursor-pointer"
                    )}
                    onClick={() => detail && setExpandedTicket(isExpanded ? null : pos.ticket)}
                  >
                    <div className="flex items-center gap-1.5">
                      <Badge variant={pos.type === "BUY" ? "success" : "danger"} className="text-xs h-5 px-1.5">{pos.type}</Badge>
                      <span className="text-sm font-number">{pos.volume} @ {pos.priceOpen.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className={cn(
                        "text-base font-bold font-number",
                        pos.profit >= 0 ? "text-apple-green" : "text-apple-red"
                      )}>
                        {pos.profit >= 0 ? "+" : ""}${pos.profit.toFixed(2)}
                      </span>
                      {detail && (isExpanded ? <ChevronUp className="h-3.5 w-3.5 text-muted-foreground/40" /> : <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/40" />)}
                    </div>
                  </div>

                  {isExpanded && detail && (
                    <div className="ml-2 mt-0.5 p-1.5 rounded bg-black/[0.02] space-y-0.5 text-sm border-l border-l-apple-cyan/20">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Peak</span>
                        <span className="font-number text-apple-green">${detail.peakProfit.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">DD</span>
                        <span className={cn("font-number", detail.drawdownFromPeak > 30 ? "text-apple-red" : "text-muted-foreground")}>{detail.drawdownFromPeak.toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">TP Prob</span>
                        <span className={cn("font-number", detail.tpProbability >= 50 ? "text-apple-green" : "text-apple-orange")}>{detail.tpProbability}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Duration</span>
                        <span className="font-number text-apple-purple">{detail.tradeHours}h</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
