"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Layers, Inbox, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Position, PositionDetail } from "@/types/trading";

interface PositionsCardProps {
  positions: Position[];
  positionDetails?: PositionDetail[];
}

export function PositionsCard({ positions, positionDetails }: PositionsCardProps) {
  const [expandedTicket, setExpandedTicket] = useState<number | null>(null);

  const getDetail = (ticket: number) =>
    positionDetails?.find((d) => d.ticket === ticket);

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
            {positions.map((pos) => {
              const detail = getDetail(pos.ticket);
              const isExpanded = expandedTicket === pos.ticket;
              const hasDetail = !!detail;

              return (
                <div key={pos.ticket}>
                  <div
                    className={cn(
                      "flex items-center justify-between p-1.5 rounded-md bg-surface-light/50",
                      pos.type === "BUY" ? "border-l-2 border-l-success" : "border-l-2 border-l-danger",
                      hasDetail && "cursor-pointer hover:bg-surface-light/80"
                    )}
                    onClick={() => hasDetail && setExpandedTicket(isExpanded ? null : pos.ticket)}
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
                    <div className="flex items-center gap-1">
                      <span className={cn(
                        "text-[11px] font-bold font-number",
                        pos.profit >= 0 ? "text-success" : "text-danger"
                      )}>
                        {pos.profit >= 0 ? "+" : ""}${pos.profit.toFixed(2)}
                      </span>
                      {hasDetail && (
                        isExpanded
                          ? <ChevronUp className="h-3 w-3 text-muted-foreground/40" />
                          : <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
                      )}
                    </div>
                  </div>

                  {/* Expandable Details */}
                  {isExpanded && detail && (
                    <div className="ml-2 mt-0.5 p-1.5 rounded bg-surface-light/30 space-y-0.5 text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Peak Profit</span>
                        <span className="font-number text-success">${detail.peakProfit.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">DD from Peak</span>
                        <span className={cn("font-number", detail.drawdownFromPeak > 30 ? "text-danger" : "text-muted-foreground")}>
                          {detail.drawdownFromPeak.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Momentum</span>
                        <span className={cn("font-number", detail.momentum > 0 ? "text-success" : detail.momentum < 0 ? "text-danger" : "text-muted-foreground")}>
                          {detail.momentum > 0 ? "+" : ""}{detail.momentum}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">TP Probability</span>
                        <span className={cn("font-number", detail.tpProbability >= 50 ? "text-success" : "text-warning")}>
                          {detail.tpProbability}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Duration</span>
                        <span className="font-number text-muted-foreground">{detail.tradeHours}h</span>
                      </div>
                      {(detail.reversalWarnings > 0 || detail.stalls > 0) && (
                        <div className="flex gap-2 pt-0.5 border-t border-border/50">
                          {detail.reversalWarnings > 0 && (
                            <span className="text-warning">Rev: {detail.reversalWarnings}</span>
                          )}
                          {detail.stalls > 0 && (
                            <span className="text-muted-foreground">Stalls: {detail.stalls}</span>
                          )}
                        </div>
                      )}
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
