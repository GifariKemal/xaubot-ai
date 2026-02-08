"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Filter, Check, X, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import type { EntryFilter } from "@/types/trading";

interface EntryFilterCardProps {
  filters: EntryFilter[];
}

export function EntryFilterCard({ filters }: EntryFilterCardProps) {
  const passedCount = filters.filter((f) => f.passed).length;
  const totalCount = filters.length;
  const hasBlocker = filters.some((f) => !f.passed);

  // Find the first blocker index — filters after it were not evaluated
  const firstBlockerIdx = filters.findIndex((f) => !f.passed);

  return (
    <Card className="glass h-full flex flex-col">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Filter className="h-3.5 w-3.5" />
          Entry Filters
          {totalCount > 0 && (
            <Badge
              variant={hasBlocker ? "danger" : "success"}
              className="ml-auto text-[10px] h-4 px-1.5"
            >
              {passedCount}/{totalCount}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-auto">
        {totalCount === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Minus className="h-4 w-4 text-muted-foreground/30 mb-1" />
            <p className="text-[10px] text-muted-foreground/60">Waiting for candle...</p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {filters.map((filter, idx) => {
              // Determine status: passed, blocked, or not evaluated
              const isNotEvaluated = firstBlockerIdx >= 0 && idx > firstBlockerIdx;
              const isBlocker = !filter.passed && idx === firstBlockerIdx;

              return (
                <div
                  key={`${filter.name}-${idx}`}
                  className={cn(
                    "flex items-center gap-1.5 px-1.5 py-0.5 rounded text-[10px]",
                    isBlocker && "bg-danger/10",
                    isNotEvaluated && "opacity-40"
                  )}
                >
                  {isNotEvaluated ? (
                    <Minus className="h-2.5 w-2.5 text-muted-foreground/40 flex-shrink-0" />
                  ) : filter.passed ? (
                    <Check className="h-2.5 w-2.5 text-success flex-shrink-0" />
                  ) : (
                    <X className="h-2.5 w-2.5 text-danger flex-shrink-0" />
                  )}
                  <span className={cn(
                    "truncate flex-1",
                    isBlocker ? "text-danger font-semibold" : "text-muted-foreground"
                  )}>
                    {filter.name}
                  </span>
                  <span className={cn(
                    "text-[9px] truncate max-w-[80px]",
                    isBlocker ? "text-danger" : "text-muted-foreground/60"
                  )}>
                    {filter.detail}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
