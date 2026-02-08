"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
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
  const firstBlockerIdx = filters.findIndex((f) => !f.passed);

  return (
    <Card className={cn("glass h-full overflow-hidden flex flex-col", hasBlocker ? "accent-top-red" : "accent-top-green")}>
      <CardHeader>
        <CardTitle className={cn(
          "text-sm font-medium flex items-center gap-1.5 uppercase tracking-wider",
          hasBlocker ? "text-apple-red" : "text-apple-green"
        )}>
          <Filter className="h-4 w-4" />
          Filters
          {totalCount > 0 && (
            <Badge variant={hasBlocker ? "danger" : "success"} className="ml-auto text-xs h-5 px-1.5">
              {passedCount}/{totalCount}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0 overflow-auto">
        {totalCount === 0 ? (
          <div className="flex flex-col items-center justify-center h-full">
            <Minus className="h-5 w-5 text-muted-foreground/30 mb-1" />
            <p className="text-sm text-muted-foreground/60">Waiting...</p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {filters.map((filter, idx) => {
              const isNotEvaluated = firstBlockerIdx >= 0 && idx > firstBlockerIdx;
              const isBlocker = !filter.passed && idx === firstBlockerIdx;

              return (
                <Tooltip key={`${filter.name}-${idx}`}>
                  <TooltipTrigger asChild>
                    <div className={cn(
                      "flex items-center gap-1.5 px-2 py-0.5 rounded text-sm",
                      isBlocker && "bg-danger/8 border-l-2 border-l-apple-red",
                      isNotEvaluated && "opacity-40",
                      filter.passed && !isNotEvaluated && "border-l-2 border-l-apple-green/30",
                      !isBlocker && !isNotEvaluated && "row-hover"
                    )}>
                      {isNotEvaluated ? (
                        <Minus className="h-3.5 w-3.5 text-muted-foreground/40 flex-shrink-0" />
                      ) : filter.passed ? (
                        <Check className="h-3.5 w-3.5 text-apple-green flex-shrink-0" />
                      ) : (
                        <X className="h-3.5 w-3.5 text-apple-red flex-shrink-0" />
                      )}
                      <span className={cn("truncate flex-1", isBlocker ? "text-apple-red font-semibold" : filter.passed ? "text-foreground/80" : "text-muted-foreground")}>
                        {filter.name}
                      </span>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    <p className="max-w-[220px]">{filter.detail || (filter.passed ? "Passed" : "Blocked")}</p>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
