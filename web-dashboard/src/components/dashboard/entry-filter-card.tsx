"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Filter, Check, X, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import { useFilterConfig } from "@/hooks/use-filter-config";
import type { EntryFilter } from "@/types/trading";

/* Map bot filter display name → filter_config.json key */
const NAME_TO_KEY: Record<string, string> = {
  "Flash Crash Guard": "flash_crash_guard",
  "Regime Filter": "regime_filter",
  "Risk Check": "risk_check",
  "Session Filter": "session_filter",
  "Spread Check": "spread_check",
  "H1 Bias (#31B)": "h1_bias",
  "ML Confidence": "ml_confidence",
  "Signal Combination": "signal_combination",
  "Time Filter (#34A)": "time_filter",
  "Cooldown": "cooldown",
  "Existing Position": "existing_position",
};

interface EntryFilterCardProps {
  filters: EntryFilter[];
}

export function EntryFilterCard({ filters }: EntryFilterCardProps) {
  const { config, updating, toggleFilter } = useFilterConfig();

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
              const isDisabled = filter.detail?.includes("[DISABLED]");

              /* Resolve filter config key for toggle */
              const configKey = NAME_TO_KEY[filter.name];
              const filterEnabled = configKey && config
                ? config.filters[configKey]?.enabled ?? true
                : true;

              return (
                <div
                  key={`${filter.name}-${idx}`}
                  className={cn(
                    "flex items-center gap-1.5 px-2 py-0.5 rounded text-sm",
                    isBlocker && "bg-danger/8 border-l-2 border-l-apple-red",
                    isNotEvaluated && "opacity-40",
                    filter.passed && !isNotEvaluated && "border-l-2 border-l-apple-green/30",
                    isDisabled && "border-l-2 border-l-orange-400/40",
                    !isBlocker && !isNotEvaluated && "row-hover"
                  )}
                >
                  {isDisabled ? (
                    <Minus className="h-3.5 w-3.5 text-orange-400 flex-shrink-0" />
                  ) : isNotEvaluated ? (
                    <Minus className="h-3.5 w-3.5 text-muted-foreground/40 flex-shrink-0" />
                  ) : filter.passed ? (
                    <Check className="h-3.5 w-3.5 text-apple-green flex-shrink-0" />
                  ) : (
                    <X className="h-3.5 w-3.5 text-apple-red flex-shrink-0" />
                  )}

                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className={cn(
                        "truncate flex-1 cursor-default",
                        isDisabled ? "text-orange-400/70" :
                        isBlocker ? "text-apple-red font-semibold" :
                        filter.passed ? "text-foreground/80" :
                        "text-muted-foreground"
                      )}>
                        {filter.name}
                      </span>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <p className="max-w-[220px]">
                        {filter.detail || (filter.passed ? "Passed" : "Blocked")}
                        {configKey && !filterEnabled && " — Filter disabled, auto-pass"}
                      </p>
                    </TooltipContent>
                  </Tooltip>

                  {/* Toggle switch — outside tooltip so clicks work */}
                  {configKey && config && (
                    <Switch
                      checked={filterEnabled}
                      onCheckedChange={() => toggleFilter(configKey)}
                      disabled={updating}
                      className="shrink-0 scale-75 origin-right"
                    />
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
