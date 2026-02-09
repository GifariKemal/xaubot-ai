"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useFilterConfig } from "@/hooks/use-filter-config";
import { AlertCircle, CheckCircle2, RefreshCcw } from "lucide-react";

export function FiltersConfigCard() {
  const { config, loading, error, updating, toggleFilter, refresh } = useFilterConfig();

  if (loading && !config) {
    return (
      <Card className="glass">
        <CardHeader>
          <CardTitle>Entry Filters</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex items-center justify-between">
              <Skeleton className="h-4 w-[200px]" />
              <Skeleton className="h-5 w-9 rounded-full" />
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="glass border-red-500/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-400">
            <AlertCircle className="h-5 w-5" />
            Entry Filters
          </CardTitle>
          <CardDescription className="text-red-400/70">{error}</CardDescription>
        </CardHeader>
        <CardContent>
          <button
            onClick={refresh}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <RefreshCcw className="h-4 w-4" />
            Retry
          </button>
        </CardContent>
      </Card>
    );
  }

  if (!config) return null;

  const filters = Object.entries(config.filters);
  const enabledCount = filters.filter(([, f]) => f.enabled).length;

  // Sort by name for consistent display
  const sortedFilters = filters.sort((a, b) => a[1].name.localeCompare(b[1].name));

  return (
    <Card className="glass">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              Entry Filters
              <Badge variant="outline" className="font-normal">
                {enabledCount}/{filters.length}
              </Badge>
            </CardTitle>
            <CardDescription>
              Toggle filters on/off — updates live without bot restart
            </CardDescription>
          </div>
          <button
            onClick={refresh}
            disabled={updating}
            className="p-2 hover:bg-white/5 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCcw className={`h-4 w-4 ${updating ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {sortedFilters.map(([key, filter]) => (
          <div
            key={key}
            className="flex items-start justify-between gap-4 py-2 border-b border-white/5 last:border-0"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">{filter.name}</span>
                {filter.enabled ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-400 shrink-0" />
                ) : (
                  <AlertCircle className="h-3.5 w-3.5 text-orange-400 shrink-0" />
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                {filter.description}
              </p>
            </div>
            <Switch
              checked={filter.enabled}
              onCheckedChange={() => toggleFilter(key)}
              disabled={updating}
              className="shrink-0"
            />
          </div>
        ))}

        {config.metadata?.updated_at && (
          <div className="text-xs text-muted-foreground pt-2 border-t border-white/5">
            Last updated: {new Date(config.metadata.updated_at).toLocaleString('id-ID', {
              timeZone: 'Asia/Jakarta',
              dateStyle: 'short',
              timeStyle: 'short'
            })} WIB
          </div>
        )}
      </CardContent>
    </Card>
  );
}
