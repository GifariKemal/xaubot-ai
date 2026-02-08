"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Terminal } from "lucide-react";
import type { LogEntry } from "@/types/trading";

interface LogCardProps {
  logs: LogEntry[];
}

export function LogCard({ logs }: LogCardProps) {
  const getLevelColor = (level: string) => {
    switch (level) {
      case "error": return "text-danger";
      case "warn": return "text-warning";
      case "trade": return "text-info";
      default: return "text-success";
    }
  };

  const getLevelBadge = (level: string) => {
    switch (level) {
      case "error": return "ERR";
      case "warn": return "WRN";
      case "trade": return "TRD";
      default: return "INF";
    }
  };

  return (
    <Card className="glass h-full flex flex-col">
      <CardHeader>
        <CardTitle className="text-[11px] font-medium text-muted-foreground flex items-center gap-1.5 uppercase tracking-wider">
          <Terminal className="h-3.5 w-3.5" />
          Activity
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <div className="h-full overflow-auto rounded-md bg-background/60 p-2 font-mono text-[10px] leading-relaxed">
          {logs.length === 0 ? (
            <p className="text-muted-foreground/60">Waiting for activity...</p>
          ) : (
            <div className="space-y-0.5">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-1.5">
                  <span className="text-muted-foreground/60 shrink-0">{log.time}</span>
                  <span className={`font-semibold shrink-0 ${getLevelColor(log.level)}`}>
                    {getLevelBadge(log.level)}
                  </span>
                  <span className="text-foreground/70 truncate">{log.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
