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
      case "error": return "text-apple-red";
      case "warn": return "text-apple-orange";
      case "trade": return "text-apple-blue";
      default: return "text-apple-green";
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
    <Card className="glass h-full overflow-hidden flex flex-col accent-top-purple">
      <CardHeader>
        <CardTitle className="text-sm font-medium text-apple-purple flex items-center gap-1.5 uppercase tracking-wider">
          <Terminal className="h-4 w-4" />
          Activity
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <div className="h-full overflow-auto rounded-md bg-white/40 backdrop-blur-sm p-2 font-mono text-sm leading-relaxed border border-apple-purple/10">
          {logs.length === 0 ? (
            <p className="text-muted-foreground/60">Waiting for activity...</p>
          ) : (
            <div className="space-y-0.5">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-1.5 rounded px-1 -mx-1 row-hover">
                  <span className="text-muted-foreground/50 shrink-0">{log.time}</span>
                  <span className={`font-semibold shrink-0 ${getLevelColor(log.level)}`}>{getLevelBadge(log.level)}</span>
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
