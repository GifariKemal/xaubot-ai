"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Terminal } from "lucide-react";
import type { LogEntry } from "@/types/trading";

interface LogCardProps {
  logs: LogEntry[];
}

export function LogCard({ logs }: LogCardProps) {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-500';
      case 'warn': return 'text-amber-500';
      case 'trade': return 'text-cyan-400';
      default: return 'text-green-400';
    }
  };

  const getLevelBadge = (level: string) => {
    switch (level) {
      case 'error': return 'ERR';
      case 'warn': return 'WRN';
      case 'trade': return 'TRD';
      default: return 'INF';
    }
  };

  return (
    <Card className="bg-card/50 backdrop-blur col-span-2">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Terminal className="h-4 w-4" />
          AI ACTIVITY LOG
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[150px] rounded-md bg-black/50 p-3 font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-muted-foreground">Waiting for activity...</p>
          ) : (
            <div className="space-y-1">
              {logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-muted-foreground">[{log.time}]</span>
                  <span className={`font-semibold ${getLevelColor(log.level)}`}>
                    [{getLevelBadge(log.level)}]
                  </span>
                  <span className="text-foreground/80">{log.message}</span>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
