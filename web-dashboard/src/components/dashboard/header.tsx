"use client";

import { Badge } from "@/components/ui/badge";
import { Bot, Wifi, WifiOff, Clock } from "lucide-react";

interface HeaderProps {
  connected: boolean;
  lastUpdate: string;
  dataAge: number;
}

export function Header({ connected, lastUpdate, dataAge }: HeaderProps) {
  const isStale = dataAge > 5;

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between">
        <div className="flex items-center gap-3">
          <Bot className="h-6 w-6 text-primary" />
          <div className="flex items-baseline gap-2">
            <h1 className="text-lg font-bold">AI TRADING BOT</h1>
            <span className="text-xs text-primary font-semibold">MONITOR</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Data Freshness */}
          <Badge variant={isStale ? "destructive" : "secondary"} className="gap-1">
            <Clock className="h-3 w-3" />
            {isStale ? `STALE (${dataAge.toFixed(0)}s)` : `LIVE (${dataAge.toFixed(1)}s)`}
          </Badge>

          {/* Connection Status */}
          <Badge variant={connected ? "default" : "destructive"} className="gap-1">
            {connected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            {connected ? 'Connected' : 'Disconnected'}
          </Badge>

          {/* Time */}
          <span className="text-sm font-medium text-muted-foreground">
            {lastUpdate || '--:--:--'} WIB
          </span>
        </div>
      </div>
    </header>
  );
}
