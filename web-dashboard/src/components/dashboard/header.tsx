"use client";

import { Badge } from "@/components/ui/badge";
import { Bot, Wifi, WifiOff, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  connected: boolean;
  lastUpdate: string;
  dataAge: number;
}

export function Header({ connected, lastUpdate, dataAge }: HeaderProps) {
  const getDataStatus = () => {
    if (dataAge > 45) return { label: "OFFLINE", variant: "danger" as const, dot: "bg-danger" };
    if (dataAge > 15) return { label: `STALE ${dataAge.toFixed(0)}s`, variant: "warning" as const, dot: "bg-warning animate-pulse" };
    return { label: `LIVE ${dataAge.toFixed(1)}s`, variant: "success" as const, dot: "bg-success" };
  };

  const status = getDataStatus();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-xl">
      <div className="flex h-10 items-center justify-between px-3">
        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-primary/10">
            <Bot className="h-4 w-4 text-primary" />
          </div>
          <h1 className="text-base font-bold text-gradient">XAUBOT AI</h1>
          <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-widest hidden sm:block">
            Monitor
          </span>
        </div>

        {/* Status */}
        <div className="flex items-center gap-2">
          <Badge variant={status.variant} className="gap-1.5 font-number text-[11px]">
            <span className={cn("w-1.5 h-1.5 rounded-full", status.dot)} />
            {status.label}
          </Badge>

          <Badge variant={connected ? "success" : "danger"} className="gap-1.5 text-[11px] hidden sm:inline-flex">
            {connected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            {connected ? "Connected" : "Disconnected"}
          </Badge>

          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-surface border border-border text-[11px]">
            <Clock className="h-3 w-3 text-muted-foreground" />
            <span className="font-number font-medium">
              {lastUpdate || "--:--:--"}
            </span>
            <span className="text-muted-foreground">WIB</span>
          </div>
        </div>
      </div>
    </header>
  );
}
