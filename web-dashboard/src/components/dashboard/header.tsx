"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "@/components/theme-toggle";
import { Bot, Wifi, WifiOff, Clock, BookOpen, Info, History, FlaskConical, Bell } from "lucide-react";
import { cn } from "@/lib/utils";
import { AboutDialog } from "@/components/about-dialog";

interface HeaderProps {
  connected: boolean;
  lastUpdate: string;
  dataAge: number;
}

const navLinks = [
  { href: "/trades", icon: History, label: "Trades" },
  { href: "/backtests", icon: FlaskConical, label: "Backtests" },
  { href: "/alerts", icon: Bell, label: "Alerts" },
  { href: "/books", icon: BookOpen, label: "Docs" },
];

export function Header({ connected, lastUpdate, dataAge }: HeaderProps) {
  const getDataStatus = () => {
    if (dataAge > 45) return { label: "OFFLINE", variant: "danger" as const, dot: "bg-danger" };
    if (dataAge > 15) return { label: `STALE ${dataAge.toFixed(0)}s`, variant: "warning" as const, dot: "bg-warning animate-pulse" };
    return { label: `LIVE ${dataAge.toFixed(1)}s`, variant: "success" as const, dot: "bg-success" };
  };

  const status = getDataStatus();

  return (
    <header className="shrink-0 w-full border-b border-border bg-white/70 dark:bg-white/[0.03] backdrop-blur-2xl"
      style={{ borderImage: "linear-gradient(90deg, rgba(0,122,255,0.2), rgba(175,82,222,0.2), rgba(52,199,89,0.2)) 1" }}
    >
      <div className="flex h-9 items-center justify-between px-3">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-6 rounded-md bg-primary/10 border border-primary/20">
            <Bot className="h-3.5 w-3.5 text-primary" />
          </div>
          <Link href="/">
            <h1 className="text-base font-bold text-gradient">XAUBOT AI</h1>
          </Link>
          <span className="text-xs text-muted-foreground/50 font-number">v2.0</span>

          <div className="w-px h-4 bg-border mx-1" />

          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/40 dark:bg-white/[0.06] backdrop-blur-sm border border-white/30 dark:border-white/10 hover:border-primary/20 transition-colors text-sm text-muted-foreground hover:text-primary"
              title={link.label}
            >
              <link.icon className="h-3.5 w-3.5" />
              <span className="hidden lg:inline">{link.label}</span>
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle />

          <AboutDialog>
            <button
              className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/40 dark:bg-white/[0.06] backdrop-blur-sm border border-white/30 dark:border-white/10 hover:border-primary/20 transition-colors text-sm text-muted-foreground hover:text-primary"
              title="About XAUBOT AI"
            >
              <Info className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">About</span>
            </button>
          </AboutDialog>

          <Badge variant={status.variant} className="gap-1.5 font-number text-sm h-6">
            <span className={cn(
              "w-2 h-2 rounded-full",
              status.dot
            )} />
            {status.label}
          </Badge>

          <Badge variant={connected ? "success" : "danger"} className="gap-1.5 text-sm h-6">
            {connected ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
            {connected ? "MT5" : "OFF"}
          </Badge>

          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/40 dark:bg-white/[0.06] backdrop-blur-sm border border-white/30 dark:border-white/10 text-sm">
            <Clock className="h-3.5 w-3.5 text-apple-cyan" />
            <span className="font-number font-medium">{lastUpdate || "--:--:--"}</span>
            <span className="text-muted-foreground text-xs">WIB</span>
          </div>
        </div>
      </div>
    </header>
  );
}
