"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Bot,
  Github,
  Shield,
  Brain,
  TrendingUp,
  Code2,
  Scale,
} from "lucide-react";

export function AboutDialog({ children }: { children: React.ReactNode }) {
  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg">
              <Bot className="h-5 w-5 text-white" />
            </div>
            <div>
              <DialogTitle className="text-xl">XAUBOT AI</DialogTitle>
              <DialogDescription className="text-sm">
                Smart Gold Trading Bot v2.0
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-5 pt-4">
          {/* Description */}
          <p className="text-sm text-muted-foreground leading-relaxed">
            Bot trading XAUUSD (Emas) otomatis berbasis AI yang menggabungkan{" "}
            <strong className="text-foreground">XGBoost Machine Learning</strong>,{" "}
            <strong className="text-foreground">Smart Money Concepts</strong> (SMC), dan{" "}
            <strong className="text-foreground">Hidden Markov Model</strong> untuk deteksi regime pasar pada
            MetaTrader 5.
          </p>

          {/* Tech Stack */}
          <div className="grid grid-cols-2 gap-2">
            {[
              { icon: Brain, label: "XGBoost ML", desc: "37-fitur prediksi sinyal" },
              { icon: TrendingUp, label: "Smart Money", desc: "OB, FVG, BOS, CHoCH" },
              { icon: Shield, label: "HMM Regime", desc: "3-state deteksi pasar" },
              { icon: Code2, label: "Polars Engine", desc: "Data processing cepat" },
            ].map((item) => (
              <div
                key={item.label}
                className="flex items-start gap-2.5 p-2.5 rounded-lg bg-surface-light border border-border"
              >
                <item.icon className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                <div>
                  <p className="text-xs font-semibold">{item.label}</p>
                  <p className="text-[11px] text-muted-foreground">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* Author */}
          <div className="p-3.5 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 border border-blue-100/60 dark:border-blue-800/30">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
                GK
              </div>
              <div>
                <p className="text-sm font-semibold">Gifari Kemal</p>
                <p className="text-xs text-muted-foreground">Developer & Maintainer</p>
              </div>
              <a
                href="https://github.com/GifariKemal/xaubot-ai"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface border border-border hover:border-primary/30 text-xs text-muted-foreground hover:text-primary transition-colors"
              >
                <Github className="h-3.5 w-3.5" />
                GitHub
              </a>
            </div>
          </div>

          {/* Legal / License */}
          <div className="space-y-3 pt-1">
            <div className="flex items-start gap-2.5">
              <Scale className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold">MIT License</p>
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  Perangkat lunak sumber terbuka — bebas digunakan, dimodifikasi, dan
                  didistribusikan sesuai ketentuan lisensi MIT.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-2.5">
              <Shield className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
              <div>
                <p className="text-xs font-semibold">Disclaimer</p>
                <p className="text-[11px] text-muted-foreground leading-relaxed">
                  Perangkat lunak ini dibuat <strong className="text-foreground">hanya untuk tujuan edukasi dan riset</strong>.
                  Trading dengan margin memiliki risiko tinggi. Kinerja masa lalu bukan
                  indikasi hasil di masa depan. Gunakan dengan risiko Anda sendiri.
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="pt-3 border-t border-border flex items-center justify-between">
            <p className="text-[11px] text-muted-foreground">
              &copy; 2025–2026 Gifari Kemal. All rights reserved.
            </p>
            <p className="text-[11px] text-muted-foreground font-mono">v2.0</p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
