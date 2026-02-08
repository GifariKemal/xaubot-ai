import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Backtest Viewer — XAUBOT AI",
  description: "Compare and analyze backtest results across strategies",
};

export default function BacktestsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
