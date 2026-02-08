import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Trade History — XAUBOT AI",
  description: "Complete trade history and performance analytics",
};

export default function TradesLayout({ children }: { children: React.ReactNode }) {
  return children;
}
