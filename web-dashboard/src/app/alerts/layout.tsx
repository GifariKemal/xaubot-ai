import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Alert / Signal Log — XAUBOT AI",
  description: "Complete signal and alert history with execution tracking",
};

export default function AlertsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
