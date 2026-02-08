import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "XAUBOT AI — Documentation",
  description: "System documentation and architecture reference for XAUBOT AI",
};

export default function BooksLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="h-full overflow-auto">{children}</div>
  );
}
