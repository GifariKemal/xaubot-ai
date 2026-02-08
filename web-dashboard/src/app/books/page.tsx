"use client";

import { useState, useMemo, useCallback } from "react";
import Link from "next/link";
import {
  BookOpen,
  Sparkles,
  LayoutDashboard,
  List,
  Brain,
  Cpu,
  TrendingUp,
  Layers,
  Shield,
  Clock,
  ShieldAlert,
  Target,
  ArrowRightCircle,
  ArrowLeftCircle,
  Newspaper,
  Send,
  RefreshCw,
  BarChart3,
  Gauge,
  GraduationCap,
  Plug,
  Settings,
  FileText,
  ListChecks,
  Calculator,
  Database,
  Play,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  ArrowLeft,
  Search,
  X,
  PanelLeftClose,
  PanelLeftOpen,
  Info,
  type LucideIcon,
} from "lucide-react";
import { books, categories, type BookEntry } from "@/data/books";
import { MarkdownRenderer } from "@/components/books/markdown-renderer";
import { AboutDialog } from "@/components/about-dialog";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const iconMap: Record<string, LucideIcon> = {
  BookOpen, Sparkles, LayoutDashboard, List, Brain, Cpu, TrendingUp, Layers,
  Shield, Clock, ShieldAlert, Target, ArrowRightCircle, ArrowLeftCircle,
  Newspaper, Send, RefreshCw, BarChart3, Gauge, GraduationCap, Plug,
  Settings, FileText, ListChecks, Calculator, Database, Play, AlertTriangle,
};

const categoryIcons: Record<string, LucideIcon> = {
  "Mulai di Sini": BookOpen,
  "AI & Analisis": Brain,
  "Risiko & Proteksi": Shield,
  "Proses Trading": TrendingUp,
  "Infrastruktur": Settings,
  "Konektor & Konfigurasi": Plug,
  "Engine & Data": Database,
  "Orkestrator": Play,
  "Analisis": AlertTriangle,
};

export default function BooksPage() {
  const [selectedSlug, setSelectedSlug] = useState<string>("readme");
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    () => new Set(categories)
  );
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [search, setSearch] = useState("");

  const selectedBook = useMemo(
    () => books.find((b) => b.slug === selectedSlug) ?? books[0],
    [selectedSlug]
  );

  const filteredBooks = useMemo(() => {
    if (!search.trim()) return books;
    const q = search.toLowerCase();
    return books.filter(
      (b) =>
        b.title.toLowerCase().includes(q) ||
        b.description.toLowerCase().includes(q) ||
        b.category.toLowerCase().includes(q)
    );
  }, [search]);

  const groupedBooks = useMemo(() => {
    const map = new Map<string, BookEntry[]>();
    for (const cat of categories) {
      const items = filteredBooks.filter((b) => b.category === cat);
      if (items.length > 0) map.set(cat, items);
    }
    return map;
  }, [filteredBooks]);

  const toggleCategory = useCallback((cat: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }, []);

  const selectBook = useCallback((slug: string) => {
    setSelectedSlug(slug);
    document.getElementById("books-content")?.scrollTo(0, 0);
  }, []);

  return (
    <div className="flex flex-col h-full min-h-0 bg-background">
      {/* ── Header ── */}
      <header className="shrink-0 w-full border-b border-border bg-white/80 dark:bg-white/[0.03] backdrop-blur-xl">
        <div className="flex h-11 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface border border-border hover:border-primary/20 transition-colors text-sm text-muted-foreground hover:text-primary"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Dashboard</span>
            </Link>

            <button
              onClick={() => setSidebarOpen((p) => !p)}
              className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-light transition-colors text-muted-foreground hover:text-foreground"
              title={sidebarOpen ? "Tutup sidebar" : "Buka sidebar"}
            >
              {sidebarOpen ? (
                <PanelLeftClose className="h-4 w-4" />
              ) : (
                <PanelLeftOpen className="h-4 w-4" />
              )}
            </button>

            <div className="w-px h-5 bg-border" />
            <div className="flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-amber-600 dark:text-amber-400" />
              <h1 className="text-base font-bold">Dokumentasi Sistem</h1>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="font-mono">{books.length} dokumen</span>
            <span className="text-border">|</span>
            <span className="font-semibold">XAUBOT AI</span>
            <span className="text-border">|</span>
            <ThemeToggle />
            <AboutDialog>
              <button
                className="flex items-center gap-1.5 px-2 py-0.5 rounded-lg bg-surface border border-border hover:border-primary/20 transition-colors text-muted-foreground hover:text-primary"
                title="About XAUBOT AI"
              >
                <Info className="h-3.5 w-3.5" />
                <span className="hidden sm:inline text-xs">About</span>
              </button>
            </AboutDialog>
          </div>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 min-h-0">
        {/* ── Sidebar ── */}
        <aside
          className={cn(
            "shrink-0 border-r border-border bg-white/60 dark:bg-white/[0.02] backdrop-blur-sm flex flex-col transition-all duration-200 ease-in-out",
            sidebarOpen ? "w-80" : "w-0 overflow-hidden"
          )}
        >
          {/* Search */}
          <div className="p-2.5 border-b border-border">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Cari dokumen..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-8 pr-8 py-1.5 rounded-lg bg-surface-light border border-border text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-amber-400/60 dark:focus:border-amber-500/40 focus:ring-1 focus:ring-amber-200/40 dark:focus:ring-amber-500/20"
              />
              {search && (
                <button
                  onClick={() => setSearch("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Category list */}
          <nav className="flex-1 overflow-y-auto py-1.5">
            {Array.from(groupedBooks.entries()).map(([cat, items]) => {
              const CatIcon = categoryIcons[cat] ?? BookOpen;
              const isExpanded = expandedCategories.has(cat);

              return (
                <div key={cat} className="mb-0.5">
                  <button
                    onClick={() => toggleCategory(cat)}
                    className="w-full flex items-center gap-2 px-3 py-2 text-[13px] font-semibold text-muted-foreground hover:text-foreground hover:bg-surface-light transition-colors"
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-3 w-3" />
                    ) : (
                      <ChevronRight className="h-3 w-3" />
                    )}
                    <CatIcon className="h-4 w-4" />
                    <span className="uppercase tracking-widest">{cat}</span>
                    <span className="ml-auto text-[11px] font-normal bg-surface-light px-1.5 rounded-full">
                      {items.length}
                    </span>
                  </button>

                  {isExpanded && (
                    <div className="pb-1">
                      {items.map((book) => {
                        const Icon = iconMap[book.icon] ?? BookOpen;
                        const isActive = book.slug === selectedSlug;
                        return (
                          <button
                            key={book.slug}
                            onClick={() => selectBook(book.slug)}
                            className={cn(
                              "w-full flex items-center gap-2.5 pl-9 pr-3 py-[7px] text-[15px] transition-all",
                              isActive
                                ? "bg-amber-50/80 dark:bg-amber-900/20 text-amber-800 dark:text-amber-300 font-medium border-r-2 border-amber-500"
                                : "text-muted-foreground hover:text-foreground hover:bg-surface-light"
                            )}
                          >
                            <Icon
                              className={cn(
                                "h-4 w-4 shrink-0",
                                isActive ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground"
                              )}
                            />
                            <span className="truncate">{book.title}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </nav>
        </aside>

        {/* ── Content ── */}
        <main id="books-content" className="flex-1 min-w-0 overflow-y-auto bg-background">
          <div className="max-w-7xl mx-auto px-10 py-8">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 mb-5 text-xs text-muted-foreground">
              <BookOpen className="h-3 w-3" />
              <span>{selectedBook.category}</span>
              <ChevronRight className="h-3 w-3" />
              <span className="text-foreground font-medium">
                {selectedBook.title}
              </span>
            </div>

            {/* Description card */}
            <div className="mb-8 px-5 py-4 rounded-xl bg-white dark:bg-white/[0.04] border border-border shadow-sm">
              <p className="text-[1rem] text-muted-foreground leading-relaxed">
                {selectedBook.description}
              </p>
            </div>

            {/* Markdown content */}
            <article className="pb-16">
              <MarkdownRenderer content={selectedBook.content} />
            </article>
          </div>
        </main>
      </div>
    </div>
  );
}
