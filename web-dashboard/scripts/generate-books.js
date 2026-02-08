/**
 * Script to generate src/data/books.ts from documentation files.
 * Run: node scripts/generate-books.js
 */
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(__dirname, "..", "src", "data", "books.ts");

// Define all books with their source files and metadata (Indonesian)
const bookDefs = [
  // Mulai di Sini
  { slug: "readme", title: "README", category: "Mulai di Sini", icon: "BookOpen", description: "Gambaran proyek, instalasi, dan panduan cepat memulai XAUBot AI", file: path.join(ROOT, "README.md") },
  { slug: "features", title: "Fitur & Komponen", category: "Mulai di Sini", icon: "Sparkles", description: "Daftar lengkap fitur — 14 filter entry, 12 kondisi exit, manajemen risiko", file: path.join(ROOT, "docs", "FEATURES.md") },
  { slug: "architecture-full", title: "Arsitektur Lengkap", category: "Mulai di Sini", icon: "LayoutDashboard", description: "Arsitektur menyeluruh sistem — alur data, komponen, dan interaksi antar modul", file: path.join(ROOT, "docs", "arsitektur-ai", "00-ARSITEKTUR-LENGKAP.md") },
  { slug: "architecture-index", title: "Indeks Arsitektur", category: "Mulai di Sini", icon: "List", description: "Daftar semua dokumen arsitektur dan status komponen terkini", file: path.join(ROOT, "docs", "arsitektur-ai", "README.md") },

  // AI & Analisis
  { slug: "hmm-regime", title: "HMM Regime Detector", category: "AI & Analisis", icon: "Brain", description: "Deteksi kondisi pasar menggunakan Hidden Markov Model 3 state", file: path.join(ROOT, "docs", "arsitektur-ai", "01-HMM-Regime-Detector.md") },
  { slug: "xgboost", title: "XGBoost Signal Predictor", category: "AI & Analisis", icon: "Cpu", description: "Model machine learning untuk prediksi sinyal BUY/SELL/HOLD", file: path.join(ROOT, "docs", "arsitektur-ai", "02-XGBoost-Signal-Predictor.md") },
  { slug: "smc", title: "SMC Analyzer", category: "AI & Analisis", icon: "TrendingUp", description: "Analisis Smart Money Concepts — Order Block, FVG, BOS, CHoCH", file: path.join(ROOT, "docs", "arsitektur-ai", "03-SMC-Analyzer.md") },
  { slug: "feature-eng", title: "Feature Engineering", category: "AI & Analisis", icon: "Layers", description: "37 fitur teknikal — RSI, ATR, MACD, Bollinger, dan lainnya", file: path.join(ROOT, "docs", "arsitektur-ai", "04-Feature-Engineering.md") },

  // Risiko & Proteksi
  { slug: "risk-management", title: "Manajemen Risiko", category: "Risiko & Proteksi", icon: "Shield", description: "Sistem manajemen risiko dinamis dengan mode kapital dan batas harian", file: path.join(ROOT, "docs", "arsitektur-ai", "05-Risk-Management.md") },
  { slug: "session-filter", title: "Filter Sesi", category: "Risiko & Proteksi", icon: "Clock", description: "Filter sesi perdagangan — Sydney, London, New York dalam zona waktu WIB", file: path.join(ROOT, "docs", "arsitektur-ai", "06-Session-Filter.md") },
  { slug: "stop-loss", title: "Stop Loss", category: "Risiko & Proteksi", icon: "ShieldAlert", description: "Proteksi SL berbasis ATR dan broker-level untuk keamanan maksimal", file: path.join(ROOT, "docs", "arsitektur-ai", "07-Stop-Loss.md") },
  { slug: "take-profit", title: "Take Profit", category: "Risiko & Proteksi", icon: "Target", description: "Target TP multi-level dengan perhitungan ATR dan struktur pasar", file: path.join(ROOT, "docs", "arsitektur-ai", "08-Take-Profit.md") },

  // Proses Trading
  { slug: "entry-trade", title: "Entry Trade", category: "Proses Trading", icon: "ArrowRightCircle", description: "14 filter entry dan logika eksekusi perdagangan — dari sinyal hingga order", file: path.join(ROOT, "docs", "arsitektur-ai", "09-Entry-Trade.md") },
  { slug: "exit-trade", title: "Exit Trade", category: "Proses Trading", icon: "ArrowLeftCircle", description: "12 kondisi exit termasuk trailing SL, batas waktu, dan perubahan regime", file: path.join(ROOT, "docs", "arsitektur-ai", "10-Exit-Trade.md") },

  // Infrastruktur
  { slug: "news-agent", title: "News Agent", category: "Infrastruktur", icon: "Newspaper", description: "Filter berita ekonomi dan penilaian dampak — saat ini nonaktif", file: path.join(ROOT, "docs", "arsitektur-ai", "11-News-Agent.md") },
  { slug: "telegram", title: "Notifikasi Telegram", category: "Infrastruktur", icon: "Send", description: "Notifikasi trade real-time dan ringkasan harian via Telegram Bot", file: path.join(ROOT, "docs", "arsitektur-ai", "12-Telegram-Notifications.md") },
  { slug: "auto-trainer", title: "Auto Trainer", category: "Infrastruktur", icon: "RefreshCw", description: "Pipeline retraining otomatis saat kondisi pasar berubah signifikan", file: path.join(ROOT, "docs", "arsitektur-ai", "13-Auto-Trainer.md") },
  { slug: "backtest", title: "Backtest", category: "Infrastruktur", icon: "BarChart3", description: "Framework backtesting yang disinkronkan dengan logika live trading", file: path.join(ROOT, "docs", "arsitektur-ai", "14-Backtest.md") },
  { slug: "dynamic-confidence", title: "Dynamic Confidence", category: "Infrastruktur", icon: "Gauge", description: "Ambang batas confidence adaptif berdasarkan kondisi dan performa pasar", file: path.join(ROOT, "docs", "arsitektur-ai", "15-Dynamic-Confidence.md") },
  { slug: "train-models", title: "Train Models", category: "Infrastruktur", icon: "GraduationCap", description: "Pipeline pelatihan model dan optimasi hyperparameter XGBoost", file: path.join(ROOT, "docs", "arsitektur-ai", "22-Train-Models.md") },

  // Konektor & Konfigurasi
  { slug: "mt5-connector", title: "Konektor MT5", category: "Konektor & Konfigurasi", icon: "Plug", description: "Lapisan koneksi MetaTrader 5 dan eksekusi order trading", file: path.join(ROOT, "docs", "arsitektur-ai", "16-MT5-Connector.md") },
  { slug: "configuration", title: "Konfigurasi", category: "Konektor & Konfigurasi", icon: "Settings", description: "Pengaturan trading, mode kapital, dan konfigurasi environment", file: path.join(ROOT, "docs", "arsitektur-ai", "17-Configuration.md") },
  { slug: "trade-logger", title: "Trade Logger", category: "Konektor & Konfigurasi", icon: "FileText", description: "Pencatatan trade ke database PostgreSQL untuk analisis historis", file: path.join(ROOT, "docs", "arsitektur-ai", "18-Trade-Logger.md") },
  { slug: "position-manager", title: "Position Manager", category: "Konektor & Konfigurasi", icon: "ListChecks", description: "Pelacakan dan manajemen posisi terbuka secara real-time", file: path.join(ROOT, "docs", "arsitektur-ai", "19-Position-Manager.md") },

  // Engine & Data
  { slug: "risk-engine", title: "Risk Engine", category: "Engine & Data", icon: "Calculator", description: "Perhitungan risiko, Kelly criterion, dan position sizing otomatis", file: path.join(ROOT, "docs", "arsitektur-ai", "20-Risk-Engine.md") },
  { slug: "database", title: "Database", category: "Engine & Data", icon: "Database", description: "Skema PostgreSQL dan penyimpanan data perdagangan", file: path.join(ROOT, "docs", "arsitektur-ai", "21-Database.md") },

  // Orkestrator
  { slug: "main-live", title: "Orkestrator Utama", category: "Orkestrator", icon: "Play", description: "Async main loop — inti dari trading bot yang mengkoordinasi semua komponen", file: path.join(ROOT, "docs", "arsitektur-ai", "23-Main-Live-Orchestrator.md") },

  // Analisis
  { slug: "weakness-analysis", title: "Analisis Kelemahan", category: "Analisis", icon: "AlertTriangle", description: "Kelemahan yang diketahui, risiko, dan prioritas perbaikan sistem", file: path.join(ROOT, "docs", "WEAKNESS_ANALYSIS.md") },
];

function escapeForTemplate(str) {
  // Escape backticks and ${} in template literals
  return str.replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
}

// Ensure output directory exists
const outDir = path.dirname(OUT);
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

let output = `// AUTO-GENERATED — do not edit manually.
// Run: node scripts/generate-books.js

export interface BookEntry {
  slug: string;
  title: string;
  category: string;
  icon: string;
  description: string;
  content: string;
}

export const categories = [
  "Mulai di Sini",
  "AI & Analisis",
  "Risiko & Proteksi",
  "Proses Trading",
  "Infrastruktur",
  "Konektor & Konfigurasi",
  "Engine & Data",
  "Orkestrator",
  "Analisis",
] as const;

export type Category = (typeof categories)[number];

export const books: BookEntry[] = [\n`;

for (const def of bookDefs) {
  let content = "";
  try {
    content = fs.readFileSync(def.file, "utf-8");
  } catch (e) {
    console.warn(`WARNING: Could not read ${def.file}: ${e.message}`);
    content = `# ${def.title}\n\n*Dokumen tidak ditemukan.*`;
  }

  output += `  {
    slug: ${JSON.stringify(def.slug)},
    title: ${JSON.stringify(def.title)},
    category: ${JSON.stringify(def.category)},
    icon: ${JSON.stringify(def.icon)},
    description: ${JSON.stringify(def.description)},
    content: \`${escapeForTemplate(content)}\`,
  },\n`;
}

output += `];\n`;

fs.writeFileSync(OUT, output, "utf-8");
console.log(`Generated ${OUT} with ${bookDefs.length} books.`);
