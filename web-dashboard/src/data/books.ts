// AUTO-GENERATED — do not edit manually.
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

export const books: BookEntry[] = [
  {
    slug: "readme",
    title: "README",
    category: "Mulai di Sini",
    icon: "BookOpen",
    description: "Gambaran proyek, instalasi, dan panduan cepat memulai XAUBot AI",
    content: `# XAUBot AI

**Bot trading XAUUSD (Emas) berbasis AI** dengan *XGBoost ML*, *Smart Money Concepts* (SMC), dan deteksi *regime* menggunakan *Hidden Markov Model* untuk *MetaTrader 5*.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MetaTrader 5](https://img.shields.io/badge/broker-MetaTrader%205-orange.svg)](https://www.metatrader5.com/)

---

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| **Model *XGBoost ML*** | Model 37-fitur yang memprediksi BUY/SELL/HOLD dengan *confidence* terkalibrasi |
| ***Smart Money Concepts*** | *Order Block*, *Fair Value Gap*, *Break of Structure*, *Change of Character* |
| **Deteksi *Regime* HMM** | *Hidden Markov Model* 3-state yang mengklasifikasikan pasar *trending*/*ranging*/*volatile* |
| **Manajemen Risiko Dinamis** | *Stop Loss* berbasis ATR, *position sizing* dengan *Kelly criterion*, batas kerugian harian |
| **Kesadaran Sesi** | Dioptimalkan untuk sesi Sydney, London, dan New York |
| **Pelatihan Ulang Otomatis** | Model secara otomatis dilatih ulang saat kondisi pasar berubah |
| **Notifikasi Telegram** | Pemberitahuan *trade* secara *real-time* dan ringkasan harian |
| ***Dashboard* Web** | Antarmuka pemantauan *Next.js* untuk pelacakan *live* |

## Arsitektur

\`\`\`mermaid
graph TD
    MT5["MetaTrader 5<br/>(XAUUSD M15)"] -->|OHLCV| DP["Data Pipeline<br/>(Polars Engine)"]
    DP --> SMC["SMC Analyzer<br/>(OB / FVG / BOS)"]
    DP --> FE["Feature Engineering<br/>(37 fitur)"]
    DP --> HMM["HMM Regime<br/>Detector"]
    SMC --> XGB["XGBoost Model<br/>(Signal + Confidence)"]
    FE --> XGB
    HMM --> XGB
    XGB --> EF["14 Entry<br/>Filters"]
    XGB --> RE["Risk Engine<br/>(ATR + Kelly)"]
    XGB --> PM["Position<br/>Manager"]
    EF --> TE["Trade Execution<br/>(MT5 + Logging)"]
    RE --> TE
    PM --> TE
\`\`\`

## Struktur Proyek

\`\`\`
xaubot-ai/
├── main_live.py              # Orkestrator trading async utama
├── train_models.py           # Skrip pelatihan model
├── src/                      # Modul inti
│   ├── config.py             #   Konfigurasi trading & mode kapital
│   ├── mt5_connector.py      #   Layer koneksi MetaTrader 5
│   ├── smc_polars.py         #   Penganalisis Smart Money Concepts
│   ├── ml_model.py           #   Model trading XGBoost
│   ├── feature_eng.py        #   Feature engineering (37 fitur)
│   ├── regime_detector.py    #   Deteksi regime pasar HMM
│   ├── risk_engine.py        #   Kalkulasi & validasi risiko
│   ├── smart_risk_manager.py #   Manajemen risiko dinamis
│   ├── session_filter.py     #   Filter sesi (Sydney/London/NY)
│   ├── position_manager.py   #   Manajemen posisi terbuka
│   ├── dynamic_confidence.py #   Threshold confidence adaptif
│   ├── auto_trainer.py       #   Pipeline pelatihan ulang otomatis
│   ├── news_agent.py         #   Filter berita ekonomi
│   ├── telegram_notifier.py  #   Notifikasi Telegram
│   ├── trade_logger.py       #   Pencatatan trade ke DB
│   └── utils.py              #   Fungsi utilitas
├── backtests/                # Backtesting
│   ├── backtest_live_sync.py #   Backtest utama (sinkron dengan live)
│   └── archive/              #   Versi historis
├── scripts/                  # Skrip utilitas
│   ├── check_market.py       #   Analisis cepat pasar SMC
│   ├── check_positions.py    #   Lihat posisi terbuka
│   ├── check_status.py       #   Cek status akun
│   ├── close_positions.py    #   Tutup semua posisi darurat
│   ├── modify_tp.py          #   Modifikasi level take-profit
│   └── get_trade_history.py  #   Tarik riwayat trade
├── tests/                    # Pengujian
├── models/                   # Model terlatih (.pkl)
├── data/                     # Data pasar & catatan trade
├── docs/                     # Dokumentasi
│   ├── arsitektur-ai/        #   Dokumen arsitektur (23 komponen)
│   └── research/             #   Riset & analisis
├── web-dashboard/            # Dashboard pemantauan Next.js
├── docker/                   # Konfigurasi & skrip Docker
│   ├── scripts/              #   Skrip pembantu (.bat/.sh)
│   └── docs/                 #   Dokumentasi Docker
└── archive/                  # File usang (gitignored)
\`\`\`

## Hasil *Backtest* (Jan 2025 - Feb 2026)

| Metrik | Nilai |
|--------|-------|
| Total *Trade* | 654 |
| *Win Rate* | 63.9% |
| *Net P/L* | $4,189.52 |
| *Profit Factor* | 2.64 |
| *Max Drawdown* | 2.2% |
| *Sharpe Ratio* | 4.83 |

## Instalasi

### Deployment *Docker* (Direkomendasikan)

**Mulai Cepat:**

\`\`\`bash
# 1. Clone repositori
git clone https://github.com/GifariKemal/xaubot-ai.git
cd xaubot-ai

# 2. Konfigurasi environment
cp docker/.env.docker.example .env
# Edit .env dengan kredensial MT5 Anda

# 3. Jalankan semua layanan (Windows)
docker\\scripts\\docker-start.bat

# 3. Jalankan semua layanan (Linux/Mac)
./docker/scripts/docker-start.sh
\`\`\`

**Layanan yang tersedia:**
- *Dashboard*: http://localhost:3000
- API: http://localhost:8000
- Dokumentasi API: http://localhost:8000/docs
- *Database*: localhost:5432

**Dokumentasi *Docker* lengkap:** Lihat [docker/docs/DOCKER.md](docker/docs/DOCKER.md)

---

### Instalasi Manual

**Prasyarat:**
- Python 3.11+
- Terminal *MetaTrader 5* (Windows)
- PostgreSQL (opsional, untuk pencatatan *trade*)

**Persiapan:**

\`\`\`bash
# Clone repositori
git clone https://github.com/GifariKemal/xaubot-ai.git
cd xaubot-ai

# Instal dependensi
pip install -r requirements.txt

# Konfigurasi environment
cp .env.example .env
# Edit .env dengan kredensial MT5 dan token Telegram Anda
\`\`\`

### Konfigurasi

Pengaturan utama di \`.env\`:

\`\`\`env
# MetaTrader 5
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# Notifikasi Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading
CAPITAL=5000
SYMBOL=XAUUSD
\`\`\`

### Menjalankan

\`\`\`bash
# Latih model terlebih dahulu
python train_models.py

# Jalankan bot
python main_live.py

# Jalankan backtest
python backtests/backtest_live_sync.py --tune
\`\`\`

## Manajemen Risiko

| Proteksi | Detail |
|----------|--------|
| ***Stop Loss* Berbasis ATR** | Jarak minimum 1.5x ATR |
| ***Stop Loss* Level Broker** | *Stop Loss* darurat diatur di level broker |
| ***Position Sizing*** | *Kelly criterion* dengan penyesuaian mode kapital |
| **Batas Kerugian Harian** | 5% dari kapital per hari |
| **Batas Kerugian Total** | 10% dari kapital |
| **Batas Posisi** | Maksimal 2 posisi bersamaan |
| ***Exit* Berbasis Waktu** | Maksimal 6 jam per *trade* |
| **Filter Sesi** | Hanya membuka *trade* saat sesi aktif |
| **Filter *Spread*** | Menolak *trade* saat *spread* tinggi |
| ***Cooldown*** | Waktu minimum antar *trade* |

## Teknologi

- **Polars** — Mesin pemrosesan data performa tinggi (bukan Pandas)
- ***XGBoost*** — Model *machine learning* berbasis *gradient boosting*
- **hmmlearn** — *Hidden Markov Model* untuk deteksi *regime* pasar
- ***MetaTrader5*** — API koneksi broker
- **asyncio** — *Event loop* asinkron untuk eksekusi latensi rendah
- **loguru** — *Logging* terstruktur
- **PostgreSQL** — *Database* pencatatan *trade*
- ***Next.js*** — *Dashboard* web

## Peringatan

> Perangkat lunak ini dibuat **hanya untuk tujuan edukasi dan riset**. Trading valuta asing (Forex) dan komoditas dengan margin memiliki tingkat risiko yang tinggi dan mungkin tidak cocok untuk semua investor. Kinerja masa lalu bukan indikasi hasil di masa depan. Anda dapat kehilangan sebagian atau seluruh investasi Anda. **Gunakan dengan risiko Anda sendiri.**

## Lisensi

[MIT License](LICENSE) - Hak Cipta (c) 2025-2026 Gifari Kemal
`,
  },
  {
    slug: "features",
    title: "Fitur & Komponen",
    category: "Mulai di Sini",
    icon: "Sparkles",
    description: "Daftar lengkap fitur — 14 filter entry, 12 kondisi exit, manajemen risiko",
    content: `# XAUBot AI — Referensi Fitur

## Gambaran Umum

XAUBot AI adalah bot *trading* XAUUSD (Emas) otomatis yang menggabungkan **XGBoost *Machine Learning***, **Smart Money Concepts (SMC)**, dan **Hidden Markov Model (HMM)** untuk deteksi *regime*. Bot ini beroperasi di MetaTrader 5 melalui *loop* Python asinkron, mengeksekusi *trade* pada *timeframe* M15 (15 menit).

Bot mengikuti *pipeline* yang ketat: data diambil, fitur direkayasa, struktur pasar dianalisis, *regime* diklasifikasikan, prediksi ML dihasilkan, dan serangkaian 14 *filter* berurutan menentukan apakah *trade* dieksekusi. Setelah posisi terbuka, 12 kondisi *exit* dipantau setiap 5-10 detik.

---

## *Pipeline* 14 *Entry Filter*

Terdapat **14 *filter*** yang berjalan secara berurutan selama \`_trading_iteration()\`. Sebuah sinyal harus melewati **SEMUA** *filter* untuk mengeksekusi *trade*.

### 1. Pengambilan Data
- Mengambil **200 *bar* M15** dari MetaTrader 5.
- Data dikonversi ke **Polars *DataFrame*** (bukan Pandas).

### 2. Rekayasa Fitur (*Feature Engineering*)
- Menghitung **37 fitur teknikal** dari data OHLCV.
- Meliputi: *RSI*, *ATR*, *MACD*, *Bollinger Bands*, *EMA* (berbagai periode), *Stochastic*, indikator berbasis volume, dan lainnya.
- Semua komputasi menggunakan Polars untuk performa.

### 3. Analisis *SMC*
- Mendeteksi struktur institusional *Smart Money Concepts*:
  - ***Order Block* (OB)** — zona *supply/demand* dari aktivitas institusional.
  - ***Fair Value Gap* (FVG)** — ketidakseimbangan dalam *price action*.
  - ***Break of Structure* (BOS)** — sinyal kelanjutan tren.
  - ***Change of Character* (CHoCH)** — sinyal pembalikan arah.

### 4. Deteksi *Regime*
- ***HMM* (*Hidden Markov Model*)** mengklasifikasikan kondisi pasar saat ini:
  - \`TRENDING\` — pergerakan searah, kondusif untuk *entry*.
  - \`RANGING\` — konsolidasi menyamping, ukuran posisi dikurangi.
  - \`HIGH_VOLATILITY\` — pergerakan tidak menentu, butuh kehati-hatian.
  - \`CRISIS\` — kondisi ekstrem, *trading* diblokir.

### 5. Pelindung *Flash Crash*
- Proteksi darurat: jika pergerakan harga melebihi ambang persentase tertentu, **semua posisi langsung ditutup**.
- Mencegah kerugian katastropik saat dislokasi pasar mendadak.

### 6. *Filter Regime*
- Memblokir *trading* sepenuhnya jika rekomendasi *regime* adalah \`SLEEP\`.
- Mencegah *entry* saat kondisi pasar tidak menguntungkan yang teridentifikasi oleh *HMM*.

### 7. Pemeriksaan Risiko
- Memblokir *trading* jika:
  - **Batas kerugian harian** telah tercapai (5% dari kapital).
  - ***Equity*** terlalu rendah relatif terhadap *margin* yang dibutuhkan.
  - **Batas kerugian total** telah dilanggar (10% dari kapital).

### 8. *Filter* Sesi
- Memfilter berdasarkan sesi *trading* **WIB (Waktu Indonesia Barat)**.
- Setiap sesi menerapkan ***lot size multiplier*** untuk mengontrol eksposur:
  - **Sydney** (06:00-13:00 WIB) — *multiplier* 0.5x (volatilitas rendah).
  - **Tokyo** (07:00-16:00 WIB) — *multiplier* 0.7x (volatilitas sedang).
  - **London** (15:00-24:00 WIB) — *multiplier* 1.0x (volatilitas tinggi).
  - **New York** (20:00-24:00 WIB) — *multiplier* 1.0x (volatilitas ekstrem).
  - ***Off-Hours*** (00:00-06:00 WIB) — **diblokir sepenuhnya**.

### 9. *Filter Bias* H1 (#31B)
- Konfirmasi *multi-timeframe* menggunakan ***EMA20* pada *chart* H1**.
- Posisi harga relatif terhadap *EMA20* H1 menentukan bias arah:
  - **BULLISH** (harga di atas *EMA20*) — hanya sinyal *BUY* yang diizinkan.
  - **BEARISH** (harga di bawah *EMA20*) — hanya sinyal *SELL* yang diizinkan.
  - **NEUTRAL** (harga dekat *EMA20*) — **semua sinyal diblokir**.
- Hasil *backtest*: **+$343 peningkatan, *win rate* 81.8%, *Sharpe* 3.97**.

### 10. Generasi Sinyal *SMC*
- Menghasilkan sinyal ***BUY* atau *SELL*** berdasarkan analisis struktur *SMC*.
- Setiap sinyal memiliki ***confidence score*** yang berasal dari kualitas struktur yang terdeteksi (kedekatan *OB*, keselarasan *FVG*, konteks *BOS*/*CHoCH*).

### 11. Kombinasi Sinyal
- Menggabungkan **sinyal *SMC* + prediksi *ML* (*XGBoost*)**.
- Menerapkan ***dynamic confidence threshold*** yang beradaptasi berdasarkan:
  - Sesi *trading* saat ini.
  - *Regime* pasar.
  - Volatilitas terkini.
- Kedua sinyal harus sepakat arah; *confidence* gabungan harus melampaui *threshold*.

### 12. *Filter* Waktu (#34A)
- Melewatkan jam WIB tertentu yang dikenal berkondisi buruk:
  - **Jam 9 WIB** — akhir sesi *New York*, likuiditas rendah.
  - **Jam 21 WIB** — transisi *London*-*New York*, rawan *whipsaw*.
- Hasil *backtest*: **+$356 peningkatan**.

### 13. *Cooldown Trade*
- Memberlakukan jeda minimum **150 detik (2.5 menit)** antara *trade* berturut-turut.
- Mencegah *overtrading* dan *entry* bertubi-tubi dari sinyal yang noisy.

### 14. Gerbang Risiko Cerdas (*Smart Risk Gate*)
- Gerbang terakhir sebelum eksekusi. Memeriksa:
  - **Mode *trading***: \`NORMAL\`, \`RECOVERY\`, \`PROTECTED\`, atau \`STOPPED\`.
  - **Perhitungan *lot size***: Berdasarkan *ATR*, mode kapital, dan *multiplier* sesi.
  - **Batas posisi**: Maksimal **2 posisi bersamaan** diizinkan.
- Jika mode \`STOPPED\`, tidak ada *trade* yang dieksekusi terlepas dari kualitas sinyal.

---

## 12 Kondisi *Exit*

**12 kondisi *exit*** diperiksa setiap **5-10 detik** selama posisi terbuka.

### 1. *Take Profit* (TP Level Broker)
- *TP* dipasang di level broker saat *entry*.
- Dihitung menggunakan rasio *risk-reward* berbasis *ATR*.

### 2. *Trailing Stop* (#24B)
- ***Trailing stop* adaptif berbasis *ATR***:
  - Jarak aktivasi: ***ATR* x 4.0**.
  - Ukuran langkah: ***ATR* x 3.0**.
- Mengunci keuntungan seiring harga bergerak menguntungkan.

### 3. Perpindahan *Breakeven* (#24B)
- Memindahkan *stop loss* ke **harga *entry*** (*breakeven*) saat keuntungan belum direalisasi melampaui ***ATR* x 2.0**.
- Menghilangkan risiko pada *trade* setelah pergerakan menguntungkan.

### 4. *Exit* Pembalikan *ML*
- Menutup posisi jika *confidence* model *ML* **berbalik arah** dengan *confidence* melebihi **75%**.
- Merespons perubahan kondisi pasar yang terdeteksi oleh *XGBoost*.

### 5. Kerugian Maksimal Per *Trade*
- ***Stop loss* level perangkat lunak** sebesar **1% dari kapital**.
- Berfungsi sebagai jaring pengaman di samping *SL* broker.

### 6. Batas Kerugian Harian
- Jika kerugian kumulatif harian mencapai **5% dari kapital**, **semua posisi ditutup** dan *trading* dihentikan untuk hari itu.

### 7. Batas Kerugian Total
- Jika kerugian kumulatif total mencapai **10% dari kapital**, ***trading* dihentikan sepenuhnya** sampai intervensi manual.

### 8. Penanganan Penutupan Pasar
- Sebelum penutupan harian atau penutupan akhir pekan:
  - Mengambil keuntungan pada posisi dengan *unrealized profit* **> $5**.
  - Mencegah risiko *gap* dari posisi yang terbawa semalam/akhir pekan.

### 9. Darurat *Flash Crash*
- Dipicu oleh pergerakan harga ekstrem secara tiba-tiba.
- **Langsung menutup semua posisi terbuka** tanpa penundaan.

### 10. Proteksi *Drawdown*
- Memantau *drawdown* dari puncak *equity*.
- Menutup semua posisi jika *drawdown* melebihi **50%** dari puncak.

### 11. *Impulse Trail* (#33B)
- *Trailing stop* yang ditingkatkan menggunakan **deteksi *impulse candle***.
- Mengidentifikasi *candle* momentum kuat dan men-*trail* *stop* di belakangnya.
- Lebih responsif dibanding *trailing ATR* standar dalam kondisi tren.

### 12. *Smart Breakeven* (#28B)
- Logika *breakeven* yang ditingkatkan dengan **pemicu *ATR multiplier***:
  - Pemicu: keuntungan melampaui ***ATR* x 2.0**.
  - Memindahkan *SL* ke *entry* + *buffer* kecil.
- Lebih adaptif dibanding *breakeven* berbasis pip tetap.

---

## Riwayat Optimasi *Backtest*

Rangkuman optimasi utama yang diterapkan ke bot *live*, diuji dan divalidasi melalui *backtest*.

| # | Nama | Perubahan Utama | Hasil |
|---|------|-----------------|-------|
| #24B | *ATR-Adaptive Exit* | *Trailing* berbasis *ATR* (4.0x) dan *breakeven* (2.0x) *multiplier* | Optimasi dasar untuk logika *exit* |
| #28B | *Smart Breakeven* | *Breakeven* yang ditingkatkan dengan pemicu *ATR* x 2.0 | Peningkatan waktu *exit* pada *trade* yang menang |
| #31B | *Filter* H1 *EMA20* | *Filter multi-timeframe* harga H1 vs *EMA20* | +$343, WR 81.8%, *Sharpe* 3.97 |
| #33B | *Impulse Trail* | *Trail* menggunakan deteksi *impulse candle* | *Trailing* lebih baik di pasar tren |
| #34A | Lewati Jam Tertentu | Lewati jam WIB 9 dan 21 | +$356, pengurangan kerugian *whipsaw* |

---

## Manajemen Risiko

### Mode Kapital

Mode kapital dikonfigurasi otomatis berdasarkan saldo akun. Setiap mode mengatur parameter risiko yang sesuai untuk ukuran akun.

| Mode | Rentang Kapital | Risiko/*Trade* | *Lot* Maks |
|------|----------------|----------------|------------|
| MICRO | < $500 | 2% | 0.02 |
| SMALL | $500 - $10,000 | 1.5% | 0.05 |
| MEDIUM | $10,000 - $100,000 | 0.5% | 0.10 |
| LARGE | > $100,000 | 0.25% | 0.50 |

### Mode *Trading*

*Smart Risk Manager* secara dinamis menyesuaikan mode *trading* berdasarkan performa terkini.

| Mode | Pemicu | Penyesuaian *Lot* |
|------|--------|-------------------|
| NORMAL | Kondisi *default* | *Lot* dasar (0.01-0.03) |
| RECOVERY | Setelah *trade* rugi | *Lot* pemulihan (0.01) |
| PROTECTED | Mendekati batas kerugian harian | *Lot* minimum (0.01) |
| STOPPED | Batas kerugian harian atau total tercapai | *Trading* tidak diizinkan |

### Batas Risiko

| Batas | Nilai | Aksi |
|-------|-------|------|
| Kerugian harian maks | 5% dari kapital | Tutup semua posisi, hentikan *trading* untuk hari itu |
| Kerugian total maks | 10% dari kapital | Hentikan semua *trading* sampai *reset* manual |
| Kerugian maks per *trade* | 1% dari kapital | *Stop loss* perangkat lunak |
| *SL* darurat broker | 2% dari kapital | *Hard stop* level broker |
| Posisi bersamaan maks | 2 | Tolak *entry* baru jika sudah di batas |

---

## *Filter* Sesi (WIB)

Semua waktu sesi dalam **WIB (Waktu Indonesia Barat, UTC+7)**.

| Sesi | Jam (WIB) | Volatilitas | *Multiplier Lot* |
|------|-----------|-------------|-------------------|
| Sydney | 06:00 - 13:00 | Rendah | 0.5x |
| Tokyo | 07:00 - 16:00 | Sedang | 0.7x |
| London | 15:00 - 24:00 | Tinggi | 1.0x |
| New York | 20:00 - 24:00 | Ekstrem | 1.0x |
| *Off-Hours* | 00:00 - 06:00 | N/A | **Diblokir** |

### *Golden Hour*
- **19:00 - 23:00 WIB** (*London*-*New York Overlap*).
- Periode likuiditas dan volatilitas tertinggi untuk XAUUSD.
- Kondisi *trading* terbaik; *multiplier lot* penuh diterapkan.

### Jam yang Dilewati (#34A)
- **Jam 9 WIB** — Akhir sesi *New York*; likuiditas rendah menyebabkan *fill* yang tidak menentu.
- **Jam 21 WIB** — Transisi *London*-*New York*; rawan *whipsaw* dan *false breakout*.

---

## *Auto-Trainer*

Bot menyertakan *pipeline* pelatihan ulang model otomatis untuk menjaga model *ML* tetap mutakhir dengan kondisi pasar.

| Parameter | Nilai |
|-----------|-------|
| Interval pemeriksaan | Setiap 20 *candle* (~5 jam pada M15) |
| Pelatihan ulang harian | 05:00 WIB (saat pasar tutup) |
| Pelatihan akhir pekan | Pelatihan mendalam dengan jendela data yang diperluas |
| *Threshold AUC* minimum | 0.65 |
| Kebijakan *rollback* | Jika model baru berkinerja lebih buruk, kembali ke *backup* |

### Alur Pelatihan Ulang
1. Setiap 20 *candle*, *auto-trainer* memeriksa metrik performa model.
2. Jika *AUC* turun di bawah **0.65**, pelatihan ulang dipicu.
3. Pada **05:00 WIB setiap hari** (pasar tutup), pelatihan ulang terjadwal berjalan.
4. Pada **akhir pekan**, pelatihan mendalam menggunakan *dataset* historis yang lebih besar.
5. Setelah pelatihan, model baru divalidasi terhadap model sebelumnya.
6. Jika model baru berkinerja lebih buruk, sistem **melakukan *rollback*** ke model *backup*.

---

## Model *ML*

### Algoritma
- ***XGBoost* *gradient-boosted decision trees***.

### Fitur
- **37 indikator teknikal** dihitung oleh \`src/feature_eng.py\`:
  - Tren: *EMA* (berbagai periode), *MACD*, *ADX*.
  - Momentum: *RSI*, *Stochastic K/D*.
  - Volatilitas: *ATR*, *Bollinger Bands* (*width*, *%B*).
  - Volume: Indikator berbasis volume.
  - Kustom: Fitur turunan *SMC*, fitur *regime*.

### Keluaran
- **Sinyal**: *BUY*, *SELL*, atau *HOLD*.
- ***Confidence score***: 0.0 hingga 1.0, digunakan dalam kombinasi dengan *confidence SMC*.

### *Threshold* Dinamis
- *Threshold confidence* untuk eksekusi *trade* tidak tetap.
- Menyesuaikan berdasarkan:
  - **Sesi**: *Threshold* lebih tinggi saat sesi volatilitas rendah.
  - ***Regime***: *Threshold* lebih tinggi saat *regime ranging*/*volatile*.
  - **Performa terkini**: Diperketat setelah kerugian, dilonggarkan setelah kemenangan.

---

## Komponen Aktif

| Komponen | File | Status | Deskripsi |
|----------|------|--------|-----------|
| Penganalisis *SMC* | \`src/smc_polars.py\` | Aktif | Deteksi *Order Block*, *FVG*, *BOS*, *CHoCH* |
| *ML XGBoost* | \`src/ml_model.py\` | Aktif | Prediksi sinyal dengan *confidence* |
| *Regime HMM* | \`src/regime_detector.py\` | Aktif | Klasifikasi *regime* pasar |
| Mesin Fitur | \`src/feature_eng.py\` | Aktif | Komputasi 37 fitur teknikal |
| Mesin Risiko | \`src/risk_engine.py\` | Aktif | *SL*/*TP* berbasis *ATR*, *position sizing* |
| *Smart Risk Manager* | \`src/smart_risk_manager.py\` | Aktif | Manajemen mode dinamis |
| Manajer Posisi | \`src/position_manager.py\` | Aktif | Pemantauan kondisi *exit* |
| *Filter* Sesi | \`src/session_filter.py\` | Aktif | *Filtering* berbasis sesi WIB |
| *Confidence* Dinamis | \`src/dynamic_confidence.py\` | Aktif | Penyesuaian *threshold* adaptif |
| *Auto Trainer* | \`src/auto_trainer.py\` | Aktif | Pelatihan ulang model terjadwal |
| Notifikasi Telegram | \`src/telegram_notifier.py\` | Aktif | Peringatan *trade* via Telegram |
| Pencatat *Trade* | \`src/trade_logger.py\` | Aktif | Pencatatan *trade* ke PostgreSQL |
| Agen Berita | \`src/news_agent.py\` | **NONAKTIF** | *Filter* berita ekonomi (mengurangi $178 profit di *backtest*) |
| Detektor *Flash Crash* | \`src/regime_detector.py\` | Aktif | Penutupan posisi darurat |

---

## Diagram Arsitektur

\`\`\`mermaid
flowchart TD
    MT5["MT5 Broker"] --> DF["Data Fetch"]
    DF --> FE["Feature Eng (37 fitur)"]
    FE --> SMC["SMC Analysis"]
    SMC --> HMM["Regime Detection (HMM)"]
    HMM --> FCG["Flash Crash Guard"]
    FCG --> RF["Regime Filter"]
    RF --> RC["Risk Check"]
    RC --> SF["Session Filter"]
    SF --> H1["H1 Bias Filter (#31B)"]
    H1 --> SG["SMC Signal Gen"]
    SG --> SC["Signal Combination (ML+SMC)"]
    SC --> TF["Time Filter (#34A)"]
    TF --> TC["Trade Cooldown"]
    TC --> SRG["Smart Risk Gate"]
    SRG --> TE["TRADE EXECUTION"]
    TE --> PM["Position Manager (12 exits)"]
    PM --> LOG["Telegram + PostgreSQL Logging"]
\`\`\`
`,
  },
  {
    slug: "architecture-full",
    title: "Arsitektur Lengkap",
    category: "Mulai di Sini",
    icon: "LayoutDashboard",
    description: "Arsitektur menyeluruh sistem — alur data, komponen, dan interaksi antar modul",
    content: `# Arsitektur Lengkap — Smart AI Trading Bot

> **Dokumen:** Arsitektur keseluruhan sistem dalam 1 file
> **Instrumen:** XAUUSD (Gold) M15
> **Platform:** MetaTrader 5
> **Bahasa:** Python 3.11+ (async, Polars, XGBoost, HMM)
> **Database:** PostgreSQL + CSV fallback
> **Notifikasi:** Telegram Bot API

---

## Daftar Isi

1. [Gambaran Umum](#1-gambaran-umum)
2. [Diagram Arsitektur](#2-diagram-arsitektur)
3. [23 Komponen](#3-23-komponen)
4. [Pipeline Data: Dari OHLCV ke Keputusan Trading](#4-pipeline-data)
5. [Alur Entry: 14 Filter](#5-alur-entry-14-filter)
6. [Alur Exit: 12 Kondisi](#6-alur-exit-12-kondisi)
7. [Sistem Proteksi Risiko 4 Lapis](#7-sistem-proteksi-risiko-4-lapis)
8. [AI/ML Engine](#8-aiml-engine)
9. [Smart Money Concepts (SMC)](#9-smart-money-concepts)
10. [Position Lifecycle](#10-position-lifecycle)
11. [Auto-Retraining & Model Management](#11-auto-retraining)
12. [Infrastruktur & Database](#12-infrastruktur--database)
13. [Konfigurasi & Parameter Kritis](#13-konfigurasi--parameter-kritis)
14. [Performa & Timing](#14-performa--timing)
15. [Error Handling & Fault Tolerance](#15-error-handling--fault-tolerance)
16. [Daftar File Source Code](#16-daftar-file-source-code)

---

## 1. Gambaran Umum

### Apa Ini?

Bot trading otomatis yang menggabungkan **3 otak kecerdasan buatan** untuk trading XAUUSD (Emas) di MetaTrader 5:

\`\`\`
OTAK 1: Smart Money Concepts (SMC)
        → Membaca pola institusi besar (bank, hedge fund)
        → Menentukan DIMANA entry, SL, dan TP

OTAK 2: XGBoost Machine Learning
        → Memprediksi ARAH harga (naik/turun)
        → Memberikan tingkat keyakinan (confidence)

OTAK 3: Hidden Markov Model (HMM)
        → Membaca KONDISI pasar (tenang/volatile/krisis)
        → Menyesuaikan ukuran posisi dan agresivitas
\`\`\`

### Filosofi Desain

\`\`\`
1. KESELAMATAN MODAL NOMOR 1
   → 4 lapis proteksi stop loss
   → Lot ultra-kecil (0.01-0.02)
   → Circuit breaker otomatis

2. TIDAK PERNAH CRASH
   → Setiap error di-catch, bot terus jalan
   → Database gagal? CSV fallback
   → MT5 putus? Auto-reconnect

3. SELF-IMPROVING
   → Model AI dilatih ulang otomatis setiap hari
   → Rollback otomatis jika model baru lebih buruk
   → Threshold confidence menyesuaikan kondisi pasar

4. TRANSPARAN
   → Setiap keputusan dicatat ke database
   → Notifikasi Telegram real-time
   → Laporan harian, jam-an, dan per-trade
\`\`\`

### Angka-Angka Kunci

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| Modal target | $5,000 | Small account mode |
| Risiko per trade | 1% ($50) | Maksimum kerugian per posisi |
| Lot size | 0.01 - 0.02 | Ultra-konservatif |
| Max daily loss | 3% ($150) | Circuit breaker harian |
| Max total loss | 10% ($500) | Stop total trading |
| Max posisi bersamaan | 2-3 | Menghindari overexposure |
| Cooldown antar trade | 5 menit | Mencegah overtrading |
| Loop speed | ~50ms | Cepat tapi efisien |
| Timeframe | M15 | 15 menit per candle |

---

## 2. Diagram Arsitektur

### Diagram Keseluruhan Sistem

\`\`\`mermaid
graph TD
    subgraph MAIN["MAIN LIVE — Orchestrator<br/>main_live.py — TradingBot<br/>Candle-based M15 + position check ~10 detik"]
        direction TB

        subgraph P1["PHASE 1: DATA"]
            MT5C["MT5 Connector<br/>(broker)"] --> FEng["Feature Engine<br/>(40+ fitur)"]
            FEng --> SMCA["SMC Analyzer<br/>(institusi)"]
            MT5C ~~~ HMMD["HMM Regime<br/>(3 state)"]
            FEng --> XGB["XGBoost ML Model<br/>(prediksi)"]
            SMCA --> XGB
            HMMD --> DC["Dynamic Confidence<br/>(threshold)"]
        end

        subgraph P2["PHASE 2: MONITORING"]
            PM["Position Manager<br/>(trailing)"]
            SRM["Smart Risk Manager"]
            RiskE["Risk Engine<br/>(Kelly)"]
        end

        subgraph P3["PHASE 3: ENTRY"]
            SF["Session Filter<br/>(waktu)"]
            NA["News Agent<br/>(berita)"]
            EF["14-Gate Entry Filter"] --> EXEC["EXECUTE"]
        end

        subgraph P4["PHASE 4: PERIODIK"]
            AT["Auto Trainer<br/>(retrain)"]
            TN["Telegram Notifier<br/>(laporan)"]
            TL["Trade Logger<br/>(DB+CSV)"]
        end

        P1 --> P2
        P2 --> P3
        P3 --> P4
        P4 --> DB["PostgreSQL + CSV Backup"]
    end
\`\`\`

### Alur Data (Data Flow)

\`\`\`mermaid
flowchart TD
    MT5B["MT5 Broker<br/>(XAUUSD M15)"] -->|"200 bar OHLCV"| MT5C["MT5 Connector<br/>numpy → Polars (tanpa Pandas)"]
    MT5C --> FE["Feature Engineer<br/>OHLCV → 40+ fitur teknikal<br/>RSI, ATR, MACD, BB, EMA, Volume"]
    FE --> SMC["SMC Analyzer"]
    FE --> HMM["HMM Regime Detect"]
    SMC --> XGB["XGBoost ML Predictor<br/>24 fitur → BUY/SELL/HOLD<br/>+ Confidence 0-100%"]
    HMM --> XGB
    XGB --> DCM["Dynamic Confidence<br/>Threshold berdasarkan sesi,<br/>regime, volatilitas, trend"]
    DCM --> SC["Signal Combiner (14 Filter)<br/>SMC + ML harus setuju<br/>+ Session + Risk + Cooldown"]
    SC --> PASS{"PASS?"}
    PASS -->|"NO"| WAIT["Tunggu loop berikutnya"]
    PASS -->|"YES"| RE["Risk Engine + Risk Manager<br/>Kelly Criterion → lot size<br/>Validasi order → approve/reject"]
    RE --> EXEC["Execute Order via MT5<br/>Kirim ke MT5 dengan SL dan TP<br/>Register ke Position Manager"]
    EXEC --> TN["Telegram Notifier"]
    EXEC --> TL["Trade Logger<br/>(DB + CSV)"]
\`\`\`

---

## 3. 23 Komponen

### Tabel Komponen Lengkap

| # | Komponen | File | Kategori | Fungsi Utama |
|---|----------|------|----------|-------------|
| 1 | HMM Regime Detector | \`src/regime_detector.py\` | AI/ML | Deteksi kondisi pasar (3 regime) |
| 2 | XGBoost Predictor | \`src/ml_model.py\` | AI/ML | Prediksi arah harga + confidence |
| 3 | SMC Analyzer | \`src/smc_polars.py\` | Analisis | Pola institusi: FVG, OB, BOS, CHoCH |
| 4 | Feature Engineering | \`src/feature_eng.py\` | Data | OHLCV → 40+ fitur numerik |
| 5 | Smart Risk Manager | \`src/smart_risk_manager.py\` | Risiko | 4 mode *trading*, 12 kondisi *exit* |
| 6 | Session Filter | \`src/session_filter.py\` | Filter | Waktu trading optimal (WIB) |
| 7 | Stop Loss (4 Lapis) | Multi-file | Proteksi | SMC → Software → Emergency → Circuit |
| 8 | Take Profit (6 Layer) | Multi-file | Proteksi | Hard → Momentum → Peak → Probability → Early → Broker |
| 9 | Entry Trade | \`main_live.py\` | Eksekusi | 14 *filter* berurutan |
| 10 | Exit Trade | \`main_live.py\` | Eksekusi | 12 kondisi *exit real-time* |
| 11 | News Agent | \`src/news_agent.py\` | Monitor | **NONAKTIF** — dikomentari di kode |
| 12 | Telegram Notifier | \`src/telegram_notifier.py\` | Notifikasi | 11 tipe notifikasi real-time |
| 13 | Auto Trainer | \`src/auto_trainer.py\` | ML Ops | Retraining harian otomatis |
| 14 | Backtest | \`backtests/backtest_live_sync.py\` | Validasi | Simulasi 100% sync dengan live |
| 15 | Dynamic Confidence | \`src/dynamic_confidence.py\` | Adaptif | Threshold ML adaptif (60-85%) |
| 16 | MT5 Connector | \`src/mt5_connector.py\` | Koneksi | Bridge ke broker, auto-reconnect |
| 17 | Configuration | \`src/config.py\` | Config | 6 sub-config, auto-adjust modal |
| 18 | Trade Logger | \`src/trade_logger.py\` | Logging | Dual storage DB + CSV |
| 19 | Position Manager | \`src/position_manager.py\` | Manajemen | Trailing SL, breakeven, market close |
| 20 | Risk Engine | \`src/risk_engine.py\` | Risiko | Kelly Criterion, circuit breaker |
| 21 | Database | \`src/db/\` | Storage | PostgreSQL, 6 repository |
| 22 | Train Models | \`train_models.py\` | Training | Script training awal |
| 23 | Main Live | \`main_live.py\` | Orchestrator | Koordinasi semua komponen |

### Hubungan Antar Komponen

\`\`\`mermaid
flowchart TD
    CONFIG["CONFIGURATION (17)<br/>Sumber parameter semua"] -->|"dikonsumsi oleh semua"| MT5_16
    CONFIG --> FE4
    CONFIG --> SMC3

    MT5_16["MT5 (16)<br/>Broker"] --> FE4["FeatEng (4)<br/>40+ fitur"]
    FE4 --> SMC3["SMC (3)<br/>Institusi"]
    FE4 --> HMM1["HMM (1)<br/>Regime"]
    FE4 --> XGB2["XGBoost (2)<br/>Prediksi"]
    SMC3 --> XGB2

    HMM1 --> DC15["Dynamic Confidence (15)<br/>Threshold adaptif"]
    XGB2 --> DC15

    SESSION6["Session (6)<br/>Waktu"] --> ENTRY9["ENTRY TRADE (9)<br/>14 Filter Gate"]
    DC15 --> ENTRY9
    NEWS11["News (11)<br/>Berita"] --> ENTRY9

    ENTRY9 --> RISK20["RiskEng (20)<br/>Kelly Lot"]
    ENTRY9 --> SRISK5["SmartRisk (5)<br/>4 Mode"]

    RISK20 --> EXEC["EXECUTE ORDER<br/>via MT5 (16)"]
    SRISK5 --> EXEC

    EXEC --> PM19["PosMgr (19)<br/>Trailing"]
    EXEC --> LOG18["Logger (18)<br/>DB+CSV"]
    EXEC --> TG12["Telegram (12)"]

    PM19 --> EXIT10["EXIT (10)<br/>12 Kondisi"]
    LOG18 --> DB21["DB (21)<br/>PostgreSQL"]

    subgraph PERIODIK["Periodik"]
        AT13["AutoTrain (13)<br/>Harian"]
        BT14["Backtest (14)<br/>Validasi"]
        TM22["Train Models (22)<br/>Setup awal"]
    end
\`\`\`

---

## 4. Pipeline Data

### Dari OHLCV Mentah ke Keputusan Trading

#### Tahap 1: Data Fetching (MT5 Connector)

\`\`\`mermaid
flowchart TD
    MT5B["MT5 Broker"] -->|"mt5.copy_rates_from_pos<br/>(XAUUSD, M15, 0, 200)"| NPA["NumPy Structured Array"]
    NPA -->|"Konversi langsung ke Polars<br/>(TANPA Pandas)"| PDF["Polars DataFrame:<br/>time (i64), open (f64), high (f64),<br/>low (f64), close (f64),<br/>tick_volume (f64), spread (f64)"]
\`\`\`

**Kenapa Polars, bukan Pandas?**
- 3-5x lebih cepat untuk operasi vectorized
- Memory-efficient (zero-copy)
- Native lazy evaluation
- Konsisten di seluruh codebase (tidak ada konversi bolak-balik)

#### Tahap 2: Feature Engineering (40+ Fitur)

\`\`\`mermaid
flowchart TD
    INPUT["Input: Polars DataFrame<br/>(200 bar OHLCV)"] --> MOM["Momentum Indicators"]
    INPUT --> VOL["Volatility Indicators"]
    INPUT --> TREND["Trend Indicators"]
    INPUT --> PA["Price Action"]
    INPUT --> VOLF["Volume Features"]
    INPUT --> LAG["Lag Features"]
    INPUT --> TIME["Time Features"]

    MOM --> M1["RSI(14) - 0-100, overbought/oversold"]
    MOM --> M2["MACD(12,26,9) - trend strength and direction"]
    MOM --> M3["MACD Histogram - momentum acceleration"]

    VOL --> V1["ATR(14) - average true range (pips)"]
    VOL --> V2["Bollinger Bands(20,2.0) - upper, lower, width"]
    VOL --> V3["Volatility(20) - rolling std of returns"]

    TREND --> T1["EMA(9) / EMA(21) - fast/slow crossover"]
    TREND --> T2["EMA Cross Signal - 1 bullish / -1 bearish"]
    TREND --> T3["SMA(20) - simple moving average"]

    PA --> P1["Returns(1,5,20) - % perubahan harga"]
    PA --> P2["Log Returns - untuk distribusi normal"]
    PA --> P3["Price Position - posisi dalam range BB"]
    PA --> P4["Higher High/Lower Low count - trend structure"]

    VOLF --> VF1["Volume SMA(20) - rata-rata volume"]
    VOLF --> VF2["Volume Ratio - current / average"]

    LAG --> L1["close_lag_1..5 - harga sebelumnya"]
    LAG --> L2["returns_lag_1..3 - return sebelumnya"]

    TIME --> TI1["Hour, Weekday - waktu candle"]
    TIME --> TI2["Session flags - london, ny, overlap"]

    M1 & M2 & M3 & V1 & V2 & V3 & T1 & T2 & T3 & P1 & P2 & P3 & P4 & VF1 & VF2 & L1 & L2 & TI1 & TI2 --> OUTPUT["Output: DataFrame + 40 kolom baru<br/>(semua numerik, siap ML)"]
\`\`\`

**Minimum data:** 26 bar untuk semua indikator stabil

#### Tahap 3: SMC Analysis (Pola Institusi)

\`\`\`mermaid
flowchart TD
    INPUT["Input: DataFrame dengan OHLCV"] --> SP["Swing Points (Fractal)<br/>Window: 11 bar (swing_length=5, +/-5 dari tengah)<br/>Output: swing_high (1/0), swing_low (-1/0), level harga"]
    INPUT --> FVG["Fair Value Gaps (FVG)<br/>Bullish: bar i-2 high < bar i low (gap up)<br/>Bearish: bar i-2 low > bar i high (gap down)<br/>Output: fvg_bull, fvg_bear, fvg_top, fvg_bottom, fvg_mid"]
    INPUT --> OB["Order Blocks (OB)<br/>Lookback: 10 bar<br/>Bullish: candle bearish terakhir sebelum move up besar<br/>Bearish: candle bullish terakhir sebelum move down besar<br/>Output: ob (1/-1), ob_top, ob_bottom, ob_mitigated"]
    INPUT --> BOS["Break of Structure (BOS)<br/>Harga break swing high/low = trend continuation<br/>Output: bos (1/-1), level yang di-break"]
    INPUT --> CHOCH["Change of Character (CHoCH)<br/>Harga break berlawanan arah trend = reversal signal<br/>Output: choch (1/-1), level yang di-break"]
    INPUT --> LIQ["Liquidity Zones<br/>BSL: Buy Side Liquidity (above swing highs)<br/>SSL: Sell Side Liquidity (below swing lows)<br/>Output: bsl_level, ssl_level"]

    SP & FVG & OB & BOS & CHOCH & LIQ --> SIG["Signal Generation<br/>Syarat: Structure break + (FVG ATAU Order Block)"]

    SIG --> ENTRY["Entry: harga saat ini"]
    SIG --> SL["SL: ATR-based, minimum 1.5 x ATR dari entry"]
    SIG --> TP["TP: 2:1 Risk-Reward minimum, cap 4 x ATR"]
    SIG --> CONF["Confidence: 40-85%<br/>(v5: calibrated weighted scoring)"]
    SIG --> REASON["Reason: BOS + Bullish FVG at 2645.50"]
\`\`\`

#### Tahap 4: Regime Detection (HMM)

\`\`\`mermaid
flowchart TD
    INPUT["Input: log_returns + normalized_range (volatilitas)"] -->|"GaussianHMM(n_components=3, lookback=500)"| REGIME["3 Regime Output"]

    REGIME --> R0["REGIME 0: Low Volatility<br/>Pasar tenang, range kecil<br/>Lot multiplier: 1.0x (normal)<br/>Rekomendasi: TRADE"]
    REGIME --> R1["REGIME 1: Medium Volatility<br/>Pasar aktif, trend jelas<br/>Lot multiplier: 1.0x (normal)<br/>Rekomendasi: TRADE"]
    REGIME --> R2["REGIME 2: High Volatility<br/>Pasar sangat volatile, berbahaya<br/>Lot multiplier: 0.5x (setengah)<br/>Rekomendasi: REDUCE"]
    REGIME --> RC["CRISIS (FlashCrashDetector)<br/>Move > 2.5% dalam 1 menit<br/>Lot multiplier: 0.0x (STOP)<br/>Rekomendasi: EMERGENCY CLOSE ALL"]

    style R0 fill:#4CAF50,color:#fff
    style R1 fill:#2196F3,color:#fff
    style R2 fill:#FF9800,color:#fff
    style RC fill:#F44336,color:#fff
\`\`\`

#### Tahap 5: ML Prediction (XGBoost)

\`\`\`mermaid
flowchart TD
    INPUT["Input: 24 fitur terpilih dari<br/>Feature Engineering + SMC + Regime"] -->|"XGBoost Binary Classifier<br/>max_depth=3, learning_rate=0.05<br/>min_child_weight=10, subsample=0.7<br/>colsample_bytree=0.6<br/>reg_alpha=1.0 (L1), reg_lambda=5.0 (L2)"| OUTPUT["Output:<br/>prob_up: 0.72 (probabilitas naik)<br/>prob_down: 0.28 (probabilitas turun)"]

    OUTPUT --> SIG["Signal: BUY (prob_up > 0.50)<br/>Confidence: 72%"]
    SIG --> TH1["prob > 0.50 - ada sinyal (minimum)"]
    SIG --> TH2["prob > 0.65 - sinyal kuat"]
    SIG --> TH3["prob > 0.75 - sinyal sangat kuat"]
    SIG --> TH4["prob > 0.80 - lot bisa naik ke 0.02"]
\`\`\`

#### Tahap 6: Dynamic Confidence (Threshold Adaptif)

\`\`\`mermaid
flowchart TD
    BASE["Base Score: 50"] --> SESS["Session Modifier<br/>Golden Time (20:00-23:59 WIB): +20<br/>London (15:00-23:59): +15<br/>New York (20:00-05:00): +10<br/>Tokyo/Sydney: +0<br/>Market Closed: -30"]
    BASE --> REG["Regime Modifier<br/>Medium Volatility: +15<br/>Low Volatility: +5<br/>High Volatility: -5<br/>Crisis: -25"]
    BASE --> VOLM["Volatility Modifier<br/>Medium (ideal): +10<br/>Low: +0<br/>High: -5<br/>Extreme: -10"]
    BASE --> TREN["Trend Modifier<br/>Trending (jelas): +10<br/>Ranging (sideways): -5"]
    BASE --> SMCC["SMC Confluence<br/>Ada konfluensi: +10<br/>Tidak ada: +0"]
    BASE --> MLC["ML Confidence<br/>ge 70%: +5<br/>ge 60%: +2<br/>lt 60%: +0"]

    SESS & REG & VOLM & TREN & SMCC & MLC --> TOTAL["Total Score (0-100)"]

    TOTAL --> EXC["EXCELLENT (ge 80 poin)<br/>Threshold: 60% (longgar)"]
    TOTAL --> GOOD["GOOD (65-79)<br/>Threshold: 65%"]
    TOTAL --> MOD["MODERATE (50-64)<br/>Threshold: 70%"]
    TOTAL --> POOR["POOR (35-49)<br/>Threshold: 80% (ketat)"]
    TOTAL --> AVOID["AVOID (lt 35)<br/>Threshold: 85% (SKIP)"]

    style EXC fill:#4CAF50,color:#fff
    style GOOD fill:#8BC34A,color:#fff
    style MOD fill:#FF9800,color:#fff
    style POOR fill:#FF5722,color:#fff
    style AVOID fill:#F44336,color:#fff
\`\`\`

Contoh: Golden Time + Medium Vol + Trending + SMC + ML 72%
= 50 + 20 + 15 + 10 + 10 + 10 + 5 = 120 (cap 100)
= EXCELLENT -> Threshold 60% -> ML 72% PASS

---

## 5. Alur Entry: 14 Filter

Setiap sinyal harus melewati **14 gerbang berurutan**. Satu saja gagal = TIDAK trading.

\`\`\`mermaid
flowchart TD
    S["SINYAL SMC + ML MASUK"] --> F1{"Filter 1: Session<br/>Jam trading dibolehkan?<br/>BLOCK: 00:00-06:00 WIB (dead zone)<br/>BLOCK: Jumat ge 23:00 WIB"}
    F1 -->|"BLOCK"| SKIP1["SKIP"]
    F1 -->|"PASS: London/NY/Golden Time"| F2{"Filter 2: Risk Mode<br/>Bukan STOPPED?<br/>BLOCK: daily/total limit hit"}
    F2 -->|"BLOCK"| SKIP2["SKIP"]
    F2 -->|"PASS: NORMAL/RECOVERY/PROTECTED"| F3{"Filter 3: SMC Signal<br/>Ada setup SMC valid?<br/>BLOCK: Tidak ada FVG/OB + BOS/CHoCH"}
    F3 -->|"BLOCK"| SKIP3["SKIP"]
    F3 -->|"PASS: Ada sinyal BUY/SELL + SL + TP"| F4{"Filter 4: ML Confidence<br/>ML confidence ge dynamic threshold?<br/>Threshold 60-85% tergantung quality"}
    F4 -->|"BLOCK"| SKIP4["SKIP"]
    F4 -->|"PASS: confidence ge threshold"| F5{"Filter 5: ML Agreement<br/>ML TIDAK strongly disagree?<br/>BLOCK: ML > 65% berlawanan arah SMC"}
    F5 -->|"BLOCK"| SKIP5["SKIP"]
    F5 -->|"PASS: ML setuju atau netral"| F6{"Filter 6: Market Quality<br/>Dynamic Confidence bukan AVOID?<br/>BLOCK: Quality == AVOID (score lt 35)"}
    F6 -->|"BLOCK"| SKIP6["SKIP"]
    F6 -->|"PASS: EXCELLENT/GOOD/MODERATE/POOR"| F7{"Filter 7: Signal Confirmation<br/>Sinyal konsisten 2x berturut?<br/>BLOCK: Baru muncul 1x"}
    F7 -->|"BLOCK"| SKIP7["SKIP"]
    F7 -->|"PASS: 2x berturut"| F8{"Filter 8: Pullback (v5: ATR-based)<br/>Momentum selaras?<br/>BLOCK: RSI overbought/oversold<br/>BLOCK: MACD berlawanan<br/>BLOCK: Bounce > 15% ATR (3 candle)"}
    F8 -->|"BLOCK"| SKIP8["SKIP"]
    F8 -->|"PASS: Momentum selaras"| F9{"Filter 9: Trade Cooldown<br/>Sudah ge 5 menit sejak trade terakhir?<br/>BLOCK: lt 300 detik"}
    F9 -->|"BLOCK"| SKIP9["SKIP"]
    F9 -->|"PASS: ge 300 detik"| F10{"Filter 10: Position Limit<br/>Posisi terbuka lt limit?<br/>BLOCK: Sudah 2+ posisi terbuka"}
    F10 -->|"BLOCK"| SKIP10["SKIP"]
    F10 -->|"PASS: lt 2 posisi"| F11{"Filter 11: Lot Size<br/>Lot size > 0?<br/>BLOCK: Lot = 0 (crisis/risk tinggi)"}
    F11 -->|"BLOCK"| SKIP11["SKIP"]
    F11 -->|"PASS: Lot ge 0.01"| F12{"Filter 12: Flash Crash Guard<br/>TIDAK ada flash crash?<br/>BLOCK: Move > 2.5% dalam 1 menit"}
    F12 -->|"BLOCK + CLOSE ALL"| SKIP12["SKIP"]
    F12 -->|"PASS: Normal"| F13{"Filter 13: H1 Bias (#31B)<br/>EMA20 H1 selaras arah sinyal?<br/>BLOCK: BUY tapi harga lt EMA20 H1<br/>BLOCK: SELL tapi harga > EMA20 H1"}
    F13 -->|"BLOCK"| SKIP13["SKIP"]
    F13 -->|"PASS: Selaras"| F14{"Filter 14: Time Filter (#34A)<br/>Bukan jam transisi?<br/>BLOCK: Jam 9 atau 21 WIB"}
    F14 -->|"BLOCK"| SKIP14["SKIP"]
    F14 -->|"PASS"| EXEC["EXECUTE TRADE<br/>BUY atau SELL<br/>via MT5 Connector"]

    EXEC --> REG["Register ke SmartRiskManager"]
    EXEC --> LOG["Log ke TradeLogger (DB + CSV)"]
    EXEC --> TG["Kirim notifikasi Telegram"]

    style EXEC fill:#4CAF50,color:#fff,stroke:#388E3C,stroke-width:3px
\`\`\`

---

## 6. Alur Exit: 12 Kondisi

Setiap posisi terbuka dievaluasi **setiap ~10 detik** (di antara candle) atau **setiap candle baru** (full analysis) terhadap 12 kondisi exit:

\`\`\`mermaid
flowchart TD
    POS["POSISI TERBUKA<br/>(dicek setiap ~10 detik)<br/>Update: profit, momentum, peak, durasi"] --> K1{"Kondisi 1: Smart Take Profit<br/>(a) Profit ge $40 = hard TP<br/>(b) Profit ge $25 + momentum lt -30<br/>(c) Peak > $30, sekarang lt 60% peak<br/>(d) Profit ge $20 + TP prob lt 25%"}
    K1 -->|"TRIGGER"| CLOSE1["TUTUP: Smart TP"]
    K1 -->|"tidak trigger"| K2{"Kondisi 2: Early Exit (Profit Kecil)<br/>Profit $5-$15 + ML reversal ge 65%<br/>+ momentum lt -50"}
    K2 -->|"TRIGGER"| CLOSE2["TUTUP: Amankan profit kecil"]
    K2 -->|"tidak trigger"| K3{"Kondisi 3: Early Cut<br/>(v4: Smart Hold DIHAPUS)<br/>Loss ge 30% max ($15)<br/>DAN momentum lt -30"}
    K3 -->|"TRIGGER"| CLOSE3["TUTUP CEPAT: Early cut"]
    K3 -->|"tidak trigger"| K4{"Kondisi 4: ML Trend Reversal<br/>ML confidence ge 65%<br/>BERLAWANAN ARAH posisi<br/>+ 3x warning berturut-turut"}
    K4 -->|"TRIGGER"| CLOSE4["TUTUP: AI deteksi pembalikan"]
    K4 -->|"tidak trigger"| K5{"Kondisi 5: Maximum Loss<br/>Loss ge 50% dari max_loss_per_trade<br/>($25 dari $50)"}
    K5 -->|"TRIGGER"| CLOSE5["TUTUP: Kerugian terlalu besar"]
    K5 -->|"tidak trigger"| K6{"Kondisi 6: Stall Detection<br/>Posisi 10+ bar<br/>tanpa profit signifikan"}
    K6 -->|"TRIGGER"| CLOSE6["TUTUP: Pasar tidak bergerak"]
    K6 -->|"tidak trigger"| K7{"Kondisi 7: Daily Limit<br/>Total daily loss<br/>mendekati limit"}
    K7 -->|"TRIGGER"| CLOSE7["TUTUP SEMUA: Proteksi modal"]
    K7 -->|"tidak trigger"| K8{"Kondisi 8: Weekend Close<br/>Dekat market close weekend<br/>+ ada posisi profit"}
    K8 -->|"TRIGGER"| CLOSE8["TUTUP: Hindari gap risk"]
    K8 -->|"tidak trigger"| K9{"Kondisi 9: Smart Time-Based<br/>(v5: Dont Cut Winners)<br/>(a) > 4 jam + no growth = TUTUP<br/>(b) > 4 jam + growing + ML = HOLD<br/>(c) > 6 jam + profit lt $10 = TUTUP<br/>(d) > 6 jam + profit > $10 = extend 8j<br/>(e) > 8 jam = TUTUP final"}
    K9 -->|"TRIGGER"| CLOSE9["TUTUP: Time-based"]
    K9 -->|"tidak trigger"| K10{"Kondisi 10: Trailing Stop Loss<br/>Profit ge 25 pips?<br/>SL ikuti harga jarak 10 pips"}
    K10 -->|"TRIGGER"| CLOSE10["TRAILING: Kunci profit"]
    K10 -->|"tidak trigger"| K11{"Kondisi 11: Breakeven Protection<br/>Profit ge 15 pips?<br/>SL ke entry + 2 poin buffer"}
    K11 -->|"TRIGGER"| CLOSE11["BREAKEVEN: Posisi aman"]
    K11 -->|"tidak trigger"| K12["Kondisi 12: Default HOLD<br/>Tidak ada kondisi terpenuhi<br/>Biarkan posisi berjalan"]

    style CLOSE1 fill:#F44336,color:#fff
    style CLOSE2 fill:#F44336,color:#fff
    style CLOSE3 fill:#F44336,color:#fff
    style CLOSE4 fill:#F44336,color:#fff
    style CLOSE5 fill:#F44336,color:#fff
    style CLOSE6 fill:#F44336,color:#fff
    style CLOSE7 fill:#F44336,color:#fff
    style CLOSE8 fill:#F44336,color:#fff
    style CLOSE9 fill:#F44336,color:#fff
    style CLOSE10 fill:#FF9800,color:#fff
    style CLOSE11 fill:#FF9800,color:#fff
    style K12 fill:#4CAF50,color:#fff
\`\`\`

### *Position Manager* (Tambahan per Posisi)

Selain 12 kondisi di atas, *Position Manager* juga menjalankan:

\`\`\`mermaid
flowchart TD
    PM["Position Manager (per posisi)"] --> MCH{"Market Close Handler<br/>(Prioritas Tertinggi)<br/>Dekat close harian/weekend?"}
    MCH -->|"Profit ge $10 + dekat close"| CLOSE_MC["TUTUP (amankan)"]
    MCH -->|"Loss + weekend + SL > 50%"| CLOSE_GAP["TUTUP (gap risk)"]
    MCH -->|"Loss kecil + weekend"| HOLD_MC["HOLD (bisa recovery)"]
    MCH -->|"Tidak dekat close"| BE{"Breakeven Protection<br/>Profit ge 15 pips?"}
    BE -->|"Ya"| BE_ACT["Pindah SL ke entry + 2 buffer<br/>(posisi tidak bisa rugi lagi)"]
    BE -->|"Tidak"| TS{"Trailing Stop<br/>Profit ge 25 pips?"}
    TS -->|"Ya"| TS_ACT["SL mengikuti harga, jarak 10 pips<br/>(kunci profit sambil biarkan berjalan)"]
    TS -->|"Tidak"| CONT["Lanjut monitoring"]
\`\`\`

---

## 7. Sistem Proteksi Risiko 4 Lapis

\`\`\`mermaid
flowchart TD
    subgraph L1["LAPIS 1: BROKER STOP LOSS (Otomatis oleh MT5)"]
        L1D["SL = Entry +/- (1.5 x ATR), minimum 10 pips<br/>Dikirim bersama order ke broker<br/>Aktif 24/7, bahkan jika bot mati<br/>Max loss: ~$50-80 per trade"]
    end
    subgraph L2["LAPIS 2: SOFTWARE SMART EXIT (Bot evaluasi setiap ~10 detik)"]
        L2D["12 kondisi exit (lihat bagian 6)<br/>Biasanya menutup SEBELUM broker SL kena<br/>Target close: loss le $25 (lebih ketat dari broker)<br/>Termasuk: momentum, ML reversal, stall, time limit"]
    end
    subgraph L3["LAPIS 3: EMERGENCY STOP LOSS (Backup jika software gagal)"]
        L3D["Max loss per trade: 2% modal ($100 untuk $5K)<br/>Diset sebagai broker SL terpisah<br/>Aktif jika software error/hang"]
    end
    subgraph L4["LAPIS 4: CIRCUIT BREAKER (Hentikan semua trading)"]
        L4D["Trigger 1: Daily loss ge 3% ($150) = Stop hari ini<br/>Trigger 2: Total loss ge 10% ($500) = Stop total<br/>Trigger 3: Flash crash > 2.5% / 1 menit = CLOSE ALL<br/>Reset: Otomatis hari baru (daily), manual (total)"]
    end

    L1 --> L2 --> L3 --> L4

    style L1 fill:#4CAF50,color:#fff
    style L2 fill:#2196F3,color:#fff
    style L3 fill:#FF9800,color:#fff
    style L4 fill:#F44336,color:#fff
\`\`\`

### 4 Mode Trading (Smart Risk Manager)

\`\`\`mermaid
flowchart TD
    NORMAL["MODE: NORMAL<br/>Kondisi: Semua aman, tidak ada masalah<br/>Lot: 0.01 - 0.02 (berdasarkan confidence)<br/>Max posisi: 2-3"]
    NORMAL -->|"3x loss berturut-turut"| RECOVERY["MODE: RECOVERY<br/>Kondisi: Setelah kerugian beruntun<br/>Lot: 0.01 (minimum saja)<br/>Max posisi: 1"]
    RECOVERY -->|"mendekati 80% daily limit"| PROTECTED["MODE: PROTECTED<br/>Kondisi: Hampir kena daily limit<br/>Lot: 0.01 (minimum saja)<br/>Max posisi: 1"]
    PROTECTED -->|"daily/total limit tercapai"| STOPPED["MODE: STOPPED<br/>Kondisi: Batas kerugian tercapai<br/>Lot: 0 (TIDAK BOLEH trading)<br/>Max posisi: 0 (tutup semua)<br/>Reset: Otomatis hari baru"]

    style NORMAL fill:#4CAF50,color:#fff
    style RECOVERY fill:#FF9800,color:#fff
    style PROTECTED fill:#FF5722,color:#fff
    style STOPPED fill:#F44336,color:#fff
\`\`\`

### Lot Sizing: Risk-Constrained Half-Kelly

\`\`\`
Langkah 1: Hitung Kelly Fraction
    f* = (win_rate × avg_rr - (1 - win_rate)) / avg_rr

    Contoh: win_rate=55%, avg_rr=2.0
    f* = (0.55 × 2.0 - 0.45) / 2.0 = 0.325 (32.5%)

Langkah 2: Cap Kelly (max 25%)
    f* = min(0.325, 0.25) = 0.25

Langkah 3: Half-Kelly (safety)
    f* = 0.25 × 0.5 = 0.125 (12.5%)

Langkah 4: Apply regime multiplier
    High volatility: × 0.5 = 0.0625
    Normal: × 1.0 = 0.125

Langkah 5: Cap di config limit
    config risk_per_trade = 1%
    actual_risk = min(0.125, 0.01) = 0.01 (1%)

Langkah 6: Hitung lot
    risk_amount = $5000 × 1% = $50
    SL distance = 50 pips → pip_value ~$1/pip/0.01lot
    lot = $50 / (50 × $1) = 0.01 lot

Langkah 7: ML Confidence boost
    ML ≥ 80% → lot × 2 = 0.02 lot (maximum)
    ML < 65% → lot = 0.01 (minimum)

Langkah 8: Session multiplier
    Golden Time: × 1.2
    Sydney: × 0.5

    Final lot: 0.01 - 0.02 (ultra-konservatif)
\`\`\`

---

## 8. AI/ML Engine

### Hidden Markov Model (HMM) — Otak Regime

\`\`\`mermaid
flowchart TD
    subgraph HMM["HIDDEN MARKOV MODEL"]
        direction TB
        INFO["Library: hmmlearn.GaussianHMM<br/>Input: log_returns + rolling_volatility (2 fitur)<br/>States: 3 (Low, Medium, High Volatility)<br/>Lookback: 500 bar untuk training<br/>Retrain: setiap 20 bar (auto-update)"]
        TRANS["Transition Matrix (contoh):<br/>Fr Low: To Low 0.85, To Med 0.12, To High 0.03<br/>Fr Med: To Low 0.10, To Med 0.80, To High 0.10<br/>Fr High: To Low 0.05, To Med 0.15, To High 0.80"]
        EMISI["Distribusi Emisi (per state):<br/>Low: mu_return ~ 0, sigma = kecil<br/>Med: mu_return ~ 0, sigma = sedang<br/>High: mu_return ~ 0, sigma = besar"]
        OUTPUT["Output:<br/>regime: 0/1/2 (low/medium/high)<br/>confidence: 0.0 - 1.0<br/>lot_multiplier: 1.0 / 0.5 / 0.0<br/>recommendation: TRADE / REDUCE / SLEEP"]
        INFO --> TRANS --> EMISI --> OUTPUT
    end
\`\`\`

### XGBoost — Otak Prediksi

\`\`\`mermaid
flowchart TD
    subgraph XGB["XGBOOST BINARY CLASSIFIER"]
        direction TB
        INFO2["Library: xgboost<br/>Objective: binary:logistic<br/>Target: UP (1) / DOWN (0) pada bar berikutnya"]
        CONFIG2["Anti-Overfitting Config:<br/>max_depth: 3 (shallow trees)<br/>learning_rate: 0.05 (slow learning)<br/>min_child_weight: 10 (min samples/leaf)<br/>subsample: 0.7 (70% data/tree)<br/>colsample_bytree: 0.6 (60% features/tree)<br/>reg_alpha: 1.0 (L1), reg_lambda: 5.0 (L2)<br/>gamma: 1.0 (min loss reduction)<br/>num_boost_round: 50 (few rounds)"]
        FITUR2["24 Fitur Input (Top 10):<br/>1. RSI(14), 2. MACD_histogram<br/>3. ATR(14), 4. bb_width<br/>5. returns_1, 6. price_position<br/>7. volatility_20, 8. returns_5<br/>9. ema_cross, 10. regime"]
        OUT2["Output:<br/>signal: BUY / SELL / HOLD<br/>probability: 0.0-1.0 (prob of UP)<br/>confidence: 0.0-1.0 (prob of winning side)"]
        VAL2["Validation:<br/>Train/Test: 70%/30% (50-bar gap, anti leakage)<br/>Walk-forward: 500 train / 50 test / 50 step<br/>Target AUC: > 0.65<br/>Rollback AUC: lt 0.60 (v4: dinaikkan dari 0.52)<br/>Overfitting ratio: train_AUC/test_AUC lt 1.15"]
        INFO2 --> CONFIG2 --> FITUR2 --> OUT2 --> VAL2
    end
\`\`\`

### Kombinasi Sinyal (SMC + ML)

\`\`\`mermaid
flowchart TD
    SMC_SIG["SMC Signal: BUY at 2645, SL 2635, TP 2665, conf 75%"]
    ML_SIG["ML Signal: BUY, confidence 72%"]

    SMC_SIG & ML_SIG --> COMBINE{"KOMBINASI SMC + ML"}

    COMBINE --> C1["CASE 1: SMC BUY + ML BUY (ge 50%)<br/>Combined confidence = (75%+72%)/2 = 73.5%<br/>= ENTRY (jika pass 14 filter)"]
    COMBINE --> C2["CASE 2: SMC BUY + ML SELL (ge 65%)<br/>ML strongly disagrees = BLOCK (filter #5)<br/>= TIDAK entry"]
    COMBINE --> C3["CASE 3: SMC BUY + ML uncertain (lt 50%)<br/>ML tidak yakin = BLOCK (filter #4)<br/>= TIDAK entry"]
    COMBINE --> C4["CASE 4: Tidak ada SMC signal<br/>Tidak ada entry point = SKIP<br/>SMC adalah sinyal PRIMER (wajib ada)"]

    C1 & C2 & C3 & C4 --> PRINSIP["PRINSIP:<br/>SMC = sinyal UTAMA (entry/SL/TP)<br/>ML = KONFIRMASI (bisa blokir, tidak bisa inisiasi)<br/>HMM = PENYESUAI (mengatur agresivitas)"]

    style C1 fill:#4CAF50,color:#fff
    style C2 fill:#F44336,color:#fff
    style C3 fill:#F44336,color:#fff
    style C4 fill:#9E9E9E,color:#fff
\`\`\`

---

## 9. Smart Money Concepts (SMC)

### 6 Konsep yang Dianalisis

\`\`\`mermaid
flowchart TD
    subgraph SP["1. SWING POINTS (Fractal High/Low)"]
        SP_D["Swing High: titik tertinggi dalam window 11 bar<br/>5 bar kiri lebih rendah, 5 bar kanan lebih rendah<br/><br/>Swing Low: titik terendah dalam window 11 bar<br/>5 bar kiri lebih tinggi, 5 bar kanan lebih tinggi"]
    end
    subgraph FVG2["2. FAIR VALUE GAP (FVG) - Ketidakseimbangan Harga"]
        FVG_D["Bullish FVG (gap up): Bar i-2 high lt Bar i low (ada gap)<br/>Bearish FVG (gap down): Bar i-2 low > Bar i high<br/>Harga cenderung kembali mengisi FVG = entry zone"]
    end
    subgraph OB2["3. ORDER BLOCK (OB) - Zona Institusi"]
        OB_D["Bullish OB: candle bearish terakhir sebelum rally<br/>(zona dimana institusi menempatkan buy order besar)<br/>Lookback: 10 bar untuk deteksi<br/>Mitigated: true jika harga sudah revisit"]
    end
    subgraph BOS2["4. BREAK OF STRUCTURE (BOS) - Kelanjutan Trend"]
        BOS_D["Uptrend BOS: harga break di ATAS swing high sebelumnya<br/>= Trend bullish berlanjut<br/>Downtrend BOS: harga break di BAWAH swing low<br/>= Trend bearish berlanjut"]
    end
    subgraph CHOCH2["5. CHANGE OF CHARACTER (CHoCH) - Perubahan Trend"]
        CHOCH_D["Uptrend ke Downtrend:<br/>Harga break di BAWAH swing low terakhir<br/>= Trend berubah dari bullish ke bearish<br/>(sinyal reversal)"]
    end
    subgraph LIQ2["6. LIQUIDITY ZONES - Target Likuiditas"]
        LIQ_D["BSL (Buy Side Liquidity): di atas swing highs<br/>(stop loss para seller berkumpul)<br/>SSL (Sell Side Liquidity): di bawah swing lows<br/>(stop loss para buyer berkumpul)<br/>Institusi sering hunt liquidity zone ini"]
    end
\`\`\`

### Signal Generation

\`\`\`mermaid
flowchart TD
    SB["Structure Break<br/>(BOS atau CHoCH)"] --> VALID["VALID SIGNAL"]
    ZN["Zone<br/>(FVG atau Order Block)"] --> VALID

    VALID --> BUY_SIG["BUY Signal:<br/>BOS bullish ATAU CHoCH bearish-to-bullish<br/>+ Bullish FVG ATAU Bullish OB di bawah harga<br/>Entry: harga saat ini<br/>SL: di bawah zone, minimum 1.5 x ATR<br/>TP: 2:1 R:R, maximum 4 x ATR<br/>Confidence: 40-85% (v5: calibrated weighted scoring)"]
    VALID --> SELL_SIG["SELL Signal:<br/>BOS bearish ATAU CHoCH bullish-to-bearish<br/>+ Bearish FVG ATAU Bearish OB di atas harga<br/>Entry: harga saat ini<br/>SL: di atas zone, minimum 1.5 x ATR<br/>TP: 2:1 R:R, maximum 4 x ATR<br/>Confidence: 40-85% (v5: calibrated weighted scoring)"]

    style BUY_SIG fill:#4CAF50,color:#fff
    style SELL_SIG fill:#F44336,color:#fff
\`\`\`

---

## 10. Position Lifecycle

### Dari Lahir Sampai Mati (Siklus Hidup Posisi)

\`\`\`mermaid
flowchart TD
    T1["TAHAP 1: SINYAL TERDETEKSI<br/>SMC menemukan setup + ML konfirmasi + 14 filter PASS<br/>Keputusan: BUKA POSISI"]
    T1 --> T2["TAHAP 2: LOT SIZE CALCULATION<br/>Risk Engine (Kelly): Balance $5000 x 1% = $50 max loss<br/>SL distance 50 pips = lot 0.01<br/>ML ge 80% = 0.02 lot, ML lt 65% = 0.01 lot<br/>Session: Golden x1.2, Sydney x0.5<br/>Regime: Normal x1.0, High Vol x0.5, Crisis x0.0"]
    T2 --> T3["TAHAP 3: ORDER VALIDATION<br/>SL di sisi benar (BUY: SL lt entry)<br/>TP di sisi benar (BUY: TP > entry)<br/>Lot dalam range (0.01 - 0.05)<br/>Entry dekat harga saat ini (lt 0.1%)<br/>Risk% le 1.5x config limit<br/>Circuit breaker TIDAK aktif"]
    T3 --> T4["TAHAP 4: ORDER EXECUTION<br/>MT5 Connector mengirim order:<br/>Symbol: XAUUSD, Type: BUY/SELL<br/>Lot: 0.01-0.02, SL: ATR-based<br/>TP: 2:1 R:R, Deviation: 20 points<br/>Retry: max 3 attempts"]
    T4 --> T4B["TAHAP 4b: POST-EXECUTION VALIDATION (v5)<br/>Slippage Validation: max 0.15% (~$4 XAUUSD)<br/>Log WARNING jika melebihi batas<br/>Gunakan harga AKTUAL untuk tracking<br/>Partial Fill: cek volume, update lot aktual"]
    T4B --> T5["TAHAP 5: POSITION REGISTERED<br/>Smart Risk Manager: catat entry AKTUAL, peak=0<br/>Trade Logger: insert PostgreSQL (30+ field) + CSV<br/>Telegram: notifikasi trade open<br/>(entry, SL, TP, R:R, confidence, regime)"]
    T5 --> T6["TAHAP 6: ACTIVE MONITORING<br/>(setiap ~10 detik + candle baru)<br/>Update profit/loss, peak profit, momentum<br/>Hitung TP probability<br/>Cek 12 kondisi exit<br/>Cek Position Manager (trailing, breakeven)<br/>Cek Market Close Handler"]
    T6 -->|"trigger close"| T7["TAHAP 7: POSITION CLOSED<br/>MT5: Close via market order<br/>Risk Manager: record P/L, update counters,<br/>check mode transition<br/>Logger: update exit price, profit, duration, reason<br/>Telegram: notifikasi trade close"]
    T6 -->|"HOLD"| T6

    style T1 fill:#1565C0,color:#fff
    style T7 fill:#C62828,color:#fff
\`\`\`

---

## 11. Auto-Retraining & Model Management

### Lifecycle Model AI

\`\`\`mermaid
flowchart TD
    INIT["INITIAL TRAINING (train_models.py)<br/>Dijalankan 1x saat setup"]
    INIT --> I1["1. Fetch 10,000 bar M15 dari MT5 (~104 hari)"]
    I1 --> I2["2. Feature Engineering = 40+ fitur"]
    I2 --> I3["3. SMC Analysis = struktur pasar"]
    I3 --> I4["4. Create target UP/DOWN (lookahead=1)<br/>4b. Split 70/30, 50-bar gap (anti leakage)"]
    I4 --> I5["5. Train HMM (3 regime, lookback=500)"]
    I5 --> I6["6. Train XGBoost (50 rounds, early_stop=5)"]
    I6 --> I7["7. Walk-forward validation (500/50/50)"]
    I7 --> I8["8. Save: hmm_regime.pkl + xgboost_model.pkl"]

    I8 --> DAILY["DAILY AUTO-RETRAINING (Auto Trainer)<br/>Otomatis setiap hari 05:00 WIB"]

    DAILY --> SCHED["Schedule:<br/>Harian (05:00 WIB): 8,000 bar, 50 rounds<br/>Weekend (Sabtu): 15,000 bar, 80 rounds (deep)<br/>Emergency: jika AUC lt 0.65"]

    SCHED --> D1["1. Backup model saat ini"]
    D1 --> D2["2. Fetch data baru dari MT5"]
    D2 --> D3["3. Feature Engineering + SMC"]
    D3 --> D4["4. Train HMM baru + XGBoost baru"]
    D4 --> D5{"5. Validasi: test AUC ge 0.60?"}
    D5 -->|"Ya"| SAVE["Save model baru, reload di memory"]
    D5 -->|"Tidak"| ROLLBACK["ROLLBACK ke model sebelumnya"]
    SAVE --> D6["6. Log hasil ke PostgreSQL"]
    ROLLBACK --> D6
    D6 --> D7["7. Kirim laporan via Telegram"]

    D7 --> SAFETY["Safety:<br/>Max 5 backup (rotasi)<br/>Min 20 jam antar retrain (cooldown)<br/>Auto-rollback jika AUC lt 0.60<br/>Model lama selalu tersedia"]
\`\`\`

### Perbandingan Initial vs Auto Training

| Aspek | train_models.py | Auto Trainer |
|-------|-----------------|-------------|
| Kapan | Manual, 1x setup | Otomatis, harian |
| Data | 10,000 bar | 8K (harian) / 15K (weekend) |
| Boost rounds | 50 | 50 (harian) / 80 (weekend) |
| Walk-forward | Ya | Tidak |
| Backup | Tidak | Ya (5 terakhir) |
| Rollback | Tidak | Ya (AUC < 0.60) |
| Database | Tidak | Ya (PostgreSQL) |
| Tujuan | Setup awal | Maintenance rutin |

---

## 12. Infrastruktur & Database

### PostgreSQL Schema

\`\`\`mermaid
flowchart TD
    DB["trading_db"] --> T["trades<br/>Semua trade: open, close, profit, SMC, ML, features"]
    DB --> TR["training_runs<br/>Log setiap training: AUC, akurasi, durasi, rollback"]
    DB --> SG["signals<br/>Setiap sinyal yang dihasilkan: executed atau tidak"]
    DB --> MS["market_snapshots<br/>Snapshot periodik: harga, regime, volatilitas"]
    DB --> BS["bot_status<br/>Status bot: uptime, loop count, balance, risk mode"]
    DB --> DS["daily_summaries<br/>Ringkasan harian: win rate, profit factor, per sesi"]
\`\`\`

### Tabel \`trades\` (Detail)

\`\`\`sql
-- Identifikasi
ticket, symbol, direction (BUY/SELL)

-- Harga
entry_price, exit_price, stop_loss, take_profit

-- Hasil
lot_size, profit_usd, profit_pips
opened_at, closed_at, duration_seconds

-- Konteks Entry
entry_regime, entry_volatility, entry_session
smc_signal, smc_confidence, smc_reason
smc_fvg_detected, smc_ob_detected, smc_bos_detected, smc_choch_detected
ml_signal, ml_confidence
market_quality, market_score, dynamic_threshold

-- Konteks Exit
exit_reason, exit_regime, exit_ml_signal

-- Keuangan
balance_before, balance_after, equity_at_entry

-- Data Lengkap
features_entry (JSON), features_exit (JSON)
bot_version, trade_mode
\`\`\`

### Connection Architecture

\`\`\`mermaid
flowchart TD
    TL["TradeLogger"] -->|"TradeRepository, SignalRepository,<br/>MarketSnapshotRepository"| DBC["DatabaseConnection (Singleton)"]
    AT2["AutoTrainer"] -->|"TrainingRepository"| DBC
    ML["main_live.py"] -->|"BotStatusRepository,<br/>DailySummaryRepository"| DBC
    DASH["Dashboard"] -->|"Semua repository (READ)"| DBC

    DBC --> POOL["ThreadedConnectionPool (1-10 koneksi)"]
    POOL --> PG["PostgreSQL Server"]
\`\`\`

### Graceful Degradation

\`\`\`mermaid
flowchart TD
    CHECK{"PostgreSQL tersedia?"}
    CHECK -->|"Ya"| DUAL["Gunakan DB + CSV backup (dual write)"]
    CHECK -->|"Tidak"| CSV["CSV saja (bot tetap berjalan 100%)"]
    DUAL --> NOTE["Bot TIDAK PERNAH crash karena database.<br/>Semua operasi DB dibungkus try-except."]
    CSV --> NOTE
\`\`\`

---

## 13. Konfigurasi & Parameter Kritis

### Configuration System

\`\`\`mermaid
flowchart TD
    ENV[".env file"] --> TC["TradingConfig.from_env()"]

    TC --> RC["RiskConfig<br/>risk_per_trade: 1.0% (SMALL) / 0.5% (MEDIUM)<br/>max_daily_loss: 3.0% (SMALL) / 2.0% (MEDIUM)<br/>max_total_loss: 10.0%<br/>max_positions: 3 (SMALL) / 5 (MEDIUM)<br/>min_lot: 0.01<br/>max_lot: 0.05 (SMALL) / 2.0 (MEDIUM)<br/>max_leverage: 1:100 (SMALL) / 1:30 (MEDIUM)"]

    TC --> SC["SMCConfig<br/>swing_length: 5<br/>fvg_min_gap_pips: 2.0<br/>ob_lookback: 10<br/>bos_close_break: true"]

    TC --> MLC2["MLConfig<br/>confidence_threshold: 0.65<br/>entry_confidence: 0.70<br/>high_confidence: 0.75<br/>very_high_confidence: 0.80<br/>retrain_frequency_days: 7"]

    TC --> THC["ThresholdsConfig<br/>ml_min_confidence: 0.65<br/>ml_high_confidence: 0.75<br/>trade_cooldown_seconds: 300<br/>min_profit_to_secure: $15<br/>good_profit: $25, great_profit: $40<br/>flash_crash_threshold: 2.5%<br/>sydney_lot_multiplier: 0.5"]

    TC --> RGC["RegimeConfig<br/>n_regimes: 3<br/>lookback: 500<br/>retrain_frequency: 20"]
\`\`\`

### Capital Mode Auto-Detection

\`\`\`mermaid
flowchart TD
    BAL{"Balance?"}
    BAL -->|"le $10,000"| SMALL["SMALL MODE<br/>Risk: 1% per trade ($50 pada $5K)<br/>Daily limit: 3% ($150)<br/>Lot: 0.01-0.05<br/>Leverage: 1:100<br/>Timeframe: M15<br/>Max posisi: 3"]
    BAL -->|"> $10,000"| MEDIUM["MEDIUM MODE<br/>Risk: 0.5% per trade<br/>Daily limit: 2%<br/>Lot: 0.01-2.0<br/>Leverage: 1:30<br/>Timeframe: H1<br/>Max posisi: 5"]
\`\`\`

### Session Schedule (WIB = GMT+7)

\`\`\`mermaid
flowchart LR
    subgraph SESSION["Session Schedule (WIB = GMT+7)"]
        direction TB
        DZ["00:00-04:00 DEAD ZONE<br/>BLOCKED - Likuiditas rendah"]
        RO["04:00-06:00 ROLLOVER<br/>BLOCKED - Spread melebar"]
        SY["06:00-07:00 Sydney<br/>0.5x - Pasar baru buka"]
        TK["07:00-13:00 Tokyo+Sydney<br/>0.7x - Asia aktif"]
        TA["13:00-15:00 Tokyo akhir<br/>0.7x - Transisi"]
        LD["15:00-20:00 London<br/>1.0x - Volatilitas naik"]
        GT["20:00-23:59 GOLDEN TIME<br/>1.2x - London+NY overlap"]
        WR["Jumat ge 23:00 WEEKEND RISK<br/>BLOCKED - Gap risk"]
    end

    style DZ fill:#F44336,color:#fff
    style RO fill:#F44336,color:#fff
    style SY fill:#FF9800,color:#fff
    style TK fill:#FFC107,color:#000
    style TA fill:#FFC107,color:#000
    style LD fill:#2196F3,color:#fff
    style GT fill:#4CAF50,color:#fff
    style WR fill:#F44336,color:#fff
\`\`\`

Golden Time (20:00-23:59 WIB) = waktu paling optimal
- Spread ketat, likuiditas maksimal, volatilitas ideal
- Lot multiplier 1.2x (bonus)
- v4: Smart Hold dihapus -- tidak ada lagi hold losers menunggu sesi tertentu

---

## 14. Performa & Timing

### Main Loop Breakdown

Target: < 50ms per iterasi analisis

**FULL ANALYSIS (saat candle baru M15):**

| Komponen | Waktu | Keterangan |
|----------|-------|------------|
| MT5 data fetch | ~10ms | 200 bar M15 via API |
| Feature engineering | ~5ms | 40+ fitur, Polars |
| SMC analysis | ~5ms | 6 konsep, Polars native |
| HMM predict | ~2ms | 2 fitur -> 1 regime |
| XGBoost predict | ~3ms | 24 fitur -> 1 signal |
| Position monitoring | ~5ms | Per posisi terbuka |
| Entry logic | ~5ms | 14 filter check |
| Overhead | ~15ms | Logging, state update |
| **TOTAL** | **~50ms** | |

**POSITION CHECK ONLY (di antara candle, setiap ~10 detik):**

| Komponen | Waktu | Keterangan |
|----------|-------|------------|
| MT5 data fetch | ~5ms | 50 bar saja |
| Feature engineering | ~3ms | Minimal fitur |
| ML prediction | ~3ms | Untuk exit evaluation |
| Position evaluation | ~5ms | 12 kondisi exit |
| Overhead | ~5ms | Logging |
| **TOTAL** | **~21ms** | |

### Timer Periodik

| Event | Interval | Cara Trigger |
|-------|----------|--------------|
| Full analysis + entry | Candle baru M15 | Deteksi candle |
| Position monitoring | ~10 detik | Di antara candle |
| Performance logging | 4 candle (~1j) | candle_count % 4 |
| Auto-retrain check | 20 candle (~5j) | candle_count % 20 |
| Market update Telegram | 30 menit | Timer |
| Hourly analysis | 1 jam | Timer |
| Daily summary + reset | Ganti hari | Date check |

---

## 15. Error Handling & Fault Tolerance

### Prinsip: Bot TIDAK PERNAH Crash

\`\`\`mermaid
flowchart TD
    subgraph LV1["LEVEL 1: Per-Loop Error Handling"]
        LV1D["try: Fetch data, analyze, trade<br/>except ConnectionError: reconnect()<br/>except Exception: Log error, lanjut loop berikutnya<br/>Bot TIDAK crash dari error tunggal"]
    end
    subgraph LV2["LEVEL 2: MT5 Auto-Reconnect"]
        LV2D["MT5 putus?<br/>Attempt 1: reconnect (tunggu 2 detik)<br/>Attempt 2: reconnect (tunggu 4 detik)<br/>Attempt 3: reconnect (tunggu 8 detik)<br/>Cooldown 60 detik, Retry cycle (max 5/cooldown)<br/>Selama disconnected: monitoring PAUSE,<br/>entry DITUNDA, posisi dilindungi broker SL"]
    end
    subgraph LV3["LEVEL 3: Database Graceful Degradation"]
        LV3D["PostgreSQL down?<br/>Switch ke CSV-only mode<br/>Semua data tetap dicatat<br/>Trading tetap berjalan normal<br/>Retry DB connection periodik"]
    end
    subgraph LV4["LEVEL 4: Telegram Failure"]
        LV4D["Telegram API error?<br/>Log error secara silent<br/>Trading tetap jalan 100%<br/>Retry di notifikasi berikutnya"]
    end
    subgraph LV5["LEVEL 5: Model File Missing"]
        LV5D[".pkl file tidak ditemukan?<br/>Log warning<br/>Skip prediksi (ML/HMM)<br/>Trading bisa jalan tanpa ML (SMC only)<br/>Trigger: jalankan train_models.py"]
    end
    subgraph LV6["LEVEL 6: Flash Crash Protection"]
        LV6D["Harga bergerak > 2.5% dalam 1 menit?<br/>EMERGENCY: Close ALL positions<br/>Circuit breaker AKTIF<br/>Kirim alert KRITIS via Telegram<br/>Bot masuk mode STOPPED"]
    end

    LV1 --> LV2 --> LV3 --> LV4 --> LV5 --> LV6
\`\`\`

### Startup & Shutdown

\`\`\`mermaid
flowchart TD
    subgraph STARTUP["STARTUP SEQUENCE"]
        direction TB
        S1["1. Load konfigurasi dari .env"] --> S2["2. Connect ke MT5 (max 3 retry)"]
        S2 --> S3["3. Load model HMM dari models/hmm_regime.pkl"]
        S3 --> S4["4. Load model XGBoost dari models/xgboost_model.pkl"]
        S4 --> S5["5. Initialize SmartRiskManager (set balance, limits)"]
        S5 --> S6["6. Initialize SessionFilter (WIB timezone)"]
        S6 --> S7["7. Initialize TelegramNotifier"]
        S7 --> S8["8. Initialize TradeLogger (connect DB)"]
        S8 --> S9["9. Initialize AutoTrainer"]
        S9 --> S10["10. Send Telegram: BOT STARTED"]
        S10 --> S11["11. Mulai main loop"]
    end

    subgraph SHUTDOWN["SHUTDOWN SEQUENCE (SIGINT/SIGTERM)"]
        direction TB
        D1["1. Signal diterima"] --> D2["2. Hentikan loop utama"]
        D2 --> D3["3. Kirim Telegram: BOT STOPPED"]
        D3 --> D4["4. Disconnect MT5"]
        D4 --> D5["5. Close database connections"]
        D5 --> D6["6. Exit"]
    end
\`\`\`

---

## 16. Daftar File Source Code

\`\`\`mermaid
flowchart TD
    ROOT["Smart Automatic Trading BOT + AI/"]
    ROOT --> MAIN["main_live.py - Orchestrator utama"]
    ROOT --> TRAIN["train_models.py - Script training awal"]
    ROOT --> ENVF[".env - Environment variables"]

    ROOT --> SRC["src/"]
    SRC --> CFG["config.py - Konfigurasi terpusat (6 sub-config)"]
    SRC --> MT5F["mt5_connector.py - Bridge ke MetaTrader 5"]
    SRC --> FEF["feature_eng.py - Feature Engineering (40+ fitur)"]
    SRC --> RDF["regime_detector.py - HMM Regime Detection"]
    SRC --> MLF["ml_model.py - XGBoost Signal Predictor"]
    SRC --> SMCF["smc_polars.py - Smart Money Concepts (6 konsep)"]
    SRC --> SRMF["smart_risk_manager.py - 4-Mode Risk Manager"]
    SRC --> REF["risk_engine.py - Kelly Criterion + Circuit Breaker"]
    SRC --> SFF["session_filter.py - Session Time Filter (WIB)"]
    SRC --> DCF["dynamic_confidence.py - Dynamic Threshold Manager"]
    SRC --> NAF["news_agent.py - News Event Monitor"]
    SRC --> TNF["telegram_notifier.py - Telegram Push Notifications"]
    SRC --> ATF["auto_trainer.py - Daily Auto-Retraining"]
    SRC --> TLF["trade_logger.py - Dual Storage Logger (DB+CSV)"]
    SRC --> PMF["position_manager.py - Position Manager + Market Close"]
    SRC --> DBD["db/"]
    DBD --> DBINIT["__init__.py - DB exports"]
    DBD --> DBCONN["connection.py - PostgreSQL Singleton + Pool"]
    DBD --> DBREPO["repository.py - 6 Repository classes"]

    ROOT --> MODELS["models/"]
    MODELS --> XGBM["xgboost_model.pkl - Trained XGBoost model"]
    MODELS --> HMMM["hmm_regime.pkl - Trained HMM model"]
    MODELS --> BKUP["backup/ - Auto-backup (5 terakhir)"]

    ROOT --> DATA["data/"]
    DATA --> TDATA["training_data.parquet"]
    DATA --> TLOGS["trade_logs/ - CSV backup (per bulan)"]

    ROOT --> BTESTS["backtests/"]
    BTESTS --> BTSYNC["backtest_live_sync.py - 100% sync live"]

    ROOT --> LOGS["logs/"]
    LOGS --> LOGF["training_YYYY-MM-DD.log"]

    ROOT --> DOCS["docs/arsitektur-ai/"]
    DOCS --> DOC0["00-ARSITEKTUR-LENGKAP.md - Dokumen ini"]
    DOCS --> DOCR["README.md - Index komponen"]
    DOCS --> DOC1["01-23 (per komponen) - Detail per modul"]
\`\`\`

---

## Ringkasan Eksekutif

**Smart AI Trading Bot** adalah sistem trading otomatis yang menggabungkan:

1. **Smart Money Concepts (SMC)** sebagai sinyal UTAMA — mendeteksi zona institusi (FVG, Order Block, BOS, CHoCH) untuk menentukan entry, SL, dan TP yang presisi.

2. **XGBoost Machine Learning** sebagai KONFIRMASI — memprediksi arah harga dengan 24 fitur teknikal, memblokir trade jika tidak setuju dengan SMC.

3. **Hidden Markov Model (HMM)** sebagai PENYESUAI — mendeteksi kondisi pasar (tenang/volatile/krisis) untuk menyesuaikan agresivitas.

4. **4-Lapis Proteksi Risiko** — dari broker SL, software smart exit, emergency stop, hingga circuit breaker. Lot ultra-kecil (0.01-0.02) memastikan kerugian per trade maximum $50 (1%).

5. **Self-Improving** — model AI dilatih ulang otomatis setiap hari dengan auto-rollback jika model baru lebih buruk.

6. **Fault-Tolerant** — bot tidak pernah crash. MT5 putus? Auto-reconnect. Database mati? CSV fallback. Error? Log dan lanjut.

Semua ini dikoordinasikan oleh **Main Live Orchestrator** yang menjalankan loop **candle-based** — analisis penuh hanya saat candle M15 baru terbentuk (~50ms per iterasi), dengan pengecekan posisi setiap ~10 detik di antara candle (~21ms). Mengevaluasi 14 *filter entry* dan 12 kondisi *exit* secara *real-time*, dengan notifikasi Telegram untuk setiap kejadian penting.

\`\`\`
TARGET: Trading XAUUSD M15 yang KONSISTEN dan AMAN
        dengan kerugian terkontrol dan profit teroptimasi.
\`\`\`
`,
  },
  {
    slug: "architecture-index",
    title: "Indeks Arsitektur",
    category: "Mulai di Sini",
    icon: "List",
    description: "Daftar semua dokumen arsitektur dan status komponen terkini",
    content: `# Dokumentasi Arsitektur — XAUBot AI

> Panduan lengkap arsitektur dan komponen sistem *trading bot* otomatis XAUUSD.

---

## Daftar Dokumen

| # | Dokumen | Deskripsi |
|---|---------|-----------|
| 00 | **Arsitektur Lengkap** | Gambaran besar seluruh sistem — *data flow*, komponen, dan interaksi |
| 01 | **HMM *Regime Detector*** | Deteksi kondisi pasar menggunakan *Hidden Markov Model* 3 *state* |
| 02 | **XGBoost *Signal Predictor*** | Model *machine learning* untuk prediksi BUY/SELL/HOLD |
| 03 | **SMC *Analyzer*** | Analisis *Smart Money Concepts* — *Order Block*, FVG, BOS, CHoCH |
| 04 | ***Feature Engineering*** | 37 fitur teknikal — RSI, ATR, MACD, *Bollinger*, dll |
| 05 | **Manajemen Risiko** | Sistem manajemen risiko dinamis dengan mode kapital |
| 06 | **Filter Sesi** | Filter sesi perdagangan — Sydney, London, New York (WIB) |
| 07 | ***Stop Loss*** | Proteksi SL berbasis ATR dan *broker-level* |
| 08 | ***Take Profit*** | Target TP multi-level dengan ATR dan struktur pasar |
| 09 | ***Entry Trade*** | 14 filter *entry* dan logika eksekusi perdagangan |
| 10 | ***Exit Trade*** | 12 kondisi *exit* termasuk *trailing* SL, batas waktu, perubahan *regime* |
| 11 | ***News Agent*** | Filter berita ekonomi dan penilaian dampak *(saat ini nonaktif)* |
| 12 | **Notifikasi Telegram** | Notifikasi *trade* dan ringkasan harian via Telegram |
| 13 | ***Auto Trainer*** | *Pipeline retraining* otomatis saat kondisi pasar berubah |
| 14 | ***Backtest*** | *Framework backtesting* yang disinkronkan dengan logika *live* |
| 15 | ***Dynamic Confidence*** | Ambang batas *confidence* adaptif berdasarkan kondisi pasar |
| 16 | **Konektor MT5** | Lapisan koneksi *MetaTrader 5* dan eksekusi *order* |
| 17 | **Konfigurasi** | Konfigurasi *trading*, mode kapital, dan pengaturan *environment* |
| 18 | ***Trade Logger*** | Pencatatan *trade* ke *database* PostgreSQL |
| 19 | ***Position Manager*** | Pelacakan dan manajemen posisi terbuka |
| 20 | ***Risk Engine*** | Perhitungan risiko, *Kelly criterion*, dan *position sizing* |
| 21 | ***Database*** | Skema PostgreSQL dan penyimpanan data perdagangan |
| 22 | ***Train Models*** | *Pipeline* pelatihan model dan optimasi *hyperparameter* |
| 23 | **Orkestrator Utama** | *Async main loop* — inti dari *trading bot* |

---

## Diagram Arsitektur

\`\`\`mermaid
graph TD
    A["MetaTrader 5<br/>XAUUSD M15"] -->|OHLCV| B["Data Pipeline<br/>Polars Engine"]
    B --> C["SMC Analyzer<br/>OB / FVG / BOS"]
    B --> D["Feature Engineering<br/>37 Fitur"]
    B --> E["HMM Regime<br/>Detector"]
    C --> F["XGBoost Model<br/>Signal + Confidence"]
    D --> F
    E --> F
    F --> G["14 Entry Filters"]
    F --> H["Risk Engine<br/>ATR + Kelly"]
    G --> I["Eksekusi Trade<br/>MT5"]
    H --> I
    I --> J["Position Manager<br/>12 Exit Conditions"]
    J --> K["Telegram + PostgreSQL<br/>Logging"]
\`\`\`

## Status Komponen (Terkini)

| Komponen | Status | Catatan |
|----------|--------|---------|
| SMC *Analyzer* | **Aktif** | *Order Block*, FVG, BOS, CHoCH |
| XGBoost *Model* | **Aktif** | 37 fitur, *confidence calibrated* |
| HMM *Regime* | **Aktif** | 3 *state* — *low/medium/high volatility* |
| *Session Filter* | **Aktif** | WIB, Tokyo-London *overlap* diblokir |
| *Smart Risk Manager* | **Aktif** | Mode NORMAL/PROTECTED/RECOVERY/COOLDOWN |
| *Dynamic Confidence* | **Aktif** | *Threshold* adaptif per kondisi pasar |
| *Auto Trainer* | **Aktif** | *Retrain* otomatis tiap 7 hari |
| *News Agent* | **Nonaktif** | Dikomentari di \`main_live.py\` baris 64 |
| Telegram | **Aktif** | Notifikasi *entry/exit* + ringkasan harian |
| *Trade Logger* | **Aktif** | *Logging* ke PostgreSQL |
`,
  },
  {
    slug: "hmm-regime",
    title: "HMM Regime Detector",
    category: "AI & Analisis",
    icon: "Brain",
    description: "Deteksi kondisi pasar menggunakan Hidden Markov Model 3 state",
    content: `# HMM (*Hidden Markov Model*) — *Regime Detector*

> **File:** \`src/regime_detector.py\`
> **Model:** \`models/hmm_regime.pkl\`
> **Library:** \`hmmlearn.GaussianHMM\`

---

## Apa Itu HMM?

*Hidden Markov Model* (HMM) adalah model statistik yang mengidentifikasi **kondisi tersembunyi** (*hidden states*) dari data yang dapat diamati. Dalam konteks *trading*, HMM mendeteksi **3 kondisi pasar** (*regime*) yang tidak terlihat langsung dari harga:

\`\`\`mermaid
graph LR
    A["Data Pasar<br/>Return, Volatilitas, Volume"] --> B["HMM<br/>GaussianHMM 3-state"]
    B --> C["Low Volatility<br/>🟢 TRADE"]
    B --> D["Medium Volatility<br/>🟡 REDUCE"]
    B --> E["High Volatility<br/>🔴 SLEEP"]
\`\`\`

---

## 3 *State* Pasar

| *State* | Label | Rekomendasi | Efek pada *Trading* |
|---------|-------|-------------|---------------------|
| **0** | *Low Volatility* | **TRADE** | *Lot* normal, semua filter aktif |
| **1** | *Medium Volatility* | **REDUCE** | *Lot* dikurangi, *entry* lebih ketat |
| **2** | *High Volatility* / Krisis | **SLEEP** | **Tidak boleh *trading*** — terlalu berisiko |

---

## Cara Kerja

### *Input Features* (3 fitur)

\`\`\`python
features = [
    "returns",      # Perubahan harga (%)
    "volatility",   # Volatilitas rolling (standar deviasi)
    "volume_change" # Perubahan volume (%)
]
\`\`\`

### Proses *Training*

\`\`\`python
class MarketRegimeDetector:
    def __init__(self,
        n_regimes=3,           # 3 state
        lookback_periods=500,  # 500 bar untuk training
        retrain_frequency=20,  # Retrain setiap 20 bar baru
        covariance_type="full",
        random_state=42,
    ):
        self.hmm = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
\`\`\`

### Proses Deteksi

\`\`\`python
# 1. Siapkan data 500 bar terakhir
X = df[["returns", "volatility", "volume_change"]].to_numpy()

# 2. Fit model (atau load dari .pkl)
self.hmm.fit(X)

# 3. Prediksi state saat ini
state = self.hmm.predict(X)[-1]  # State terakhir

# 4. Hitung probabilitas tiap state
probs = self.hmm.predict_proba(X)[-1]
# Contoh: [0.85, 0.10, 0.05] = 85% low vol
\`\`\`

---

## Output: \`RegimeState\`

\`\`\`python
@dataclass
class RegimeState:
    regime: MarketRegime       # LOW/MEDIUM/HIGH_VOLATILITY atau CRISIS
    confidence: float          # Probabilitas state terpilih (0-1)
    probabilities: Dict        # Probabilitas semua state
    volatility: float          # Level volatilitas saat ini
    recommendation: str        # "TRADE", "REDUCE", atau "SLEEP"
\`\`\`

---

## Integrasi dengan Sistem

\`\`\`mermaid
graph TD
    A["HMM Regime Detector"] --> B{"Regime?"}
    B -->|LOW VOL| C["✅ TRADE<br/>Lot normal, semua filter aktif"]
    B -->|MEDIUM VOL| D["⚠️ REDUCE<br/>Lot dikurangi, entry lebih ketat"]
    B -->|HIGH VOL| E["🛑 SLEEP<br/>Blokir semua entry baru"]
    C --> F["Entry Filter #2"]
    D --> F
    E -->|Blokir| G["Skip — tidak boleh trading"]
\`\`\`

**Penggunaan dalam *main_live.py*:**
- *Regime* **SLEEP** → blokir semua *entry* baru (Filter #2)
- *Regime* memengaruhi *lot sizing* — \`SmartRiskManager\` mengurangi *lot* pada *medium volatility*
- *Regime* dicatat di setiap *trade log* untuk analisis historis

---

## Penyimpanan Model

- **Format:** \`.pkl\` (*pickle*)
- **Lokasi:** \`models/hmm_regime.pkl\`
- **Ukuran:** ~50-100 KB
- ***Retrain*:** Otomatis setiap 20 bar baru ATAU melalui \`AutoTrainer\` setiap 7 hari
- ***Auto-retrain* dipicu juga saat:** Akurasi deteksi turun atau distribusi *return* berubah signifikan

---

## Konfigurasi

Dari \`src/config.py\` → \`RegimeConfig\`:

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| \`n_regimes\` | **3** | Jumlah *state* HMM |
| \`lookback_periods\` | **500** | Bar untuk *training* HMM |
| \`retrain_frequency\` | **20** | *Retrain* setiap 20 bar baru |
`,
  },
  {
    slug: "xgboost",
    title: "XGBoost Signal Predictor",
    category: "AI & Analisis",
    icon: "Cpu",
    description: "Model machine learning untuk prediksi sinyal BUY/SELL/HOLD",
    content: `# XGBoost — *Signal Predictor*

> **File:** \`src/ml_model.py\`
> **Model:** \`models/xgboost_model.pkl\`
> **Library:** \`xgboost\`

---

## Apa Itu XGBoost *Signal Predictor*?

XGBoost (*Extreme Gradient Boosting*) adalah model *machine learning* yang memprediksi **sinyal *trading*** — BUY, SELL, atau HOLD — berdasarkan **37 fitur teknikal**. Model ini berfungsi sebagai **konfirmasi kedua** setelah analisis SMC.

---

## Alur Prediksi

\`\`\`mermaid
graph LR
    A["37 Fitur Teknikal"] --> B["XGBoost Model"]
    B --> C["Probabilitas per Kelas"]
    C --> D["BUY: 72%"]
    C --> E["SELL: 18%"]
    C --> F["HOLD: 10%"]
    D --> G["Signal: BUY<br/>Confidence: 72%"]
\`\`\`

---

## 37 *Features* (Fitur *Input*)

Model menerima 37 fitur yang dihitung oleh \`FeatureEngineer\`:

| Grup | Fitur | Jumlah |
|------|-------|--------|
| **Momentum** | RSI 14, RSI 7, MACD, *MACD Signal*, *MACD Histogram* | 5 |
| **Volatilitas** | ATR 14, *Bollinger Upper/Lower/Width*, *Keltner Channel* | 5 |
| **Trend** | EMA 9/20/50, SMA 20/50, *EMA Crossover* | 6 |
| **Volume** | *Volume Ratio*, *Volume MA*, *OBV*, *Volume Change* | 4 |
| ***Price Action*** | *Body Size*, *Shadow Ratio*, *Candle Pattern*, Jarak dari EMA | 5 |
| **Struktur** | *Higher High/Lower Low*, *Swing Detection*, BOS/CHoCH | 4 |
| ***Derived*** | *Returns* (1/3/5 bar), *Volatility Ratio*, *Momentum Score* | 5 |
| ***Lagged*** | Fitur-fitur di atas dengan *lag* 1-3 bar | 3 |

---

## *Output*: Prediksi

\`\`\`python
@dataclass
class MLPrediction:
    signal: str          # "BUY", "SELL", atau "HOLD"
    confidence: float    # 0.0 - 1.0
    probabilities: Dict  # {"BUY": 0.72, "SELL": 0.18, "HOLD": 0.10}
\`\`\`

---

## Peran dalam Sistem

XGBoost **bukan pembuat keputusan utama** — fungsinya adalah **konfirmasi dan filter**:

\`\`\`mermaid
graph TD
    SMC["SMC Analyzer<br/>Sinyal Utama"] --> COMBINE["Kombinasi Sinyal"]
    ML["XGBoost<br/>Konfirmasi"] --> COMBINE
    COMBINE --> CHECK{"ML setuju?"}
    CHECK -->|"Ya (≥50%)"| PASS["✅ Lanjut ke filter berikutnya"]
    CHECK -->|"Sangat tidak setuju (>65%)"| VETO["🛑 VETO — blokir entry"]
    CHECK -->|"Ragu (<50%)"| SKIP["⚠️ Skip — confidence terlalu rendah"]
\`\`\`

### Aturan Kombinasi:

| Kondisi | Aksi |
|---------|------|
| SMC = BUY, ML = BUY (≥50%) | ✅ **Konfirmasi** — lanjut |
| SMC = BUY, ML = HOLD | ⚠️ *Skip* — ML tidak yakin |
| SMC = BUY, ML = SELL (≥65%) | 🛑 **VETO** — ML *strongly disagree* |
| SMC = BUY, ML = SELL (<65%) | ✅ *Pass* — ML kurang yakin untuk veto |

---

## *Confidence Threshold*

| Level | Nilai | Penggunaan |
|-------|-------|------------|
| **Minimum** | **0.50** | Batas paling rendah untuk diterima |
| ***Entry*** | **0.65-0.70** | *Default* dari \`DynamicConfidence\` |
| ***High*** | **0.75** | Sinyal kuat — *lot multiplier* aktif |
| ***Very High*** | **0.80** | Sangat yakin — batas atas |

*Threshold* disesuaikan secara dinamis oleh \`DynamicConfidenceManager\` berdasarkan kondisi pasar.

---

## *Training*

\`\`\`python
# train_models.py
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,    # L1 regularization
    reg_lambda=1.0,   # L2 regularization
)

# Training data: 1000+ bar XAUUSD M15
# Label: Pergerakan harga setelah N bar
# Validasi: Walk-forward dengan 80/20 split
\`\`\`

### ***Auto-Retrain***

Model otomatis di-*retrain* oleh \`AutoTrainer\` setiap **7 hari** atau saat:
- Akurasi prediksi turun signifikan
- Distribusi pasar berubah
- *Confidence calibration* menyimpang

---

## Penyimpanan Model

| Properti | Nilai |
|----------|-------|
| **Format** | \`.pkl\` (*pickle*) via \`xgboost\` |
| **Lokasi** | \`models/xgboost_model.pkl\` |
| **Ukuran** | ~1-5 MB |
| **Fitur** | 37 kolom (harus identik saat *training* dan *inference*) |
| ***Retrain*** | Otomatis tiap 7 hari |
`,
  },
  {
    slug: "smc",
    title: "SMC Analyzer",
    category: "AI & Analisis",
    icon: "TrendingUp",
    description: "Analisis Smart Money Concepts — Order Block, FVG, BOS, CHoCH",
    content: `# SMC Analyzer (*Smart Money Concepts*)

> **File:** \`src/smc_polars.py\`
> **Framework:** Pure Polars (vectorized, tanpa loop)

---

## Pipeline Analisis SMC

Berikut adalah alur lengkap pipeline analisis *Smart Money Concepts*, dari data OHLCV mentah hingga menghasilkan sinyal trading:

\`\`\`mermaid
flowchart TD
    A["Data OHLCV\\n(Polars DataFrame)"] --> B["calculate_all(df)"]

    B --> C["calculate_swing_points()\\nDeteksi swing points\\n(fractal high/low)"]
    C --> D["calculate_fvg()\\nDeteksi Fair Value Gap\\n(imbalance harga)"]
    D --> E["calculate_order_blocks()\\nDeteksi Order Block\\n(zona institusi)\\n-- butuh swing points --"]
    E --> F["calculate_bos_choch()\\nDeteksi BOS dan CHoCH\\n(struktur pasar)\\n-- butuh swing points --"]

    F --> G["DataFrame + semua kolom SMC"]

    G --> H["generate_signal(df)"]
    H --> I["Cek struktur & zona\\ndalam 10 candle terakhir"]
    I --> J["Hitung entry, stop loss, take profit\\n(ATR-based, min 1:2 R:R)"]
    J --> K["calculate_confidence()\\nScoring 40% – 85%"]
    K --> L["SMCSignal\\n(signal_type, entry, SL, TP,\\nconfidence, reason)"]
    L --> M["Dikombinasikan dengan\\nXGBoost + HMM"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#0f3460,color:#fff
    style G fill:#16213e,stroke:#0f3460,color:#fff
    style H fill:#16213e,stroke:#0f3460,color:#fff
    style L fill:#1a1a2e,stroke:#e94560,color:#fff
    style M fill:#0f3460,stroke:#e94560,color:#fff
\`\`\`

---

## Apa Itu SMC?

*Smart Money Concepts* adalah metode analisis berdasarkan **cara institusi besar (bank, hedge fund) trading**. SMC membaca **struktur pasar** dan **jejak uang besar** untuk menemukan zona *entry* yang presisi.

**Analogi:** SMC adalah **peta jalan** — menunjukkan zona penting, rambu lalu lintas, dan rute terbaik.

---

## 6 Konsep yang Diimplementasikan

| # | Konsep | Fungsi | Lines |
|---|--------|--------|-------|
| 1 | *Swing Points* | Puncak & lembah penting | 185-261 |
| 2 | *Fair Value Gap* (FVG) | Imbalance/gap harga | 84-183 |
| 3 | *Order Block* (OB) | Zona order institusi | 263-368 |
| 4 | *Break of Structure* (BOS) | Kelanjutan tren | 370-457 |
| 5 | *Change of Character* (CHoCH) | Pembalikan tren | 370-457 |
| 6 | *Liquidity Zones* | Kumpulan *stop loss* | 459-551 |

---

## 1. *Swing Points* (Fractal High/Low)

**Fungsi:** Mendeteksi puncak dan lembah penting di chart.

### Algoritma

\`\`\`
Window: 2 x swing_length + 1 = 11 candle (default swing_length=5)

Deteksi TANPA LOOKAHEAD (no future data):
  - Rolling max/min menggunakan center=False (hanya data masa lalu)
  - Swing point dikonfirmasi swing_length bar SETELAH terjadi
  - Menggunakan shift(swing_length) untuk melihat "center" point

Swing High: High di center point = Maximum dalam window 11 bar ke belakang
Swing Low:  Low di center point = Minimum dalam window 11 bar ke belakang

Catatan: Deteksi terlambat swing_length bar (5 bar), tapi
         TIDAK menggunakan data masa depan (zero *lookback*).
\`\`\`

### Visualisasi

\`\`\`
         /\\ <- Swing High (high = max 11 candle)
        /  \\
       /    \\
      /      \\
     /        \\/ <- Swing Low (low = min 11 candle)
    /
\`\`\`

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| \`swing_high\` | 1 / 0 | 1 jika *swing high* |
| \`swing_low\` | -1 / 0 | -1 jika *swing low* |
| \`swing_high_level\` | float | Harga di *swing high* |
| \`swing_low_level\` | float | Harga di *swing low* |
| \`last_swing_high\` | float | *Swing high* terakhir (forward fill) |
| \`last_swing_low\` | float | *Swing low* terakhir (forward fill) |

---

## 2. *Fair Value Gap* (FVG)

**Fungsi:** Mendeteksi **imbalance/gap** di harga — zona yang belum "diisi" oleh pasar.

### Algoritma

\`\`\`
TANPA LOOKAHEAD — Deteksi pada candle KETIGA (setelah pola selesai):

Bullish FVG:                    Bearish FVG:
Candle T-2: ████ high           Candle T-2: ████ low
                  |                           |
Candle T-1:    (middle)         Candle T-1:    (middle)
                  |                           |
Candle T:   ████ low (SAAT INI) Candle T:   ████ high (SAAT INI)

Syarat Bullish: high[T-2] < low[T]   (gap antara candle pertama & ketiga)
Syarat Bearish: low[T-2]  > high[T]  (gap antara candle pertama & ketiga)

Catatan: TIDAK menggunakan shift(-1) / data masa depan.
         FVG dideteksi pada candle saat ini (T) setelah pola terkonfirmasi.
\`\`\`

### Zona FVG

\`\`\`
Bullish FVG Zone:
  Top    = low[T]      (low candle saat ini = batas atas gap)
  Bottom = high[T-2]   (high candle pertama = batas bawah gap)
  Mid    = (top + bottom) / 2  (50% retracement)
\`\`\`

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| \`fvg_signal\` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |
| \`fvg_top\` | float | Batas atas gap |
| \`fvg_bottom\` | float | Batas bawah gap |
| \`fvg_mid\` | float | Titik tengah (target retracement) |

**Peran:** Zona *entry* ideal — harga cenderung **kembali mengisi gap** sebelum melanjutkan.

---

## 3. *Order Block* (OB)

**Fungsi:** Mendeteksi candle terakhir sebelum pergerakan besar — zona dimana institusi menaruh order.

### Algoritma

\`\`\`
TANPA LOOKAHEAD — Validasi menggunakan candle SAAT INI:

Bullish OB:
  1. Temukan swing low
  2. Lihat 10 candle ke belakang
  3. Cari candle bearish terakhir (close < open)
  4. Jika candle SAAT INI close di atas high candle tersebut:
     -> Candle itu = Bullish Order Block
     (validasi di bar saat ini, BUKAN bar berikutnya)

Bearish OB:
  1. Temukan swing high
  2. Lihat 10 candle ke belakang
  3. Cari candle bullish terakhir (close > open)
  4. Jika candle SAAT INI close di bawah low candle tersebut:
     -> Candle itu = Bearish Order Block
     (validasi di bar saat ini, BUKAN bar berikutnya)
\`\`\`

### Visualisasi

\`\`\`
Bullish OB:                     Bearish OB:
                                ████ <- Bullish candle terakhir
████ <- Bearish candle terakhir        sebelum jatuh
        sebelum naik           ═══════════════
═══════════════                     ||| turun
     ||| naik
\`\`\`

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| \`ob\` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |
| \`ob_top\` | float | Batas atas zona OB |
| \`ob_bottom\` | float | Batas bawah zona OB |
| \`ob_mitigated\` | bool | True jika OB sudah dikunjungi ulang |

**Peran:** Zona support/resistance berdasarkan aksi institusi besar.

---

## 4. *Break of Structure* (BOS)

**Fungsi:** Mendeteksi **kelanjutan tren** — harga menembus *swing point* searah tren.

### Algoritma

\`\`\`python
# Tren sudah BULLISH, lalu:
if close > last_swing_high:
    bos = 1  # Bullish BOS — tren naik BERLANJUT

# Tren sudah BEARISH, lalu:
if close < last_swing_low:
    bos = -1  # Bearish BOS — tren turun BERLANJUT
\`\`\`

### Visualisasi

\`\`\`
Bullish BOS:
     SH1        SH2 (baru ditembus!)
    /    \\      / close >>>
   /      \\    /
  /        SL1           -> BOS! Tren naik lanjut

Bearish BOS:
  \\        SH1
   \\      /    \\
    \\    /      \\ close <<<
     SL1        SL2 (baru ditembus!)  -> BOS! Tren turun lanjut
\`\`\`

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| \`bos\` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |

**Peran:** Konfirmasi bahwa **tren masih kuat** dan lanjut.

---

## 5. *Change of Character* (CHoCH)

**Fungsi:** Mendeteksi **pembalikan tren** — harga menembus *swing point* berlawanan tren.

### Algoritma

\`\`\`python
# Tren sedang BEARISH, lalu:
if close > last_swing_high:
    choch = 1  # Bullish CHoCH — REVERSAL naik!

# Tren sedang BULLISH, lalu:
if close < last_swing_low:
    choch = -1  # Bearish CHoCH — REVERSAL turun!
\`\`\`

### Visualisasi

\`\`\`
Bearish CHoCH (tren naik -> balik turun):
     SH <- gagal naik
    /  \\
   /    \\
  /      close menembus SL >>> CHoCH! Reversal turun!
 SL

Bullish CHoCH (tren turun -> balik naik):
 SH
  \\      close menembus SH >>> CHoCH! Reversal naik!
   \\    /
    \\  /
     SL <- gagal turun
\`\`\`

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| \`choch\` | 1 / -1 / 0 | Bullish / Bearish / Tidak ada |
| \`market_structure\` | 1 / -1 / 0 | Bullish / Bearish / Netral |

**Peran:** **Early warning** perubahan arah tren.

---

## 6. *Liquidity Zones*

**Fungsi:** Mendeteksi kumpulan *stop loss* (equal highs/lows) yang bisa "disapu" oleh institusi.

### Algoritma

\`\`\`
1. Hitung rolling std & mean dari highs dan lows (window=20)
2. Coefficient of Variation = std / mean
3. Jika CV < 0.001 (0.1%):
   -> Harga sangat mirip = cluster likuiditas
   -> BSL (Buy Side Liquidity) = level high
   -> SSL (Sell Side Liquidity) = level low
4. Deteksi sweep:
   -> BSL sweep: High > BSL lalu close < BSL
   -> SSL sweep: Low < SSL lalu close > SSL
\`\`\`

### Visualisasi

\`\`\`
Buy Side Liquidity (BSL):        Sell Side Liquidity (SSL):
═══════ equal highs ═══════
████ ████ ████ ████              ████ ████ ████ ████
                                 ═══════ equal lows ═══════
^ Stop loss short sellers        ^ Stop loss long traders
^ Institusi sweep ke atas        ^ Institusi sweep ke bawah
\`\`\`

### Output
| Kolom | Nilai | Keterangan |
|-------|-------|-----------|
| \`bsl_level\` | float | Level buy side liquidity |
| \`ssl_level\` | float | Level sell side liquidity |
| \`liquidity_sweep\` | "BSL" / "SSL" / None | Sweep terdeteksi |

---

## Signal Generation

### Logika Pembentukan Sinyal

Sinyal trading dihasilkan dari kombinasi **struktur pasar** dan **zona harga**. Diagram berikut menunjukkan bagaimana komponen SMC digabungkan menjadi sinyal akhir:

\`\`\`mermaid
flowchart LR
    subgraph Struktur["Struktur Pasar"]
        MS["market_structure\\n(bullish / bearish)"]
        BOS["BOS\\n(kelanjutan tren)"]
        CHoCH["CHoCH\\n(pembalikan tren)"]
    end

    subgraph Zona["Zona Harga"]
        FVG["Fair Value Gap\\n(imbalance)"]
        OB["Order Block\\n(zona institusi)"]
    end

    MS --> COND{"Struktur ATAU\\nBreak searah?"}
    BOS --> COND
    CHoCH --> COND

    FVG --> ZONE{"Ada FVG ATAU\\nOrder Block?"}
    OB --> ZONE

    COND -->|Ya| COMBINE{"Struktur + Zona\\n= Valid Setup?"}
    ZONE -->|Ya| COMBINE

    COMBINE -->|Bullish| BULL["BUY Signal"]
    COMBINE -->|Bearish| BEAR["SELL Signal"]
    COMBINE -->|Tidak lengkap| NONE["None\\n(tidak ada sinyal)"]

    BULL --> CALC["Hitung entry, SL, TP\\n(ATR-based)"]
    BEAR --> CALC

    CALC --> CONF["calculate_confidence()\\nScoring 40%–85%"]
    CONF --> SIGNAL["SMCSignal"]

    style Struktur fill:#1a1a2e,stroke:#e94560,color:#fff
    style Zona fill:#16213e,stroke:#0f3460,color:#fff
    style SIGNAL fill:#0f3460,stroke:#e94560,color:#fff
    style NONE fill:#333,stroke:#666,color:#aaa
\`\`\`

### ATR-Based Dynamic SL/TP (v4 Update)

Sebelum menghitung *stop loss* dan *take profit*, sistem mengambil nilai ATR untuk kalkulasi dinamis:

\`\`\`python
atr = latest["atr"]              # Dari Feature Engineering

# Sanity check ATR (v4: validasi ketat)
if atr is None or atr <= 0 or atr > current_close * 0.05:
    atr = 12.0                   # Default realistis untuk XAUUSD (~$12-15 tipikal)

min_sl_distance = 1.5 * atr      # Minimum jarak SL = 1.5 ATR
min_rr_ratio    = 2.0            # ENFORCED: Minimum Risk:Reward 1:2
\`\`\`

### Kondisi Bullish Signal

\`\`\`
IF (market_structure == BULLISH ATAU ada BOS/CHoCH bullish)
AND (ada FVG bullish ATAU Order Block bullish):

  Entry  = current_close  (v4: SELALU harga saat ini, bukan harga zone lama)

  SL (v4 - ATR-based, lebih protektif):
    swing_sl  = last_swing_low (jika ada & di bawah entry)
    atr_sl    = entry - 1.5 * ATR
    SL        = MIN(swing_sl, atr_sl)  <- pilih yang LEBIH JAUH
    IF entry - SL < min_sl_distance:
       SL = entry - min_sl_distance    <- enforce jarak minimum

  TP (v4 - ENFORCED minimum 1:2 RR):
    risk = entry - SL
    tp   = entry + (risk * 2.0)        <- TEPAT 2:1 R:R
    IF actual_rr < 2.0:
       -> SKIP sinyal (tidak valid)    <- sinyal ditolak jika RR terlalu kecil
\`\`\`

### Kondisi Bearish Signal

\`\`\`
IF (market_structure == BEARISH ATAU ada BOS/CHoCH bearish)
AND (ada FVG bearish ATAU Order Block bearish):

  Entry  = current_close  (v4: SELALU harga saat ini, bukan harga zone lama)

  SL (v4 - ATR-based, lebih protektif):
    swing_sl  = last_swing_high (jika ada & di atas entry)
    atr_sl    = entry + 1.5 * ATR
    SL        = MAX(swing_sl, atr_sl)  <- pilih yang LEBIH JAUH
    IF SL - entry < min_sl_distance:
       SL = entry + min_sl_distance    <- enforce jarak minimum

  TP (v4 - ENFORCED minimum 1:2 RR):
    risk = SL - entry
    tp   = entry - (risk * 2.0)        <- TEPAT 2:1 R:R
    IF actual_rr < 2.0:
       -> SKIP sinyal (tidak valid)    <- sinyal ditolak jika RR terlalu kecil
\`\`\`

### Perbandingan Evolusi SL/TP

| Komponen | v2 (lama) | v3 | v4 (sekarang) |
|----------|-----------|-----|---------------|
| *Entry* | Zone price (FVG/OB) | Zone price (FVG/OB) | **SELALU** current_close |
| *SL* (BUY) | entry × 0.995 (bisa terlalu dekat) | MIN(swing, 1.5 ATR) | MIN(swing, 1.5 ATR) + enforce min distance |
| *TP* | risk × 2 (tanpa batas) | MIN(risk×2, 4×ATR) (dibatasi) | risk × 2.0 (**ENFORCED**), SKIP jika RR < 2.0 |
| ATR *default* | close × 1% | close × 1% | $12 (realistis XAUUSD) |
| *Lookahead* | Ada (shift -1) | Ada (shift -1) | **TIDAK ADA** (zero future) |

### Sistem *Confidence* (v5: Calibrated Weighted Scoring)

\`\`\`
Sebelum (v4):                     Sesudah (v5):
  Base: 55%                         Base: 40%
  + BOS/CHoCH: +10%                 + Structure aligned: +15%
  + FVG: +10%                       + BOS/CHoCH: +12%
  + OB: +10%                        + FVG: +8%
  Max: 85%                          + Order Block: +10%
                                    + Trend strength: +10%
                                    + Fresh level: +5%
                                    Max: 85%

Kalkulasi confidence sekarang menggunakan metode calculate_confidence():
  1. Base = 40% (minimum, selalu ada)
  2. Structure aligned = +15% (market_structure searah sinyal)
  3. BOS/CHoCH = +12% (ada *Break of Structure* / *Change of Character*)
  4. FVG = +8% (ada *Fair Value Gap*)
  5. Order Block = +10% (ada *Order Block*)
  6. Trend strength = +10% (ada >=2 BOS searah dalam 20 bar terakhir)
  7. Fresh level = +5% (first touch of key level)
  8. Cap di 85% (tidak pernah 100% yakin)

Contoh:
  BUY signal, structure bullish, ada BOS + FVG + OB, trend kuat:
  = 40% + 15% + 12% + 8% + 10% + 10% = 95% -> cap 85%

  BUY signal, structure bearish, ada CHoCH + FVG saja:
  = 40% + 0% + 12% + 8% + 0% + 0% = 60%
\`\`\`

### Output Signal

\`\`\`python
SMCSignal:
  signal_type: "BUY" / "SELL"
  entry_price: float
  stop_loss: float        # ATR-based (min 1.5 ATR dari entry)
  take_profit: float      # 2:1 RR, capped di 4 ATR
  confidence: 0.40 - 0.85  # v5: base diturunkan ke 40% (calibrated)
  reason: "Bullish BOS + FVG + OB"
  risk_reward: float      # Minimum 2.0
\`\`\`

---

## Konfigurasi

\`\`\`python
SMCConfig:
  swing_length: 5          # Window untuk deteksi swing (11 bar total)
  fvg_min_gap_pips: 2.0    # Minimum ukuran FVG
  ob_lookback: 10          # Berapa jauh cari OB ke belakang
  bos_close_break: True    # Harus close (bukan wick) yang break
\`\`\`

---

## Integrasi dalam Pipeline

\`\`\`mermaid
flowchart TD
    A["Data OHLCV"] --> B["smc.calculate_all(df)"]
    B --> B1["calculate_swing_points()"]
    B --> B2["calculate_fvg()"]
    B --> B3["calculate_order_blocks()\\n(butuh swing points)"]
    B --> B4["calculate_bos_choch()\\n(butuh swing points)"]
    B1 --> C["smc.generate_signal(df)"]
    B2 --> C
    B3 --> C
    B4 --> C
    C --> D["SMCSignal\\n(entry, stop loss, take profit, confidence)"]
    D --> E["Dikombinasikan dengan\\nXGBoost + HMM"]
\`\`\`
`,
  },
  {
    slug: "feature-eng",
    title: "Feature Engineering",
    category: "AI & Analisis",
    icon: "Layers",
    description: "37 fitur teknikal — RSI, ATR, MACD, Bollinger, dan lainnya",
    content: `# *Feature Engineering*

> **File:** \`src/feature_eng.py\`
> **Class:** \`FeatureEngineer\`
> **Framework:** Pure Polars (vectorized, tanpa loop, tanpa TA-Lib — bukan Pandas)

---

## Pipeline *Feature Engineering*

\`\`\`mermaid
flowchart LR
    A["OHLCV Data\\n(open, high, low,\\nclose, volume, time)"] --> B["calculate_all()"]

    subgraph B["calculate_all()"]
        direction TB
        B1["calculate_rsi()"]
        B2["calculate_atr()"]
        B3["calculate_macd()"]
        B4["calculate_bollinger_bands()"]
        B5["calculate_ema_crossover()"]
        B6["calculate_volume_features()"]
        B7["calculate_ml_features()\\n(returns, volatility,\\nlags, trend, time)"]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7
    end

    B --> C["40+ Fitur Numerik"]
    C --> D["ML Ready\\n(XGBoost Input)"]

    style A fill:#2d3748,stroke:#63b3ed,color:#fff
    style C fill:#2d3748,stroke:#48bb78,color:#fff
    style D fill:#2d3748,stroke:#f6ad55,color:#fff
\`\`\`

> **Performa:** Seluruh pipeline dijalankan dalam **< 100ms** untuk 5000 bar menggunakan Pure Polars (vectorized, 10-100x lebih cepat dari Pandas loop). Menghasilkan **40+ fitur** yang siap digunakan model ML.

---

## Apa Itu *Feature Engineering*?

*Feature Engineering* adalah proses **mengubah data harga mentah (OHLCV) menjadi 40+ fitur numerik** yang bisa dibaca oleh model machine learning. Ini adalah "mata" dari AI -- tanpa fitur yang baik, model tidak bisa belajar apapun.

**Analogi:** *Feature Engineering* adalah **alat ukur** -- thermometer, barometer, kompas -- yang mengubah data mentah menjadi informasi bermakna.

---

## Flow Utama: \`calculate_all()\`

\`\`\`
Input: DataFrame OHLCV (open, high, low, close, volume)
    |
    |-- calculate_rsi()               -> rsi
    |-- calculate_atr()               -> atr, atr_percent
    |-- calculate_macd()              -> macd, macd_signal, macd_histogram
    |-- calculate_bollinger_bands()   -> bb_upper, bb_lower, bb_width, bb_percent_b
    |-- calculate_ema_crossover()     -> ema_9, ema_21, ema_cross_bull/bear
    |-- calculate_volume_features()   -> volume_ratio, high_volume
    |
    |-- [jika include_ml_features=True]
    |   calculate_ml_features()       -> returns, volatility, lags, trends, time
    |
    v
Output: DataFrame dengan 40+ kolom fitur
\`\`\`

**Data minimum:** 26 bar (kebutuhan MACD slow EMA) agar semua fitur stabil.

---

## Kategori 1: Indikator Teknikal

### RSI (*Relative Strength Index*) -- Period 14

\`\`\`
Formula: RSI = 100 - (100 / (1 + RS))
         RS  = Average Gain / Average Loss
Smoothing: Wilder's EMA (alpha = 1/14)
\`\`\`

| Nilai | Interpretasi |
|-------|-------------|
| RSI > 70 | *Overbought* (potensi turun) |
| RSI < 30 | *Oversold* (potensi naik) |
| RSI ~ 50 | Netral |

**Output:** \`rsi\`

---

### ATR (*Average True Range*) -- Period 14

\`\`\`
True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
ATR = Wilder's EMA dari True Range
ATR% = (ATR / Close) * 100
\`\`\`

| Kondisi | Interpretasi |
|---------|-------------|
| ATR tinggi | Pasar *volatile* (pergerakan besar) |
| ATR rendah | Pasar tenang (pergerakan kecil) |

**Output:** \`atr\`, \`atr_percent\`

---

### MACD (*Moving Average Convergence Divergence*) -- 12/26/9

\`\`\`
MACD Line  = EMA(12) - EMA(26)
Signal     = EMA(MACD Line, 9)
Histogram  = MACD Line - Signal
\`\`\`

| Kondisi | Interpretasi |
|---------|-------------|
| Histogram > 0 & naik | Bullish *momentum* menguat |
| Histogram < 0 & turun | Bearish *momentum* menguat |
| MACD cross Signal ke atas | Potensi reversal naik |
| MACD cross Signal ke bawah | Potensi reversal turun |

**Output:** \`macd\`, \`macd_signal\`, \`macd_histogram\`

---

### *Bollinger Bands* -- Period 20, StdDev 2.0

\`\`\`
Middle = SMA(20)
Upper  = Middle + 2 * StdDev
Lower  = Middle - 2 * StdDev
Width  = (Upper - Lower) / Middle
%B     = (Close - Lower) / (Upper - Lower)
\`\`\`

| Kondisi | Interpretasi |
|---------|-------------|
| %B > 1 | Harga di atas upper band (extreme bullish) |
| %B < 0 | Harga di bawah lower band (extreme bearish) |
| %B ~ 0.5 | Harga di tengah |
| Width melebar | *Volatility* meningkat |
| Width menyempit | *Volatility* menurun (squeeze) |

**Output:** \`bb_middle\`, \`bb_upper\`, \`bb_lower\`, \`bb_width\`, \`bb_percent_b\`

---

### EMA *Crossover* -- 9/21

\`\`\`
EMA9  = Exponential Moving Average (cepat)
EMA21 = Exponential Moving Average (lambat)
\`\`\`

*EMA* (*Exponential Moving Average*) memberikan bobot lebih besar pada data terbaru, sehingga lebih responsif terhadap perubahan harga dibanding SMA.

| Kondisi | Interpretasi |
|---------|-------------|
| EMA9 > EMA21 | *Trend* naik |
| EMA9 < EMA21 | *Trend* turun |
| EMA9 cross atas EMA21 | Sinyal beli (*bullish crossover*) |
| EMA9 cross bawah EMA21 | Sinyal jual (*bearish crossover*) |

**Output:** \`ema_9\`, \`ema_21\`, \`ema_cross_bull\`, \`ema_cross_bear\`

---

## Kategori 2: Volume Features -- Period 20

\`\`\`
volume_sma        = Rolling Mean(volume, 20)
volume_ratio      = volume / volume_sma
volume_increasing = 1 jika volume > volume sebelumnya
high_volume       = 1 jika volume_ratio > 1.5
\`\`\`

**Fungsi:** Konfirmasi breakout -- pergerakan besar harus didukung volume tinggi.

**Catatan:** Jika kolom volume tidak ada di data, fitur ini di-skip (graceful degradation).

---

## Kategori 3: ML-Specific Features

### *Returns* & *Momentum*

\`\`\`
returns_1   = (Close[t] / Close[t-1]) - 1     # Return 1 bar
returns_5   = (Close[t] / Close[t-5]) - 1     # Return 5 bar
returns_20  = (Close[t] / Close[t-20]) - 1    # Return 20 bar
log_returns = ln(Close[t] / Close[t-1])        # Log return
\`\`\`

**Fungsi:** Mengukur kecepatan dan arah pergerakan harga (*momentum*) dalam berbagai timeframe.

---

### Price Position

\`\`\`
price_position   = (Close - Low) / (High - Low)   # Posisi 0-1 dalam range candle
dist_from_sma_20 = (Close / SMA20) - 1            # Jarak (%) dari rata-rata
\`\`\`

**Fungsi:** Mengukur dimana harga relatif terhadap range dan rata-rata.

---

### *Volatility*

\`\`\`
volatility_20       = StdDev(log_returns, 20)       # Realized volatility
normalized_range    = (High - Low) / Close           # Range sebagai % harga
avg_normalized_range = SMA(normalized_range, 14)     # Rata-rata range 14 bar
\`\`\`

**Fungsi:** Input penting untuk HMM regime detection dan risk sizing. *Volatility* yang tinggi menandakan pasar bergejolak dan mempengaruhi ukuran posisi.

---

### *Lag Features*

\`\`\`
close_lag_1 = Close[t-1]
close_lag_2 = Close[t-2]
close_lag_3 = Close[t-3]
close_lag_5 = Close[t-5]
\`\`\`

**Fungsi:** Auto-regressive features -- menangkap pola harga berulang. *Lag features* memberikan konteks historis langsung kepada model.

---

### *Trend* Features

\`\`\`
higher_high = 1 jika High[t] > High[t-1], else 0
lower_low   = 1 jika Low[t] < Low[t-1], else 0
hh_count_5  = Sum(higher_high, 5 bar)   # Berapa kali HH dalam 5 bar
ll_count_5  = Sum(lower_low, 5 bar)     # Berapa kali LL dalam 5 bar
\`\`\`

**Fungsi:** Mengukur konsistensi *trend* -- banyak HH = strong uptrend.

---

### *Time Features*

\`\`\`
hour           = Jam (0-23)
weekday        = Hari (0=Senin, 6=Minggu)
london_session = 1 jika jam 08:00-16:00 UTC
ny_session     = 1 jika jam 13:00-21:00 UTC
\`\`\`

**Fungsi:** Pasar berperilaku berbeda tiap sesi -- London *volatile*, Asian tenang. *Time features* membantu model mengenali pola berbasis waktu.

**Catatan:** Hanya dihitung jika kolom \`time\` bertipe Datetime.

---

## Kategori 4: SMC sebagai Fitur Numerik

Dari SMC Analyzer, dikonversi jadi angka untuk XGBoost:

\`\`\`
swing_high       = 1 / 0
swing_low        = -1 / 0
fvg_signal       = 1 (bull) / -1 (bear) / 0
ob               = 1 (bull) / -1 (bear) / 0
bos              = 1 (bull) / -1 (bear) / 0
choch            = 1 (bull) / -1 (bear) / 0
market_structure = 1 (bull) / -1 (bear) / 0
regime           = 0 / 1 / 2 (dari HMM)
\`\`\`

---

## Target Variable (Label Training)

\`\`\`python
create_target(df, lookahead=1, threshold=0.0):
    target = 1 jika close[t+1] > close[t]   # Harga naik
    target = 0 jika close[t+1] <= close[t]   # Harga turun/tetap
    target_return = close[t+1] / close[t] - 1  # Return kontinu
\`\`\`

**Catatan:** Target dibuat saat training saja, tidak saat live trading.

---

## Preprocessing untuk ML

### Penanganan Null
\`\`\`python
# Bar awal memiliki NaN karena lookback period
# Saat training: baris dengan NaN di-drop
df_clean = df.select(features + [target]).drop_nulls()
\`\`\`

### Penanganan Infinity
\`\`\`python
# Saat prediksi: NaN & infinity diganti 0
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
\`\`\`

### Normalisasi
**Tidak dilakukan** -- XGBoost berbasis tree, scale-invariant (tidak perlu scaling).

### Cleanup Kolom Temporary
Setiap method membersihkan kolom sementara yang diawali \`_\` (misal \`_delta\`, \`_avg_gain\`, dll).

---

## Fitur yang Digunakan vs Tidak

### Digunakan oleh XGBoost (24+ fitur)
Semua indikator teknikal, *returns*, *volatility*, *trend*, *time features*, SMC numerik, regime.

### Tidak Digunakan (Excluded)
- Kolom OHLCV asli: \`time\`, \`open\`, \`high\`, \`low\`, \`close\`, \`volume\`
- Kolom meta: \`spread\`, \`real_volume\`, \`target\`, \`target_return\`
- Kolom SMC level: \`fvg_top\`, \`fvg_bottom\`, \`ob_top\`, \`ob_bottom\`, dll
- Kolom temporary: apapun yang diawali \`_\`

---

## Parameter Konfigurasi

| Indikator | Parameter | Default | Configurable |
|-----------|-----------|---------|-------------|
| RSI (*Relative Strength Index*) | period | 14 | Ya |
| ATR (*Average True Range*) | period | 14 | Ya |
| MACD (*Moving Average Convergence Divergence*) | fast/slow/signal | 12/26/9 | Ya |
| *Bollinger Bands* | period, std_dev | 20, 2.0 | Ya |
| EMA *Crossover* | fast/slow | 9/21 | Ya |
| Volume | period | 20 | Ya |
| *Returns* | lookback | [1, 5, 20] | Hardcoded |
| *Volatility* | window | 20 | Hardcoded |
| Session | London hours | 08-16 UTC | Hardcoded |
| Session | NY hours | 13-21 UTC | Hardcoded |

---

## Performa

- **5000 bar features:** < 100ms (sangat cepat)
- **Framework:** Pure Polars vectorized (10-100x lebih cepat dari Pandas loop) -- **bukan Pandas**
- **Memory:** ~1.6MB untuk 40+ fitur x 5000 bar
- **Total fitur:** 40+ kolom numerik siap ML
`,
  },
  {
    slug: "risk-management",
    title: "Manajemen Risiko",
    category: "Risiko & Proteksi",
    icon: "Shield",
    description: "Sistem manajemen risiko dinamis dengan mode kapital dan batas harian",
    content: `# Manajemen Risiko — *Smart Risk Manager*

> **File utama:** \`src/smart_risk_manager.py\`
> **File pendukung:** \`src/risk_engine.py\`, \`src/position_manager.py\`
> **Konfigurasi:** \`src/config.py\`

---

## Gambaran Umum

Manajemen risiko adalah **fondasi terpenting** dari sistem *trading*. Bot menggunakan pendekatan berlapis — dari kalkulasi ukuran posisi hingga perlindungan otomatis saat kondisi pasar memburuk.

---

## 4 Mode *Trading*

\`\`\`mermaid
graph LR
    A["NORMAL<br/>🟢 Trading normal"] -->|"Kerugian > 50% batas harian"| B["PROTECTED<br/>🟡 Lot dikurangi 50%"]
    B -->|"Kerugian > 80% batas harian"| C["RECOVERY<br/>🟠 Lot minimal, sangat ketat"]
    C -->|"Kerugian = batas harian"| D["COOLDOWN/STOPPED<br/>🔴 Berhenti total"]
    D -->|"Hari baru (reset)"| A
    B -->|"Profit pulih"| A
\`\`\`

| Mode | Kondisi | Efek |
|------|---------|------|
| **NORMAL** | Kerugian < 50% batas harian | *Lot* normal, semua filter standar |
| **PROTECTED** | Kerugian 50-80% batas harian | *Lot* dikurangi 50%, *entry* lebih ketat |
| **RECOVERY** | Kerugian 80-100% batas harian | *Lot* minimal, hanya sinyal sangat kuat |
| **COOLDOWN / STOPPED** | Kerugian = batas harian | **Berhenti total** — tidak boleh *trading* |

---

## Batas Risiko (Akun *Small* $5.000)

| Batas | Nilai | Kalkulasi |
|-------|-------|-----------|
| **Risiko per *trade*** | 1.0% | $50 |
| **Kerugian harian** | 3.0% | $150 |
| **Posisi bersamaan** | Max 3 | Terbatas oleh risiko |
| **Max *lot*** | 0.05 | Batas keras |
| **Min *lot*** | 0.01 | *Lot* paling kecil |
| ***Cooldown*** | 5 menit | Antar *trade* |

---

## Kalkulasi Ukuran Posisi

Bot menggunakan **metode *Half-Kelly Criterion***:

\`\`\`python
# 1. Hitung jumlah risiko
risk_amount = balance * risk_per_trade / 100
# $5.000 * 1% = $50

# 2. Hitung jarak SL
sl_distance = abs(entry - stop_loss)
sl_pips = sl_distance / 0.1  # XAUUSD

# 3. Hitung lot
lot = risk_amount / (sl_pips * pip_value)

# 4. Half-Kelly (keamanan)
lot *= 0.5

# 5. Apply multiplier
lot *= session_multiplier  # 0.5x - 1.2x
lot *= regime_multiplier   # Dikurangi saat volatile

# 6. Batasi
lot = max(0.01, min(lot, 0.05))
\`\`\`

---

## Proteksi Berlapis

### Lapis 1: *Entry Filter* (Sebelum *Trade*)

14 filter *entry* harus lolos — termasuk *session filter*, *regime check*, dan *smart risk gate*.

### Lapis 2: *Position Monitoring* (Saat *Trade* Aktif)

12 kondisi *exit* diperiksa setiap ~10 detik:
- *Smart Take Profit* (4 sub-kondisi)
- *Early Cut* (momentum negatif)
- *Trend Reversal* (ML sinyal balik)
- *Max Loss* per *trade*
- *Stall Detection*
- Batas harian
- *Weekend close*
- *Time-based exit* (4-8 jam)
- *Trailing SL* + *Breakeven*

### Lapis 3: *Broker-Level SL*

\`\`\`python
# SL dikirim ke broker sebagai proteksi darurat
result = mt5.send_order(
    sl=emergency_sl,  # Berbasis ATR — proteksi server-side
    tp=signal.take_profit,
)
\`\`\`

Jika koneksi internet terputus, **SL di *broker* tetap aktif**.

### Lapis 4: *Circuit Breaker*

\`\`\`python
# Flash crash detection — hentikan semua trading
flash_crash_threshold = 2.5  # Pergerakan 2.5% dalam 1 menit
if move_percent > threshold:
    HALT_ALL_TRADING
\`\`\`

---

## 7 Alasan *Exit* (*ExitReason*)

| Kode | Deskripsi |
|------|-----------|
| \`TAKE_PROFIT\` | Target profit tercapai atau profit diamankan |
| \`TREND_REVERSAL\` | ML mendeteksi pembalikan *trend* |
| \`DAILY_LIMIT\` | Batas kerugian harian tercapai |
| \`POSITION_LIMIT\` | Batas kerugian per posisi (S/L) |
| \`TOTAL_LIMIT\` | Batas kerugian total tercapai |
| \`WEEKEND_CLOSE\` | Mendekati penutupan *weekend* |
| \`MANUAL\` | Penutupan manual oleh pengguna |

---

## *State* Risiko Harian

\`\`\`python
@dataclass
class RiskState:
    mode: TradingMode        # NORMAL/PROTECTED/RECOVERY/COOLDOWN
    daily_profit: float      # Total profit hari ini ($)
    daily_loss: float        # Total kerugian hari ini ($)
    daily_trades: int        # Jumlah trade hari ini
    consecutive_losses: int  # Kerugian berturut-turut
    last_loss_amount: float  # Kerugian terakhir ($)
    can_trade: bool          # Boleh trading atau tidak
\`\`\`

*State* **di-reset setiap hari baru** (00:00 WIB) — hari baru, kesempatan baru.

---

## *Position Guard* (Per Posisi)

Setiap posisi yang terbuka memiliki *guard* sendiri yang melacak:

| Properti | Keterangan |
|----------|------------|
| \`entry_price\` | Harga masuk |
| \`peak_profit\` | Profit tertinggi yang pernah dicapai |
| \`profit_history\` | Riwayat profit (untuk *stall detection*) |
| \`reversal_warnings\` | Jumlah peringatan *reversal* dari ML |
| \`stall_count\` | Berapa kali harga *stuck* |
| \`entry_time\` | Waktu masuk (untuk *time-based exit*) |
`,
  },
  {
    slug: "session-filter",
    title: "Filter Sesi",
    category: "Risiko & Proteksi",
    icon: "Clock",
    description: "Filter sesi perdagangan — Sydney, London, New York dalam zona waktu WIB",
    content: `# *Session Filter* — Filter Sesi Perdagangan

> **File:** \`src/session_filter.py\`
> **Class:** \`SessionFilter\`
> **Zona Waktu:** WIB (Waktu Indonesia Barat / GMT+7)

---

## Apa Itu *Session Filter*?

*Session Filter* mengontrol **kapan bot boleh *trading*** berdasarkan sesi pasar global. Setiap sesi memiliki karakteristik volatilitas yang berbeda — bot memilih **sesi terbaik** untuk memaksimalkan peluang.

---

## Peta Sesi *Trading* (WIB)

\`\`\`mermaid
gantt
    title Sesi Trading dalam WIB (GMT+7)
    dateFormat HH:mm
    axisFormat %H:%M

    section Sesi
    Sydney (Low Vol)        :06:00, 13:00
    Tokyo (Medium Vol)      :07:00, 16:00
    London (High Vol)       :15:00, 23:59
    New York (Extreme Vol)  :20:00, 23:59

    section Overlap
    Tokyo-London DIBLOKIR   :crit, 15:00, 16:00
    London-NY GOLDEN        :active, 20:00, 23:59

    section Bahaya
    Dead Zone               :crit, 00:00, 04:00
    Rollover                :crit, 04:00, 06:00
\`\`\`

---

## Konfigurasi Sesi (Terkini)

| Sesi | Jam WIB | Volatilitas | *Trading* | *Lot Multiplier* | Catatan |
|------|---------|-------------|-----------|-------------------|---------|
| **Sydney** | 06:00 - 13:00 | *Low* | **Ya** | 0.5x | *Backtest* membuktikan profit $5.934 |
| **Tokyo** | 07:00 - 16:00 | *Medium* | **Ya** | 0.7x | Volume cukup |
| **Tokyo-London *Overlap*** | 15:00 - 16:00 | *High* | **Tidak** | 0.0x | #24B: *Backtest* +$345 tanpa sesi ini |
| **London** | 15:00 - 23:59 | *High* | **Ya** | 1.0x | Sesi utama Eropa |
| **London-NY *Overlap*** | 20:00 - 23:59 | *Extreme* | **Ya** | **1.2x** | **Waktu emas** — volatilitas tertinggi |
| **New York** | 20:00 - 23:59 | *Extreme* | **Ya** | 1.0x | Sesi utama AS |
| ***Off Hours*** | Lainnya | *Low* | **Tidak** | 0.0x | Di luar semua sesi |

---

## Zona Bahaya

| Zona | Jam WIB | Alasan | Aksi |
|------|---------|--------|------|
| ***Dead Zone*** | 00:00 - 04:00 | Likuiditas rendah, *spread* tinggi | **Blokir *trading*** |
| ***Rollover*** | 04:00 - 06:00 | *Spread* sangat lebar saat *rollover* | **Blokir *trading*** |
| **Jumat *Close*** | Sabtu 04:30+ | Mendekati tutup *weekend* | **Blokir *entry* baru** |
| ***Weekend*** | Sabtu - Minggu | Pasar tutup | **Blokir sepenuhnya** |

---

## Jam *Skip* Khusus (#34A)

Selain filter sesi, ada filter waktu tambahan dari optimasi *backtest*:

\`\`\`python
# main_live.py — Filter #34A
# Skip jam 9 dan 21 WIB — backtest menambah +$356 profit
wib_hour = datetime.now(ZoneInfo("Asia/Jakarta")).hour
if wib_hour in (9, 21):
    return  # Jam transisi — volatilitas tidak optimal
\`\`\`

---

## Cara Kerja

\`\`\`python
def can_trade(self) -> Tuple[bool, str, float]:
    """
    Returns:
        can_trade: Boleh trading atau tidak
        reason: Alasan dalam bahasa Indonesia
        multiplier: Pengali lot size (0.0 - 1.2)
    """
    # 1. Cek weekend
    if self.is_weekend():
        return False, "Market tutup (weekend)", 0.0

    # 2. Cek Friday close
    if self.is_friday_close():
        return False, "Mendekati penutupan Jumat", 0.0

    # 3. Cek zona bahaya
    if self.is_danger_zone():
        return False, "Zona bahaya: spread melebar", 0.0

    # 4. Cek sesi saat ini
    session, config = self.get_current_session()
    if not config.allow_trading:
        return False, f"Trading tidak diizinkan saat {config.name}", 0.0

    return True, f"Trading OK - {config.name}", config.position_size_multiplier
\`\`\`

---

## Pengaruh pada *Position Sizing*

*Session multiplier* langsung mengubah ukuran *lot*:

\`\`\`python
# main_live.py
safe_lot = smart_risk.calculate_lot_size(...)
safe_lot = max(0.01, safe_lot * session_multiplier)

# Contoh (akun $5.000):
# London-NY Overlap: 0.02 * 1.2 = 0.024 → 0.02 lot (setelah rounding)
# Sydney:           0.02 * 0.5 = 0.010 → 0.01 lot (half size)
# Tokyo:            0.02 * 0.7 = 0.014 → 0.01 lot
\`\`\`

---

## Waktu Berita Dampak Tinggi

Bot juga memiliki daftar waktu berita ekonomi penting:

| Berita | Jam WIB | *Buffer* Sebelum | *Buffer* Setelah |
|--------|---------|------------------|------------------|
| NFP (*Non-Farm Payrolls*) | 19:30 | 15 menit | 30 menit |
| FOMC (*Federal Reserve*) | 01:00 | 15 menit | 45 menit |
| CPI (*Consumer Price Index*) | 19:30 | 15 menit | 30 menit |

> **Catatan:** *News Agent* saat ini **nonaktif** (\`main_live.py\` baris 64). Filter berita direncanakan untuk diaktifkan kembali di versi mendatang.
`,
  },
  {
    slug: "stop-loss",
    title: "Stop Loss",
    category: "Risiko & Proteksi",
    icon: "ShieldAlert",
    description: "Proteksi SL berbasis ATR dan broker-level untuk keamanan maksimal",
    content: `# *Stop Loss* (S/L) — Sistem Proteksi Berlapis

> **File terkait:** \`src/smc_polars.py\`, \`main_live.py\`, \`src/smart_risk_manager.py\`

---

\`\`\`mermaid
block-beta
    columns 1
    block:layer1["Layer 1 — SMC ATR-Based Stop Loss"]:1
        A["Dikirim ke broker sebagai SL aktif\\n1.5 ATR dari entry (~$15-$30)"]
    end
    block:layer2["Layer 2 — Software Smart Exit"]:1
        B["Bot monitor posisi & tutup otomatis\\nDinamis berdasarkan konteks ($25-$50)"]
    end
    block:layer3["Layer 3 — Emergency Broker SL"]:1
        C["Safety net 2% modal\\nAktif jika software gagal ($100)"]
    end
    block:layer4["Layer 4 — Circuit Breaker"]:1
        D["Halt total semua trading\\nFlash crash 2.5% / daily limit -5%"]
    end

    style layer1 fill:#2d7d46,color:#fff
    style layer2 fill:#2d6a9f,color:#fff
    style layer3 fill:#b8860b,color:#fff
    style layer4 fill:#a82020,color:#fff
\`\`\`

---

## Apa Itu *Stop Loss* di Bot Ini?

*Stop Loss* bukan hanya satu angka — ini adalah **sistem proteksi 4 lapis** yang bekerja bersamaan. Jika satu layer gagal, layer berikutnya siap melindungi.

**Analogi:** SL di bot ini seperti sistem keamanan gedung — ada CCTV (software monitoring), security (broker SL), alarm kebakaran (*Emergency* SL), dan sprinkler otomatis (*Circuit Breaker*).

---

## 4 Layer *Stop Loss*

\`\`\`
Layer 1: SMC ATR-Based SL      <- Dikirim ke broker sebagai SL aktif
Layer 2: Software Smart Exit    <- Bot monitor & tutup posisi secara cerdas
Layer 3: Emergency Broker SL    <- Safety net 2% jika software gagal
Layer 4: Circuit Breaker        <- Halt total jika flash crash / daily limit
\`\`\`

\`\`\`
Harga masuk (Entry)
    |
    |-- Layer 1: SMC SL (1.5 ATR)         contoh: -$15 ~ -$30
    |
    |-- Layer 2: Software SL ($25-$50)     50% dari max loss
    |
    |-- Layer 3: Emergency SL ($100)       2% modal (jaring terakhir)
    |
    |-- Layer 4: Circuit Breaker           Flash crash 2.5% -> HALT
    |
    v
Semakin jauh = semakin jarang tercapai (backup)
\`\`\`

---

## Layer 1: SMC *ATR*-Based *Stop Loss*

**Sumber:** \`smc_polars.py\` (Lines 631-652, 694-702)
**Dikirim ke:** Broker MT5 sebagai SL order aktif

### Perhitungan

\`\`\`python
# Ambil ATR dari Feature Engineering
atr = latest["atr"]                    # Contoh: ATR = $8.50
min_sl_distance = 1.5 * atr            # 1.5 * 8.50 = $12.75

# Untuk BUY:
swing_sl = last_swing_low              # Contoh: $4935.00
atr_sl   = entry - min_sl_distance     # $4950 - $12.75 = $4937.25
SL       = MIN(swing_sl, atr_sl)       # $4935.00 (pilih yang LEBIH JAUH)

# Untuk SELL:
swing_sl = last_swing_high             # Contoh: $4965.00
atr_sl   = entry + min_sl_distance     # $4950 + $12.75 = $4962.75
SL       = MAX(swing_sl, atr_sl)       # $4965.00 (pilih yang LEBIH JAUH)
\`\`\`

### Kenapa MIN/MAX (Pilih yang Lebih Jauh)?

\`\`\`
Sebelum (v2): SL = swing_low ATAU entry * 0.995
  -> Bisa sangat dekat, gampang kena *whipsaw*

Sesudah (v3): SL = MIN(swing_low, entry - 1.5*ATR)
  -> Selalu minimal 1.5 ATR dari entry
  -> Lebih protektif terhadap noise pasar
\`\`\`

---

## Layer 2: Software *Smart Exit*

**Sumber:** \`smart_risk_manager.py\` (Lines 559-724)
**Mekanisme:** Bot monitor posisi setiap detik dan tutup otomatis

### Kondisi Software SL

\`\`\`
Max loss per trade: $50 (1% dari modal $5,000)

Trigger exit jika:
  1. Loss >= $25 (50% dari max)
     Kecuali golden time DAN momentum > -40 -> hold

  2. Loss >= $20 (40%) + ML reversal 65%+ berlawanan
     -> Tutup karena trend reversal

  3. Loss >= $15 + harga stall 10+ candle
     stall_count >= 5 -> tutup

  4. 4+ jam terbuka + profit < $5
     -> Time-based exit

  5. 6+ jam terbuka
     -> Force exit (apapun kondisinya)
\`\`\`

### Kelebihan Software SL vs Hard SL

\`\`\`
Hard SL (broker):
  - Kaku, tidak bisa diubah
  - Bisa kena *whipsaw* lalu harga balik
  - Tidak bisa mempertimbangkan konteks

Software SL (bot):
  - Dinamis, mempertimbangkan momentum
  - Bisa hold jika golden time & momentum positif
  - Bisa exit lebih cepat jika ML deteksi reversal
  - Mempertimbangkan durasi posisi
\`\`\`

---

## Layer 3: *Emergency* Broker *Stop Loss*

**Sumber:** \`smart_risk_manager.py\` (Lines 305-346)
**Fungsi:** Jaring pengaman TERAKHIR jika software gagal (disconnect, crash, dll)

### Perhitungan

\`\`\`python
# Konfigurasi: 2% dari modal
emergency_sl_percent = 2.0
emergency_sl_usd = 5000 * 0.02 = $100    # Max loss jika software gagal

# Hitung jarak SL
pip_value = lot_size * 10                 # 0.01 lot -> $0.10/pip
emergency_pips = $100 / $0.10 = 1000 pips
price_distance = 1000 * 0.01 = $10.00

# SL price
BUY:  SL = entry - $10.00 = $4940.00
SELL: SL = entry + $10.00 = $4960.00
\`\`\`

### Kapan *Emergency* SL Tercapai?

Seharusnya **tidak pernah** — software SL ($50) akan menutup jauh sebelum *Emergency* SL ($100). *Emergency* SL hanya tercapai jika:
- Bot crash / disconnect
- Server bermasalah
- Internet putus
- Harga *gap* melewati semua level

---

## Layer 4: *Circuit Breaker*

**Sumber:** \`risk_engine.py\` (Lines 143-151)
**Fungsi:** Halt trading total saat kondisi darurat

\`\`\`python
# Flash crash: Pergerakan > 2.5% dalam waktu singkat
if price_move > flash_crash_threshold:
    activate_circuit_breaker("Flash crash detected")
    # Tutup SEMUA posisi
    # Block semua trade baru
    # Kirim alert Telegram

# Daily loss limit
if daily_pnl_percent <= -5.0%:
    activate_circuit_breaker("Daily loss limit breached")
\`\`\`

---

## Pengiriman SL ke Broker (main_live.py)

### Flow Pengiriman

\`\`\`python
# Step 1: Ambil SL dari SMC signal (ATR-based)
broker_sl = signal.stop_loss

# Step 2: Validasi jarak minimum (10 pips untuk XAUUSD)
min_sl_distance = 1.0  # $1 = 10 pips

if direction == "BUY":
    if current_price - broker_sl < 1.0:
        broker_sl = current_price - 2.0    # Paksa lebih lebar

if direction == "SELL":
    if broker_sl - current_price < 1.0:
        broker_sl = current_price + 2.0    # Paksa lebih lebar

# Step 3: Kirim order DENGAN SL
result = mt5.send_order(
    sl=broker_sl,       # SL AKTIF di broker
    tp=signal.take_profit,
    ...
)

# Step 4: Fallback jika broker reject (error 10016)
if not result.success and retcode == 10016:
    # SL terlalu dekat / tidak valid
    result = mt5.send_order(
        sl=0,            # Tanpa broker SL
        comment="AI Safe v3 NoSL"
    )
    # Software SL tetap aktif sebagai proteksi
\`\`\`

---

## Tabel Ringkasan Layer SL

| Layer | Sumber | Jarak dari Entry | Max Loss | Kondisi Trigger |
|-------|--------|-----------------|----------|-----------------|
| **1. SMC *ATR*** | Broker SL aktif | 1.5 *ATR* (~$12-15) | ~$15-30 | Harga hit SL level |
| **2. Software** | Bot monitoring | Dinamis | $25-50 | Loss threshold + konteks |
| **3. *Emergency*** | Broker safety net | 2% modal ($10) | $100 | Software gagal |
| **4. *Circuit Breaker*** | Halt total | Semua posisi | Unlimited cap | *Flash crash* / daily limit |

---

## Skenario Proteksi

### Skenario 1: Trading Normal

\`\`\`
Entry BUY @ $4950, SL broker @ $4937 (1.5 ATR)
  -> Harga turun ke $4938 -> Masih aman
  -> Harga turun ke $4936 -> BROKER SL HIT -> Tutup otomatis
  -> Loss: ~$14 (0.01 lot)
\`\`\`

### Skenario 2: Connection Lost

\`\`\`
Entry BUY @ $4950, SL broker @ $4937
  -> Bot disconnect
  -> Harga turun drastis ke $4920
  -> BROKER SL sudah aktif di $4937 -> Tutup otomatis
  -> Loss: ~$13 (bukan unlimited!)
\`\`\`

### Skenario 3: Weekend *Gap*

\`\`\`
Jumat: Entry BUY @ $4950, SL broker @ $4937
  -> Senin buka *gap* di $4910 (melewati SL)
  -> Broker eksekusi SL di harga terbaik ~$4910
  -> Loss: ~$40 (lebih dari SL tapi terproteksi)
\`\`\`

### Skenario 4: *Flash Crash* (Tanpa Broker SL Fallback)

\`\`\`
Entry BUY @ $4950, sl=0 (broker reject)
  -> Harga jatuh cepat ke $4925
  -> Software: loss = $25 >= 50% max -> TUTUP
  -> Loss: ~$25 (software protect)

  -> Jika software juga gagal:
     Emergency SL @ $4940 -> TUTUP
     -> Loss: ~$100 max
\`\`\`
`,
  },
  {
    slug: "take-profit",
    title: "Take Profit",
    category: "Risiko & Proteksi",
    icon: "Target",
    description: "Target TP multi-level dengan perhitungan ATR dan struktur pasar",
    content: `# *Take Profit* (T/P) — Sistem Pengambilan Profit Cerdas

> **File terkait:** \`src/smc_polars.py\`, \`main_live.py\`, \`src/smart_risk_manager.py\`

---

## Flowchart Prioritas *Take Profit*

\`\`\`mermaid
flowchart TD
    A["Evaluasi Posisi Terbuka"] --> B{"profit >= $40?"}
    B -- Ya --> B1["Layer 1: Hard TP\\nTutup langsung"]
    B -- Tidak --> C{"profit >= $25\\nDAN momentum < -30?"}
    C -- Ya --> C1["Layer 2: Momentum TP\\nAmankan profit"]
    C -- Tidak --> D{"peak > $30\\nDAN current < 60% peak?"}
    D -- Ya --> D1["Layer 3: Peak Protection\\nKunci sisa profit"]
    D -- Tidak --> E{"profit >= $20\\nDAN TP probability < 25%?"}
    E -- Ya --> E1["Layer 4: Probability TP\\nAmbil sekarang"]
    E -- Tidak --> F{"profit $5-15\\nDAN ML reversal\\nDAN momentum < -50?"}
    F -- Ya --> F1["Layer 5: Early Exit\\nProfit kecil > loss"]
    F -- Tidak --> G["Layer 6: Broker TP\\nHarga hit level otomatis"]

    style B1 fill:#16a34a,color:#fff
    style C1 fill:#2563eb,color:#fff
    style D1 fill:#7c3aed,color:#fff
    style E1 fill:#d97706,color:#fff
    style F1 fill:#dc2626,color:#fff
    style G fill:#64748b,color:#fff
\`\`\`

---

## Apa Itu *Take Profit* di Bot Ini?

*Take Profit* bukan hanya satu target harga — ini adalah **sistem multi-layer** yang secara cerdas memutuskan kapan mengambil profit berdasarkan *momentum*, *probability*, dan *peak protection*.

**Analogi:** TP di bot ini seperti **pemanen buah pintar** — tahu kapan buah sudah matang (*Hard TP*), kapan cuaca akan buruk (*momentum* drop), dan kapan panen sebelum busuk (*Peak Protection*).

---

## Layer *Take Profit*

\`\`\`
Layer 1: Broker TP           <- Target harga dikirim ke broker (SMC-generated)
Layer 2: Hard TP             <- Software tutup jika profit >= $40
Layer 3: Momentum TP         <- Tutup jika profit bagus tapi momentum turun
Layer 4: Peak Protection     <- Tutup jika profit turun dari peak
Layer 5: Probability TP      <- Tutup jika probabilitas capai TP rendah
Layer 6: Early Exit          <- Tutup profit kecil jika reversal terdeteksi
\`\`\`

---

## Layer 1: Broker TP (SMC-Generated)

**Sumber:** \`smc_polars.py\` (Lines 654-659, 704-709)
**Dikirim ke:** Broker MT5 sebagai TP order aktif

### Perhitungan

\`\`\`python
# ATR-based TP cap
atr = latest["atr"]                    # Contoh: ATR = $8.50
max_tp_distance = 4.0 * atr            # 4 * 8.50 = $34.00

# Untuk BUY:
risk = entry - sl                      # $4950 - $4937 = $13
tp   = entry + (risk * 2)              # $4950 + $26 = $4976 (2:1 RR)
if tp > entry + max_tp_distance:       # $4976 vs $4950 + $34 = $4984
    tp = entry + max_tp_distance       # Tidak kena cap, tetap $4976

# Untuk SELL:
risk = sl - entry                      # $4963 - $4950 = $13
tp   = entry - (risk * 2)              # $4950 - $26 = $4924 (2:1 RR)
if tp < entry - max_tp_distance:       # $4924 vs $4950 - $34 = $4916
    tp = entry - max_tp_distance       # Tidak kena cap, tetap $4924
\`\`\`

### Kenapa TP Di-cap 4 ATR?

\`\`\`
Sebelum (v2): TP = risk * 2 (tanpa batas)
  -> Bisa sangat jauh ($50+ dari entry)
  -> Jarang tercapai, posisi terbuka terlalu lama

Sesudah (v3): TP = MIN(risk * 2, 4 * ATR)
  -> Dibatasi maksimal 4x ATR
  -> Target lebih realistis, lebih sering tercapai
\`\`\`

### Dikirim ke Broker

\`\`\`python
# main_live.py
result = mt5.send_order(
    sl=broker_sl,
    tp=signal.take_profit,    # <- TP dari SMC (ATR-capped)
    ...
)
\`\`\`

Jika harga mencapai TP level, broker otomatis menutup posisi — tidak perlu bot online.

---

## Layer 2: *Hard Take Profit* ($40)

**Sumber:** \`smart_risk_manager.py\` (Lines 595-599)

\`\`\`python
# Profit mencapai $40+ -> langsung tutup
if current_profit >= 40:
    return True, ExitReason.TAKE_PROFIT,
           "[TP] Target profit reached: $40.00"
\`\`\`

**Kenapa $40?** Ini threshold profit yang cukup besar untuk diamankan, terlepas dari kondisi pasar.

---

## Layer 3: *Momentum*-Based TP ($25+)

**Sumber:** \`smart_risk_manager.py\` (Lines 601-603)

\`\`\`python
# Profit $25+ tapi momentum turun -> amankan profit
if current_profit >= 25 and momentum < -30:
    return True, ExitReason.TAKE_PROFIT,
           "[SECURE] Securing $25.00 (momentum dropping)"
\`\`\`

### Bagaimana *Momentum* Dihitung

\`\`\`python
# PositionGuard.calculate_momentum() (Lines 113-131)
# Melihat 5 profit history terakhir

recent_profits = profit_history[-5:]
profit_change = recent_profits[-1] - recent_profits[0]

# Normalisasi: $10 change = 50 poin
momentum = (profit_change / 10) * 50
# Range: -100 sampai +100

# momentum < -30 artinya profit sedang TURUN cukup cepat
\`\`\`

**Visualisasi:**

\`\`\`
Profit ($)
  40 |
  35 |        /\\
  30 |       /  \\  <- Momentum mulai negatif
  25 |------/----\\------ Layer 3 trigger: amankan!
  20 |     /      \\
  15 |    /        \\
  10 |   /          \\
   5 |  /
   0 |_/________________________> waktu
\`\`\`

---

## Layer 4: *Peak Protection* ($30+ peak)

**Sumber:** \`smart_risk_manager.py\` (Lines 605-607)

\`\`\`python
# Profit pernah $30+ tapi sekarang turun ke 60% dari peak
if guard.peak_profit > 30 and current_profit < guard.peak_profit * 0.6:
    return True, ExitReason.TAKE_PROFIT,
           "[LOCK] Securing profit (was $35 peak)"
\`\`\`

### Cara Kerja Peak Tracking

\`\`\`python
# Setiap evaluasi, update peak profit
guard.peak_profit = max(guard.peak_profit, current_profit)

# Contoh:
# Peak: $35 -> 60% = $21
# Current: $18 (turun dari $35)
# $18 < $21 -> TUTUP, lindungi sisa profit
\`\`\`

**Visualisasi:**

\`\`\`
Profit ($)
  35 |    * <- peak_profit = $35
  30 |   / \\
  25 |  /   \\
  21 |./.....\\....... 60% threshold ($21)
  18 |        \\* <- current = $18, TUTUP!
  15 |         \\
  10 |          (kehilangan lebih banyak dihindari)
\`\`\`

---

## Layer 5: *Probability*-Based TP ($20+)

**Sumber:** \`smart_risk_manager.py\` (Lines 609-611)

\`\`\`python
# Probabilitas capai TP rendah + profit cukup -> ambil sekarang
if tp_probability < 25 and current_profit >= 20:
    return True, ExitReason.TAKE_PROFIT,
           "[PROB] Taking profit $20 (TP prob: 15%)"
\`\`\`

### Cara Hitung TP *Probability*

\`\`\`python
# PositionGuard.get_tp_probability() (Lines 133-168)
# Score 0-100% berdasarkan 4 faktor:

Factor 1: Progress ke TP (0-40 poin)
  progress = (current_profit / target_tp_profit) * 100
  -> Makin dekat ke TP = skor tinggi

Factor 2: Momentum (0-30 poin)
  -> *Momentum* positif = skor tinggi

Factor 3: ML *Confidence* Trend (0-20 poin)
  -> ML *confidence* naik = skor tinggi

Factor 4: Time Penalty (0-10 poin DIKURANGI)
  -> 2 poin per jam (makin lama = makin rendah)

probability = factor1 + factor2 + factor3 - time_penalty
\`\`\`

---

## Layer 6: *Early Exit* (Profit Kecil + Reversal)

**Sumber:** \`smart_risk_manager.py\` (Lines 617-627)

\`\`\`python
# Profit $5-$15 + momentum sangat buruk + ML reversal
if 5 <= current_profit < 15:
    if momentum < -50 and ml_confidence >= 0.65:
        if ml_signal berlawanan dengan posisi:
            return True, ExitReason.TAKE_PROFIT,
                   "Early exit - reversal detected"
\`\`\`

**Logika:** Lebih baik ambil profit kecil ($5-$15) daripada menunggu profit hilang karena reversal.

---

## Prioritas Exit (Urutan Pengecekan)

\`\`\`
1. Hard TP ($40+)              <- Paling prioritas
2. Momentum TP ($25+, mom<-30)
3. Peak Protection ($30+ peak, <60%)
4. Probability TP ($20+, prob<25%)
5. Early Exit ($5-15, reversal)
6. Broker TP (harga hit level)  <- Independen dari software
\`\`\`

**Catatan:** Broker TP berjalan independen — jika harga hit TP level di broker, posisi tertutup otomatis meskipun bot offline.

---

## Contoh Skenario

### Skenario 1: TP Broker Hit

\`\`\`
Entry BUY @ $4950, TP broker @ $4976
  -> Harga naik ke $4976
  -> BROKER TP HIT -> Tutup otomatis
  -> Profit: ~$26 (0.01 lot = $2.60)
\`\`\`

### Skenario 2: Software TP Lebih Cepat

\`\`\`
Entry BUY @ $4950, TP broker @ $4990
  -> Harga naik ke $4990 (profit $40)
  -> Software: profit >= $40 -> HARD TP
  -> Tutup sebelum broker TP level
\`\`\`

### Skenario 3: *Momentum* Drop

\`\`\`
Entry BUY @ $4950
  -> Profit naik: $10 -> $20 -> $28 -> $25
  -> momentum = -35 (turun)
  -> Software: profit $25 + momentum < -30
  -> MOMENTUM TP: amankan $25
\`\`\`

### Skenario 4: *Peak Protection*

\`\`\`
Entry BUY @ $4950
  -> Profit naik: $15 -> $25 -> $35 (peak!)
  -> Profit turun: $35 -> $30 -> $22 -> $19
  -> 60% dari $35 = $21
  -> $19 < $21 -> PEAK PROTECTION: amankan $19
  (tanpa ini, profit bisa turun ke $0 atau bahkan loss)
\`\`\`

---

## Tabel Ringkasan Layer TP

| Layer | Trigger | Profit Min | Kondisi Tambahan |
|-------|---------|-----------|------------------|
| **1. Broker TP** | Harga hit level | - | Otomatis, independen |
| **2. *Hard TP*** | profit >= $40 | $40 | Tidak ada |
| **3. *Momentum* TP** | profit >= $25 | $25 | *momentum* < -30 |
| **4. *Peak Protection*** | peak > $30 | ~$18+ | current < 60% peak |
| **5. *Probability* TP** | profit >= $20 | $20 | TP *probability* < 25% |
| **6. *Early Exit*** | profit $5-15 | $5 | ML reversal + *momentum* < -50 |
`,
  },
  {
    slug: "entry-trade",
    title: "Entry Trade",
    category: "Proses Trading",
    icon: "ArrowRightCircle",
    description: "14 filter entry dan logika eksekusi perdagangan — dari sinyal hingga order",
    content: `# *Entry Trade* — Proses Masuk Posisi

> **File utama:** \`main_live.py\`
> **File pendukung:** \`src/smc_polars.py\`, \`src/ml_model.py\`, \`src/smart_risk_manager.py\`, \`src/session_filter.py\`

---

## Apa Itu *Entry Trade*?

*Entry Trade* adalah keseluruhan proses dari **mendeteksi peluang** hingga **mengirim *order* ke *broker***. Bot menggunakan **14 filter** yang harus **SEMUA lolos** sebelum satu *trade* dieksekusi.

**Analogi:** *Entry Trade* seperti **proses *boarding* pesawat** — harus punya tiket (*signal*), *passport* valid (*confirmation*), lulus *security check* (risiko), tepat waktu (sesi), dan *gate* terbuka (*position limit*).

---

## Daftar *Checklist Entry* (Semua Harus PASS)

| # | Filter | Keterangan | Status |
|---|--------|------------|--------|
| 1 | ***Flash Crash Guard*** | Apakah ada pergerakan harga ekstrem? | **Aktif** |
| 2 | ***Regime Filter*** | Apakah *regime* HMM bukan SLEEP? | **Aktif** |
| 3 | ***Risk Check*** | Apakah \`risk_metrics.can_trade\` = \`true\`? | **Aktif** |
| 4 | ***Session Filter*** | Apakah sesi perdagangan mengizinkan *trading*? | **Aktif** |
| 5 | ***SMC Signal*** | Apakah ada sinyal valid dari SMC *Analyzer*? | **Aktif** |
| 6 | ***Signal Combination*** | Apakah kombinasi SMC + ML menghasilkan sinyal akhir? | **Aktif** |
| 7 | **H1 *Bias* (#31B)** | Apakah *bias* H1 EMA20 sejalan dengan sinyal? | **Aktif** |
| 8 | **Filter Waktu (#34A)** | Apakah bukan jam 9 atau 21 WIB? | **Aktif** |
| 9 | ***Trade Cooldown*** | Sudah 5 menit sejak *trade* terakhir? | **Aktif** |
| 10 | ***Pullback Filter*** | Apakah bukan sedang *pullback/retrace*? | **Nonaktif** |
| 11 | ***Smart Risk Gate*** | Mode *trading* bukan STOPPED/COOLDOWN? | **Aktif** |
| 12 | **Kalkulasi *Lot*** | Apakah *lot size* > 0 setelah semua *adjustment*? | **Aktif** |
| 13 | ***Spread* Validasi** | Apakah *spread* tidak terlalu lebar? | **Aktif** |
| 14 | **Batas Posisi** | Posisi terbuka < 2? | **Aktif** |

> **Semua PASS** → Eksekusi *Trade*
> **Satu GAGAL** → *Skip*, tunggu *loop* berikutnya

---

## *Step-by-Step Flow*

### Filter 1: *Flash Crash Guard*

\`\`\`python
# main_live.py
is_flash, move_pct = self.flash_crash.detect(df.tail(5))
if is_flash:
    return  # Pergerakan harga ekstrem terdeteksi
\`\`\`

**Bisa *block*:** Pergerakan harga > 2.5% dalam 1 menit (*flash crash threshold* dari \`config.py\`).

---

### Filter 2: *Regime Filter*

\`\`\`python
regime_sleep = regime_state and regime_state.recommendation == "SLEEP"
if regime_sleep:
    return  # HMM mendeteksi kondisi krisis
\`\`\`

**Bisa *block*:** *Regime* HIGH_VOLATILITY / CRISIS — pasar terlalu bergejolak.

---

### Filter 3: *Risk Check*

\`\`\`python
if not risk_metrics.can_trade:
    return  # Risiko di luar batas
\`\`\`

---

### Filter 4: *Session Filter*

\`\`\`python
session_ok, session_reason, session_multiplier = self.session_filter.can_trade()
if not session_ok:
    return  # Bukan waktu trading
\`\`\`

**Bisa *block*:** *Weekend*, Jumat > 23:00, zona bahaya (00:00-06:00), sesi *low volatility*.
**Tokyo-London *overlap*** (15:00-16:00 WIB) **diblokir** — hasil optimasi *backtest* #24B.

---

### Filter 5: *SMC Signal*

\`\`\`python
smc_signal = self.smc.generate_signal(df)
if smc_signal is None:
    return  # Tidak ada setup SMC yang valid
\`\`\`

**SMC membutuhkan:**
- Struktur pasar (*bullish/bearish*) ATAU BOS/CHoCH
- DAN (FVG ATAU *Order Block*)
- Minimum 2:1 *risk/reward*

**Output:** *Entry price*, SL, TP, *confidence* (55-85%), alasan.

---

### Filter 6: *Signal Combination*

\`\`\`python
final_signal = self._combine_signals(smc_signal, ml_prediction, regime_state)
if final_signal is None:
    return  # Sinyal terfilter
\`\`\`

Menggabungkan **SMC + ML + *Regime*** menjadi satu sinyal akhir. ML harus *agree* atau minimal tidak *strongly disagree* (> 65% *confidence* berlawanan).

---

### Filter 7: H1 *Bias* (#31B)

\`\`\`python
# Backtest #31B: H1 EMA20 filter menambah +$345 profit
if h1_bias == "BULLISH" and final_signal.signal_type == "SELL":
    return  # BUY signal vs H1 bullish = blokir
if h1_bias == "BEARISH" and final_signal.signal_type == "BUY":
    return  # SELL signal vs H1 bearish = blokir
if h1_bias == "NEUTRAL":
    return  # Tidak ada bias jelas = blokir
\`\`\`

**Tujuan:** Hanya masuk posisi yang sejalan dengan *trend* H1.

---

### Filter 8: Filter Waktu (#34A)

\`\`\`python
# Backtest #34A: skip jam 9 dan 21 WIB menambah +$356 profit
wib_hour = datetime.now(ZoneInfo("Asia/Jakarta")).hour
if wib_hour in (9, 21):
    return  # Jam transisi — volatilitas tidak optimal
\`\`\`

**Tujuan:** Menghindari jam transisi sesi yang berpotensi *whipsaw*.

---

### Filter 9: *Trade Cooldown*

\`\`\`python
trade_cooldown = 300  # 5 menit
if last_trade_time and (now - last_trade_time).total_seconds() < 300:
    return  # Tunggu cooldown selesai
\`\`\`

**Tujuan:** Mencegah *overtrading* — minimal 5 menit antar *trade*.

---

### Filter 10: *Pullback Filter* (NONAKTIF)

\`\`\`python
# DISABLED — mode SMC-only
# Struktur SMC sudah memvalidasi zona entry
\`\`\`

> Filter ini dinonaktifkan karena analisis SMC sudah mencakup validasi *pullback* dalam logika *Order Block* dan FVG.

---

### Filter 11: *Smart Risk Gate*

\`\`\`python
risk_rec = self.smart_risk.get_trading_recommendation()
if not risk_rec["can_trade"]:
    return  # Mode STOPPED/COOLDOWN
\`\`\`

**4 mode** *Smart Risk*: NORMAL → PROTECTED → RECOVERY → COOLDOWN/STOPPED.

---

### Filter 12-14: *Lot*, *Spread*, dan Batas Posisi

\`\`\`python
# Kalkulasi lot
safe_lot = self.smart_risk.calculate_lot_size(...)
safe_lot = max(0.01, safe_lot * session_multiplier)
if safe_lot <= 0:
    return  # Lot 0 = tidak boleh trade

# Validasi spread
if spread > max_allowed:
    return  # Spread terlalu lebar

# Batas posisi (max 2)
can_open, limit_reason = self.smart_risk.can_open_position()
if not can_open:
    return  # Sudah 2 posisi terbuka
\`\`\`

---

## Eksekusi *Order*

Setelah semua **14 filter** lolos:

\`\`\`python
# Step A: Ambil harga real-time
tick = mt5.get_tick(symbol)
current_price = tick.ask if BUY else tick.bid

# Step B: Validasi broker SL (min 10 pips)
broker_sl = signal.stop_loss
if jarak_terlalu_dekat:
    broker_sl = paksa_lebih_lebar

# Step C: Kirim order
result = mt5.send_order(
    symbol="XAUUSD",
    order_type="BUY" / "SELL",
    volume=0.01 - 0.05,
    sl=broker_sl,             # SL berbasis ATR
    tp=signal.take_profit,    # TP dari SMC (ATR-capped)
    magic=123456,
    comment="AI Safe v3",
)

# Step D: Fallback jika broker reject SL
if gagal dan error 10016:
    result = mt5.send_order(sl=0, ...)

# Step E: Validasi slippage
if result.success:
    slippage = abs(result.price - signal.entry_price)
    max_slippage = signal.entry_price * 0.0015  # 0.15%
    if slippage > max_slippage:
        log WARNING "HIGH SLIPPAGE"

# Step F: Register posisi (gunakan nilai AKTUAL)
    smart_risk.register_position(
        ticket=result.order_id,
        entry_price=result.price,      # Harga aktual
        lot_size=result.volume,        # Volume aktual
        direction=signal.signal_type,
    )
\`\`\`

---

## *Post-Entry*

\`\`\`python
# Log trade detail ke PostgreSQL
trade_logger.log_trade_open(signal, ml_prediction, regime, market_quality, ...)

# Kirim notifikasi Telegram
await telegram.send_trade_open(trade_info)

# Update cooldown timer
last_trade_time = now
\`\`\`

---

## Diagram *Flow* Lengkap

\`\`\`mermaid
graph TD
    A["Loop Setiap ~30 Detik"] --> B["Fetch 200 Bar M15"]
    B --> C["Feature Eng + SMC + HMM + XGBoost"]
    C --> F1{"1. Flash Crash?"}
    F1 -->|Ya| SKIP["Skip ↩"]
    F1 -->|Tidak| F2{"2. Regime SLEEP?"}
    F2 -->|Ya| SKIP
    F2 -->|Tidak| F3{"3. Risk OK?"}
    F3 -->|Tidak| SKIP
    F3 -->|Ya| F4{"4. Session OK?"}
    F4 -->|Tidak| SKIP
    F4 -->|Ya| F5{"5. SMC Signal?"}
    F5 -->|Tidak| SKIP
    F5 -->|Ya| F6{"6. Signal Combo?"}
    F6 -->|Tidak| SKIP
    F6 -->|Ya| F7{"7. H1 Bias OK?"}
    F7 -->|Tidak| SKIP
    F7 -->|Ya| F8{"8. Jam OK?"}
    F8 -->|Tidak| SKIP
    F8 -->|Ya| F9{"9. Cooldown OK?"}
    F9 -->|Tidak| SKIP
    F9 -->|Ya| F11{"10. Risk Gate?"}
    F11 -->|Tidak| SKIP
    F11 -->|Ya| F12{"11-14. Lot/Spread/Pos?"}
    F12 -->|Tidak| SKIP
    F12 -->|Ya| EXEC["EKSEKUSI TRADE"]
    EXEC --> POST["Register + Log + Telegram"]
\`\`\`

---

## Statistik Filter

Dalam kondisi normal, dari ratusan *loop* per jam:

| Sumber *Block* | Persentase | Keterangan |
|-----------------|-----------|------------|
| Tidak ada sinyal SMC | **~95%** | Pasar *sideways*, tidak ada *setup* |
| ML *disagreement* / *low confidence* | **~3%** | ML tidak yakin atau berlawanan |
| *Pullback*, sesi, H1 *bias* | **~1%** | Filter waktu dan arah |
| **Lolos semua → *Trade*** | **< 1%** | Sangat selektif |

**Rata-rata:** 3-8 *trade* per hari.
`,
  },
  {
    slug: "exit-trade",
    title: "Exit Trade",
    category: "Proses Trading",
    icon: "ArrowLeftCircle",
    description: "12 kondisi exit termasuk trailing SL, batas waktu, dan perubahan regime",
    content: `# *Exit Trade* — Proses Keluar Posisi

> **File utama:** \`main_live.py\`, \`src/smart_risk_manager.py\`
> **File pendukung:** \`src/position_manager.py\`

---

## Apa Itu *Exit Trade*?

*Exit Trade* adalah keseluruhan proses **monitoring posisi terbuka** dan **memutuskan kapan menutup**. Bot memeriksa setiap posisi terbuka **setiap ~10 detik** (di antara *candle*) atau **setiap *candle* baru** (*full analysis*) dengan **12 kondisi *exit*** berbeda.

**Prinsip utama:** **Jangan biarkan *winner* menjadi *loser***, tapi juga **jangan potong *winner* terlalu cepat**.

---

## 12 Kondisi *Exit*

\`\`\`mermaid
graph TD
    POS["Posisi Terbuka"] --> C1{"1. Smart TP<br/>Profit ≥ $15?"}
    C1 -->|Ya & kondisi| CLOSE["TUTUP POSISI"]
    C1 -->|Tidak| C2{"2. Early Exit<br/>Profit $5-15?"}
    C2 -->|Ya & reversal| CLOSE
    C2 -->|Tidak| C3{"3. Early Cut<br/>Loss + momentum?"}
    C3 -->|Ya| CLOSE
    C3 -->|Tidak| C4{"4. Trend Reversal<br/>ML sinyal balik?"}
    C4 -->|Ya| CLOSE
    C4 -->|Tidak| C5{"5. Max Loss<br/>Loss > 50% max?"}
    C5 -->|Ya| CLOSE
    C5 -->|Tidak| C6{"6. Stall<br/>Stuck + rugi?"}
    C6 -->|Ya| CLOSE
    C6 -->|Tidak| C7{"7. Daily Limit<br/>Batas harian?"}
    C7 -->|Ya| CLOSE
    C7 -->|Tidak| C8{"8. Weekend Close"}
    C8 -->|Ya| CLOSE
    C8 -->|Tidak| C9{"9. Time-Based<br/>4-8 jam?"}
    C9 -->|Ya & kondisi| CLOSE
    C9 -->|Tidak| C10{"10-12. Trailing SL<br/>Breakeven, dll"}
    C10 -->|Ya| CLOSE
    C10 -->|Tidak| HOLD["TAHAN POSISI ↩"]
\`\`\`

---

### CHECK 1: *Smart Take Profit* (Profit ≥ $15)

| Kondisi | Aksi | Keterangan |
|---------|------|------------|
| Profit ≥ **$40** | **Tutup** langsung | Target tercapai — *hard TP* |
| Profit ≥ **$25** dan momentum < -30 | **Tutup** | Profit bagus tapi momentum turun |
| *Peak profit* > $30, profit turun ke < 60% *peak* | **Tutup** | Lindungi profit dari *peak* |
| Probabilitas TP < 25%, profit ≥ **$20** | **Tutup** | Kemungkinan TP rendah |
| Momentum ≥ 0 | **Tahan** | Masih bagus, biarkan berjalan |

---

### CHECK 2: *Smart Early Exit* (Profit $5-15)

\`\`\`python
if 5 <= current_profit < 15:
    if momentum < -50 and ml_confidence >= 0.65:
        if ml_signal berlawanan dengan arah posisi:
            TUTUP  # Sinyal reversal kuat + profit kecil
\`\`\`

**Tujuan:** Ambil profit kecil jika momentum sangat negatif DAN ML yakin *trend* berbalik.

---

### CHECK 3: *Early Cut* (Loss + Momentum Negatif)

\`\`\`python
if current_profit < 0:
    loss_pct = abs(current_profit) / max_loss_per_trade * 100
    if momentum < -50 and loss_pct >= 30:
        TUTUP  # Potong kerugian sebelum makin besar
\`\`\`

**Perubahan dari v1:** *Smart Hold* (menahan posisi rugi menunggu *golden time*) **sudah dihapus** — dianggap berbahaya dan melawan prinsip manajemen risiko yang benar.

---

### CHECK 4: *Trend Reversal* (Sinyal ML Berbalik)

| Kondisi | Aksi |
|---------|------|
| ML sinyal **berbalik** + *confidence* ≥ 75% + loss > 40% max + loss > $8 | **Tutup** |
| **3x** peringatan *reversal* berturut-turut + loss > $10 | **Tutup** |

**Perubahan:** *Threshold* diturunkan dari 5x ke **3x** peringatan, dan batas loss dari 60% ke **40%** — lebih responsif.

---

### CHECK 5: *Maximum Loss Per Trade*

\`\`\`python
if current_profit <= -(max_loss_per_trade * 0.50):
    TUTUP  # 50% dari batas max — tanpa pengecualian
\`\`\`

Untuk akun *small* ($5.000): max loss per *trade* = ~$50, *trigger* di $25.

---

### CHECK 6: *Stall Detection*

\`\`\`python
if len(profit_history) >= 10:
    recent_range = max(last_10) - min(last_10)
    if recent_range < $3 and current_profit < -$15:
        stall_count += 1
        if stall_count >= 5:
            TUTUP  # Harga stuck, posisi rugi
\`\`\`

**Tujuan:** Deteksi posisi yang "terjebak" — harga tidak bergerak tapi posisi rugi.

---

### CHECK 7: Batas Kerugian Harian

\`\`\`python
potensi_loss = daily_loss + abs(current_profit)
if potensi_loss >= max_daily_loss:
    TUTUP  # Akan melebihi batas kerugian harian
\`\`\`

---

### CHECK 8: *Weekend Close*

\`\`\`python
# Sabtu 04:30+ WIB (30 menit sebelum market tutup)
if near_weekend_close:
    if profit > 0:
        TUTUP  # Amankan profit
    elif loss > -$10:
        TUTUP  # Loss kecil — hindari gap weekend
\`\`\`

**Tujuan:** Hindari risiko *gap weekend* — posisi tanpa proteksi selama 2 hari.

---

### CHECK 9: *Smart Time-Based Exit*

| Durasi | Kondisi | Aksi |
|--------|---------|------|
| **4+ jam** | Profit < $5, momentum tidak tumbuh | **Tutup** — posisi *stuck* |
| **4+ jam** | Profit ≥ $5, momentum positif, ML sejalan | **Tahan** — perpanjang waktu |
| **6+ jam** | Profit < $10 ATAU momentum negatif | **Tutup** — terlalu lama |
| **8+ jam** | Apapun kondisinya | **Tutup** — batas waktu absolut |

**Perubahan:** Tidak lagi memotong *winner* secara paksa — posisi yang masih tumbuh bisa diperpanjang.

---

### CHECK 10-12: *Position Manager* (Tambahan)

Selain 9 kondisi di atas dari \`SmartRiskManager\`, \`SmartPositionManager\` juga menjalankan:

| # | Kondisi | Keterangan |
|---|---------|------------|
| 10 | ***Trailing Stop Loss*** | SL mengikuti harga naik (berbasis ATR) — \`atr_trail_start_mult=4.0\`, \`atr_trail_step_mult=3.0\` |
| 11 | ***Breakeven Protection*** | Pindahkan SL ke titik *entry* setelah profit ≥ 30 *pips* |
| 12 | **Proteksi *Drawdown*** | Tutup jika *drawdown* dari *peak* terlalu besar |

---

## Alasan *Exit* (*ExitReason Enum*)

| Kode | Deskripsi |
|------|-----------|
| \`TAKE_PROFIT\` | Target profit tercapai atau profit diamankan |
| \`TREND_REVERSAL\` | ML mendeteksi pembalikan *trend* |
| \`DAILY_LIMIT\` | Batas kerugian harian tercapai |
| \`POSITION_LIMIT\` | Batas kerugian per posisi tercapai (S/L) |
| \`TOTAL_LIMIT\` | Batas kerugian total tercapai |
| \`WEEKEND_CLOSE\` | Mendekati penutupan *weekend* |
| \`MANUAL\` | Penutupan manual |

---

## Diagram *Flow* Evaluasi Posisi

\`\`\`mermaid
graph LR
    A["Setiap ~10 detik"] --> B["Hitung Profit/Loss<br/>Momentum, TP Prob"]
    B --> C["SmartRiskManager<br/>evaluate_position()"]
    C --> D{"Harus<br/>Tutup?"}
    D -->|Ya| E["Tutup via MT5<br/>+ Log + Telegram"]
    D -->|Tidak| F["SmartPositionManager<br/>Trailing SL, Breakeven"]
    F --> G{"SL Perlu<br/>Digeser?"}
    G -->|Ya| H["Modify SL<br/>via MT5"]
    G -->|Tidak| I["Tahan Posisi ↩"]
\`\`\`

---

## Statistik *Exit*

Berdasarkan data *backtest* (Jan 2025 - Feb 2026):

| Alasan *Exit* | Persentase | Rata-rata P/L |
|----------------|-----------|----------------|
| *Take Profit* (semua jenis) | **~40%** | **+$18.50** |
| *Trend Reversal* / *Early Cut* | **~25%** | **-$12.30** |
| *Time-Based Exit* | **~15%** | **+$3.20** |
| *Trailing SL* hit | **~10%** | **+$8.70** |
| Max Loss / *Daily Limit* | **~8%** | **-$22.50** |
| *Weekend Close* | **~2%** | **+$5.10** |
`,
  },
  {
    slug: "news-agent",
    title: "News Agent",
    category: "Infrastruktur",
    icon: "Newspaper",
    description: "Filter berita ekonomi dan penilaian dampak — saat ini nonaktif",
    content: `# *News Agent* — Monitoring Berita Ekonomi

> **File:** \`src/news_agent.py\`
> **Class:** \`NewsAgent\`
> **Status: NONAKTIF** — dikomentari di \`main_live.py\` baris 64

---

## Status Saat Ini

> **PENTING:** Modul *News Agent* saat ini **tidak aktif** dalam sistem *live*. *Import* dikomentari di \`main_live.py\`:
>
> \`\`\`python
> # from src.news_agent import NewsAgent, create_news_agent, MarketCondition  # DISABLED
> \`\`\`
>
> Modul ini tersedia dalam *codebase* untuk aktivasi di masa mendatang.

---

## Apa Itu *News Agent*?

*News Agent* adalah modul yang **memonitor berita ekonomi** berdampak tinggi dan menilai potensi dampaknya terhadap perdagangan XAUUSD. Ketika aktif, modul ini dapat:

1. **Mengambil kalender ekonomi** dari sumber eksternal
2. **Menilai dampak berita** terhadap pasar emas
3. **Memberi peringatan** atau **memblokir *trading*** saat berita berdampak tinggi

---

## *Design* Modul

\`\`\`mermaid
graph TD
    A["Sumber Berita<br/>API Kalender Ekonomi"] --> B["NewsAgent"]
    B --> C{"Dampak?"}
    C -->|"Rendah"| D["✅ Trading normal"]
    C -->|"Sedang"| E["⚠️ Kurangi lot"]
    C -->|"Tinggi"| F["🛑 Blokir trading"]
    F --> G["NFP, FOMC, CPI<br/>Buffer 15-45 menit"]
\`\`\`

---

## Berita Berdampak Tinggi (Tersimpan di \`session_filter.py\`)

Meskipun *News Agent* nonaktif, daftar waktu berita sudah tersimpan di *Session Filter*:

| Berita | Jam WIB | *Buffer* Sebelum | *Buffer* Setelah |
|--------|---------|------------------|------------------|
| **NFP** (*Non-Farm Payrolls*) | 19:30 | 15 menit | 30 menit |
| **FOMC** (*Federal Reserve*) | 01:00 | 15 menit | 45 menit |
| **CPI** (*Consumer Price Index*) | 19:30 | 15 menit | 30 menit |

---

## Rencana Aktivasi

Langkah-langkah untuk mengaktifkan kembali *News Agent*:

1. Hapus komentar di \`main_live.py\` baris 64
2. Konfigurasi sumber data berita di \`.env\`
3. Integrasikan pengecekan berita ke *entry filter pipeline*
4. Uji coba dengan mode *monitoring only* (tidak memblokir, hanya *log*)
5. Aktifkan *blocking* setelah validasi

---

## Mengapa Dinonaktifkan?

- **Ketergantungan API eksternal** — memerlukan *API key* dan koneksi internet stabil
- **Latensi tambahan** — setiap pengecekan berita menambah waktu *loop*
- **Hasil *backtest* tanpa *News Agent* sudah baik** — sistem sudah terproteksi oleh *session filter* dan *regime detector*
`,
  },
  {
    slug: "telegram",
    title: "Notifikasi Telegram",
    category: "Infrastruktur",
    icon: "Send",
    description: "Notifikasi trade real-time dan ringkasan harian via Telegram Bot",
    content: `# *Telegram Notifications* — Sistem Notifikasi *Real-Time*

> **File:** \`src/telegram_notifier.py\`
> **Class:** \`TelegramNotifier\`
> **API:** Telegram Bot API (*async* via aiohttp)

---

## Arsitektur Notifikasi

\`\`\`mermaid
flowchart LR
    A["Event\\n(Trade / Alert / Timer)"] --> B["TelegramNotifier\\n(async aiohttp)"]
    B --> C["Telegram Bot API\\n(/sendMessage\\n/sendPhoto\\n/sendDocument)"]
    C --> D["User / Grup Telegram"]

    style A fill:#2d2d2d,stroke:#f5a623,color:#fff
    style B fill:#2d2d2d,stroke:#4a9eff,color:#fff
    style C fill:#2d2d2d,stroke:#50c878,color:#fff
    style D fill:#2d2d2d,stroke:#ff6b6b,color:#fff
\`\`\`

\`\`\`mermaid
flowchart TD
    LOOP["Main Loop\\n(setiap 1 detik)"] --> NEW_DAY{"New day?"}
    NEW_DAY -- Ya --> DAILY["Daily Summary + Reset"]
    NEW_DAY -- Tidak --> HOURLY{"Hourly timer?"}
    HOURLY -- Ya --> HOUR_MSG["Hourly Analysis"]
    HOURLY -- Tidak --> HALF{"30-min timer?"}
    HALF -- Ya --> MARKET["Market Update"]
    HALF -- Tidak --> TRADE{"Trade executed?"}
    TRADE -- Ya --> OPEN["Trade Open Notification"]
    TRADE -- Tidak --> CLOSE{"Position closed?"}
    CLOSE -- Ya --> CLOSE_MSG["Trade Close Notification"]
    CLOSE -- Tidak --> LIMIT{"Limit hit?"}
    LIMIT -- Ya --> CRIT["Critical Limit Alert"]
    LIMIT -- Tidak --> FLASH{"Flash crash?"}
    FLASH -- Ya --> EMERG["Emergency Close Alert"]
    FLASH -- Tidak --> LOOP

    style LOOP fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style DAILY fill:#1a1a2e,stroke:#50c878,color:#fff
    style HOUR_MSG fill:#1a1a2e,stroke:#50c878,color:#fff
    style MARKET fill:#1a1a2e,stroke:#50c878,color:#fff
    style OPEN fill:#1a1a2e,stroke:#f5a623,color:#fff
    style CLOSE_MSG fill:#1a1a2e,stroke:#f5a623,color:#fff
    style CRIT fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style EMERG fill:#1a1a2e,stroke:#ff6b6b,color:#fff
\`\`\`

---

## Apa Itu *Telegram Notifications*?

*Telegram Notifications* mengirimkan **laporan *real-time*** ke grup Telegram setiap kali terjadi event penting — trade dibuka/ditutup, laporan harian, alert darurat, dan status sistem.

**Analogi:** *Telegram Notifications* seperti **dashboard pilot di cockpit** — menampilkan semua informasi penting secara *real-time* tanpa harus melihat layar trading.

---

## Konfigurasi

\`\`\`
Bot Token:  Dari environment variable TELEGRAM_BOT_TOKEN
Chat ID:    Dari environment variable TELEGRAM_CHAT_ID
Format:     HTML (parse_mode)
Transport:  Async HTTP POST via aiohttp
Timezone:   WIB (Asia/Jakarta)
\`\`\`

\`\`\`python
# Inisialisasi
from dotenv import load_dotenv
load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
enabled = bool(bot_token and chat_id)  # Auto-disable jika tidak dikonfigurasi
\`\`\`

---

## 11 Tipe Notifikasi

| # | Tipe | Trigger | Frekuensi |
|---|------|---------|-----------|
| 1 | *Trade Open* | Order berhasil dieksekusi | Per trade |
| 2 | *Trade Close* | Posisi ditutup | Per trade |
| 3 | *Market Update* | Timer 30 menit | Setiap 30 menit |
| 4 | *Hourly Analysis* | Timer 1 jam | Setiap 1 jam |
| 5 | *Daily Summary* | Pergantian hari | 1x per hari |
| 6 | *Startup* | Bot dinyalakan | 1x per sesi |
| 7 | *Shutdown* | Bot dimatikan | 1x per sesi |
| 8 | *News Alert* | Event ekonomi terdeteksi | Per event |
| 9 | *Critical Limit* | Daily/total loss limit | Per event |
| 10 | *Emergency Close* | *Flash crash* / darurat | Per event |
| 11 | *System Status* | Status berkala | Per request |

---

## Format Pesan

### 1. *Trade Open*

\`\`\`
🟢 LONG #123456
├ XAUUSD
├ Entry: 4950.00
├ Lot: 0.02
├ SL: 4937.00 (-$13)
├ TP: 4976.00 (+$26)
├ R:R: 1:2.0
├ AI: 75% | medium_volatility
└ SMC Bullish BOS + FVG
⏰ 14:35 WIB
\`\`\`

| Elemen | Arti |
|--------|------|
| 🟢/🔴 | BUY (hijau) / SELL (merah) |
| LONG/SHORT | Arah posisi |
| #123456 | Ticket ID dari broker |
| R:R | *Risk to Reward ratio* |
| AI: 75% | ML *confidence* |
| medium_volatility | HMM *regime* |

---

### 2. *Trade Close*

\`\`\`
✅ WIN #123456
├ XAUUSD BUY
├ Entry: 4950.00
├ Exit: 4965.00
├ Lot: 0.02
├ P/L: +$30.00 (+0.49%)
├ Pips: +150.0
├ Duration: 2m
├ Bal Before: $6130.00
└ Bal After: $6160.00
⏰ 14:40 WIB
\`\`\`

| Emoji | Arti |
|-------|------|
| ✅ | WIN (profit) |
| ❌ | LOSS (rugi) |
| ➖ | BREAKEVEN (impas) |

---

### 3. *Market Update* (Setiap 30 Menit)

\`\`\`
📊 XAUUSD $4965.00
├ 🟢 BUY 75%
├ UPTREND
├ medium_volatility
├ London-NY Overlap
└ ✅
⏰ 14:45
\`\`\`

---

### 4. *Hourly Analysis* (Setiap 1 Jam)

\`\`\`
📊 HOURLY 14:00 WIB

Account
├ Bal: $5,094.68
├ Eq: $5,120.50
├ Float: +$25.82
└ Day: +$150.00 (12 trades)

Positions (2)
├ #123456 BUY: +$30.00 M:+45
└ #123457 SELL: -$15.00 M:-20

Market
├ XAUUSD $4,965.00
├ London-NY Overlap
└ medium_volatility | high

AI Signal
├ BUY 75% / thresh 70%
└ Quality: EXCELLENT (score:85) → READY

Risk NORMAL
└ Daily Loss: $0.00 / $148.34

✅ News: SAFE
\`\`\`

---

### 5. *Daily Summary*

\`\`\`
🎉 DAILY REPORT 2025-02-06

Result
├ P/L: +$150.00 (+3.03%)
├ Gross Win: +$500.00
├ Gross Loss: -$350.00
├ Bal Start: $4,944.68
└ Bal End: $5,094.68

Stats
├ Total: 12 trades
├ Wins: 8 | Losses: 4
├ Win Rate: 66.7%
├ Profit Factor: 1.43
└ Avg/Trade: $12.50

Recent Trades
├ ✅ BUY: +$30.00
├ ❌ SELL: -$25.00
├ ✅ BUY: +$45.00
├ ➖ SELL: $0.00
└ ✅ BUY: +$100.00
\`\`\`

| Emoji Hari | Arti |
|-----------|------|
| 🎉 | Hari profit |
| 📉 | Hari loss |
| ➖ | Hari breakeven |

---

### 6. *Startup*

\`\`\`
🚀 BOT STARTED

Config
├ Symbol: XAUUSD
├ Mode: small
├ Capital: $5,000.00
├ Balance: $4,944.68
└ ML: Loaded (37 features)

Risk Settings
├ Risk/Trade: 1%
├ Max Daily Loss: 5%
├ Max Total Loss: 10%
└ SL: Smart (ATR-based)

✅ News: SAFE
⏰ 2025-02-06 08:15 WIB
\`\`\`

---

### 7. *Shutdown*

\`\`\`
🔴 BOT STOPPED

Session Summary
├ Balance: $5,094.68
├ Total Trades: 12
├ ✅ P/L: +$150.00
└ Uptime: 8.5h

⏰ 2025-02-06 16:45 WIB
\`\`\`

---

### 8. *News Alert*

\`\`\`
🚨 NEWS DANGER_NEWS
├ NFP (Non-Farm Payroll) - HIGH IMPACT
├ High volatility expected during release
└ Buffer: 60m
⏰ 20:25
\`\`\`

| Emoji | Kondisi |
|-------|---------|
| 🚨 | DANGER_NEWS |
| ⚠️ | CAUTION / DANGER_SENTIMENT |
| ✅ | SAFE |

---

### 9. *Critical Limit Alert*

\`\`\`
🚨 DAILY LOSS LIMIT REACHED 🚨

Daily Loss: $250.00
Limit: $250.00 (5%)

⛔ TRADING STOPPED FOR TODAY
Will resume tomorrow automatically.
\`\`\`

---

### 10. *Emergency Close*

\`\`\`
🚨 EMERGENCY CLOSE COMPLETE

Closed 3 positions due to flash crash detection
Total P/L: -$45.00
\`\`\`

---

### 11. Alert (Berbagai Tipe)

| Alert Type | Emoji | Contoh |
|-----------|-------|--------|
| *flash_crash* | 🚨 | "Flash crash detected on XAUUSD" |
| *high_volatility* | ⚡ | "Volatility spike detected" |
| *connection_error* | 📡 | "MT5 connection lost" |
| *model_retrain* | 🔄 | "ML model retrained successfully" |
| *market_close* | 🔔 | "Market closing in 30 minutes" |
| *low_balance* | 💰 | "Account balance below threshold" |

---

## 3 Metode Pengiriman

\`\`\`mermaid
flowchart LR
    N["TelegramNotifier"] --> SM["send_message()\\n/sendMessage\\nTeks biasa"]
    N --> SP["send_photo()\\n/sendPhoto\\nChart / grafik"]
    N --> SD["send_document()\\n/sendDocument\\nFile PDF"]

    style N fill:#2d2d2d,stroke:#4a9eff,color:#fff
    style SM fill:#2d2d2d,stroke:#50c878,color:#fff
    style SP fill:#2d2d2d,stroke:#f5a623,color:#fff
    style SD fill:#2d2d2d,stroke:#ff6b6b,color:#fff
\`\`\`

| Metode | Endpoint | Kegunaan |
|--------|----------|---------|
| \`send_message()\` | \`/sendMessage\` | Teks biasa (semua notifikasi) |
| \`send_photo()\` | \`/sendPhoto\` | Chart/grafik (*daily report*) |
| \`send_document()\` | \`/sendDocument\` | File PDF (laporan detail) |

---

## *Error Handling*

\`\`\`mermaid
flowchart TD
    SEND["send_message() / send_photo()"] --> TRY{"Try-Except"}
    TRY -- Berhasil --> CHECK{"HTTP Status\\n== 200?"}
    CHECK -- Ya --> OK["Return True\\n(Terkirim)"]
    CHECK -- Tidak --> LOG_ERR["Log Error\\nReturn False"]
    TRY -- Exception --> LOG_WARN["Log Warning\\nLanjut Trading"]

    DISABLED{"Token / ChatID\\nkosong?"} --> AUTO["Auto-disable\\nReturn True"]

    EMERG["Emergency Close"] --> CLOSE_POS["Tutup Posisi Dulu"]
    CLOSE_POS --> TRY_NOTIF{"Kirim Notifikasi"}
    TRY_NOTIF -- Gagal --> IGNORE["pass\\n(Trading > Notifikasi)"]
    TRY_NOTIF -- Berhasil --> OK2["Notifikasi Terkirim"]

    style SEND fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style OK fill:#1a1a2e,stroke:#50c878,color:#fff
    style OK2 fill:#1a1a2e,stroke:#50c878,color:#fff
    style LOG_ERR fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style LOG_WARN fill:#1a1a2e,stroke:#f5a623,color:#fff
    style IGNORE fill:#1a1a2e,stroke:#f5a623,color:#fff
    style AUTO fill:#1a1a2e,stroke:#888,color:#fff
\`\`\`

\`\`\`
Strategi: GRACEFUL DEGRADATION

1. Try-Except di setiap send method
   -> Gagal kirim? Log warning, lanjut trading

2. HTTP status check
   -> Status != 200? Log error, return False

3. Emergency close
   -> Telegram gagal? TETAP tutup posisi
   -> Trading > notifikasi dalam prioritas

4. Disabled mode
   -> Token/ChatID kosong? Auto-disable, return True
   -> Bot tetap berjalan tanpa notifikasi
\`\`\`

Strategi ini menerapkan pola *graceful degradation* — kegagalan notifikasi **tidak pernah** menghentikan proses trading. Sistem *emergency close* akan tetap menutup posisi meskipun Telegram tidak responsif, menerapkan prinsip *circuit breaker* di mana komponen non-kritis diisolasi dari jalur kritis.

\`\`\`python
# Contoh: Emergency close TIDAK boleh gagal karena Telegram
try:
    await telegram.send_message("Emergency close...")
except:
    pass  # Jangan biarkan Telegram failure menghentikan close
\`\`\`

---

## *Rate Limiting*

| Notifikasi | Interval |
|-----------|----------|
| *Trade Open/Close* | Langsung (per event) |
| *Market Update* | 30 menit |
| *Hourly Analysis* | 1 jam |
| *Daily Summary* | 1x per hari |
| *Startup* / *Shutdown* | 1x per sesi |
| Min *message interval* | 1 detik (variable) |

*Rate limiting* mencegah flooding ke Telegram Bot API yang memiliki batas ~30 pesan/detik per grup. Interval minimum 1 detik antar pesan menjaga bot tetap dalam batas aman.

---

## Kapan Notifikasi Dikirim di *Main Loop*

\`\`\`
Main Loop (setiap 1 detik)
    |
    |-- Cek new day? ------> Daily Summary + Reset
    |
    |-- Cek hourly timer? -> Hourly Analysis (setiap 1 jam)
    |
    |-- Cek 30min timer? --> Market Update (setiap 30 menit)
    |
    |-- Trade executed? ---> Trade Open notification
    |
    |-- Position closed? --> Trade Close notification
    |
    |-- Limit hit? --------> Critical Limit Alert
    |
    |-- Flash crash? ------> Emergency Close Alert
    |
    |-- (startup) ---------> Startup message
    |
    |-- (shutdown) --------> Shutdown message
\`\`\`

---

## *Formatting* HTML

Semua pesan menggunakan HTML *parse mode*:

\`\`\`html
<b>Bold</b>           -> Label penting
<code>Monospace</code> -> Angka, harga, nilai
<i>Italic</i>          -> Info tambahan, alasan signal
\`\`\`

Tree structure menggunakan *box-drawing characters*:

\`\`\`
├  -> Item tengah
└  -> Item terakhir
\`\`\`
`,
  },
  {
    slug: "auto-trainer",
    title: "Auto Trainer",
    category: "Infrastruktur",
    icon: "RefreshCw",
    description: "Pipeline retraining otomatis saat kondisi pasar berubah signifikan",
    content: `# *Auto Trainer* --- Sistem *Retraining* Otomatis

> **File:** \`src/auto_trainer.py\`
> **Class:** \`AutoTrainer\`
> **Database:** PostgreSQL (opsional, fallback ke file)

---

## Apa Itu *Auto Trainer*?

*Auto Trainer* adalah sistem yang **melatih ulang model AI secara otomatis** agar tetap up-to-date dengan kondisi pasar terbaru. *Retraining* dilakukan saat market tutup (05:00 WIB) untuk menghindari gangguan saat trading aktif.

**Analogi:** *Auto Trainer* seperti **pelatih yang membuat atlet berlatih setiap malam** --- setelah pertandingan selesai, atlet (model AI) dilatih dengan data terbaru agar siap menghadapi tantangan esok hari.

---

## Jadwal *Retraining*

| Tipe | Waktu | Data | Boost Rounds | Kondisi |
|------|-------|------|-------------|---------|
| **Daily** | 05:00 WIB (market close) | 8.000 bar | 50 | Senin--Jumat |
| **Weekend** | 05:00 WIB Sabtu/Minggu | 15.000 bar | 80 | *Deep training* |
| **Emergency** | Kapan saja | 8.000 bar | 50 | *AUC* < 0.65 |
| **Initial** | Pertama kali | 8.000 bar | 50 | Belum pernah training |

\`\`\`
Visualisasi Jadwal (WIB):

Sen  Sel  Rab  Kam  Jum  Sab  Min
 |    |    |    |    |    |    |
05:00 05:00 05:00 05:00 05:00 05:00 05:00
Daily Daily Daily Daily Daily DEEP  DEEP
8K    8K    8K    8K    8K    15K   15K
\`\`\`

---

## Konfigurasi

\`\`\`python
AutoTrainer(
    models_dir="models",                  # Folder simpan model
    data_dir="data",                      # Folder data training
    daily_retrain_hour_wib=5,             # Jam retrain: 05:00 WIB
    weekend_retrain=True,                 # Deep training weekend
    min_hours_between_retrain=20,         # Min 20 jam antar retrain
    backup_models=True,                   # Backup model lama
    use_db=True,                          # Simpan history ke PostgreSQL
    min_auc_threshold=0.65,               # Alert jika AUC < 0.65
    auto_retrain_on_low_auc=True,         # Auto retrain saat AUC rendah
)
\`\`\`

---

## Proses *Retraining* (Step-by-Step)

### Flowchart Keputusan *Retraining*

\`\`\`mermaid
flowchart TD
    A[Cek should_retrain] --> B{Sudah >= 20 jam\\nsejak retrain terakhir?}
    B -- Tidak --> Z[Skip retrain]
    B -- Ya --> C{Jam 05:00 WIB\\natau AUC < 0.65?}
    C -- Tidak --> Z
    C -- Ya --> D{Weekend?}
    D -- Ya --> E[Deep training:\\n15K bar, 80 rounds]
    D -- Tidak --> F[Daily training:\\n8K bar, 50 rounds]
    E --> G[Backup model lama]
    F --> G
    G --> H[Fetch data dari MT5]
    H --> I[Feature Engineering\\n+ SMC Analysis]
    I --> J[Train HMM + XGBoost]
    J --> K[Validasi AUC]
    K --> L{Test AUC >= 0.60?}
    L -- Ya --> M[Simpan model baru]
    L -- Tidak --> N[Rollback ke model lama]
    M --> O[Record hasil ke DB]
    N --> O
    O --> P[Selesai]
\`\`\`

### Detail Langkah-Langkah

\`\`\`
1. SHOULD RETRAIN CHECK
   +-- Sudah >= 20 jam sejak retrain terakhir?
   +-- Sekarang jam 05:00 WIB (+-30 menit)?
   +-- Weekend? -> Deep training (15K bar)
   +-- AUC < 0.65? -> Emergency retrain

2. BACKUP MODEL LAMA
   +-- Copy xgboost_model.pkl -> backups/YYYYMMDD_HHMMSS/
   +-- Copy hmm_regime.pkl -> backups/YYYYMMDD_HHMMSS/
   +-- Bersihkan backup lama (simpan 5 terakhir)

3. FETCH DATA TERBARU
   +-- Ambil 8K bar (daily) atau 15K bar (weekend) dari MT5
   +-- Symbol: XAUUSD, Timeframe: M15
   +-- Validasi: minimal 1000 bar

4. FEATURE ENGINEERING
   +-- FeatureEngineer.calculate_all() -> 40+ fitur
   +-- SMCAnalyzer.calculate_all() -> struktur pasar
   +-- create_target(lookahead=1) -> label UP/DOWN

5. TRAINING HMM
   +-- MarketRegimeDetector(n_regimes=3, lookback=500)
   +-- hmm.fit(df)
   +-- Save -> models/hmm_regime.pkl

6. TRAINING XGBOOST
   +-- TradingModel(confidence_threshold=0.60)
   +-- xgb.fit(train_ratio=0.7, num_boost_round=50/80)
   +-- Early stopping: 5 rounds
   +-- Save -> models/xgboost_model.pkl

7. VALIDASI
   +-- Cek Train AUC & Test AUC
   +-- Test AUC < 0.60? -> ROLLBACK ke model lama (v4: dinaikkan dari 0.52)
   +-- Test AUC < 0.65? -> WARNING (alert)
   +-- Test AUC >= 0.65? -> SUCCESS

8. RECORD HASIL
   +-- Simpan ke PostgreSQL (training_runs table)
   +-- Backup ke file (retrain_history.txt)
   +-- Log: durasi, AUC, accuracy, status
\`\`\`

---

## *Backup* & *Rollback*

### Sistem *Backup*

\`\`\`
models/
+-- xgboost_model.pkl          # Model aktif
+-- hmm_regime.pkl             # Model aktif
+-- backups/
    +-- 20250206_050015/       # Backup terbaru
    |   +-- xgboost_model.pkl
    |   +-- hmm_regime.pkl
    +-- 20250205_050012/       # Backup kemarin
    |   +-- xgboost_model.pkl
    |   +-- hmm_regime.pkl
    +-- ... (max 5 backup)
\`\`\`

### Kapan *Rollback*?

#### Diagram Validasi *AUC*

\`\`\`mermaid
flowchart TD
    A[Model baru selesai di-training] --> B[Hitung Test AUC]
    B --> C{Test AUC >= 0.65?}
    C -- Ya --> D[KEEP model baru]
    D --> D1[Status: SUCCESS]
    C -- Tidak --> E{Test AUC >= 0.60?}
    E -- Ya --> F[KEEP model baru\\ndengan WARNING]
    F --> F1[Status: WARNING]
    F1 --> F2[Akan trigger\\nemergency retrain nanti]
    E -- Tidak --> G[ROLLBACK ke model lama]
    G --> G1[Status: ROLLBACK]
    G1 --> G2[v4: threshold dinaikkan\\ndari 0.52 ke 0.60]

    style D fill:#22c55e,color:#fff
    style D1 fill:#22c55e,color:#fff
    style F fill:#eab308,color:#000
    style F1 fill:#eab308,color:#000
    style F2 fill:#eab308,color:#000
    style G fill:#ef4444,color:#fff
    style G1 fill:#ef4444,color:#fff
    style G2 fill:#ef4444,color:#fff
\`\`\`

#### Ringkasan Keputusan

| Kondisi | Aksi | Status |
|---------|------|--------|
| *AUC* >= 0.65 | KEEP model baru | SUCCESS |
| *AUC* 0.60--0.65 | KEEP tapi WARNING (akan trigger *emergency* retrain nanti) | WARNING |
| *AUC* < 0.60 | *ROLLBACK* ke model lama (v4: dinaikkan dari 0.52, karena 0.52 hampir = acak) | ROLLBACK |

### Method *Rollback*

\`\`\`python
def rollback_models(reason="Manual rollback"):
    """
    1. Ambil backup terbaru dari models/backups/
    2. Copy xgboost_model.pkl kembali ke models/
    3. Copy hmm_regime.pkl kembali ke models/
    4. Record rollback di database
    """
\`\`\`

---

## *AUC* Monitoring

### Apa Itu *AUC*?

*AUC* (*Area Under Curve*) mengukur **seberapa baik model membedakan sinyal BUY vs SELL**:

| *AUC* | Arti | Aksi |
|-----|------|------|
| 0.80+ | Sangat bagus | Model dalam kondisi prima |
| 0.65-0.80 | Bagus | Normal, lanjut trading |
| 0.60-0.65 | Minimum | Warning, pertimbangkan *retraining* |
| < 0.60 | Buruk | **ROLLBACK** + *retraining* segera (v4 threshold) |
| 0.50 | Sama dengan tebak koin | Model tidak berguna |

### Auto-Retrain on Low *AUC*

\`\`\`python
def should_retrain_due_to_low_auc():
    """
    Cek AUC saat ini:
      AUC < 0.65? -> Perlu retrain
      Tapi: sudah retrain < 4 jam lalu? -> Tunggu
      (mencegah retrain loop)
    """
\`\`\`

---

## Database Storage

### PostgreSQL (Primary)

\`\`\`
Table: training_runs
+-- id                 # Auto-increment
+-- training_type      # "daily" / "weekend"
+-- bars_used          # 8000 / 15000
+-- num_boost_rounds   # 50 / 80
+-- started_at         # Timestamp mulai
+-- completed_at       # Timestamp selesai
+-- duration_seconds   # Durasi training
+-- hmm_trained        # Boolean
+-- xgb_trained        # Boolean
+-- train_auc          # AUC di data training
+-- test_auc           # AUC di data test
+-- train_accuracy     # Akurasi training
+-- test_accuracy      # Akurasi test
+-- model_path         # Path model disimpan
+-- backup_path        # Path backup model lama
+-- success            # Boolean
+-- error_message      # Pesan error (jika gagal)
\`\`\`

### File Fallback

Jika PostgreSQL tidak tersedia:
\`\`\`
data/retrain_history.txt
+-- 2025-02-06T05:00:15+07:00
+-- 2025-02-05T05:00:12+07:00
+-- ... (append per retrain)
\`\`\`

---

## Integrasi di Main Loop

\`\`\`python
# main_live.py — dicek setiap 20 candle M15 (~5 jam)
# v4: candle-based, bukan time-based (sebelumnya: loop_count % 300)

if candle_count % 20 == 0:  # Setiap 20 candle baru
    should_train, reason = auto_trainer.should_retrain()

    if should_train:
        logger.info(f"Auto-retraining: {reason}")

        # Retrain (blocking — tapi hanya di jam 05:00 saat market tutup)
        results = auto_trainer.retrain(
            connector=mt5,
            symbol="XAUUSD",
            timeframe="M15",
            is_weekend=(now.weekday() >= 5),
        )

        if results["success"]:
            # Reload model di memory
            ml_model.load()
            regime_detector.load()
            logger.info("Models reloaded after retraining")
        else:
            logger.error(f"Retraining failed: {results['error']}")
\`\`\`

---

## Parameter Training

### Daily Training (Senin-Jumat)

| Parameter | Nilai |
|-----------|-------|
| Data | 8.000 bar M15 (~83 hari) |
| Train/Test Split | 70% / 30% |
| XGBoost Rounds | 50 |
| *Early Stopping* | 5 rounds |
| HMM Regimes | 3 |
| HMM Lookback | 500 bar |

### Weekend *Deep Training* (Sabtu-Minggu)

| Parameter | Nilai |
|-----------|-------|
| Data | 15.000 bar M15 (~156 hari) |
| Train/Test Split | 70% / 30% |
| XGBoost Rounds | 80 |
| *Early Stopping* | 5 rounds |
| HMM Regimes | 3 |
| HMM Lookback | 500 bar |

---

## Safety Guards

\`\`\`
1. MIN 20 JAM ANTAR RETRAIN
   -> Mencegah retrain terlalu sering
   -> Exception: emergency retrain (min 4 jam)

2. VALIDASI DATA MINIMUM
   -> Butuh minimal 1000 bar
   -> Kurang dari itu? Skip retrain

3. BACKUP SEBELUM RETRAIN
   -> Model lama selalu di-backup
   -> Bisa rollback kapan saja

4. AUTO-ROLLBACK
   -> AUC < 0.60? Otomatis rollback (v4: dinaikkan dari 0.52)
   -> Model buruk tidak akan dipakai

5. CLEANUP BACKUP
   -> Hanya simpan 5 backup terakhir
   -> Mencegah disk penuh

6. GRACEFUL DEGRADATION
   -> DB tidak tersedia? Pakai file
   -> Retrain gagal? Model lama tetap aktif
\`\`\`

---

## Contoh Output Log

\`\`\`
[05:00] ==================================================
[05:00] AUTO-RETRAINING STARTED
[05:00] Type: daily, Bars: 8000, Boost Rounds: 50
[05:00] ==================================================
[05:00] Models backed up to models/backups/20250206_050015
[05:00] Fetching 8000 bars of XAUUSD M15 data...
[05:01] Received 8000 bars
[05:01] Date range: 2024-11-15 to 2025-02-06
[05:01] Applying feature engineering...
[05:01] Training HMM Regime Model...
[05:01] HMM model trained and saved
[05:02] Training XGBoost Model...
[05:02] XGBoost trained: Train AUC=0.7234, Test AUC=0.6891
[05:02] Training data saved to data/training_data.parquet
[05:02] ==================================================
[05:02] AUTO-RETRAINING COMPLETED SUCCESSFULLY
[05:02] Duration: 125s
[05:02] ==================================================
\`\`\`
`,
  },
  {
    slug: "backtest",
    title: "Backtest",
    category: "Infrastruktur",
    icon: "BarChart3",
    description: "Framework backtesting yang disinkronkan dengan logika live trading",
    content: `# Backtest — Engine Simulasi Live-Sync

> **File:** \`backtests/backtest_live_sync.py\`
> **Class:** \`LiveSyncBacktest\`
> **Prinsip:** 100% identik dengan \`main_live.py\`

---

## Apa Itu Backtest?

Backtest adalah sistem **simulasi trading pada data historis** yang logikanya 100% disinkronkan dengan trading live. Tujuannya menguji strategi sebelum dipakai uang sungguhan dan memvalidasi perubahan kode.

**Analogi:** Backtest seperti **simulator penerbangan** — pilot (bot) berlatih di kondisi realistis tanpa risiko jatuh. Setiap instrumen, prosedur, dan respons sama persis dengan pesawat asli.

---

## Prinsip Sinkronisasi

\`\`\`
ATURAN UTAMA: Backtest HARUS identik dengan live.

Setiap perubahan di main_live.py → HARUS di-mirror di backtest_live_sync.py

Yang disinkronkan:
├── ML Model: XGBoost dengan fitur yang sama
├── SMC Analyzer: Swing length & OB lookback sama
├── Regime Detection: HMM MarketRegimeDetector
├── Session Filter: Golden Time 19:00-23:00 WIB
├── Signal Logic: Semua filter entry
├── Position Sizing: Berdasarkan ML confidence tier
├── Trade Cooldown: 300 detik (5 menit)
└── Exit Logic: TP, ML reversal, max loss, time-based
\`\`\`

---

## Komponen yang Dimuat

\`\`\`python
# Sama persis dengan main_live.py
self.smc = SMCAnalyzer(swing_length=config.smc.swing_length, ob_lookback=config.smc.ob_lookback)
self.features = FeatureEngineer()
self.regime_detector = MarketRegimeDetector(model_path="models/hmm_regime.pkl")
self.ml_model = TradingModel(model_path="models/xgboost_model.pkl")
self.dynamic_confidence = create_dynamic_confidence()
\`\`\`

---

## Entry Logic (Sama dengan Live)

Semua filter entry di-replikasi:

\`\`\`mermaid
flowchart TD
    START["Untuk setiap bar dalam data historis"] --> F1{"1. COOLDOWN\\n>= 20 bar dari trade terakhir?"}
    F1 -->|YES| F2{"2. SESSION\\nBukan Off Hours 04:00-06:00?"}
    F1 -->|NO| SKIP["SKIP"]
    F2 -->|YES| F3{"3. GOLDEN TIME\\nHanya 19:00-23:00? (opsional)"}
    F2 -->|NO| SKIP
    F3 -->|YES| F4{"4. REGIME\\nBukan CRISIS?"}
    F3 -->|NO| SKIP
    F4 -->|YES| F5{"5. SMC SIGNAL\\nAda signal?"}
    F4 -->|NO| SKIP
    F5 -->|YES| F6{"6. DYNAMIC CONFIDENCE\\nBukan AVOID?"}
    F5 -->|NO| SKIP
    F6 -->|YES| F7{"7. ML THRESHOLD\\nConfidence >= 50-65%?"}
    F6 -->|NO| SKIP
    F7 -->|YES| F8{"8. ML AGREEMENT\\nTidak strongly disagree?"}
    F7 -->|NO| SKIP
    F8 -->|YES| F9{"9. SIGNAL CONFIRMATION\\n2x berturut?"}
    F8 -->|NO| SKIP
    F9 -->|YES| F10{"10. PULLBACK FILTER\\nMomentum tidak berlawanan?"}
    F9 -->|NO| SKIP
    F10 -->|YES| EXEC["EXECUTE SIMULATED TRADE"]
    F10 -->|NO| SKIP
\`\`\`

---

## Session Mapping

\`\`\`python
# Sama dengan session_filter.py
if 6 <= hour < 15:     # Sydney-Tokyo        → lot 0.5x
if 15 <= hour < 16:    # Tokyo-London Overlap → lot 0.75x
if 16 <= hour < 19:    # London Early         → lot 0.8x
if 19 <= hour < 24:    # London-NY (Golden)   → lot 1.0x  ← TERBAIK
if 0 <= hour < 4:      # NY Session           → lot 0.9x
if 4 <= hour < 6:      # Off Hours            → SKIP
\`\`\`

---

## Exit Logic (5 Kondisi)

Untuk setiap bar setelah entry (max 100 bar):

### EXIT 1: Take Profit

\`\`\`
IF harga hit TP level:
  BUY: high >= take_profit
  SELL: low <= take_profit
  -> EXIT dengan profit penuh
\`\`\`

### EXIT 2: Maximum Loss

\`\`\`
IF current_profit < -$50 (max_loss_per_trade):
  -> EXIT, potong kerugian
\`\`\`

### EXIT 3: Time-Based (Synced dengan Live v3)

\`\`\`
IF 16+ bar (4 jam) DAN profit < $5:
  a) profit >= $0 → EXIT (breakeven setelah 4 jam)
  b) profit > -$15 → EXIT (loss kecil, daripada stuck)

IF 24+ bar (6 jam):
  -> FORCE EXIT (apapun profitnya)
\`\`\`

**Visualisasi:**

\`\`\`
Bar:  0     5    10    15    16   20    24
      |-----|-----|-----|-----|-----|-----|
    entry                   |           |
                            |           |
                      4h check:    6h FORCE EXIT
                      profit<$5?
                      Ya -> exit
\`\`\`

### EXIT 4: ML Reversal

\`\`\`
Setiap 5 bar, cek prediksi ML:

IF direction BUY DAN ML bilang SELL dengan confidence > 65%:
  -> EXIT (ML mendeteksi reversal)

IF direction SELL DAN ML bilang BUY dengan confidence > 65%:
  -> EXIT (ML mendeteksi reversal)
\`\`\`

### EXIT 5: Trend Reversal (Momentum)

\`\`\`
Setelah 10+ bar, cek momentum 5 bar terakhir:

IF BUY DAN momentum < -$5 DAN current_profit < -$10:
  -> EXIT (tren berbalik + sudah rugi)

IF SELL DAN momentum > +$5 DAN current_profit < -$10:
  -> EXIT (tren berbalik + sudah rugi)
\`\`\`

---

## Lot Sizing

\`\`\`python
# Berdasarkan ML confidence tier (sama dengan live)
if ml_confidence >= 0.65:
    lot_size = 0.02        # High confidence → lot lebih besar
elif ml_confidence >= 0.55:
    lot_size = 0.01        # Medium confidence → lot standar
else:
    lot_size = 0.01        # Low confidence → lot minimum

# Apply session multiplier
lot_size = max(0.01, lot_size * session_lot_multiplier)
\`\`\`

---

## Pullback Filter

\`\`\`
Sama persis dengan main_live.py:

Untuk signal SELL, block jika:
  - Harga naik > $2 dalam 3 candle terakhir
  - MACD histogram rising + harga naik
  - Harga di atas EMA9 dan masih naik

Untuk signal BUY, block jika:
  - Harga turun > $2 dalam 3 candle terakhir
  - MACD histogram falling + harga turun
  - Harga di bawah EMA9 dan masih turun

Exception (tetap boleh entry):
  - Konsolidasi (pergerakan < $1.50)
  - Momentum searah signal
\`\`\`

---

## Metrik Performa

| Metrik | Rumus | Keterangan |
|--------|-------|------------|
| **Win Rate** | Wins / Total × 100% | Persentase trade profit |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.0 = profitable |
| **Expectancy** | (WR × Avg Win) - (LR × Avg Loss) | Rata-rata per trade |
| **Max Drawdown** | (Peak - Trough) / Peak × 100% | Penurunan terbesar |
| **Sharpe Ratio** | (Avg Return / Std Dev) × √252 | Risk-adjusted return |
| **Net P/L** | Total Profit - Total Loss | Keuntungan bersih |

---

## Threshold Tuning

Mode \`--tune\` menguji beberapa ML threshold secara otomatis:

\`\`\`python
ml_thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

# Untuk setiap threshold:
#   1. Jalankan full backtest
#   2. Catat: trades, win rate, net P/L, profit factor, drawdown
#   3. Ranking berdasarkan net P/L

# Output:
# ML Thresh  Trades  Win Rate    Net P/L       PF      DD
# --------------------------------------------------------
#       55%     145     64.8%    $1,250.00     1.85    3.2%
#       52%     178     62.1%    $1,100.00     1.72    4.1%
#       60%     112     67.0%    $  980.00     1.95    2.8%
#       ...
\`\`\`

---

## Cara Penggunaan

\`\`\`bash
# Backtest standar dengan threshold default (55%)
python backtests/backtest_live_sync.py

# Backtest dengan threshold custom
python backtests/backtest_live_sync.py --threshold 0.60

# Hanya golden time
python backtests/backtest_live_sync.py --golden-only

# Threshold tuning (cari optimal)
python backtests/backtest_live_sync.py --tune

# Simpan hasil ke CSV
python backtests/backtest_live_sync.py --save
\`\`\`

---

## Output Backtest

### Laporan Performa

\`\`\`
==================================================================
BACKTEST RESULTS
==================================================================

Configuration:
  ML Threshold: 55%
  Signal Confirmation: 2 consecutive
  Pullback Filter: Enabled
  Golden Time Only: False

Performance:
  Total Trades: 145
  Wins: 94
  Losses: 51
  Win Rate: 64.8%

Profit/Loss:
  Total Profit: $2,850.00
  Total Loss: $1,600.00
  Net P/L: $1,250.00
  Profit Factor: 1.78

Risk Metrics:
  Max Drawdown: 3.2% ($160.00)
  Avg Win: $30.32
  Avg Loss: $31.37
  Expectancy: $8.62
  Sharpe Ratio: 1.45
\`\`\`

### Breakdown Exit Reason

\`\`\`
Exit Reasons:
  take_profit: 72 (49.7%)
  timeout: 35 (24.1%)
  ml_reversal: 18 (12.4%)
  max_loss: 12 (8.3%)
  trend_reversal: 8 (5.5%)
\`\`\`

### Breakdown Session

\`\`\`
Session Performance:
  London-NY Overlap (Golden): 65 trades, 69.2% WR, $820.00
  NY Session: 32 trades, 62.5% WR, $280.00
  London Early: 28 trades, 60.7% WR, $120.00
  Sydney-Tokyo: 20 trades, 55.0% WR, $30.00
\`\`\`

---

## File Output

\`\`\`
backtests/results/
├── backtest_20250206_143000.csv          # Detail semua trade
│   ├── ticket, entry_time, exit_time
│   ├── direction, entry_price, exit_price
│   ├── stop_loss, take_profit, lot_size
│   ├── profit_usd, profit_pips, result
│   ├── exit_reason, ml_confidence, smc_confidence
│   └── regime, session, signal_reason
│
└── backtest_20250206_143000_summary.csv  # Ringkasan metrik
    ├── total_trades, wins, losses, win_rate
    ├── total_profit, total_loss, net_pnl
    ├── profit_factor, avg_win, avg_loss
    └── max_drawdown, expectancy, sharpe_ratio
\`\`\`

---

## Data Flow

\`\`\`mermaid
flowchart TD
    A["MT5 Connected"] --> B["Fetch 50.000 bar M15 XAUUSD"]
    B --> C["FeatureEngineer.calculate_all() → 40+ fitur\\nSMCAnalyzer.calculate_all() → Struktur pasar\\nRegimeDetector.predict() → Regime label"]
    C --> D["Filter: Jan 2025 - Now"]
    D --> E["Loop setiap bar"]
    E --> E1["Entry check (14 filter)"]
    E --> E2["Simulate exit (5 kondisi)"]
    E --> E3["Record trade result"]
    E --> E4["Update statistics"]
    E1 --> F["Print laporan + Save CSV"]
    E2 --> F
    E3 --> F
    E4 --> F
\`\`\`
`,
  },
  {
    slug: "dynamic-confidence",
    title: "Dynamic Confidence",
    category: "Infrastruktur",
    icon: "Gauge",
    description: "Ambang batas confidence adaptif berdasarkan kondisi dan performa pasar",
    content: `# *Dynamic Confidence* --- Penyesuaian *Threshold* Otomatis

> **File:** \`src/dynamic_confidence.py\`
> **Class:** \`DynamicConfidenceManager\`
> **Digunakan di:** \`main_live.py\`, \`backtest_live_sync.py\`

---

## Apa Itu *Dynamic Confidence*?

*Dynamic Confidence* adalah sistem yang **menyesuaikan confidence *threshold* ML secara otomatis** berdasarkan kondisi pasar saat ini. Saat kondisi ideal, *threshold* diturunkan agar lebih banyak peluang. Saat kondisi buruk, *threshold* dinaikkan untuk lebih selektif.

**Analogi:** *Dynamic Confidence* seperti **termometer yang mengatur AC otomatis** --- saat cuaca panas (pasar bagus), AC diset dingin (*threshold* rendah, lebih banyak trade). Saat cuaca dingin (pasar buruk), AC dimatikan (*threshold* tinggi, kurangi trade).

---

## Flowchart

\`\`\`mermaid
flowchart TD
    A["Kondisi Market Saat Ini"] --> B["6 Faktor Dianalisis"]

    B --> F1["1. Session<br/>+/- 20 poin"]
    B --> F2["2. Regime<br/>+/- 15 poin"]
    B --> F3["3. Volatility<br/>+/- 10 poin"]
    B --> F4["4. Trend Clarity<br/>+/- 10 poin"]
    B --> F5["5. SMC Confluence<br/>+/- 10 poin"]
    B --> F6["6. ML Alignment<br/>+/- 5 poin"]

    F1 --> S["Score (0 - 100)"]
    F2 --> S
    F3 --> S
    F4 --> S
    F5 --> S
    F6 --> S

    S --> Q{"Quality Level?"}

    Q -->|"Score >= 80"| E["EXCELLENT<br/>Threshold: 60%"]
    Q -->|"Score 65-79"| G["GOOD<br/>Threshold: 65%"]
    Q -->|"Score 50-64"| M["MODERATE<br/>Threshold: 70%"]
    Q -->|"Score 35-49"| P["POOR<br/>Threshold: 80%"]
    Q -->|"Score < 35"| AV["AVOID<br/>Threshold: 85%"]

    E --> D{"ML Confidence<br/>>= Threshold?"}
    G --> D
    M --> D
    P --> D
    AV --> SKIP["SKIP --- Jangan Trade"]

    D -->|"YES"| ENTRY["ENTRY Diizinkan"]
    D -->|"NO"| WAIT["TUNGGU --- Confidence Kurang"]

    style A fill:#4a90d9,color:#fff
    style S fill:#f5a623,color:#fff
    style Q fill:#7b68ee,color:#fff
    style E fill:#27ae60,color:#fff
    style G fill:#2ecc71,color:#fff
    style M fill:#f39c12,color:#fff
    style P fill:#e67e22,color:#fff
    style AV fill:#e74c3c,color:#fff
    style ENTRY fill:#27ae60,color:#fff
    style WAIT fill:#e67e22,color:#fff
    style SKIP fill:#e74c3c,color:#fff
\`\`\`

---

## Prinsip Dasar

\`\`\`
Market BAGUS (trending, session bagus)  --> Threshold RENDAH (60%)  --> Lebih banyak trade
Market BIASA (normal)                   --> Threshold SEDANG (70%)  --> Trade normal
Market JELEK (choppy, low liquidity)    --> Threshold TINGGI (80%)  --> Sangat selektif
Market BERBAHAYA (crisis, weekend)      --> Threshold MAXIMUM (85%) --> Hindari trading
\`\`\`

---

## Konfigurasi

\`\`\`python
DynamicConfidenceManager(
    base_threshold=0.70,   # Default threshold 70%
    min_threshold=0.60,    # Minimum (kondisi terbaik): 60%
    max_threshold=0.85,    # Maximum (kondisi terburuk): 85%
)
\`\`\`

---

## 6 Faktor Penilaian

Score dimulai dari **50** (tengah), lalu disesuaikan oleh 6 faktor:

### Faktor 1: *Session* (+/- 20 poin)

| *Session* | Poin | Alasan |
|---------|------|--------|
| London-NY Overlap / Golden | **+20** | Likuiditas tertinggi, spread rendah |
| London | **+15** | Volume tinggi |
| New York | **+10** | Volume tinggi |
| Asia/Tokyo | **+0** | *Volatility* rendah |
| Market Closed/Weekend | **-30** | Tidak ada likuiditas |
| Lainnya | **+5** | Default |

### Faktor 2: *Regime* (+/- 15 poin)

| *Regime* | Poin | Alasan |
|--------|------|--------|
| Medium *Volatility* | **+15** | Kondisi ideal untuk trading |
| Low *Volatility* | **+5** | Hati-hati *ranging* |
| High *Volatility* | **-5** | Perlu lot kecil |
| Crisis | **-25** | Hindari trading |

### Faktor 3: *Volatility* (+/- 10 poin)

| *Volatility* | Poin | Alasan |
|-----------|------|--------|
| Medium | **+10** | Pergerakan cukup, bisa diprediksi |
| Low | **+0** | Pergerakan terlalu kecil |
| High | **-5** | Sulit diprediksi |
| Extreme | **-10** | Sangat berbahaya |

### Faktor 4: Trend Clarity (+/- 10 poin)

| Trend | Poin | Alasan |
|-------|------|--------|
| Uptrend / Downtrend (*trending*) | **+10** | Arah jelas, sinyal lebih akurat |
| Neutral / *Ranging* | **-5** | Sinyal sering whipsaw |

### Faktor 5: SMC *Confluence* (+/- 10 poin)

| Kondisi | Poin | Alasan |
|---------|------|--------|
| Ada sinyal SMC (OB/FVG/BOS) | **+10** | Konfirmasi tambahan |
| Tidak ada sinyal | **+0** | Tanpa konfirmasi |

### Faktor 6: ML Alignment (+/- 5 poin)

| ML Confidence | Poin | Alasan |
|--------------|------|--------|
| >= 70% | **+5** | ML sangat yakin |
| >= 60% | **+2** | ML cukup yakin |
| < 60% | **+0** | ML kurang yakin |

---

## Pemetaan Score ke *Market Quality*

Score dihitung (0--100), lalu dipetakan ke **5 level kualitas**:

\`\`\`
Score:  0    10    20    30    35    50    65    80    100
        |-----|-----|-----|-----|-----|-----|-----|-----|
        |     AVOID      |POOR |  MODERATE  |GOOD | EXCELLENT
        |    (< 35)      |     | (50-64)    |     | (80+)
        |  thresh: 85%   |80%  |   70%      |65%  |  60%
\`\`\`

| Score | Quality | *Threshold* | Aksi |
|-------|---------|-----------|------|
| **80+** | EXCELLENT | 60% | Trade dengan percaya diri |
| **65-79** | GOOD | 65% | Trade normal |
| **50-64** | MODERATE | 70% | Trade hati-hati |
| **35-49** | POOR | 80% | Sangat selektif |
| **< 35** | AVOID | 85% | Jangan trade |

---

## Contoh Perhitungan

### Contoh 1: Kondisi Ideal (Score: 95)

\`\`\`
Base score:                        50

[+20] Session: London-NY Overlap   --> 70
[+15] Regime: Medium Volatility    --> 85
[+10] Volatility: Medium           --> 95
[+10] Trend: UPTREND               --> 105 --> cap 100
[+10] SMC: Ada FVG + BOS           --> 100
[+5]  ML: 72% confidence           --> 100

Score: 100 --> EXCELLENT --> Threshold: 60%
\`\`\`

**Artinya:** ML cukup confidence 60% saja untuk entry. Lebih banyak trade opportunity.

### Contoh 2: Kondisi Jelek (Score: 50)

\`\`\`
Base score:                        50

[+0]  Session: Asia                --> 50
[+5]  Regime: Low Volatility       --> 55
[+0]  Volatility: Low              --> 55
[-5]  Trend: RANGING               --> 50
[+0]  SMC: Tidak ada signal        --> 50
[+0]  ML: 58% confidence           --> 50

Score: 50 --> MODERATE --> Threshold: 70%
\`\`\`

**Artinya:** ML harus confidence 70% untuk entry. Lebih selektif.

### Contoh 3: Kondisi Berbahaya (Score: 0)

\`\`\`
Base score:                        50

[-30] Session: Weekend             --> 20
[-25] Regime: Crisis               --> -5 --> cap 0
[-10] Volatility: Extreme          --> 0
[-5]  Trend: Ranging               --> 0
[+0]  SMC: Tidak ada               --> 0
[+0]  ML: 55%                      --> 0

Score: 0 --> AVOID --> Threshold: 85% (praktis tidak trade)
\`\`\`

---

## Integrasi di Entry Flow

\`\`\`python
# main_live.py --- Step 6 dari 11 filter entry

# 1. Analisis kondisi market
market_analysis = dynamic_confidence.analyze_market(
    session=session_name,           # "London-NY Overlap"
    regime=regime_name,             # "medium_volatility"
    volatility=volatility_level,    # "medium"
    trend_direction=trend,          # "UPTREND"
    has_smc_signal=True,            # Ada SMC signal
    ml_signal=ml_pred.signal,       # "BUY"
    ml_confidence=ml_pred.confidence, # 0.68
)

# 2. Cek quality
if market_analysis.quality == MarketQuality.AVOID:
    return  # SKIP --- market tidak layak

# 3. Cek apakah ML confidence memenuhi threshold dinamis
can_entry, reason = dynamic_confidence.get_entry_decision(
    ml_confidence=0.68,
    analysis=market_analysis,
)

# can_entry = True (0.68 >= 0.60 threshold untuk EXCELLENT)
# reason = "Entry OK: ML 68% >= threshold 60% (score=95)"
\`\`\`

---

## Integrasi di Backtest

\`\`\`python
# backtest_live_sync.py --- identik dengan live

market_analysis = self.dynamic_confidence.analyze_market(
    session=session_name,
    regime=regime,
    volatility="medium",
    trend_direction=regime,
    has_smc_signal=True,
    ml_signal=ml_pred.signal,
    ml_confidence=ml_pred.confidence,
)

if market_analysis.quality == MarketQuality.AVOID:
    continue  # Skip bar ini
\`\`\`

---

## Method \`get_entry_decision()\`

\`\`\`python
def get_entry_decision(ml_confidence, analysis) -> (bool, str):
    """
    Keputusan final entry berdasarkan analisis.

    1. Quality == AVOID? --> False (jangan trade)
    2. ML confidence >= threshold? --> True (entry OK)
    3. ML confidence < threshold? --> False (tunggu)
    """

    # Contoh output:
    # True,  "Entry OK: ML 68% >= threshold 60% (score=95)"
    # False, "Wait: ML 55% < threshold 70% (need +15%)"
    # False, "Market quality: AVOID (score=20)"
\`\`\`

---

## Logging

\`\`\`python
def get_threshold_summary(analysis) -> str:
    """
    Output: "Market: EXCELLENT (score=95) --> Threshold: 60%"
    """
\`\`\`

Contoh log di main_live.py:

\`\`\`
[14:30] Market: EXCELLENT (score=95) --> Threshold: 60%
[14:35] Entry OK: ML 68% >= threshold 60% (score=95)
[15:00] Market: MODERATE (score=55) --> Threshold: 70%
[15:05] Wait: ML 62% < threshold 70% (need +8%)
[04:00] Market: AVOID (score=15) --> Threshold: 85%
\`\`\`

---

## Ringkasan Visual

\`\`\`
Kondisi Market Saat Ini
    |
    v
6 Faktor Dianalisis:
+-- Session     +/- 20 poin
+-- Regime      +/- 15 poin
+-- Volatility  +/- 10 poin
+-- Trend       +/- 10 poin
+-- SMC         +/- 10 poin
+-- ML          +/- 5 poin
    |
    v
Score (0-100)
    |
    v
Quality Level:
+-- EXCELLENT (80+)  --> Threshold 60%
+-- GOOD (65-79)     --> Threshold 65%
+-- MODERATE (50-64) --> Threshold 70%
+-- POOR (35-49)     --> Threshold 80%
+-- AVOID (<35)      --> Threshold 85% / SKIP
    |
    v
ML Confidence >= Threshold?
+-- YES --> ENTRY diizinkan
+-- NO  --> TUNGGU
\`\`\`
`,
  },
  {
    slug: "train-models",
    title: "Train Models",
    category: "Infrastruktur",
    icon: "GraduationCap",
    description: "Pipeline pelatihan model dan optimasi hyperparameter XGBoost",
    content: `# Train Models — Script Training Awal

> **File:** \`train_models.py\`
> **Tipe:** Script CLI (bukan modul)
> **Output:** \`models/xgboost_model.pkl\`, \`models/hmm_regime.pkl\`

---

## Apa Itu *Train Models*?

*Train Models* adalah script **pelatihan awal** yang dijalankan sekali sebelum bot mulai trading. Mengambil data historis dari MT5, melatih HMM dan XGBoost, lalu menyimpan model ke file \`.pkl\`.

**Analogi:** *Train Models* seperti **sekolah penerbangan** — melatih pilot (model AI) sebelum terbang pertama kali. Setelah itu, pelatihan rutin dilakukan oleh Auto Trainer (13).

---

## Cara Penggunaan

\`\`\`bash
python train_models.py
\`\`\`

---

## Pipeline Training

\`\`\`mermaid
flowchart TD
    A[Load Config] --> B[Connect MT5]
    B --> C[Fetch Data]
    C --> D[Feature Engineering]
    D --> E[Train HMM]
    E --> F[Train XGBoost]
    F --> G[Save Models]

    A:::config
    B:::mt5
    C:::data
    D:::data
    E:::model
    F:::model
    G:::save

    classDef config fill:#4a90d9,color:#fff
    classDef mt5 fill:#50c878,color:#fff
    classDef data fill:#f5a623,color:#fff
    classDef model fill:#d0021b,color:#fff
    classDef save fill:#7b68ee,color:#fff
\`\`\`

\`\`\`
1. LOAD CONFIG
   ├── get_config() dari .env
   └── Symbol, capital, mode

2. CONNECT MT5
   ├── Login, password, server
   └── Verifikasi: balance, equity

3. FETCH DATA
   ├── 10.000 bar XAUUSD M15
   └── ~104 hari data historis

4. FEATURE ENGINEERING
   ├── FeatureEngineer.calculate_all()  → 40+ fitur teknikal
   ├── SMCAnalyzer.calculate_all()      → Struktur pasar
   └── create_target(lookahead=1)       → Label UP/DOWN

5. SAVE DATA
   └── data/training_data.parquet

6. TRAIN HMM
   ├── MarketRegimeDetector(n_regimes=3, lookback=500)
   ├── fit(df)
   ├── Log: distribusi regime, transition matrix
   └── Save → models/hmm_regime.pkl

7. TRAIN XGBOOST
   ├── TradingModel(confidence_threshold=0.60)
   ├── fit(train_ratio=0.7, boost_rounds=50, early_stop=5)
   ├── Log: top 10 *feature importance*
   ├── *Walk-forward* validation (train=500, test=50, step=50)
   ├── Log: avg train/test *AUC*, overfitting ratio
   └── Save → models/xgboost_model.pkl

8. DISCONNECT
\`\`\`

---

## Parameter Training

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Data | 10.000 bar M15 | ~104 hari |
| Train/Test Split | 70% / 30% | Lebih banyak test data |
| XGBoost Rounds | 50 | *Anti-overfitting* |
| *Early Stopping* | 5 rounds | Stop lebih awal |
| HMM Regimes | 3 | Low/Medium/High volatility |
| HMM Lookback | 500 bar | Window training |
| *Walk-forward* Window | 500 train / 50 test | Validasi robustness |

---

## Output

\`\`\`
models/
├── xgboost_model.pkl    # Model XGBoost (*binary classifier*)
└── hmm_regime.pkl       # Model HMM (regime detector)

data/
└── training_data.parquet  # Data training (untuk referensi)

logs/
└── training_YYYY-MM-DD.log  # Log training detail
\`\`\`

---

## Contoh Output Log

\`\`\`
[08:00] ============================================================
[08:00] SMART TRADING BOT - MODEL TRAINING
[08:00] ============================================================
[08:00] Symbol: XAUUSD
[08:00] Capital: $5,000.00
[08:00] Mode: small
[08:00] Connecting to MT5...
[08:00] MT5 connected successfully!
[08:00] Account Balance: $5,094.68
[08:00] Fetching 10000 bars of XAUUSD M15 data...
[08:01] Received 10000 bars
[08:01] Date range: 2024-10-25 to 2025-02-06
[08:01] Applying feature engineering...
[08:01] Total features created: 52
[08:01] ============================================================
[08:01] Training HMM Regime Model
[08:01] ============================================================
[08:01] Regime Distribution:
[08:01]   low_volatility: 3200 bars
[08:01]   medium_volatility: 4500 bars
[08:01]   high_volatility: 2300 bars
[08:02] ============================================================
[08:02] Training XGBoost Model (Anti-Overfit Config)
[08:02] ============================================================
[08:02] Available features: 37/40
[08:02] Top 10 Feature Importance:
[08:02]   rsi: 0.0842
[08:02]   macd_histogram: 0.0756
[08:02]   atr: 0.0689
[08:02]   ...
[08:03] Walk-forward Results:
[08:03]   Avg Train AUC: 0.7234
[08:03]   Avg Test AUC: 0.6891
[08:03]   Overfitting ratio: 1.05
[08:03] ============================================================
[08:03] TRAINING COMPLETE
[08:03] ============================================================
[08:03] HMM Model: SAVED
[08:03] XGBoost Model: SAVED
\`\`\`

---

## Kapan Dijalankan?

| Situasi | Script |
|---------|--------|
| **Pertama kali setup** | \`train_models.py\` (wajib) |
| **Setelah update kode** | \`train_models.py\` (opsional) |
| **Rutin harian** | Auto Trainer (otomatis) |
| **Model buruk** | \`train_models.py\` (manual retrain) |

---

## Perbedaan dengan Auto Trainer

| Aspek | train_models.py | Auto Trainer |
|-------|-----------------|-------------|
| **Kapan** | Manual, 1x | Otomatis, harian |
| **Data** | 10K bar | 8K (daily) / 15K (weekend) |
| **Backup** | Tidak | Ya (5 terakhir) |
| **Rollback** | Tidak | Ya (*AUC* < 0.60) |
| **Database** | Tidak | Ya (PostgreSQL) |
| *Walk-forward* | Ya | Tidak |
| **Tujuan** | Setup awal | Maintenance rutin |
`,
  },
  {
    slug: "mt5-connector",
    title: "Konektor MT5",
    category: "Konektor & Konfigurasi",
    icon: "Plug",
    description: "Lapisan koneksi MetaTrader 5 dan eksekusi order trading",
    content: `# *MT5 Connector* — Jembatan ke *MetaTrader* 5

> **File:** \`src/mt5_connector.py\`
> **Class:** \`MT5Connector\`, \`MT5SimulationConnector\`
> **Library:** MetaTrader5 (Python API)

---

## Apa Itu *MT5 Connector*?

*MT5 Connector* adalah **jembatan komunikasi** antara bot AI dan terminal *MetaTrader* 5. Semua interaksi dengan broker — ambil data harga, kirim order, cek posisi — dilakukan melalui modul ini.

**Analogi:** *MT5 Connector* seperti **penerjemah di bandara** — menerjemahkan perintah bot (Python) ke bahasa yang dipahami broker (MT5 API), dan sebaliknya.

---

## Fungsi Utama

| Method | Fungsi | Return |
|--------|--------|--------|
| \`connect()\` | Koneksi ke MT5 terminal | \`bool\` |
| \`disconnect()\` | Putus koneksi | - |
| \`reconnect()\` | Reconnect otomatis | \`bool\` |
| \`ensure_connected()\` | Cek & *auto-reconnect* | \`bool\` |
| \`get_market_data()\` | Ambil data OHLCV | \`pl.DataFrame\` |
| \`get_tick()\` | Ambil harga real-time | \`TickData\` |
| \`send_order()\` | Kirim order BUY/SELL | \`OrderResult\` |
| \`close_position()\` | Tutup posisi | \`OrderResult\` |
| \`get_open_positions()\` | Cek posisi terbuka | \`pl.DataFrame\` |
| \`get_symbol_info()\` | Info simbol (spread, dll) | \`Dict\` |
| \`get_multi_timeframe_data()\` | Ambil data multi-timeframe | \`Dict[str, pl.DataFrame]\` |

---

## Koneksi & *Auto-Reconnect*

### Connection Flow

\`\`\`mermaid
flowchart TD
    A([Start connect]) --> B[Shutdown koneksi lama]
    B --> C[mt5.initialize\\nlogin, password, server]
    C --> D{Initialize\\nberhasil?}
    D -- Ya --> E[Tunggu 2 detik\\nstabilisasi terminal]
    D -- Tidak --> K{Attempt\\n< max_retries?}
    E --> F{terminal_info\\n!= None?}
    F -- Ya --> G{terminal\\n.connected?}
    F -- Tidak --> J[Shutdown & retry]
    G -- Ya --> H[Set _connected = True\\nAmbil account_info\\nSelect symbol XAUUSD]
    G -- Tidak --> I[Tunggu 3 detik\\nCek ulang terminal]
    I --> G2{Masih belum\\nconnected?}
    G2 -- Ya --> J
    G2 -- Tidak --> H
    H --> Z([Connected!])
    J --> K
    K -- Ya --> L[Exponential backoff\\n2s, 4s, 8s]
    L --> B
    K -- Tidak --> M([ConnectionError\\nRaise exception])

    style A fill:#4CAF50,color:#fff
    style Z fill:#4CAF50,color:#fff
    style M fill:#f44336,color:#fff
    style H fill:#2196F3,color:#fff
\`\`\`

### Mekanisme *Auto-Reconnect*

\`\`\`mermaid
flowchart TD
    A([ensure_connected\\ndipanggil]) --> B{Flag\\n_connected?}
    B -- False --> C[reconnect]
    B -- True --> D[Cek mt5.account_info]
    D --> E{Info\\nvalid?}
    E -- Ya --> F([Tetap connected\\nReset attempt counter])
    E -- Tidak --> G[Set _connected = False\\nIncrement attempt]
    G --> H{Attempt >\\nmax 5?}
    H -- Ya --> I{Sudah lewat\\n60 detik cooldown?}
    I -- Ya --> J[Reset attempt = 0]
    I -- Tidak --> K([Return False\\nMasih dalam cooldown])
    J --> C
    H -- Tidak --> C
    C --> L{reconnect\\nberhasil?}
    L -- Ya --> M([Reconnected!\\nReset attempt counter])
    L -- Tidak --> N([Return False])

    style A fill:#FF9800,color:#fff
    style F fill:#4CAF50,color:#fff
    style M fill:#4CAF50,color:#fff
    style K fill:#f44336,color:#fff
    style N fill:#f44336,color:#fff
\`\`\`

**Detail *Auto-Reconnect*:**

\`\`\`python
ensure_connected():
    """
    Dipanggil sebelum setiap operasi penting.

    1. Cek flag _connected
    2. Coba mt5.account_info()
    3. Gagal? → reconnect()
    4. Max 5 attempts, lalu cooldown 60 detik
    """
\`\`\`

Mekanisme *exponential backoff* memastikan bot tidak membombardir server broker dengan request berulang. Setiap kali koneksi gagal, waktu tunggu berlipat ganda (2s, 4s, 8s). Setelah 5 kali gagal berturut-turut, bot masuk fase *cooldown* selama 60 detik sebelum mencoba lagi.

---

## Data Fetching (Polars Native)

\`\`\`python
get_market_data(symbol="XAUUSD", timeframe="M15", count=1000, max_retries=3)
\`\`\`

**Proses:**

\`\`\`
MT5 Terminal
    |
    v
ensure_connected() → auto-reconnect jika putus
    |
    v
mt5.symbol_select() → pastikan simbol aktif di Market Watch
    |
    v
mt5.copy_rates_from_pos() → numpy structured array
    |
    v
LANGSUNG ke Polars DataFrame (TANPA Pandas)
    |
    v
Cast types:
├── time: Unix timestamp → Datetime
├── open/high/low/close: Float64
├── tick_volume → volume (Int64)
└── spread, real_volume: Int64
    |
    v
Return pl.DataFrame
\`\`\`

> **Catatan penting:** Data dikonversi langsung dari NumPy structured array ke Polars DataFrame. Tidak ada konversi perantara via Pandas. Ini adalah optimisasi kritis untuk performa — menjaga target **< 50ms per loop**.

**Kolom output:**

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| \`time\` | Datetime | Waktu candle |
| \`open\` | Float64 | Harga buka |
| \`high\` | Float64 | Harga tertinggi |
| \`low\` | Float64 | Harga terendah |
| \`close\` | Float64 | Harga tutup |
| \`volume\` | Int64 | Tick volume |
| \`spread\` | Int64 | Spread |
| \`real_volume\` | Int64 | Real volume |

---

## Order Execution

\`\`\`python
send_order(
    symbol="XAUUSD",
    order_type="BUY",     # atau "SELL"
    volume=0.01,          # Lot size
    sl=4937.00,           # Stop Loss
    tp=4976.00,           # Take Profit
    deviation=20,         # Max slippage (points)
    magic=123456,         # Bot ID
    comment="AI Bot",
    max_retries=3,
)
\`\`\`

### Order Execution Flow dengan *Retry* Logic

\`\`\`mermaid
flowchart TD
    A([send_order\\ndipanggil]) --> B[Ambil tick data\\nmt5.symbol_info_tick]
    B --> C{Tick\\nvalid?}
    C -- Tidak --> D([Return: Failed\\nNo tick data])
    C -- Ya --> E[Tentukan harga\\nBUY → ask / SELL → bid]
    E --> F[Build request:\\naction, symbol, volume,\\ntype, price, SL, TP,\\ndeviation, magic]
    F --> G[mt5.order_send]
    G --> H{Result\\n== None?}
    H -- Ya --> I[Log error]
    I --> P
    H -- Tidak --> J{RETCODE?}
    J -- 10009 DONE --> K([Order berhasil!\\nReturn OrderResult])
    J -- "10013-10016\\nINVALID" --> L([Return: Failed\\nNon-retryable error])
    J -- 10027\\nTRADE_DISABLED --> M([Raise RuntimeError\\nAutoTrading off])
    J -- "10004 REQUOTE\\n10006 REJECT\\nlainnya" --> N[Log warning\\nTunggu 0.5 detik]
    N --> P{Attempt\\n< max_retries?}
    P -- Ya --> Q[Refresh harga\\nUlangi order]
    Q --> G
    P -- Tidak --> R([Return: Failed\\nMax retries exceeded])

    style A fill:#FF9800,color:#fff
    style K fill:#4CAF50,color:#fff
    style D fill:#f44336,color:#fff
    style L fill:#f44336,color:#fff
    style M fill:#f44336,color:#fff
    style R fill:#f44336,color:#fff
\`\`\`

Parameter \`deviation\` mengontrol toleransi *slippage* maksimum dalam poin. Jika harga bergeser melebihi batas ini saat eksekusi, broker akan menolak order (REQUOTE) dan bot akan melakukan *retry* otomatis dengan harga terbaru.

**Close Position** juga menggunakan logika *retry* yang sama — setiap attempt mengambil ulang harga terbaru untuk memastikan akurasi.

---

## Timeframe Mapping

| String | MT5 Constant | Penggunaan |
|--------|-------------|------------|
| \`M1\` | TIMEFRAME_M1 | 1 menit |
| \`M5\` | TIMEFRAME_M5 | 5 menit |
| \`M15\` | TIMEFRAME_M15 | **Utama** (execution) |
| \`M30\` | TIMEFRAME_M30 | 30 menit |
| \`H1\` | TIMEFRAME_H1 | 1 jam (EMA20 filter) |
| \`H4\` | TIMEFRAME_H4 | Trend analysis |
| \`D1\` | TIMEFRAME_D1 | 1 hari |
| \`W1\` | TIMEFRAME_W1 | 1 minggu |

---

## Error Codes

| Code | Nama | Aksi |
|------|------|------|
| 10009 | DONE | Order berhasil |
| 10004 | REQUOTE | *Retry* — harga berubah |
| 10006 | REJECT | *Retry* — ditolak server |
| 10013 | INVALID | Stop, order salah |
| 10014 | INVALID_VOLUME | Stop, lot salah |
| 10015 | INVALID_PRICE | Stop, harga salah |
| 10016 | INVALID_STOPS | Stop, SL/TP salah |
| 10027 | TRADE_DISABLED | AutoTrading off |
| -10001 | COMMON_ERROR | Reconnect |
| -10002 | INVALID_PARAMS | Reconnect |
| -10003 | NO_CONNECTION | Reconnect |
| -10004 | NO_IPC | Reconnect |
| -1 | TERMINAL_CALL_FAILED | Reconnect |

Error code -10003, -10004, -10001, -10002, dan -1 termasuk dalam \`CONNECTION_ERRORS\` dan secara otomatis memicu mekanisme *auto-reconnect*.

---

## *Simulation Mode*

\`\`\`python
class MT5SimulationConnector(MT5Connector):
    """
    Untuk testing tanpa MT5 terminal.

    - connect() selalu berhasil
    - get_market_data() generate data sintetis (random walk)
    - Base price XAUUSD: $2000
    - Berguna untuk development & unit testing
    """
\`\`\`

*Simulation mode* memungkinkan pengembangan dan testing tanpa perlu koneksi ke terminal *MetaTrader* yang sebenarnya. Connector ini menghasilkan data OHLCV sintetis menggunakan random walk dari harga dasar $2000.

> **Catatan:** *Simulation mode* secara otomatis aktif jika library MetaTrader5 tidak terinstal di environment.

---

## Konfigurasi Koneksi

\`\`\`python
MT5Connector(
    login=12345678,                    # Dari .env MT5_LOGIN
    password="password123",            # Dari .env MT5_PASSWORD
    server="BrokerServer-Live",        # Dari .env MT5_SERVER
    path="C:/Program Files/MT5/...",   # Dari .env MT5_PATH (opsional)
    timeout=60000,                     # 60 detik timeout
)
\`\`\`

### Context Manager Support

*MT5 Connector* mendukung penggunaan sebagai context manager:

\`\`\`python
with MT5Connector(login, password, server) as mt5_conn:
    data = mt5_conn.get_market_data("XAUUSD", "M15")
    # Otomatis disconnect saat keluar blok
\`\`\`

---

## Catatan Teknis

- **Polars, bukan Pandas:** Semua konversi data dari MT5 menggunakan Polars secara langsung. Tidak ada *connection pooling* atau konversi via Pandas.
- **Thread Safety:** *MT5 Connector* berjalan di satu thread utama. Library MT5 Python API tidak thread-safe, jadi semua operasi dilakukan secara sekuensial.
- **Password Security:** Password disimpan dengan prefix \`_\` (\`self._password\`) sebagai konvensi private attribute.
- **Symbol Pre-selection:** Setelah koneksi berhasil, simbol XAUUSD otomatis di-select di Market Watch untuk memastikan data siap diambil.
`,
  },
  {
    slug: "configuration",
    title: "Konfigurasi",
    category: "Konektor & Konfigurasi",
    icon: "Settings",
    description: "Pengaturan trading, mode kapital, dan konfigurasi environment",
    content: `# Konfigurasi — Pusat Pengaturan Bot

> **File:** \`src/config.py\`
> **Class:** \`TradingConfig\`, \`RiskConfig\`, \`SMCConfig\`, \`MLConfig\`, \`ThresholdsConfig\`, \`RegimeConfig\`
> **Sumber:** *Environment variables* (\`.env\`)

---

## Struktur Konfigurasi

\`\`\`mermaid
graph TD
    A[".env File"] --> B["TradingConfig.from_env()"]
    B --> C["CapitalMode<br/>SMALL / MEDIUM"]
    C -->|"≤ $10.000"| D["SMALL<br/>Risk 1%, Max Lot 0.05"]
    C -->|"> $10.000"| E["MEDIUM<br/>Risk 0.5%, Max Lot 2.0"]
    B --> F["RiskConfig"]
    B --> G["SMCConfig"]
    B --> H["MLConfig"]
    B --> I["ThresholdsConfig"]
    B --> J["RegimeConfig"]
\`\`\`

---

## 2 Mode Kapital

### Mode SMALL (≤ $10.000) — *Growth Mode*

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| \`risk_per_trade\` | **1.0%** | $50 risiko per *trade* (akun $5.000) |
| \`max_daily_loss\` | **3.0%** | Batas kerugian harian $150 |
| \`max_leverage\` | **1:100** | *Leverage* tinggi untuk akun kecil |
| \`max_positions\` | **3** | Maksimal 3 posisi bersamaan |
| \`max_lot_size\` | **0.05** | Batas atas *lot* per *trade* |
| \`min_lot_size\` | **0.01** | *Lot* minimum |
| \`execution_timeframe\` | **M15** | *Scalping* / *day trading* |

### Mode MEDIUM (> $10.000) — *Preservation Mode*

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| \`risk_per_trade\` | **0.5%** | $250 risiko per *trade* (akun $50.000) |
| \`max_daily_loss\` | **2.0%** | Batas kerugian harian $1.000 |
| \`max_leverage\` | **1:30** | *Leverage* konservatif |
| \`max_positions\` | **5** | Lebih banyak diversifikasi |
| \`max_lot_size\` | **2.0** | Batas atas *lot* |
| \`execution_timeframe\` | **H1** | *Swing trading* |
| \`trend_timeframe\` | **H4** | Analisis *trend* jangka menengah |

---

## *Thresholds* (Ambang Batas)

### *ML Confidence*

| Parameter | Nilai | Fungsi |
|-----------|-------|--------|
| \`ml_min_confidence\` | **0.65** | Minimum *confidence* untuk pertimbangkan sinyal |
| \`ml_entry_confidence\` | **0.70** | *Default confidence* untuk *entry* |
| \`ml_high_confidence\` | **0.75** | *High confidence* — sinyal kuat |
| \`ml_very_high_confidence\` | **0.80** | Sangat yakin — *lot multiplier* aktif |

### *Dynamic Threshold*

| Parameter | Nilai | Kondisi |
|-----------|-------|---------|
| \`dynamic_threshold_aggressive\` | **0.65** | Pasar *trending* kuat |
| \`dynamic_threshold_moderate\` | **0.70** | Kondisi normal |
| \`dynamic_threshold_conservative\` | **0.75** | Pasar bergejolak |

### *Profit/Loss* ($)

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| \`min_profit_to_secure\` | **$15** | Mulai pertimbangkan *take profit* |
| \`good_profit_level\` | **$25** | Level profit bagus |
| \`great_profit_level\` | **$40** | *Hard take profit* — ambil profit |

### *Trading Timing*

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| \`trade_cooldown_seconds\` | **300** | 5 menit antar *trade* |
| \`loop_interval_seconds\` | **30** | Interval *main loop* |
| \`sydney_lot_multiplier\` | **0.5** | *Lot* dikurangi 50% saat Sydney |

---

## *Environment Variables* (\`.env\`)

\`\`\`env
# MetaTrader 5 — WAJIB
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# Telegram — OPSIONAL
TELEGRAM_BOT_TOKEN=bot123:ABC-DEF
TELEGRAM_CHAT_ID=123456789

# Trading
CAPITAL=5000              # Menentukan mode (SMALL/MEDIUM)
SYMBOL=XAUUSD             # Pair yang diperdagangkan

# Override (opsional)
RISK_PER_TRADE=1.0        # Override risk per trade (%)
MAX_DAILY_LOSS_PERCENT=3.0
MAX_POSITION_SIZE=0.05
AI_CONFIDENCE_THRESHOLD=0.65
FLASH_CRASH_THRESHOLD=2.5
\`\`\`

---

## Kalkulasi *Position Sizing*

Bot menggunakan **metode *Half-Kelly Criterion*** untuk menghitung ukuran *lot*:

\`\`\`python
def calculate_position_size(self, entry_price, stop_loss_price, account_balance=None):
    # 1. Hitung jumlah risiko dalam $
    risk_amount = balance * (risk_per_trade / 100)

    # 2. Hitung jarak SL dalam pips
    sl_distance = abs(entry_price - stop_loss_price)
    sl_pips = sl_distance / 0.1  # XAUUSD: 1 pip = $0.1

    # 3. Hitung lot size
    lot_size = risk_amount / (sl_pips * pip_value_per_lot)

    # 4. Apply Half-Kelly (keamanan)
    lot_size *= 0.5

    # 5. Round dan batasi
    lot_size = round(lot_size / lot_step) * lot_step
    lot_size = max(min_lot, min(lot_size, max_lot))
\`\`\`

**Contoh:** Akun $5.000, SL 50 *pips*, risiko 1%:
- Risiko = $50
- *Lot* = $50 / (50 × $1) = 0.01 *lot* (setelah *Half-Kelly*)

---

## Validasi Konfigurasi

\`\`\`python
def _validate_required_settings(self):
    """Validasi environment variables wajib saat startup."""
    # MT5_LOGIN harus ada dan > 0
    # MT5_PASSWORD harus ada dan tidak kosong
    # MT5_SERVER harus ada dan tidak kosong
    # CAPITAL harus > 0
    # Jika ada yang hilang → ValueError dengan pesan jelas
\`\`\`

Bot **tidak bisa berjalan** tanpa kredensial MT5 yang valid.
`,
  },
  {
    slug: "trade-logger",
    title: "Trade Logger",
    category: "Konektor & Konfigurasi",
    icon: "FileText",
    description: "Pencatatan trade ke database PostgreSQL untuk analisis historis",
    content: `# Trade Logger — Pencatat Trade Otomatis

> **File:** \`src/trade_logger.py\`
> **Class:** \`TradeLogger\`
> **Storage:** PostgreSQL (primary) + CSV (*fallback*)

---

## Apa Itu *Trade Logger*?

*Trade Logger* mencatat **setiap trade, sinyal, dan kondisi pasar** secara otomatis ke database dan file CSV. Data ini digunakan untuk analisis performa, retraining ML model, dan debugging.

**Analogi:** *Trade Logger* seperti ***black box* di pesawat** — merekam semua yang terjadi untuk analisis setelah penerbangan (trading).

---

## Alur *Dual Storage*

\`\`\`mermaid
flowchart TD
    A["Event Terjadi\\n(trade / signal / snapshot)"] --> B[TradeLogger]
    B --> C{DB tersedia?}
    C -- Ya --> D["PostgreSQL\\n(Primary)"]
    C -- Ya --> E["CSV\\n(Backup)"]
    C -- Tidak --> E
    D --> F["trades table\\nsignals table\\nmarket_snapshots\\nbot_status"]
    E --> G["data/trade_logs/\\ntrades/ | signals/ | snapshots/\\n(file bulanan YYYY_MM.csv)"]
    style A fill:#2d333b,stroke:#adbac7,color:#adbac7
    style B fill:#1f6feb,stroke:#58a6ff,color:#fff
    style C fill:#3d444d,stroke:#adbac7,color:#adbac7
    style D fill:#238636,stroke:#3fb950,color:#fff
    style E fill:#9e6a03,stroke:#d29922,color:#fff
    style F fill:#238636,stroke:#3fb950,color:#fff
    style G fill:#9e6a03,stroke:#d29922,color:#fff
\`\`\`

**Prinsip *dual storage*:**

- **DB tersedia?** — Tulis ke PostgreSQL **DAN** CSV (double safety)
- **DB tidak tersedia?** — CSV saja (*graceful degradation*)
- CSV **selalu** ditulis sebagai *fallback*, tidak peduli status DB

---

## 3 Tipe Data yang Dicatat

### 1. Trade Record (Per Trade)

Setiap trade dibuka/ditutup dicatat lengkap:

| Kategori | Field |
|----------|-------|
| **Identitas** | ticket, symbol |
| **Trade** | direction, lot_size, entry_price, exit_price, SL, TP |
| **Hasil** | profit_usd, profit_pips, duration_seconds |
| **Waktu** | open_time, close_time |
| **Market** | regime, volatility, session, spread, ATR |
| **SMC** | *signal*, confidence, reason, FVG/OB/BOS/CHoCH flags |
| **ML** | *signal*, confidence |
| **Dynamic** | market_quality, market_score, threshold |
| **Exit** | exit_reason, exit_regime, exit_ml_signal |
| **Balance** | balance_before, balance_after, equity_at_entry |
| **Features** | JSON *snapshot* fitur saat entry & exit |

### 2. *Signal* Record (Per Sinyal)

Setiap sinyal yang dihasilkan (termasuk yang **tidak** dieksekusi):

\`\`\`
timestamp, symbol, price
signal_type, signal_source, confidence
smc_*, ml_*
regime, session, volatility, market_score
trade_executed (bool)
execution_reason ("executed" / "below_threshold" / "max_positions" / ...)
\`\`\`

### 3. Market *Snapshot* (Periodik)

*Snapshot* kondisi pasar secara berkala:

\`\`\`
timestamp, symbol, price, OHLC
regime, volatility, session, ATR, spread
ml_signal, ml_confidence
smc_signal, smc_confidence
open_positions, floating_pnl
features (JSON)
\`\`\`

---

## *Dual Storage*

\`\`\`mermaid
flowchart TD
    EV["Event Terjadi<br/>(trade / signal / snapshot)"] --> PG["PostgreSQL (Primary)<br/>trades, signals,<br/>market_snapshots, bot_status<br/>Cepat, queryable, thread-safe pooling"]
    EV --> CSV["CSV (Fallback)<br/>data/trade_logs/<br/>trades/, signals/, snapshots/<br/>Selalu ditulis (backup)"]
\`\`\`

- **DB tidak tersedia?** → CSV saja (*graceful degradation*)
- **DB tersedia?** → Tulis ke DB **DAN** CSV (double safety)

---

## Proses Log Trade

\`\`\`mermaid
flowchart TD
    OPEN["Trade Dibuka"] --> LOG_OPEN["log_trade_open()<br/>ticket, entry_price, regime, smc, ml"]
    LOG_OPEN --> MEM["Simpan ke _pending_trades di memory"]
    LOG_OPEN --> DB_INS["INSERT ke database (trades table)"]
    MEM --> WAIT["... trading berjalan ..."]
    DB_INS --> WAIT
    WAIT --> CLOSE["Trade Ditutup"]
    CLOSE --> LOG_CLOSE["log_trade_close()<br/>ticket, exit_price, profit, exit_reason"]
    LOG_CLOSE --> FETCH["Ambil data pending dari memory"]
    LOG_CLOSE --> DUR["Hitung durasi: close - open"]
    FETCH --> UPD["UPDATE database<br/>(exit_price, profit, duration)"]
    DUR --> UPD
    UPD --> CSV["APPEND ke CSV<br/>(trades_YYYY_MM.csv)"]
\`\`\`

Data *pending* disimpan dalam dictionary \`_pending_trades[ticket]\` selama trade masih terbuka. Ketika trade ditutup, data entry digabung dengan data exit menjadi satu \`TradeRecord\` lengkap sebelum ditulis ke CSV.

---

## Analisis Helper

| Method | Fungsi |
|--------|--------|
| \`get_recent_trades(10)\` | 10 trade terakhir |
| \`get_win_rate(30)\` | Win rate 30 hari |
| \`get_smc_performance(30)\` | Performa per pattern SMC |
| \`get_trades_for_training(30)\` | Data untuk ML retraining |
| \`get_stats()\` | Statistik logger |

Setiap helper method mencoba query dari PostgreSQL terlebih dahulu. Jika DB tidak tersedia, otomatis *fallback* ke pembacaan file CSV — konsisten dengan prinsip *graceful degradation*.

---

## *Thread Safety*

\`\`\`python
self._lock = threading.Lock()

# Setiap operasi CSV dilindungi lock
with self._lock:
    # Write to CSV
\`\`\`

Semua operasi tulis ke file CSV dilindungi oleh \`threading.Lock()\` untuk menjamin *thread safety*. Ini mencegah korupsi data ketika multiple thread mencoba menulis ke file yang sama secara bersamaan (misalnya log trade close dan log *signal* terjadi hampir bersamaan).

---

## File CSV (Terorganisir per Bulan)

\`\`\`
data/trade_logs/
├── trades/
│   ├── trades_2025_01.csv
│   └── trades_2025_02.csv
├── signals/
│   ├── signals_2025_01.csv
│   └── signals_2025_02.csv
└── snapshots/
    ├── snapshots_2025_01.csv
    └── snapshots_2025_02.csv
\`\`\`
`,
  },
  {
    slug: "position-manager",
    title: "Position Manager",
    category: "Konektor & Konfigurasi",
    icon: "ListChecks",
    description: "Pelacakan dan manajemen posisi terbuka secara real-time",
    content: `# Position Manager — Manajemen Posisi Cerdas

> **File:** \`src/position_manager.py\`
> **Class:** \`SmartPositionManager\`, \`SmartMarketCloseHandler\`
> **Fitur:** *Trailing Stop*, Profit Protection, *Market Close Handler*

---

## Apa Itu *Position Manager*?

*Position Manager* mengelola posisi terbuka secara **aktif dan cerdas** — *trailing stop* loss, proteksi profit, dan keputusan otomatis saat market mendekati penutupan.

**Analogi:** *Position Manager* seperti **co-pilot yang mengawasi perjalanan** — mengamankan keuntungan saat angin baik, dan mengambil tindakan darurat saat cuaca memburuk.

---

## 2 Komponen Utama

### A. SmartPositionManager

Mengelola posisi aktif: *trailing stop*, *breakeven*, profit protection.

### B. SmartMarketCloseHandler

Keputusan cerdas saat market mendekati penutupan (harian/weekend) — *market close handler*.

---

## SmartPositionManager — 7 Kondisi Aksi

Untuk setiap posisi terbuka, dicek berurutan berdasarkan prioritas:

\`\`\`mermaid
flowchart TD
    START([Posisi Terbuka]) --> C0{0. Market Close Check}
    C0 -->|Profit >= $10\\n+ dekat close| CLOSE0[CLOSE\\nAmankan profit]
    C0 -->|Loss + weekend\\n+ SL >50% hit| CLOSE0W[CLOSE\\nHindari gap risk]
    C0 -->|Loss + weekend\\n+ loss > $100| CLOSE0W
    C0 -->|Tidak terpicu| C1

    C1{1. Regime Danger?}
    C1 -->|Crisis/High Vol\\n+ profit > $50| CLOSE1[CLOSE\\nAmankan dari volatilitas]
    C1 -->|Tidak| C2

    C2{2. Opposite Signal?}
    C2 -->|Sinyal berlawanan kuat\\n+ profit > $25| CLOSE2[CLOSE\\nAmankan sebelum reversal]
    C2 -->|Tidak| C3

    C3{3. Drawdown from Peak?}
    C3 -->|Peak > $50\\n+ drawdown > 30%| CLOSE3[CLOSE\\nProfit sudah turun]
    C3 -->|Tidak| C4

    C4{4. High Urgency?}
    C4 -->|Urgency >= 7\\n+ profit > 0| CLOSE4[CLOSE\\nBanyak sinyal bahaya]
    C4 -->|Tidak| C5

    C5{5. Breakeven\\nProtection?}
    C5 -->|Profit >= BE pips| TRAIL5[TRAIL SL\\nPindah ke breakeven]
    C5 -->|Tidak| C6

    C6{6. Trailing Stop?}
    C6 -->|Profit >= trail pips| TRAIL6[TRAIL SL\\nIkuti harga]
    C6 -->|Tidak| HOLD

    HOLD([7. Default: HOLD])

    style CLOSE0 fill:#e74c3c,color:#fff
    style CLOSE0W fill:#e74c3c,color:#fff
    style CLOSE1 fill:#e74c3c,color:#fff
    style CLOSE2 fill:#e74c3c,color:#fff
    style CLOSE3 fill:#e74c3c,color:#fff
    style CLOSE4 fill:#e74c3c,color:#fff
    style TRAIL5 fill:#f39c12,color:#fff
    style TRAIL6 fill:#f39c12,color:#fff
    style HOLD fill:#27ae60,color:#fff
\`\`\`

### 0. Market Close Check (Prioritas Tertinggi)

\`\`\`
Dekat market close?
├── Profit >= $10 + dekat close → CLOSE (amankan profit)
├── Loss + dekat weekend + SL >50% hit → CLOSE (gap risk)
├── Loss + dekat weekend + loss > $100 → CLOSE (gap risk)
└── Loss kecil + dekat weekend → HOLD (bisa recovery Senin)
\`\`\`

### 1. Regime Danger

\`\`\`
Regime CRISIS atau HIGH_VOLATILITY + profit > $50:
  → CLOSE (amankan profit dari volatilitas)
\`\`\`

### 2. Opposite Signal

\`\`\`
Posisi BUY + sinyal bearish kuat + profit > $25:
  → CLOSE (amankan sebelum reversal)

Posisi SELL + sinyal bullish kuat + profit > $25:
  → CLOSE (amankan sebelum reversal)
\`\`\`

### 3. *Drawdown* from Peak

\`\`\`
Peak profit > $50 DAN drawdown > 30% dari peak:
  → CLOSE (profit sudah turun terlalu banyak)

Contoh: Peak $80, sekarang $50 → drawdown 37.5% → CLOSE
\`\`\`

### 4. High Urgency

\`\`\`
Urgency score >= 7 (dari 10) DAN profit > 0:
  → CLOSE (banyak sinyal bahaya bersamaan)
\`\`\`

### 5. *Breakeven* Protection

\`\`\`
Profit >= BE pips (ATR * 2.0, fallback 15 pips):
  → Pindah SL ke breakeven + 0.5*ATR buffer
  (tidak bisa rugi lagi)
\`\`\`

### 6. *Trailing Stop*

\`\`\`
Profit >= trail start pips (ATR * 4.0, fallback 25 pips):
  → SL mengikuti harga dengan jarak ATR * 3.0 (fallback 10 pips)
  (kunci profit sambil biarkan profit berjalan)

Impulse candle (range > 1.5x ATR):
  → Trail diperketat ke 1.5x ATR
\`\`\`

### 7. Default: HOLD

\`\`\`
Tidak ada kondisi terpenuhi → HOLD posisi
\`\`\`

---

## SmartMarketCloseHandler

*Market close handler* menentukan aksi cerdas saat market mendekati penutupan harian atau weekend.

### Market Hours (XAUUSD)

\`\`\`
Minggu 17:00 EST (Senin 05:00 WIB) → Jumat 17:00 EST (Sabtu 05:00 WIB)
24 jam, 5 hari seminggu

Daily close:  05:00 WIB (= 17:00 EST hari sebelumnya)
Weekend close: Sabtu 05:00 WIB (= Jumat 17:00 EST)
\`\`\`

### Diagram Keputusan

\`\`\`mermaid
flowchart TD
    START([Posisi Terbuka\\nDekat Close?]) --> NEAR{Dekat market close?\\n2 jam sebelum}

    NEAR -->|Tidak| NORMAL([NORMAL\\nLanjut trading biasa])

    NEAR -->|Ya| WEEKEND{Dekat weekend?\\nJumat < 30 menit}

    WEEKEND -->|Tidak, daily close| PROFIT_D{Profit >= $10?}
    PROFIT_D -->|Ya| CLOSE_P([CLOSE_PROFIT\\nAmankan profit\\nsebelum daily close])
    PROFIT_D -->|Tidak, profit kecil| VCLOSE{< 30 menit\\nke close?}
    VCLOSE -->|Ya, profit > 0| CLOSE_P
    VCLOSE -->|Tidak / loss| LOSS_D{Loss > $100?}
    LOSS_D -->|Ya| CUT_D([CUT_LOSS\\nLoss terlalu besar])
    LOSS_D -->|Tidak| HOLD_D([HOLD_LOSS\\nBisa recovery besok])

    WEEKEND -->|Ya, weekend| PROFIT_W{Profit >= $10?}
    PROFIT_W -->|Ya| CLOSE_W([CLOSE_PROFIT\\nAmankan sebelum\\nweekend])
    PROFIT_W -->|Tidak, loss| SL_CHECK{SL > 50% hit?}
    SL_CHECK -->|Ya| CUT_W([CUT_LOSS_WEEKEND\\nHindari gap risk])
    SL_CHECK -->|Tidak| BIG_LOSS{Loss > $100?}
    BIG_LOSS -->|Ya| CUT_W
    BIG_LOSS -->|Tidak| HOLD_W([HOLD_LOSS\\nBisa recovery Senin])

    style CLOSE_P fill:#27ae60,color:#fff
    style CLOSE_W fill:#27ae60,color:#fff
    style CUT_D fill:#e74c3c,color:#fff
    style CUT_W fill:#e74c3c,color:#fff
    style HOLD_D fill:#3498db,color:#fff
    style HOLD_W fill:#3498db,color:#fff
    style NORMAL fill:#95a5a6,color:#fff
\`\`\`

### Konfigurasi

\`\`\`python
SmartMarketCloseHandler(
    daily_close_hour_wib=5,          # 05:00 WIB
    hours_before_close=2.0,          # "Dekat close" = 2 jam sebelumnya
    min_profit_to_take=10.0,         # Ambil profit >= $10 sebelum close
    max_loss_to_hold=100.0,          # Hold loss sampai $100
    weekend_loss_cut_percent=50.0,   # Cut jika SL >50% hit sebelum weekend
)
\`\`\`

### 4 Rekomendasi

| Rekomendasi | Kondisi | Aksi |
|-------------|---------|------|
| **CLOSE_PROFIT** | Profit + dekat close | Tutup, amankan profit |
| **CUT_LOSS_WEEKEND** | Loss besar + dekat weekend | Tutup, hindari *gap risk* |
| **HOLD_LOSS** | Loss kecil + dekat close | Hold, bisa recovery |
| **NORMAL** | Belum dekat close | Lanjut normal |

---

## Market Analysis (*Urgency Score*)

*Urgency score* menghitung tingkat bahaya pasar pada skala 0-10. Skor tinggi memicu exit otomatis.

\`\`\`
Score dimulai dari 0, lalu ditambah:

Regime crisis/high_vol:     +3
ML opposite >75% confidence: +2
RSI >75 (overbought):       +2
RSI <25 (oversold):         +2
Trend + momentum berlawanan: +3

Total max: ~10
Score >= 7 = HIGH URGENCY → tutup jika ada profit
\`\`\`

| Komponen | Skor | Kondisi |
|----------|------|---------|
| Regime berbahaya | +3 | Crisis atau high volatility |
| ML sinyal berlawanan | +2 | Confidence > 75% + arah berlawanan posisi |
| RSI *overbought* | +2 | RSI > 75 (bahaya untuk posisi BUY) |
| RSI *oversold* | +2 | RSI < 25 (bahaya untuk posisi SELL) |
| Trend + momentum berlawanan | +3 | Trend dan momentum searah melawan posisi |
| **Threshold** | **>= 7** | **Tutup posisi jika ada profit** |

---

## Konfigurasi SmartPositionManager

\`\`\`python
SmartPositionManager(
    # Fallback jika ATR tidak tersedia
    breakeven_pips=15.0,             # Breakeven setelah 15 pips profit
    trail_start_pips=25.0,           # Mulai trailing setelah 25 pips
    trail_step_pips=10.0,            # Trail distance: 10 pips

    # ATR-adaptive exit multipliers (#24B)
    atr_be_mult=2.0,                 # Breakeven = ATR * 2.0
    atr_trail_start_mult=4.0,        # Trail start = ATR * 4.0
    atr_trail_step_mult=3.0,         # Trail step = ATR * 3.0

    # Proteksi profit
    min_profit_to_protect=50.0,      # Min $50 untuk proteksi profit
    max_drawdown_from_peak=30.0,     # Max 30% drawdown dari peak

    # Market close handler
    enable_market_close_handler=True, # Aktifkan market close handler
    min_profit_before_close=10.0,    # Take profit $10+ sebelum close
    max_loss_to_hold=100.0,          # Hold loss sampai $100
)
\`\`\`
`,
  },
  {
    slug: "risk-engine",
    title: "Risk Engine",
    category: "Engine & Data",
    icon: "Calculator",
    description: "Perhitungan risiko, Kelly criterion, dan position sizing otomatis",
    content: `# Risk Engine — Mesin Risiko & Circuit Breaker

> **File:** \`src/risk_engine.py\`
> **Class:** \`RiskEngine\`
> **Digunakan oleh:** \`main_live.py\`

---

## Apa Itu Risk Engine?

Risk Engine adalah **lapisan proteksi fundamental** yang menghitung ukuran posisi, memvalidasi order, dan mengaktifkan circuit breaker saat batas risiko terlampaui.

**Analogi:** Risk Engine seperti **sistem rem ABS di mobil** — menghitung kecepatan aman, memvalidasi manuver, dan menghentikan paksa jika ada bahaya.

---

## 4 Fungsi Utama

### 1. Position Sizing (Kelly Criterion)

\`\`\`
Risk-Constrained Half-Kelly:

1. Hitung Kelly fraction:
   f* = (p × b - q) / b
   dimana:
     p = win rate (misal 0.55)
     q = 1 - p (0.45)
     b = avg win/loss ratio (misal 2.0)

2. Cap Kelly: max 25%

3. Half-Kelly: f* × 0.5 (safety)

4. Apply regime multiplier (0.5x - 1.0x)

5. Cap di config limit: max risk_per_trade%

6. Hitung lot:
   risk_amount = balance × actual_risk%
   lot = risk_amount / (SL_pips × pip_value)

7. Round ke lot_step, clamp ke min/max
\`\`\`

**Contoh:**

\`\`\`
Balance: $5,000
Win rate: 55%
Win/Loss ratio: 2.0
Kelly: (0.55 × 2.0 - 0.45) / 2.0 = 0.325 (32.5%)
Half-Kelly: 16.25%
Cap: min(16.25%, 1.0%) = 1.0%
Risk amount: $50
SL distance: 50 pips ($5 per pip per 0.01 lot)
Lot: $50 / (50 × $1) = 0.01 lot (menambahkan regime multiplier)
\`\`\`

### 2. Risk Check (Real-time)

\`\`\`python
check_risk(balance, equity, open_positions, current_price)
    |
    v
Hitung daily P/L: equity - starting_balance
    |
    v
Cek circuit breaker aktif? → can_trade = False
    |
    v
Daily loss >= max_daily_loss%? → CIRCUIT BREAKER
    |
    v
Posisi >= max_positions? → can_trade = False
    |
    v
Return RiskMetrics(daily_pnl, drawdown, can_trade, reason)
\`\`\`

### 3. Order Validation

\`\`\`python
validate_order(type, entry, sl, tp, lot, price, balance)
    |
    ├── Circuit breaker aktif? → REJECT
    ├── BUY: SL >= entry? → REJECT ("SL harus di bawah entry")
    ├── BUY: TP <= entry? → REJECT ("TP harus di atas entry")
    ├── Lot < minimum? → REJECT
    ├── Lot > maximum? → REJECT
    ├── Entry terlalu jauh dari current price (>0.1%)? → REJECT
    ├── Risk% > 1.5× config limit? → REJECT
    └── Semua OK → APPROVED
\`\`\`

### 4. Circuit Breaker

\`\`\`
TRIGGER:
  Daily loss >= max_daily_loss% (3% untuk $5K account)

EFEK:
  → can_trade = False
  → Semua entry baru DITOLAK
  → TIDAK menutup posisi yang ada

RESET:
  → Otomatis pada hari baru
  → Manual via reset_circuit_breaker()
\`\`\`

---

## Daily Stats Tracking

\`\`\`python
# Auto-initialize setiap hari baru
_daily_stats[today] = {
    "starting_balance": equity,   # Basis untuk % hitung
    "trades": 0,                  # Total trade hari ini
    "wins": 0,                    # Trade profit
    "losses": 0,                  # Trade loss
}
\`\`\`

---

## Return Types

### RiskMetrics

\`\`\`python
@dataclass
class RiskMetrics:
    daily_pnl: float           # P/L hari ini ($)
    daily_pnl_percent: float   # P/L hari ini (%)
    open_exposure: float       # Total exposure ($)
    max_drawdown: float        # Drawdown dari peak (%)
    position_count: int        # Jumlah posisi terbuka
    can_trade: bool            # Boleh buka posisi baru?
    reason: str                # Alasan
\`\`\`

### PositionSizeResult

\`\`\`python
@dataclass
class PositionSizeResult:
    lot_size: float            # Ukuran lot yang dihitung
    risk_amount: float         # Risk dalam USD
    risk_percent: float        # Risk dalam %
    stop_distance: float       # Jarak SL (harga)
    take_profit_distance: float # Jarak TP (harga)
    approved: bool             # Disetujui?
    rejection_reason: str      # Alasan penolakan
\`\`\`

---

## Hubungan dengan Smart Risk Manager

\`\`\`
RiskEngine (modul ini)
├── Kelly Criterion position sizing
├── Circuit breaker (daily loss limit)
├── Order validation
└── Foundational risk checks

SmartRiskManager (05-Risk-Management.md)
├── 4 trading modes (NORMAL/RECOVERY/PROTECTED/STOPPED)
├── Smart exit logic (12 kondisi)
├── Position monitoring per-detik
└── Higher-level risk decisions
\`\`\`

**RiskEngine** adalah mesin kalkulasi dasar, **SmartRiskManager** adalah manajer tingkat tinggi yang menggunakannya.
`,
  },
  {
    slug: "database",
    title: "Database",
    category: "Engine & Data",
    icon: "Database",
    description: "Skema PostgreSQL dan penyimpanan data perdagangan",
    content: `# *Database Module* — *PostgreSQL* Integration

> **File:** \`src/db/connection.py\`, \`src/db/repository.py\`
> **Database:** *PostgreSQL*
> **Library:** psycopg2 (*connection pooling*)

---

## Apa Itu *Database Module*?

*Database Module* menyediakan **penyimpanan persisten** untuk semua data trading — trade history, training log, sinyal, snapshot pasar, dan status bot. Menggunakan *PostgreSQL* dengan *connection pooling* untuk performa tinggi.

**Analogi:** *Database Module* seperti **arsip perpustakaan** — menyimpan semua catatan trading secara terorganisir, bisa dicari kapan saja, dan tidak hilang meski bot di-restart.

---

## Arsitektur

\`\`\`mermaid
graph TD
    TL[TradeLogger] -->|write| TR[TradeRepository]
    TL -->|write| SigR[SignalRepository]
    TL -->|write| MSR[MarketSnapshotRepository]
    AT[AutoTrainer] -->|write| TrR[TrainingRepository]
    ML[main_live.py] -->|write| BSR[BotStatusRepository]
    ML -->|write| DSR[DailySummaryRepository]
    DASH[Dashboard] -.->|read| TR
    DASH -.->|read| SigR
    DASH -.->|read| MSR
    DASH -.->|read| TrR
    DASH -.->|read| BSR
    DASH -.->|read| DSR

    TR --> DC[DatabaseConnection<br/><i>Singleton</i>]
    SigR --> DC
    MSR --> DC
    TrR --> DC
    BSR --> DC
    DSR --> DC

    DC --> POOL[ThreadedConnectionPool<br/>1 – 10 koneksi]
    POOL --> PG[(PostgreSQL Server)]

    style DC fill:#2d6a4f,stroke:#1b4332,color:#fff
    style PG fill:#1b4332,stroke:#081c15,color:#fff
    style POOL fill:#40916c,stroke:#2d6a4f,color:#fff
\`\`\`

---

## Connection (*Singleton* + Pooling)

\`\`\`python
class DatabaseConnection:
    """
    Thread-safe singleton dengan connection pooling.

    - Hanya 1 instance (singleton pattern)
    - Pool: 1-10 koneksi (ThreadedConnectionPool)
    - Auto-reconnect jika putus
    - Context manager support
    """
\`\`\`

\`DatabaseConnection\` menerapkan pola *singleton* yang *thread-safe* — hanya satu instance yang pernah dibuat selama proses berjalan. Akses ke database dilakukan melalui *context manager* (\`with db.get_cursor() as cur\`) sehingga koneksi selalu dikembalikan ke pool setelah selesai.

### Konfigurasi

\`\`\`
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=trading_bot
DB_PASSWORD=trading_bot_2026
\`\`\`

### Penggunaan

\`\`\`python
from src.db import get_db, init_db

# Initialize
if init_db():
    db = get_db()

    # Query dengan context manager
    with db.get_cursor() as cur:
        cur.execute("SELECT * FROM trades WHERE profit_usd > 0")
        rows = cur.fetchall()

    # Simple execute
    result = db.execute("SELECT count(*) FROM trades", fetch=True)
\`\`\`

---

## 6 *Repository*

Setiap *repository* bertanggung jawab atas satu tabel dan menyediakan method khusus untuk operasi CRUD.

### 1. TradeRepository

| Method | Fungsi |
|--------|--------|
| \`insert_trade()\` | Insert trade baru (saat open) |
| \`update_trade_close()\` | Update exit data (saat close) |
| \`get_trade_by_ticket()\` | Cari trade per ticket |
| \`get_open_trades()\` | Trade yang belum ditutup |
| \`get_recent_trades(100)\` | 100 trade terakhir |
| \`get_trades_for_training(30)\` | Trade 30 hari untuk ML |
| \`get_daily_stats(date)\` | Statistik per hari |
| \`get_session_stats("London", 30)\` | Statistik per sesi |
| \`get_smc_pattern_stats(30)\` | Performa per pola SMC |

### 2. TrainingRepository

| Method | Fungsi |
|--------|--------|
| \`insert_training_run()\` | Catat mulai training |
| \`update_training_complete()\` | Update hasil training |
| \`mark_rollback()\` | Tandai model di-rollback |
| \`get_latest_successful()\` | Training sukses terakhir |
| \`get_training_history(20)\` | 20 training terakhir |

### 3. SignalRepository

| Method | Fungsi |
|--------|--------|
| \`insert_signal()\` | Catat sinyal yang dihasilkan |
| \`mark_executed()\` | Tandai sinyal yang dieksekusi |
| \`get_recent_signals(100)\` | 100 sinyal terakhir |
| \`get_signal_stats(24)\` | Statistik 24 jam |

### 4. MarketSnapshotRepository

| Method | Fungsi |
|--------|--------|
| \`insert_snapshot()\` | Simpan snapshot pasar |
| \`get_recent_snapshots(60)\` | Snapshot 60 menit terakhir |

### 5. BotStatusRepository

| Method | Fungsi |
|--------|--------|
| \`insert_status()\` | Catat status bot |
| \`get_latest_status()\` | Status terbaru |

### 6. DailySummaryRepository

| Method | Fungsi |
|--------|--------|
| \`upsert_summary()\` | Insert/update ringkasan harian |
| \`get_summary(date)\` | Ringkasan per tanggal |
| \`get_recent_summaries(30)\` | 30 hari terakhir |

---

## Tabel Database

### Entity-Relationship Diagram

\`\`\`mermaid
erDiagram
    trades {
        bigint ticket PK
        varchar symbol
        varchar direction
        float entry_price
        float exit_price
        float stop_loss
        float take_profit
        float lot_size
        float profit_usd
        float profit_pips
        timestamp opened_at
        timestamp closed_at
        int duration_seconds
        varchar entry_regime
        float entry_volatility
        varchar entry_session
        varchar smc_signal
        float smc_confidence
        text smc_reason
        bool smc_fvg_detected
        bool smc_ob_detected
        bool smc_bos_detected
        bool smc_choch_detected
        varchar ml_signal
        float ml_confidence
        varchar market_quality
        float market_score
        float dynamic_threshold
        varchar exit_reason
        varchar exit_regime
        varchar exit_ml_signal
        float balance_before
        float balance_after
        float equity_at_entry
        json features_entry
        json features_exit
        varchar bot_version
        varchar trade_mode
    }

    training_runs {
        serial id PK
        varchar training_type
        int bars_used
        int num_boost_rounds
        bool hmm_trained
        int hmm_n_regimes
        bool xgb_trained
        float train_auc
        float test_auc
        float train_accuracy
        float test_accuracy
        varchar model_path
        varchar backup_path
        bool success
        text error_message
        timestamp started_at
        timestamp completed_at
        int duration_seconds
        bool rolled_back
        text rollback_reason
        timestamp rollback_at
    }

    signals {
        serial id PK
        timestamp signal_time
        varchar symbol
        float price
        varchar signal_type
        varchar signal_source
        float combined_confidence
        varchar regime
        varchar session
        float volatility
        float market_score
        bool executed
        text execution_reason
        bigint trade_ticket FK
    }

    market_snapshots {
        serial id PK
        timestamp snapshot_time
        varchar symbol
        float price
        float open
        float high
        float low
        float close
        varchar regime
        float volatility
        varchar session
        float atr
        float spread
        varchar ml_signal
        float ml_confidence
        varchar smc_signal
        float smc_confidence
        int open_positions
        float floating_pnl
        json features
    }

    bot_status {
        serial id PK
        timestamp status_time
        bool is_running
        varchar status
        int loop_count
        float avg_execution_ms
        int uptime_seconds
        float balance
        float equity
        float margin_used
        int open_positions
        float floating_pnl
        float daily_pnl
        varchar risk_mode
        varchar current_session
        bool is_golden_time
    }

    daily_summaries {
        date summary_date PK
        int total_trades
        int winning_trades
        int losing_trades
        int breakeven_trades
        float gross_profit
        float gross_loss
        float net_profit
        float start_balance
        float end_balance
        float win_rate
        float profit_factor
        float avg_win
        float avg_loss
        int sydney_trades
        int tokyo_trades
        int london_trades
        int ny_trades
        int golden_trades
        int fvg_trades
        int fvg_wins
        int ob_trades
        int ob_wins
    }

    trades ||--o{ signals : "trade_ticket"
    daily_summaries ||--o{ trades : "summary_date covers opened_at"
\`\`\`

### trades

\`\`\`sql
├── ticket, symbol, direction
├── entry_price, exit_price, stop_loss, take_profit
├── lot_size, profit_usd, profit_pips
├── opened_at, closed_at, duration_seconds
├── entry_regime, entry_volatility, entry_session
├── smc_signal, smc_confidence, smc_reason
├── smc_fvg_detected, smc_ob_detected, smc_bos_detected, smc_choch_detected
├── ml_signal, ml_confidence
├── market_quality, market_score, dynamic_threshold
├── exit_reason, exit_regime, exit_ml_signal
├── balance_before, balance_after, equity_at_entry
├── features_entry (JSON), features_exit (JSON)
└── bot_version, trade_mode
\`\`\`

### training_runs

\`\`\`sql
├── training_type, bars_used, num_boost_rounds
├── hmm_trained, hmm_n_regimes
├── xgb_trained, train_auc, test_auc
├── train_accuracy, test_accuracy
├── model_path, backup_path
├── success, error_message
├── started_at, completed_at, duration_seconds
└── rolled_back, rollback_reason, rollback_at
\`\`\`

### signals

\`\`\`sql
├── signal_time, symbol, price
├── signal_type, signal_source, combined_confidence
├── smc_*, ml_*
├── regime, session, volatility, market_score
└── executed, execution_reason, trade_ticket
\`\`\`

### market_snapshots

\`\`\`sql
├── snapshot_time, symbol, price, OHLC
├── regime, volatility, session, ATR, spread
├── ml_signal, ml_confidence, smc_signal, smc_confidence
└── open_positions, floating_pnl, features (JSON)
\`\`\`

### bot_status

\`\`\`sql
├── status_time, is_running, status
├── loop_count, avg_execution_ms, uptime_seconds
├── balance, equity, margin_used
├── open_positions, floating_pnl, daily_pnl
└── risk_mode, current_session, is_golden_time
\`\`\`

### daily_summaries

\`\`\`sql
├── summary_date
├── total/winning/losing/breakeven_trades
├── gross_profit, gross_loss, net_profit
├── start_balance, end_balance
├── win_rate, profit_factor, avg win/loss
├── trades per session (sydney/tokyo/london/ny/golden)
└── SMC pattern stats (fvg/ob trades & wins)
\`\`\`

---

## *Graceful Degradation*

\`\`\`
PostgreSQL tersedia?
├── Ya → Gunakan DB + CSV backup
└── Tidak → CSV saja (semua tetap berjalan)

Bot TIDAK pernah crash karena database.
\`\`\`

*Graceful degradation* memastikan bot tetap beroperasi penuh meskipun *PostgreSQL* tidak tersedia. Semua operasi database dibungkus dengan \`try/except\` — jika koneksi gagal, data ditulis ke CSV sebagai fallback. Saat database kembali online, bot otomatis menggunakan koneksi pool kembali tanpa restart.
`,
  },
  {
    slug: "main-live",
    title: "Orkestrator Utama",
    category: "Orkestrator",
    icon: "Play",
    description: "Async main loop — inti dari trading bot yang mengkoordinasi semua komponen",
    content: `# Main Live — *Orchestrator* Utama

> **File:** \`main_live.py\`
> **Class:** \`TradingBot\`
> **Runtime:** *Async event loop* (asyncio)
> **Mode:** *Candle-based* (analisis penuh hanya saat candle baru M15)
> **Target:** < 0.05 detik per iterasi analisis

---

## Apa Itu Main Live?

Main Live adalah **otak pusat** yang mengorkestrasi semua komponen bot. Menjalankan loop *candle-based* — analisis penuh hanya dijalankan saat candle M15 baru terbentuk, dengan pengecekan posisi setiap 10 detik di antara candle. Mengkoordinasikan 15+ komponen dari data fetching hingga order execution.

**Analogi:** Main Live seperti **konduktor orkestra** — tidak memainkan alat musik sendiri, tapi mengarahkan semua pemain (komponen) agar bermain harmonis pada waktu yang tepat.

---

## Diagram Alur Main Loop

\`\`\`mermaid
flowchart TD
    A[STARTUP] --> B{Candle Baru?}
    B -- Ya --> C["Phase 1: DATA\\nFetch 200 bar, Feature Eng,\\nSMC, HMM, XGBoost"]
    B -- Tidak --> G["Position Check Only\\n(setiap ~10 detik)"]
    G --> H[Sleep ~5 detik]
    C --> D["Phase 2: MONITORING\\nCek posisi terbuka,\\n12 kondisi exit"]
    D --> E["Phase 3: ENTRY\\n14 Filter harus PASS,\\nExecute trade"]
    E --> F["Phase 4: PERIODIK\\nAuto-retrain, Market update,\\nHourly analysis, Daily summary"]
    F --> H
    H --> B

    style A fill:#2d6a4f,color:#fff
    style C fill:#1b4332,color:#fff
    style D fill:#40916c,color:#fff
    style E fill:#52b788,color:#000
    style F fill:#74c69d,color:#000
    style G fill:#b7e4c7,color:#000
    style H fill:#d8f3dc,color:#000
\`\`\`

---

## Diagram *Startup* / *Shutdown*

\`\`\`mermaid
flowchart LR
    subgraph STARTUP
        direction TB
        S1[Load .env Config] --> S2[Connect MT5\\nmax 3 retry]
        S2 --> S3[Load HMM Model]
        S3 --> S4[Load XGBoost Model]
        S4 --> S5[Init SmartRiskManager]
        S5 --> S6[Init SessionFilter\\nWIB timezone]
        S6 --> S7[Init Telegram + Logger]
        S7 --> S8[Init AutoTrainer]
        S8 --> S9["Telegram: BOT STARTED"]
        S9 --> S10[Mulai Main Loop]
    end

    subgraph SHUTDOWN
        direction TB
        X1[Signal SIGINT/SIGTERM] --> X2[Hentikan Loop]
        X2 --> X3["Telegram: BOT STOPPED\\n(balance, trades, uptime)"]
        X3 --> X4[Disconnect MT5]
        X4 --> X5[Close DB Connections]
        X5 --> X6[Exit]
    end

    STARTUP -.->|"Runtime\\n(loop berjalan)"| SHUTDOWN

    style S1 fill:#2d6a4f,color:#fff
    style S10 fill:#52b788,color:#000
    style X1 fill:#9d0208,color:#fff
    style X6 fill:#d00000,color:#fff
\`\`\`

---

## Komponen yang Dimuat

\`\`\`python
class TradingBot:
    def __init__(self):
        # Koneksi
        self.mt5 = MT5Connector(...)              # Jembatan ke broker
        self.telegram = TelegramNotifier(...)       # Notifikasi

        # AI Models
        self.ml_model = TradingModel(...)           # XGBoost predictor
        self.regime_detector = MarketRegimeDetector(...) # HMM regime
        self.smc = SMCAnalyzer(...)                 # Smart Money Concepts

        # Analisis
        self.features = FeatureEngineer()           # 40+ fitur
        self.dynamic_confidence = DynamicConfidenceManager(...) # Threshold
        self.session_filter = SessionFilter(...)     # Waktu trading
        # self.news_agent = NewsAgent(...)           # NONAKTIF (line 64)
        #   -> Dikomentari karena backtest membuktikan
        #      News Agent merugikan $178 profit.
        #      ML model sudah menangani volatilitas.

        # Risiko
        self.smart_risk = SmartRiskManager(...)     # Risk management
        self.risk_engine = RiskEngine(...)          # Kelly criterion
        self.position_manager = SmartPositionManager(...) # Position mgmt

        # Logging & Training
        self.trade_logger = TradeLogger(...)        # Pencatat trade
        self.auto_trainer = AutoTrainer(...)        # Retraining otomatis

        # State
        self.flash_crash_detector = FlashCrashDetector(...) # Proteksi
\`\`\`

> **Catatan:** \`NewsAgent\` **NONAKTIF** — import dikomentari di \`main_live.py\` line 64 (\`# DISABLED\`). Backtest membuktikan News Agent justru mengurangi profit sebesar $178 karena ML model sudah cukup menangani volatilitas pasar. Variabel \`self.news_agent\` di-set \`None\` (line 165).

---

## Main Loop (*Candle-Based*)

\`\`\`
STARTUP:
  Load models -> Connect MT5 -> Send Telegram startup
    |
    v
LOOP UTAMA (cek setiap ~5 detik, analisis pada candle baru):
    |
    +-- Fetch 2 bar terakhir -> cek apakah candle baru terbentuk
    |
    |===[CANDLE BARU? YA -> FULL ANALYSIS]=================
    |
    |  |===[PHASE 1: DATA]================================
    |  |
    |  +-- Fetch 200 bar M15 XAUUSD dari MT5
    |  +-- Feature Engineering (40+ fitur)
    |  +-- SMC Analysis (Swing, FVG, OB, BOS, CHoCH)
    |  +-- HMM Regime Detection
    |  +-- XGBoost Prediction
    |  |
    |  |===[PHASE 2: MONITORING]===========================
    |  |
    |  +-- Cek posisi terbuka
    |  |   +-- Untuk setiap posisi:
    |  |       +-- Update profit & momentum
    |  |       +-- 12 kondisi exit (smart_risk.evaluate_position)
    |  |       +-- Jika should_close -> tutup -> log -> Telegram
    |  |
    |  +-- Position Manager (trailing SL, breakeven)
    |  |   +-- Smart Market Close Handler
    |  |
    |  |===[PHASE 3: ENTRY]================================
    |  |
    |  +-- [1]  Flash Crash Guard -> tidak ada crash?
    |  +-- [2]  Regime Filter -> bukan SLEEP / CRISIS?
    |  +-- [3]  Risk Check -> equity & drawdown aman?
    |  +-- [4]  Session Filter -> boleh trading (WIB)?
    |  +-- [5]  SMC Signal -> ada setup?
    |  +-- [6]  Signal Combination -> sinyal valid (quality != AVOID)?
    |  +-- [7]  H1 Bias (#31B) -> H1 EMA20 selaras?
    |  +-- [8]  Time Filter (#34A) -> bukan jam 9/21 WIB?
    |  +-- [9]  Trade Cooldown -> cukup jeda sejak trade terakhir?
    |  +-- [10] Smart Risk Gate -> mode bukan STOPPED?
    |  +-- [11] Lot Size -> > 0 setelah kalkulasi?
    |  +-- [12] Position Limit -> < 2 posisi terbuka?
    |  +-- [13] Slippage Validation -> harga aktual vs expected
    |  +-- [14] Partial Fill Check -> volume aktual vs requested
    |  +-- SEMUA PASS -> Execute trade
    |  +-- Register position (harga & volume AKTUAL) -> Log -> Telegram
    |  |
    |  |===[PHASE 4: PERIODIK]=============================
    |  |
    |  +-- Setiap 20 candle (~5 jam M15): Cek auto-retrain
    |  +-- Setiap 30 menit: Market update (Telegram)
    |  +-- Setiap 1 jam: Hourly analysis (Telegram)
    |  +-- Pergantian hari: Daily summary + reset
    |
    |===[CANDLE BARU? TIDAK -> POSITION CHECK ONLY]========
    |
    +-- Setiap 10 detik: cek posisi terbuka saja
    |   +-- Fetch 50 bar (minimal data)
    |   +-- Hitung fitur untuk ML check
    |   +-- Evaluasi exit conditions per posisi
    |
    v
    Tunggu ~5 detik -> Loop lagi
\`\`\`

---

## Detail 14 *Entry* Filter

| # | Filter | Deskripsi | Sumber |
|---|--------|-----------|--------|
| 1 | Flash Crash Guard | Deteksi pergerakan harga ekstrem (>X%) dalam 5 bar terakhir | \`FlashCrashDetector\` |
| 2 | Regime Filter | Blok *entry* jika HMM regime = SLEEP atau CRISIS | \`regime_detector\` |
| 3 | Risk Check | Validasi equity, drawdown, dan balance aman | \`risk_engine\` |
| 4 | Session Filter | Hanya trading di sesi aktif (Sydney/London/NY, WIB) | \`session_filter\` |
| 5 | SMC Signal | Harus ada setup SMC (Order Block, FVG, BOS, CHoCH) | \`smc.generate_signal()\` |
| 6 | Signal Combination | Gabung SMC + ML, blok jika market quality AVOID | \`_combine_signals()\` |
| 7 | H1 Bias (#31B) | BUY hanya jika H1 BULLISH, SELL hanya jika H1 BEARISH | \`_get_h1_bias()\` |
| 8 | Time Filter (#34A) | Skip jam 9 dan 21 WIB (likuiditas rendah / whipsaw) | Hardcoded WIB check |
| 9 | Trade Cooldown | Jeda minimum antar trade (mencegah overtrade) | \`_last_trade_time\` |
| 10 | Smart Risk Gate | Cek mode risk (NORMAL/CAUTIOUS/STOPPED) | \`smart_risk\` |
| 11 | Lot Size > 0 | Pastikan kalkulasi lot menghasilkan size > 0 | \`smart_risk.calculate_lot_size()\` |
| 12 | Position Limit | Maksimal 2 posisi terbuka bersamaan | \`smart_risk.can_open_position()\` |
| 13 | Slippage Validation | Cek harga eksekusi aktual vs harga yang diharapkan | Post-execution check |
| 14 | Partial Fill Check | Cek volume aktual yang terisi vs volume yang diminta | Post-execution check |

---

## *Startup* Sequence

\`\`\`
1.  Load konfigurasi dari .env
2.  Connect ke MT5 (max 3 retry)
3.  Load model HMM dari models/hmm_regime.pkl
4.  Load model XGBoost dari models/xgboost_model.pkl
5.  Initialize SmartRiskManager (set balance, limits)
6.  Initialize SessionFilter (WIB timezone)
7.  Initialize TelegramNotifier
8.  Initialize TradeLogger
9.  Initialize AutoTrainer
10. Send Telegram: "BOT STARTED" (config, balance, risk settings)
11. Mulai main loop
\`\`\`

> **Note:** NewsAgent **tidak** diinisialisasi saat *startup* — import dan inisialisasi dikomentari sejak backtest membuktikan kerugian $178.

---

## *Shutdown* Sequence

\`\`\`
1. Signal SIGINT/SIGTERM diterima
2. Hentikan loop utama
3. Kirim Telegram: "BOT STOPPED" (balance, trades, uptime)
4. Disconnect MT5
5. Close database connections
6. Exit
\`\`\`

---

## Error Handling (*Fault Tolerance*)

\`\`\`
Setiap iterasi loop dibungkus try-except:

try:
    # Fetch data, analyze, trade
except ConnectionError:
    # MT5 disconnected -> reconnect()
except Exception as e:
    # Log error -> lanjut loop berikutnya
    # Bot TIDAK crash dari error tunggal

Prinsip: NEVER STOP TRADING karena error non-kritis
\`\`\`

Bot dirancang dengan prinsip *fault tolerance* — satu error tidak menghentikan seluruh sistem. Setiap iterasi loop dibungkus \`try-except\` sehingga error pada satu candle tidak mempengaruhi candle berikutnya. Koneksi MT5 yang putus akan otomatis di-reconnect.

---

## Timer *Periodik*

| Event | Interval | Aksi |
|-------|----------|------|
| Full analysis + *entry* | Setiap candle baru M15 | Saat candle terbentuk |
| Position *monitoring* | ~10 detik | Di antara candle |
| Performance logging | 4 candle (~1 jam) | \`loop_count % 4\` |
| Auto-retrain check | 20 candle (~5 jam) | \`loop_count % 20\` |
| Market update Telegram | 30 menit | Timer |
| Hourly analysis Telegram | 1 jam | Timer |
| Daily summary | Pergantian hari | Date check |

---

## Performa Target

\`\`\`
Target: < 0.05 detik per iterasi analisis (50ms)

Full Analysis (saat candle baru):
+-- MT5 data fetch:      ~10ms  (200 bar)
+-- Feature engineering:  ~5ms  (Polars, vectorized)
+-- SMC analysis:         ~5ms  (Polars native)
+-- HMM predict:          ~2ms
+-- XGBoost predict:      ~3ms
+-- Position monitoring:  ~5ms
+-- Entry logic:          ~5ms
+-- Overhead:             ~15ms
                          ------
                          ~50ms total

Position Check Only (di antara candle):
+-- MT5 data fetch:      ~5ms   (50 bar saja)
+-- Feature engineering:  ~3ms
+-- ML prediction:        ~3ms
+-- Position evaluation:  ~5ms
+-- Overhead:             ~5ms
                          ------
                          ~21ms total
\`\`\`

---

## Hubungan Semua Komponen

\`\`\`mermaid
flowchart TD
    subgraph BOT["main_live.py — TradingBot"]
        direction TB
        MT5["MT5 Connector"] --> FE["Feature Eng"]
        FE --> SMC["SMC Analyzer"]
        SMC --> HMM["HMM Detector"]
        HMM --> DC["Dynamic Confidence"]
        DC --> XGB["XGBoost Model"]
        XGB --> ENTRY["Entry Logic (14 Filters)<br/>Flash, Regime, Risk, Session,<br/>SMC, H1 Bias, Time, Cooldown,<br/>Smart Risk, Lot, Pos Limit"]
        ENTRY --> RE["Risk Engine"]
        ENTRY --> SRM["Smart Risk Mgr"]
        RE --> EXEC["Execute Order<br/>(BUY/SELL)"]
        SRM --> EXEC
        EXEC --> MT5
        EXEC --> PM["Position Manager"]
        EXEC --> TL["Trade Logger<br/>(DB + CSV)"]
        EXEC --> TG["Telegram Notifier"]
        PM ~~~ AT["Auto Trainer<br/>(retraining)"]
        TG ~~~ NA["News Agent<br/>(NONAKTIF)"]
        AT ~~~ SF["Session Filter<br/>(waktu trading)"]
    end
\`\`\`
`,
  },
  {
    slug: "weakness-analysis",
    title: "Analisis Kelemahan",
    category: "Analisis",
    icon: "AlertTriangle",
    description: "Kelemahan yang diketahui, risiko, dan prioritas perbaikan sistem",
    content: `# Analisis Kelemahan Sistem — *Weakness Analysis*

## Tanggal Analisis Awal: 6 Februari 2026
## Terakhir Diperbarui: 8 Februari 2026

---

## Status Perbaikan

\`\`\`mermaid
pie title Status Kelemahan (8 Item Asli)
    "Sudah Diperbaiki" : 6
    "Sebagian Diperbaiki" : 1
    "Masih Terbuka" : 1
\`\`\`

| # | Item | Status | Tanggal Fix |
|---|------|--------|-------------|
| 1 | *Broker* SL | **DIPERBAIKI** | 7 Feb 2026 |
| 2 | ATR-*based* SL | **DIPERBAIKI** | 7 Feb 2026 |
| 3 | *Faster reversal exit* | **DIPERBAIKI** | 7 Feb 2026 |
| 4 | *Time-based exit* | **DIPERBAIKI** | 7 Feb 2026 |
| 5 | *Dynamic* ML *threshold* | **DIPERBAIKI** | 7 Feb 2026 |
| 6 | *Breakeven logic* | **DIPERBAIKI** | 7 Feb 2026 |
| 7 | *Partial* TP | Sebagian (*Smart TP* 4 level) | 7 Feb 2026 |
| 8 | *Backtest sync* | Masih terbuka | — |

---

## 1. STOP LOSS — ~~KELEMAHAN KRITIS~~ DIPERBAIKI

### 1.1 ~~Tidak Ada *Broker Stop Loss*~~ — DIPERBAIKI

**Sebelum:**
\`\`\`python
result = self.mt5.send_order(
    sl=0,  # MASALAH: Tidak ada SL di broker!
    tp=signal.take_profit,
)
\`\`\`

**Sesudah (sistem saat ini):**
\`\`\`python
# Hitung emergency SL berdasarkan ATR
atr = df["atr"].tail(1).item()
emergency_sl = entry_price - (3.0 * atr) if direction == "BUY" else entry_price + (3.0 * atr)

result = self.mt5.send_order(
    sl=emergency_sl,  # BROKER-LEVEL PROTECTION — aktif!
    tp=signal.take_profit,
)
\`\`\`

**Status:** SL dikirim ke *broker* sebagai proteksi darurat. Jika koneksi internet terputus, **SL di *broker* tetap aktif**.

### 1.2 ~~*Smart Hold* Terlalu Agresif~~ — DIHAPUS

**Status:** Fitur *Smart Hold* telah **dihapus** dari sistem. *Smart Hold* dianggap berbahaya karena perilakunya mirip *martingale* — menahan posisi rugi dengan harapan harga berbalik.

### 1.3 ~~SL Berbasis *Swing* Terlalu Dekat~~ — DIPERBAIKI

**Sesudah (v4 — sistem saat ini):**
\`\`\`python
# SL sekarang menggunakan ATR-based minimum
atr = df["atr"].tail(1).item()
min_sl_distance = 1.5 * atr  # Minimum SL = 1.5 * ATR

if direction == "BUY":
    swing_sl = last_swing_low
    atr_sl = entry - min_sl_distance
    sl = min(swing_sl, atr_sl)  # Pilih yang LEBIH JAUH (lebih aman)
    if entry - sl < min_sl_distance:
        sl = entry - min_sl_distance  # Enforce jarak minimum
\`\`\`

**Status:** SL sekarang MIN(*swing*, 1.5×ATR) dengan jarak minimum yang di-*enforce*.

---

## 2. TAKE PROFIT — SEBAGIAN DIPERBAIKI

### 2.1 ~~TP *Fixed* 2:1 RR~~ — DIPERBAIKI

**Sesudah:**
- TP menggunakan **ENFORCED minimum 1:2 R:R** — sinyal ditolak jika RR < 2.0
- *Smart Take Profit* memiliki **4 level exit** berdasarkan profit:
  - Level 1: $15 — mulai pertimbangkan *take profit*
  - Level 2: $25 — level profit bagus
  - Level 3: $40 — *hard take profit*
  - Level 4: *Peak profit declining* — profit turun dari puncak

### 2.2 Tidak Ada *Partial Take Profit* — SEBAGIAN

**Status:** Sistem tidak memiliki *partial close* yang sebenarnya (tutup 25%/50%/75% posisi), tetapi *Smart TP* dengan 4 level sudah memberikan mekanisme serupa — posisi ditutup seluruhnya pada level profit yang optimal berdasarkan kondisi pasar.

**Saran ke depan:**
\`\`\`python
# Partial TP levels (belum diimplementasikan)
tp_25 = entry + (risk * 0.5)   # 25% posisi di 0.5 RR
tp_50 = entry + (risk * 1.0)   # 25% posisi di 1.0 RR
tp_75 = entry + (risk * 1.5)   # 25% posisi di 1.5 RR
tp_100 = entry + (risk * 2.0)  # 25% posisi di 2.0 RR
\`\`\`

---

## 3. ENTRY TRADE — DIPERBAIKI

### 3.1 ~~ML *Threshold* 50% = *Coin Flip*~~ — DIPERBAIKI

**Sesudah (sistem saat ini):**
- \`DynamicConfidenceManager\` menyesuaikan *threshold* secara otomatis:

| Kondisi Pasar | *Threshold* | Alasan |
|---------------|-------------|--------|
| *Trending* kuat | **0.65** | Sinyal lebih jelas, *threshold* lebih rendah |
| Normal | **0.70** | *Default* standar |
| Bergejolak | **0.75** | Butuh kepastian lebih tinggi |

### 3.2 *Signal Key Reset* Terus — MASIH ADA (Risiko Rendah)

**Status:** *Signal key* masih menggunakan \`int(entry_price)\`, yang bisa berubah antar candle. Risiko rendah karena *cooldown* 5 menit sudah mencegah duplikasi *trade*.

### 3.3 ~~*Pullback Filter Fixed* $2~~ — TIDAK RELEVAN

**Status:** *Pullback Filter* dinonaktifkan (SMC-only mode). Sistem menggunakan **14 *entry filter*** lain yang lebih robust.

---

## 4. EXIT TRADE — DIPERBAIKI

### 4.1 ~~ML *Reversal* Butuh 75% *Confidence*~~ — DIPERBAIKI

**Status:** Sistem sekarang memiliki **12 kondisi *exit*** termasuk:
- *Early Cut* — momentum negatif, tidak menunggu ML *confidence* tinggi
- *Trend Reversal* — ML mendeteksi pembalikan
- *Stall Detection* — harga *stuck* terlalu lama

### 4.2 ~~Tidak Ada *Time-Based Exit*~~ — DIPERBAIKI

**Sesudah:**
\`\`\`python
# Time-based exit sudah aktif
# 4-8 jam maximum duration per trade
trade_duration = (datetime.now() - entry_time).total_seconds() / 3600

if trade_duration > 4 and abs(current_profit) < 5:  # 4 jam tanpa progress
    return True, ExitReason.TIMEOUT
if trade_duration > 8:  # Maximum 8 jam
    return True, ExitReason.TIMEOUT
\`\`\`

### 4.3 ~~Tidak Ada *Breakeven Protection*~~ — DIPERBAIKI

**Sesudah:**
- *Breakeven Protection* aktif sebagai **kondisi *exit* #11**
- *Smart Breakeven* (#28B) — pindah SL ke *breakeven* setelah profit tertentu tercapai
- *Trailing Stop Loss* juga aktif sebagai kondisi *exit* #10

---

## 5. *BACKTEST* vs *LIVE* — MASIH TERBUKA

### 5.1 *Exit Timing* Berbeda

| Aspek | *Backtest* | *Live* |
|-------|------------|--------|
| *Check interval* | Per bar (15 min) | Per 10 detik |
| ML *reversal check* | Setiap 5 bar | Setiap *loop* |
| *Smart Hold* | Tidak ada | **Dihapus juga** |

**Status:** \`backtest_live_sync.py\` disinkronkan dengan \`main_live.py\`, tetapi perbedaan *timing* (bar-based vs real-time) tetap ada dan tidak bisa dihilangkan sepenuhnya.

### 5.2 *Slippage* Tidak Dihitung

**Status:** Masih belum ada simulasi *slippage* di *backtest*. Ini bisa menyebabkan *backtest* terlalu optimis.

**Saran:**
\`\`\`python
SLIPPAGE_PIPS = 0.5  # 0.5 pip slippage

def simulate_entry(entry_price, direction):
    if direction == "BUY":
        return entry_price + SLIPPAGE_PIPS * 0.1
    else:
        return entry_price - SLIPPAGE_PIPS * 0.1
\`\`\`

---

## 6. PRIORITAS PERBAIKAN (DIPERBARUI)

| # | Item | Status | Prioritas |
|---|------|--------|-----------|
| 1 | *Broker* SL | **SELESAI** | ~~P0~~ |
| 2 | ATR-*based* SL | **SELESAI** | ~~P1~~ |
| 3 | *Faster reversal exit* | **SELESAI** | ~~P1~~ |
| 4 | *Time-based exit* | **SELESAI** | ~~P2~~ |
| 5 | *Dynamic* ML *threshold* | **SELESAI** | ~~P2~~ |
| 6 | *Breakeven logic* | **SELESAI** | ~~P3~~ |
| 7 | *Partial* TP | Sebagian | **P3** |
| 8 | *Backtest sync* (*slippage*) | Terbuka | **P3** |

---

## 7. SKENARIO TERBURUK — MITIGASI

### Skenario 1: *Weekend Gap*
- **Sebelum:** Loss *unlimited* tanpa SL di *broker*
- **Sesudah:** *Broker* SL (3× ATR) melindungi posisi. Juga ada *weekend close* — bot menutup semua posisi sebelum penutupan *weekend*

### Skenario 2: *Flash Crash*
- **Sebelum:** *Flash crash* = posisi tidak terproteksi
- **Sesudah:** *Flash Crash Guard* (filter #12) mendeteksi pergerakan >2.5% dalam 1 menit dan **menghentikan semua trading + menutup posisi**

### Skenario 3: *Connection Lost*
- **Sebelum:** Tanpa *broker* SL = loss *unlimited*
- **Sesudah:** *Broker* SL (3× ATR) tetap aktif di *server broker* meskipun koneksi internet terputus

---

## 8. KELEMAHAN BARU YANG TERIDENTIFIKASI

### 8.1 *News Agent* Nonaktif
- Bot tidak memfilter berita berdampak tinggi (NFP, FOMC, CPI)
- *Session Filter* sudah memiliki daftar waktu berita, tapi *News Agent* yang mengambil data *real-time* dinonaktifkan
- **Risiko:** Pasar bisa sangat *volatile* saat berita besar
- **Mitigasi:** ML model dan HMM *regime detector* sudah menangani volatilitas ($178 lebih baik tanpa *News Agent*)

### 8.2 Tidak Ada *Partial Close*
- Posisi selalu ditutup 100% — tidak ada opsi tutup sebagian
- **Risiko:** Kehilangan potensi profit jika harga terus bergerak setelah *take profit*
- **Prioritas:** P3 (nice to have)

### 8.3 *Single Symbol* (XAUUSD)
- Bot hanya trading satu instrumen
- **Risiko:** Bergantung sepenuhnya pada kondisi pasar emas
- **Mitigasi:** XAUUSD adalah instrumen paling *liquid* dan *volatile*, cocok untuk *scalping* M15
`,
  },
];
