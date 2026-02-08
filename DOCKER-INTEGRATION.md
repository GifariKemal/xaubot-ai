# Dashboard Integration with Existing Docker Setup

## 🎯 Overview

Dashboard dan API telah diintegrasikan ke dalam Docker setup yang **sudah ada**. Database PostgreSQL yang sudah running **TIDAK AKAN DIGANGGU**.

## ✅ Existing Setup (Tidak Berubah)

Yang sudah jalan dan **tetap aman**:
- ✅ `trading_bot_db` - PostgreSQL database
- ✅ `trading_bot_network` - Docker network
- ✅ Database schema dengan 7 tables (trades, signals, dll)
- ✅ Volume `postgres_data` untuk persistence

## 🆕 New Services Added

Layanan baru yang ditambahkan:
1. **trading-api** - FastAPI backend untuk dashboard
2. **dashboard** - Next.js web interface
3. **pgadmin** - Database management (optional)

## 🚀 Quick Start

### Option 1: Gunakan Helper Script (Recommended)

```cmd
# Tambahkan dashboard ke setup yang sudah ada
docker-add-dashboard.bat
```

Script ini akan:
1. Check database yang sudah running
2. Build API & Dashboard services
3. Start kedua services baru
4. Connect ke database & network yang sudah ada

### Option 2: Manual Docker Compose

```cmd
# Build hanya services baru
docker-compose build trading-api dashboard

# Start hanya services baru
docker-compose up -d trading-api dashboard
```

## 📊 Access Points

Setelah services running:
- **Dashboard:** http://localhost:3000
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Database:** localhost:5432 (sudah running)

## 🔧 Service Management

### Check Status
```cmd
# Lihat status semua services
docker-status.bat

# Atau manual
docker-compose ps
```

### View Logs
```cmd
# Logs dashboard
docker-compose logs -f dashboard

# Logs API
docker-compose logs -f trading-api

# Logs database
docker-compose logs -f postgres
```

### Restart Services
```cmd
# Restart hanya dashboard
docker-compose restart dashboard

# Restart hanya API
docker-compose restart trading-api

# Restart semua (termasuk database)
docker-compose restart
```

### Remove Dashboard (Keep Database)
```cmd
# Hapus dashboard tapi tetap keep database
docker-remove-dashboard.bat

# Atau manual
docker-compose stop trading-api dashboard
docker-compose rm -f trading-api dashboard
```

## 🔗 Service Architecture

```
┌─────────────────────────────────────────────┐
│         trading_bot_network                  │
├─────────────────────────────────────────────┤
│                                              │
│  📊 Dashboard (NEW)                         │
│     Port: 3000                               │
│     └─> http://trading-api:8000             │
│                                              │
│  🔌 Trading API (NEW)                       │
│     Port: 8000                               │
│     └─> postgres:5432                       │
│                                              │
│  🗄️ PostgreSQL (EXISTING - NO CHANGE)      │
│     Port: 5432                               │
│     Status: Already Running                  │
│     Volume: postgres_data                    │
│                                              │
└─────────────────────────────────────────────┘
```

## 📝 Environment Variables

Edit `.env` untuk konfigurasi:

```env
# MT5 (Required for API)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# Trading
SYMBOL=XAUUSD
CAPITAL=10000

# Database (Already configured)
DB_USER=trading_bot
DB_PASSWORD=trading_bot_2026
DB_NAME=trading_db

# Ports
API_PORT=8000
DASHBOARD_PORT=3000
DB_PORT=5432
```

## 🐛 Troubleshooting

### Dashboard tidak bisa connect ke API

**Check API health:**
```cmd
curl http://localhost:8000/api/health
```

**View API logs:**
```cmd
docker-compose logs -f trading-api
```

### API tidak bisa connect ke database

**Check database:**
```cmd
docker exec trading_bot_db pg_isready -U trading_bot
```

**Check network:**
```cmd
docker network inspect trading_bot_network
```

### Port conflict

Edit `.env` untuk ganti port:
```env
API_PORT=8001
DASHBOARD_PORT=3001
```

Then restart:
```cmd
docker-compose down trading-api dashboard
docker-compose up -d trading-api dashboard
```

## 💾 Data Persistence

**Database data tetap aman:**
- Volume `postgres_data` tetap ada
- Hapus container tidak hapus data
- Data tersimpan di Docker volume

**Check volume:**
```cmd
docker volume ls | findstr postgres
docker volume inspect trading_bot_postgres_data
```

## 🔄 Updates

**Update code dan rebuild:**
```cmd
# Pull latest code
git pull

# Rebuild services baru
docker-compose build trading-api dashboard

# Restart
docker-compose up -d trading-api dashboard
```

**Database tidak perlu rebuild** karena schema sudah ada.

## ⚠️ Important Notes

1. **Database tidak boleh dihapus** - Data trades ada di sini
2. **Jangan run `docker-compose down -v`** - Ini akan hapus volumes
3. **Untuk stop semua:** `docker-compose stop` (data aman)
4. **Untuk restart:** `docker-compose restart` atau `docker-compose up -d`

## 📚 Files Structure

```
xaubot-ai/
├── docker-compose.yml           # Main orchestration (UPDATED)
├── Dockerfile                   # API image (NEW)
├── .env                        # Environment config
├── .dockerignore               # Build exclusions
├── docker-add-dashboard.bat    # Add dashboard script (NEW)
├── docker-remove-dashboard.bat # Remove dashboard script (NEW)
├── docker-status.bat           # Status check script (NEW)
├── docker/
│   └── init-db/
│       └── 01-schema.sql       # Database schema (EXISTING)
└── web-dashboard/
    ├── Dockerfile              # Dashboard image (NEW)
    └── .dockerignore          # Build exclusions
```

## 🎯 Summary

✅ **Database tetap jalan** - Tidak ada perubahan
✅ **Services baru ditambahkan** - API & Dashboard
✅ **Data aman** - Volume persistence
✅ **Easy management** - Helper scripts
✅ **Independent** - Bisa start/stop tanpa ganggu database

---

**Integration completed:** Feb 6, 2026
**Status:** Dashboard integrated with existing Docker setup ✨
