# Quick Start - Tambah Dashboard ke Docker Existing

## Status Saat Ini

✅ **Docker Compose sudah ada**
✅ **Service `postgres` sudah running** (container: `trading_bot_db`)
✅ **Service `trading-api` dan `dashboard` sudah didefinisikan** tapi belum di-build

## 🚀 Cara Menjalankan

### 1. Setup Environment (Kalau Belum)

```cmd
cd "C:\Users\Administrator\Videos\Smart Automatic Trading BOT + AI"

REM Copy environment template kalau belum ada
copy .env.docker.example .env

REM Edit dengan MT5 credentials Anda
notepad .env
```

Pastikan isi `.env`:
```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

SYMBOL=XAUUSD
CAPITAL=10000
```

### 2. Build Services Baru

```cmd
REM Build trading-api dan dashboard
docker-compose build trading-api dashboard
```

Ini akan:
- Build Dockerfile untuk Python API
- Build Dockerfile untuk Next.js Dashboard
- Tidak ganggu database yang sudah running

### 3. Start Services Baru

```cmd
REM Start trading-api dan dashboard
docker-compose up -d trading-api dashboard
```

### 4. Check Status

```cmd
docker-compose ps
```

Output akan menunjukkan:
```
NAME                     STATUS                 PORTS
trading_bot_db           Up (healthy)          0.0.0.0:5432->5432/tcp
trading_bot_api          Up (healthy)          0.0.0.0:8000->8000/tcp
trading_bot_dashboard    Up (healthy)          0.0.0.0:3000->3000/tcp
```

### 5. Akses Dashboard

Buka browser:
- **Dashboard:** http://localhost:3000
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## 📋 Commands Penting

```cmd
# Lihat logs
docker-compose logs -f dashboard
docker-compose logs -f trading-api

# Restart service
docker-compose restart trading-api
docker-compose restart dashboard

# Stop service
docker-compose stop trading-api dashboard

# Start lagi
docker-compose up -d trading-api dashboard

# Rebuild setelah update code
docker-compose build trading-api dashboard
docker-compose up -d trading-api dashboard
```

## 🔍 Troubleshooting

### Build Error

```cmd
# Clean build
docker-compose build --no-cache trading-api dashboard
```

### Service Tidak Start

```cmd
# Check logs
docker-compose logs trading-api
docker-compose logs dashboard

# Check health
curl http://localhost:8000/api/health
curl http://localhost:3000
```

### Port Conflict

Edit `.env`:
```env
API_PORT=8001
DASHBOARD_PORT=3001
```

Lalu restart:
```cmd
docker-compose down trading-api dashboard
docker-compose up -d trading-api dashboard
```

## ⚡ One-Liner (All in One)

```cmd
cd "C:\Users\Administrator\Videos\Smart Automatic Trading BOT + AI" && docker-compose build trading-api dashboard && docker-compose up -d trading-api dashboard && docker-compose ps
```

## 📊 Arsitektur

```
Docker Compose Project: "smart-automatic-trading-bot-ai"
├── postgres (RUNNING) ✅
│   └── trading_bot_db
├── trading-api (BUILD & START) ⚡
│   └── trading_bot_api
└── dashboard (BUILD & START) ⚡
    └── trading_bot_dashboard
```

## ✅ Checklist

- [ ] Copy `.env.docker.example` ke `.env`
- [ ] Edit `.env` dengan MT5 credentials
- [ ] Run: `docker-compose build trading-api dashboard`
- [ ] Run: `docker-compose up -d trading-api dashboard`
- [ ] Check: `docker-compose ps`
- [ ] Open: http://localhost:3000
- [ ] Test API: http://localhost:8000/api/health

---

**That's it!** Simple kan? 🎉
