# Simple Start Guide - XAUBot AI Dashboard

## 🎯 Cara Tercepat (1 Command)

```cmd
start-all.bat
```

Script ini akan:
1. ✅ Check database Docker container
2. 🚀 Start Trading API di http://localhost:8000
3. 🚀 Start Dashboard di http://localhost:3000

Dua window akan terbuka otomatis!

## 📋 Manual Start (Jika Perlu)

### Option 1: Start Semua Sekaligus
```cmd
start-all.bat
```

### Option 2: Start Satu-satu

**Terminal 1: API**
```cmd
start-api.bat
```

**Terminal 2: Dashboard**
```cmd
start-dashboard.bat
```

## ✅ Pre-requisites

### 1. Database (Docker)
Database harus sudah running:
```cmd
# Check status
docker ps | findstr trading_bot_db

# Start jika belum running
docker-compose up -d postgres
```

### 2. Python Environment
- Python 3.11+ installed
- Virtual environment akan dibuat otomatis

### 3. Node.js
- Node.js 18+ installed
- npm dependencies akan diinstall otomatis

## 🌐 Access Points

Setelah start:
- **Dashboard:** http://localhost:3000
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health
- **Status:** http://localhost:8000/api/status

## 🛑 Stop Services

Close kedua command windows atau tekan `Ctrl+C` di masing-masing window.

## 🔍 Troubleshooting

### API Error: "Module not found"

Install dependencies:
```cmd
pip install -r requirements.txt
```

### Dashboard Error: "Module not found"

Install dependencies:
```cmd
cd web-dashboard
npm install
```

### Port Already in Use

**Change API Port:**
Edit `web-dashboard/api/main.py` line terakhir:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Change 8000 to 8001
```

**Change Dashboard Port:**
Edit `web-dashboard/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8001
```

Then start dashboard on different port:
```cmd
cd web-dashboard
set PORT=3001 && npm run dev
```

### Database Not Running

Start database:
```cmd
docker-compose up -d postgres

# Check status
docker ps
```

## 📊 Architecture

```
┌─────────────────────────────────────┐
│      Windows Host Machine            │
├─────────────────────────────────────┤
│                                      │
│  📊 Dashboard (Port 3000)           │
│      npm run dev                     │
│      ↓ HTTP                          │
│  🔌 API (Port 8000)                 │
│      uvicorn main:app                │
│      ↓ PostgreSQL                    │
│  🗄️ Database (Docker)               │
│      trading_bot_db                  │
│                                      │
└─────────────────────────────────────┘
```

## 🎨 Features

Dashboard akan menampilkan:
- ⏰ Real-time XAUUSD price
- 💰 Account balance & equity
- 📈 Price history chart
- 🎯 Trading signals (SMC + ML)
- 🌊 Market regime
- ⚠️ Risk status
- 📋 Open positions
- 📝 Activity logs

## 💡 Tips

1. **Auto-start Database:**
   Tambahkan Docker Desktop ke Windows startup

2. **Keep API Running:**
   Minimize command windows, jangan close

3. **Monitor Logs:**
   Lihat output di command windows untuk debug

4. **Quick Restart:**
   Close windows dan run `start-all.bat` lagi

## 📝 Files

```
start-all.bat         # Start API + Dashboard
start-api.bat         # Start API only
start-dashboard.bat   # Start Dashboard only
```

---

**Super Simple!** Tinggal double-click `start-all.bat` 🎉
