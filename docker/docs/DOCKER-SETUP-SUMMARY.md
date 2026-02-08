# XAUBot AI - Docker Integration Summary

## ✅ Completed Tasks

### 1. **Created Dockerfile for Next.js Dashboard**
- Multi-stage build for optimization
- Standalone output for minimal image size
- Production-ready configuration
- Non-root user for security

**Location:** `web-dashboard/Dockerfile`

### 2. **Created Dockerfile for Python Trading API**
- Python 3.11-slim base image
- FastAPI server with health checks
- Proper dependency management
- Volume mounts for data/logs/models

**Location:** `Dockerfile` (root directory)

### 3. **Updated Docker Compose Configuration**
- 4 services: postgres, trading-api, dashboard, pgadmin
- Proper service dependencies and health checks
- Custom bridge network for inter-service communication
- Environment variable support via .env file
- Volume persistence for database and pgadmin

**Location:** `docker-compose.yml`

### 4. **Created Environment Configuration**
- Template with all required variables
- Clear documentation for each setting
- Default values for non-sensitive configs

**Location:** `docker/.env.docker.example`

### 5. **Created Docker Ignore Files**
- Excludes unnecessary files from images
- Reduces build context size
- Improves build performance

**Locations:**
- `web-dashboard/.dockerignore`
- `.dockerignore` (root)

### 6. **Created Helper Scripts**

#### Windows Batch Scripts:
- `docker\scripts\docker-start.bat` - Start all services
- `docker\scripts\docker-stop.bat` - Stop services with options
- `docker\scripts\docker-logs.bat` - View service logs

#### Linux/Mac Shell Scripts:
- `docker/scripts/docker-start.sh` - Start all services
- `docker/scripts/docker-stop.sh` - Stop services with options
- `docker/scripts/docker-logs.sh` - View service logs

### 7. **Updated Next.js Configuration**
- Enabled standalone output for Docker
- Optimized for production builds

**Location:** `web-dashboard/next.config.ts`

### 8. **Created Comprehensive Documentation**
- Complete Docker setup guide
- Architecture diagram
- Service management commands
- Troubleshooting section
- Security best practices
- Performance tuning tips

**Location:** `DOCKER.md`

### 9. **Updated Main README**
- Added Docker deployment section as recommended method
- Clear quick start instructions
- Links to full documentation

**Location:** `README.md`

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Docker Network                     │
│              (trading_bot_network)                    │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────┐         ┌──────────────────┐  │
│  │   Dashboard      │────────▶│   Trading API    │  │
│  │   (Next.js)      │  HTTP   │   (FastAPI)      │  │
│  │   Port: 3000     │         │   Port: 8000     │  │
│  └─────────────────┘         └────────┬─────────┘  │
│                                        │             │
│                                        │ PostgreSQL  │
│                                        │ Protocol    │
│                                        │             │
│                               ┌────────▼─────────┐  │
│                               │   PostgreSQL     │  │
│                               │   Database       │  │
│                               │   Port: 5432     │  │
│                               └──────────────────┘  │
│                                                       │
│  ┌─────────────────┐ (Optional - Admin Profile)     │
│  │    pgAdmin       │                                │
│  │   Port: 5050     │                                │
│  └─────────────────┘                                │
└──────────────────────────────────────────────────────┘
         ↕ Exposed Ports
    localhost:3000 (Dashboard)
    localhost:8000 (API)
    localhost:5432 (Database)
    localhost:5050 (pgAdmin)
```

## 🚀 Quick Start Guide

### 1. Initial Setup (One-time)

```bash
# Navigate to project
cd "Smart Automatic Trading BOT + AI"

# Create environment file
copy docker\.env.docker.example .env

# Edit .env with your MT5 credentials
notepad .env
```

**Required credentials in .env:**
```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=/path/to/mt5/terminal
```

### 2. Start Services (Windows)

**Option A: Using helper script (Recommended)**
```cmd
REM Start core services
docker\scripts\docker-start.bat

REM Or start with pgAdmin
docker\scripts\docker-start.bat --admin
```

**Option B: Manual docker-compose**
```cmd
REM Build and start
docker-compose up -d

REM With pgAdmin
docker-compose --profile admin up -d
```

### 3. Access the Dashboard

Open your browser and go to:
- **Dashboard:** http://localhost:3000

You'll see:
- Real-time price updates
- Account balance and equity
- Trading signals (SMC + ML)
- Market regime
- Open positions
- Risk status
- Activity logs

### 4. Check Other Services

- **API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/api/health
- **API Status:** http://localhost:8000/api/status
- **pgAdmin:** http://localhost:5050 (if started with --admin)

## 📋 Common Commands

### View Logs
```cmd
REM All services
docker\scripts\docker-logs.bat

REM Specific service
docker\scripts\docker-logs.bat trading-api
docker\scripts\docker-logs.bat dashboard
docker\scripts\docker-logs.bat postgres
```

### Check Status
```cmd
docker-compose ps
```

### Restart Services
```cmd
REM Restart all
docker-compose restart

REM Restart specific
docker-compose restart trading-api
docker-compose restart dashboard
```

### Stop Services
```cmd
REM Stop (keeps data)
docker\scripts\docker-stop.bat

REM Stop and remove containers (keeps data)
docker\scripts\docker-stop.bat --remove

REM Stop and remove everything including data (⚠️ DANGER!)
docker\scripts\docker-stop.bat --clean
```

### Update Code and Rebuild
```cmd
REM Pull latest code
git pull

REM Rebuild and restart
docker-compose build
docker-compose up -d
```

## 🔧 Configuration

### Port Configuration

Default ports can be changed in `.env`:

```env
API_PORT=8000          # Trading API
DASHBOARD_PORT=3000    # Web Dashboard
DB_PORT=5432          # PostgreSQL
PGADMIN_PORT=5050     # pgAdmin
```

### Environment Variables

All configuration is in `.env`:

| Category | Variables |
|----------|-----------|
| **MT5** | MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH |
| **Trading** | SYMBOL, CAPITAL |
| **Database** | DB_USER, DB_PASSWORD, DB_NAME |
| **Telegram** | TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID |
| **Ports** | API_PORT, DASHBOARD_PORT, DB_PORT, PGADMIN_PORT |

## 🐛 Troubleshooting

### Dashboard Shows "Connection Error"

**Check if API is running:**
```cmd
curl http://localhost:8000/api/health
```

**View API logs:**
```cmd
docker\scripts\docker-logs.bat trading-api
```

### Port Already in Use

**Find what's using the port:**
```cmd
netstat -ano | findstr :3000
netstat -ano | findstr :8000
```

**Change port in .env:**
```env
DASHBOARD_PORT=3001
API_PORT=8001
```

**Restart services:**
```cmd
docker-compose down
docker-compose up -d
```

### Can't Connect to MT5

1. Check credentials in `.env`
2. Ensure MT5 terminal is accessible
3. View API logs for connection errors:
   ```cmd
   docker\scripts\docker-logs.bat trading-api
   ```

### Database Connection Issues

**Check database health:**
```cmd
docker-compose ps postgres
```

**Test connection:**
```cmd
docker exec -it trading_bot_db pg_isready -U trading_bot
```

**View database logs:**
```cmd
docker\scripts\docker-logs.bat postgres
```

## 📊 Monitoring

### View Real-time Logs
```cmd
REM Follow all logs
docker-compose logs -f

REM Follow specific service
docker-compose logs -f trading-api
```

### Check Resource Usage
```cmd
docker stats
```

### Service Health
```cmd
REM All services
docker-compose ps

REM Detailed info
docker inspect trading_bot_api
docker inspect trading_bot_dashboard
```

## 🔐 Security Notes

1. **Never commit .env file** - It contains sensitive credentials
2. **Change default passwords** - Especially for database and pgAdmin
3. **Use strong passwords** - For all services
4. **Limit port exposure** - Only expose ports you need
5. **Keep Docker updated** - Regular security updates

## 📁 File Structure

```
xaubot-ai/
├── Dockerfile                    # Python API Docker image
├── docker-compose.yml            # Service orchestration
├── .env                         # Environment variables (DO NOT COMMIT)
├── .dockerignore               # Files to exclude from build
├── docker/
│   ├── .env.docker.example     # Environment template
│   ├── requirements-docker.txt # Docker-specific Python deps
│   ├── init-db/01-schema.sql   # Database schema
│   ├── scripts/                # Helper scripts
│   │   ├── docker-start.bat/.sh
│   │   ├── docker-stop.bat/.sh
│   │   ├── docker-logs.bat/.sh
│   │   ├── docker-status.bat
│   │   ├── docker-add-dashboard.bat
│   │   ├── docker-remove-dashboard.bat
│   │   ├── start-all.bat
│   │   ├── start-api.bat
│   │   └── start-dashboard.bat
│   └── docs/                   # Docker documentation
│       ├── DOCKER.md
│       ├── DOCKER-INTEGRATION.md
│       ├── DOCKER-SETUP-SUMMARY.md
│       ├── QUICK-START.md
│       └── SIMPLE-START.md
└── web-dashboard/
    ├── Dockerfile              # Next.js dashboard image
    ├── .dockerignore          # Dashboard build exclusions
    └── next.config.ts         # Next.js config (standalone output)
```

## 🎯 Benefits of Docker Setup

✅ **Easy Setup** - One command to start everything
✅ **Consistent Environment** - Same setup on any machine
✅ **Isolated Services** - No conflicts with other software
✅ **Easy Updates** - Rebuild and restart to update
✅ **Production Ready** - Same setup for dev and production
✅ **Automatic Restarts** - Services auto-restart on crash
✅ **Health Monitoring** - Built-in health checks
✅ **Volume Persistence** - Data survives container restarts

## 📚 Additional Resources

- **Full Documentation:** [DOCKER.md](DOCKER.md)
- **Styling Guide:** [web-dashboard/STYLING-GUIDE.md](web-dashboard/STYLING-GUIDE.md)
- **Docker Docs:** https://docs.docker.com
- **Docker Compose:** https://docs.docker.com/compose

## 🆘 Support

If you encounter issues:

1. Check the logs: `docker\scripts\docker-logs.bat`
2. Verify services: `docker-compose ps`
3. Review troubleshooting section in [DOCKER.md](DOCKER.md)
4. Check service health: `curl http://localhost:8000/api/health`

---

**Setup completed:** Feb 6, 2026
**Ready to deploy!** 🚀
