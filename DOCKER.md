# XAUBot AI - Docker Setup Guide

Complete guide to running the XAUBot AI trading system with Docker.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- MetaTrader 5 account credentials

## 🏗️ Architecture

The Docker setup includes 4 services:

```
┌─────────────────────────────────────────────────┐
│                   Host Machine                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐      ┌──────────────┐        │
│  │  Dashboard   │─────▶│ Trading API  │        │
│  │   Next.js    │      │   FastAPI    │        │
│  │  Port: 3000  │      │  Port: 8000  │        │
│  └──────────────┘      └──────┬───────┘        │
│                                │                 │
│                         ┌──────▼───────┐        │
│                         │  PostgreSQL  │        │
│                         │  Port: 5432  │        │
│                         └──────────────┘        │
│                                                  │
│  ┌──────────────┐ (Optional - Profile: admin)  │
│  │   pgAdmin    │                                │
│  │  Port: 5050  │                                │
│  └──────────────┘                                │
└─────────────────────────────────────────────────┘
```

### Services

1. **postgres** - PostgreSQL 16 database for trade logging
2. **trading-api** - Python FastAPI backend serving trading data
3. **dashboard** - Next.js web interface for monitoring
4. **pgadmin** - Database management UI (optional, admin profile)

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd "Smart Automatic Trading BOT + AI"

# Copy environment template
cp .env.docker.example .env
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```bash
# Required
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
MT5_PATH=/path/to/mt5

# Optional - adjust ports if needed
API_PORT=8000
DASHBOARD_PORT=3000
DB_PORT=5432
```

### 3. Start Services

**Option A: All services (without pgAdmin)**
```bash
docker-compose up -d
```

**Option B: All services including pgAdmin**
```bash
docker-compose --profile admin up -d
```

**Option C: Specific services only**
```bash
# Just database and API
docker-compose up -d postgres trading-api

# Add dashboard
docker-compose up -d dashboard
```

### 4. Access Services

- **Dashboard**: http://localhost:3000
- **Trading API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:5050 (if using admin profile)
- **PostgreSQL**: localhost:5432

## 📊 Service Management

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dashboard
docker-compose logs -f trading-api
docker-compose logs -f postgres

# Last 50 lines
docker-compose logs --tail=50 trading-api
```

### Check Status

```bash
# List running containers
docker-compose ps

# Check health
docker-compose ps --format json | jq '.[].Health'

# Detailed status
docker inspect trading_bot_api
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart trading-api
docker-compose restart dashboard
```

### Stop Services

```bash
# Stop all (keeps data)
docker-compose stop

# Stop and remove containers (keeps data)
docker-compose down

# Stop and remove everything including volumes (⚠️ deletes data!)
docker-compose down -v
```

## 🔧 Development & Debugging

### Access Container Shell

```bash
# Trading API container
docker exec -it trading_bot_api bash

# Dashboard container
docker exec -it trading_bot_dashboard sh

# Database
docker exec -it trading_bot_db psql -U trading_bot -d trading_db
```

### Rebuild After Code Changes

```bash
# Rebuild all
docker-compose build

# Rebuild specific service
docker-compose build trading-api
docker-compose build dashboard

# Rebuild and restart
docker-compose up -d --build
```

### View Resource Usage

```bash
# CPU, Memory, Network
docker stats

# Specific container
docker stats trading_bot_api
```

## 🗄️ Database Management

### Connect to PostgreSQL

```bash
# Via Docker
docker exec -it trading_bot_db psql -U trading_bot -d trading_db

# Via host (if port exposed)
psql -h localhost -p 5432 -U trading_bot -d trading_db
```

### Backup Database

```bash
# Create backup
docker exec trading_bot_db pg_dump -U trading_bot trading_db > backup_$(date +%Y%m%d).sql

# Restore backup
docker exec -i trading_bot_db psql -U trading_bot -d trading_db < backup_20260206.sql
```

### Using pgAdmin

1. Start with admin profile:
   ```bash
   docker-compose --profile admin up -d
   ```

2. Open http://localhost:5050

3. Login:
   - Email: admin@trading.local
   - Password: admin123

4. Add Server:
   - Host: postgres
   - Port: 5432
   - Database: trading_db
   - Username: trading_bot
   - Password: trading_bot_2026

## 🔍 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs trading-api

# Check events
docker events --filter container=trading_bot_api

# Inspect container
docker inspect trading_bot_api
```

### Port Already in Use

```bash
# Find what's using the port
netstat -ano | findstr :3000
netstat -ano | findstr :8000

# Change port in .env
DASHBOARD_PORT=3001
API_PORT=8001

# Restart
docker-compose down
docker-compose up -d
```

### API Can't Connect to MT5

1. Check MT5 credentials in `.env`
2. Ensure MT5 terminal is running (if running on host)
3. Check container logs:
   ```bash
   docker-compose logs trading-api | grep MT5
   ```

### Dashboard Shows Connection Error

1. Check if API is healthy:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. Check API logs:
   ```bash
   docker-compose logs trading-api
   ```

3. Verify API_URL in dashboard:
   ```bash
   docker exec -it trading_bot_dashboard env | grep API
   ```

### Database Connection Issues

```bash
# Check if postgres is healthy
docker-compose ps postgres

# Test connection
docker exec -it trading_bot_db pg_isready -U trading_bot

# Check logs
docker-compose logs postgres
```

## 🔐 Security Best Practices

1. **Change Default Passwords**
   ```bash
   # In .env
   DB_PASSWORD=strong_password_here
   PGADMIN_PASSWORD=another_strong_password
   ```

2. **Don't Expose Unnecessary Ports**
   ```yaml
   # In docker-compose.yml, comment out if not needed:
   # ports:
   #   - "5432:5432"  # Only if you need external DB access
   ```

3. **Use Secrets for Production**
   ```bash
   # Use Docker secrets instead of .env
   docker secret create mt5_password password.txt
   ```

4. **Restrict Network Access**
   ```bash
   # Only expose dashboard port
   docker-compose up -d postgres trading-api
   # Then separately: docker-compose up -d dashboard
   ```

## 📈 Performance Tuning

### Allocate More Resources

```yaml
# In docker-compose.yml
services:
  trading-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Optimize Database

```bash
# Connect to DB
docker exec -it trading_bot_db psql -U trading_bot -d trading_db

# Run vacuum
VACUUM ANALYZE;

# Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## 🔄 Updates & Maintenance

### Update Images

```bash
# Pull latest base images
docker-compose pull

# Rebuild
docker-compose build --no-cache

# Restart
docker-compose up -d
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes (⚠️ careful!)
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

## 📝 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_LOGIN` | - | MT5 account login |
| `MT5_PASSWORD` | - | MT5 account password |
| `MT5_SERVER` | - | MT5 server name |
| `MT5_PATH` | - | Path to MT5 terminal |
| `SYMBOL` | XAUUSD | Trading symbol |
| `CAPITAL` | 10000 | Trading capital |
| `API_PORT` | 8000 | API port on host |
| `DASHBOARD_PORT` | 3000 | Dashboard port on host |
| `DB_PORT` | 5432 | Database port on host |
| `DB_USER` | trading_bot | Database username |
| `DB_PASSWORD` | trading_bot_2026 | Database password |
| `DB_NAME` | trading_db | Database name |
| `TELEGRAM_BOT_TOKEN` | - | Telegram bot token (optional) |
| `TELEGRAM_CHAT_ID` | - | Telegram chat ID (optional) |

## 📚 Additional Resources

- **Docker Docs**: https://docs.docker.com
- **Docker Compose**: https://docs.docker.com/compose
- **FastAPI**: https://fastapi.tiangolo.com
- **Next.js**: https://nextjs.org

## 🆘 Getting Help

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify services: `docker-compose ps`
3. Check health: `curl http://localhost:8000/api/health`
4. Review this guide's troubleshooting section
5. Open an issue on GitHub

---

**Last Updated:** Feb 6, 2026
