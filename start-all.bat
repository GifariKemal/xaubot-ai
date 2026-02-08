@echo off
REM Start API and Dashboard in separate windows

echo.
echo ========================================
echo XAUBot AI - Starting All Services
echo ========================================
echo.

cd "%~dp0"

REM Check database
echo [1/3] Checking database...
docker ps --filter "name=trading_bot_db" --format "{{.Names}}: {{.Status}}" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: Database not running!
    echo Please start with: docker-compose up -d postgres
    echo.
    pause
    exit /b 1
)

echo.
echo [2/3] Starting API...
start "Trading API" cmd /k start-api.bat

echo Waiting for API to start...
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Starting Dashboard...
start "Web Dashboard" cmd /k start-dashboard.bat

echo.
echo ========================================
echo All Services Started!
echo ========================================
echo.
echo Access Points:
echo   - Dashboard: http://localhost:3000
echo   - API:       http://localhost:8000
echo   - API Docs:  http://localhost:8000/docs
echo   - Database:  localhost:5432
echo.
echo Two windows will open:
echo   1. Trading API (FastAPI)
echo   2. Web Dashboard (Next.js)
echo.
echo Close this window when done.
echo.
pause
