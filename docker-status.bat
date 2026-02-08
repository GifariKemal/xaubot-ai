@echo off
REM Check status of all trading bot services

echo.
echo ========================================
echo Trading Bot Docker Services Status
echo ========================================
echo.

docker-compose ps

echo.
echo ========================================
echo Service Health Checks
echo ========================================
echo.

echo [Database]
docker exec trading_bot_db pg_isready -U trading_bot 2>nul
if %errorlevel%==0 (
    echo   Status: HEALTHY
) else (
    echo   Status: NOT RUNNING
)

echo.
echo [API]
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel%==0 (
    echo   Status: HEALTHY
    echo   URL: http://localhost:8000
) else (
    echo   Status: NOT RUNNING or UNHEALTHY
)

echo.
echo [Dashboard]
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel%==0 (
    echo   Status: HEALTHY
    echo   URL: http://localhost:3000
) else (
    echo   Status: NOT RUNNING or UNHEALTHY
)

echo.
echo ========================================
echo Quick Commands
echo ========================================
echo   View logs:     docker-compose logs -f
echo   Restart:       docker-compose restart
echo   Stop all:      docker-compose stop
echo   Start all:     docker-compose up -d
echo.
pause
