@echo off
REM Remove Dashboard & API while keeping database

echo.
echo ========================================
echo Removing Dashboard & API Services
echo ========================================
echo.
echo This will stop and remove:
echo   - trading_bot_dashboard
echo   - trading_bot_api
echo.
echo Database (trading_bot_db) will remain running.
echo.
pause

echo Stopping services...
docker-compose stop trading-api dashboard

echo.
echo Removing containers...
docker-compose rm -f trading-api dashboard

echo.
echo ========================================
echo Dashboard Removed!
echo ========================================
echo.
echo Database is still running:
docker ps --filter "name=trading_bot_db" --format "table {{.Names}}\t{{.Status}}"
echo.
echo To add dashboard back: docker-add-dashboard.bat
echo.
pause
