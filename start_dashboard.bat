@echo off
echo ========================================
echo   AI Trading Bot - Web Dashboard
echo ========================================
echo.

:: Start API Server
echo Starting API Server on port 8000...
start "API Server" cmd /k "cd /d %~dp0 && python web-dashboard\api\main.py"

:: Wait a bit for API to start
timeout /t 3 /nobreak > nul

:: Start Next.js Frontend
echo Starting Web Dashboard on port 3000...
start "Web Dashboard" cmd /k "cd /d %~dp0\web-dashboard && npm run dev"

echo.
echo ========================================
echo Dashboard starting...
echo.
echo API Server:  http://localhost:8000
echo Web Dashboard: http://localhost:3000
echo ========================================
echo.
echo Press any key to open dashboard in browser...
pause > nul

start http://localhost:3000
