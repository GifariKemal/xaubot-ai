@echo off
REM Start Next.js Dashboard

echo.
echo ========================================
echo Starting Web Dashboard
echo ========================================
echo.

cd /d "%~dp0..\..\web-dashboard"

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    echo.
)

echo.
echo ========================================
echo Dashboard Starting on http://localhost:3000
echo ========================================
echo.
echo Press Ctrl+C to stop
echo.

REM Start dashboard
npm run dev
