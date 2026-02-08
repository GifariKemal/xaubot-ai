@echo off
REM Add Dashboard & API to existing Docker setup
cd /d "%~dp0..\.."

echo.
echo ========================================
echo Adding Dashboard to Existing Setup
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo WARNING: .env file not found!
    echo Creating .env from template...
    copy docker\.env.docker.example .env
    echo.
    echo Please edit .env with your MT5 credentials:
    echo    - MT5_LOGIN
    echo    - MT5_PASSWORD
    echo    - MT5_SERVER
    echo    - MT5_PATH
    echo.
    pause
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/4] Checking existing services...
docker ps --filter "name=trading_bot_db" --format "table {{.Names}}\t{{.Status}}"

echo.
echo [2/4] Building new services API and Dashboard...
docker-compose build trading-api dashboard

echo.
echo [3/4] Starting new services...
docker-compose up -d trading-api dashboard

echo.
echo [4/4] Checking all services...
docker-compose ps

echo.
echo ========================================
echo Dashboard Added Successfully!
echo ========================================
echo.
echo Access Points:
echo    Dashboard:  http://localhost:3000
echo    API:        http://localhost:8000
echo    API Docs:   http://localhost:8000/docs
echo    Database:   localhost:5432 (already running)
echo.
echo View logs:
echo    docker-compose logs -f dashboard
echo    docker-compose logs -f trading-api
echo.
pause
