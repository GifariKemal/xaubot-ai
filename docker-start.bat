@echo off
REM XAUBot AI - Docker Start Script (Windows)

echo.
echo Starting XAUBot AI Docker Services...
echo.

REM Check if .env exists
if not exist .env (
    echo WARNING: .env file not found!
    echo Creating .env from template...
    copy .env.docker.example .env
    echo.
    echo .env created. Please edit it with your MT5 credentials:
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

echo Docker is running
echo.

REM Parse arguments
set PROFILE_FLAG=
if "%1"=="--admin" set PROFILE_FLAG=--profile admin
if "%1"=="-a" set PROFILE_FLAG=--profile admin

if defined PROFILE_FLAG (
    echo Starting with pgAdmin admin profile...
) else (
    echo Starting core services postgres, api, dashboard...
    echo Tip: Use 'docker-start.bat --admin' to include pgAdmin
)

echo.
echo Pulling latest base images...
docker-compose pull

echo.
echo Building services...
docker-compose build

echo.
echo Starting services...
docker-compose %PROFILE_FLAG% up -d

echo.
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

echo.
echo Checking service health...
docker-compose ps

echo.
echo ========================================
echo Services started successfully!
echo ========================================
echo.
echo Access Points:
echo    Dashboard:  http://localhost:3000
echo    API:        http://localhost:8000
echo    API Docs:   http://localhost:8000/docs
echo    Database:   localhost:5432

if defined PROFILE_FLAG (
    echo    pgAdmin:    http://localhost:5050
)

echo.
echo Useful Commands:
echo    View logs:       docker-compose logs -f
echo    View API logs:   docker-compose logs -f trading-api
echo    Stop services:   docker-compose down
echo    Restart:         docker-compose restart
echo.
echo Full documentation: DOCKER.md
echo.
pause
