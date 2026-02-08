@echo off
REM XAUBot AI - Docker Logs Viewer (Windows)

set SERVICE=%1
set LINES=%2
if "%LINES%"=="" set LINES=100

if "%SERVICE%"=="" (
    echo Viewing logs for all services...
    echo Tip: Use 'docker-logs.bat SERVICE [LINES]' to view specific service
    echo    Available: trading-api, dashboard, postgres, pgadmin
    echo.
    docker-compose logs -f --tail=%LINES%
) else (
    echo Viewing logs for: %SERVICE% last %LINES% lines
    echo.
    docker-compose logs -f --tail=%LINES% %SERVICE%
)
