@echo off
REM XAUBot AI - Docker Stop Script (Windows)
cd /d "%~dp0..\.."

echo.
echo Stopping XAUBot AI Docker Services...
echo.

if "%1"=="--remove" goto remove
if "%1"=="-r" goto remove
if "%1"=="--clean" goto clean
if "%1"=="-c" goto clean
goto stop

:remove
echo Stopping and removing containers...
docker-compose down
echo.
echo Containers stopped and removed
goto end

:clean
echo WARNING: This will remove all data including database!
set /p confirm="Are you sure? (yes/no): "
if /i "%confirm%"=="yes" (
    docker-compose down -v
    echo.
    echo Containers, networks, and volumes removed
) else (
    echo.
    echo Cancelled
)
goto end

:stop
echo Stopping containers data will be preserved...
docker-compose stop
echo.
echo Containers stopped

:end
echo.
echo To restart: docker\scripts\docker-start.bat
echo.
pause
