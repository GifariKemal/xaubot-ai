@echo off
REM Hourly Bot Monitoring Script
REM Run this in Windows Task Scheduler every 1 hour

cd /d "C:\Users\Administrator\Videos\Smart Automatic Trading BOT + AI"

echo.
echo ============================================================
echo HOURLY MONITORING - %date% %time%
echo ============================================================
echo.

REM Run monitoring script
python scripts\monitor_bot.py

REM Log to file
python scripts\monitor_bot.py >> logs\monitor_hourly.log 2>&1

echo.
echo Monitoring complete. Check logs\monitor_hourly.log for history.
echo Next check in 1 hour.
echo.

REM Optional: pause if running manually
REM pause
