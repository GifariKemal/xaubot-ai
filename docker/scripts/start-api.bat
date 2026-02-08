@echo off
REM Start Trading API (FastAPI)

echo.
echo ========================================
echo Starting Trading API
echo ========================================
echo.

cd /d "%~dp0..\.."

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -q fastapi uvicorn pydantic python-dotenv aiohttp

echo.
echo ========================================
echo API Starting on http://localhost:8000
echo ========================================
echo.
echo Press Ctrl+C to stop
echo.

REM Start the API
python web-dashboard\api\main.py
