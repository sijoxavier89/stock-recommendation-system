@echo off
REM Stock Recommendation System - Startup Script (Windows)

setlocal enabledelayedexpansion

echo ========================================
echo Stock Recommendation System - Startup
echo ========================================
echo.

REM Create logs directory
if not exist logs mkdir logs

REM Check and start Backend
echo [Backend] Starting FastAPI server on port 8000...
cd backend

REM Check if venv exists
if not exist venv (
    echo [Backend] Creating Python virtual environment...
    python -m venv venv
)

REM Activate venv and start
call venv\Scripts\activate.bat
pip install -q -r requirements.txt
start "Backend Server" cmd /k "uvicorn app.main:app --reload"

cd ..
timeout /t 2 /nobreak

REM Check and start Frontend
echo [Frontend] Starting React dev server on port 5173...
cd frontend

REM Check if node_modules exists
if not exist node_modules (
    echo [Frontend] Installing dependencies...
    call npm install
)

start "Frontend Server" cmd /k "npm run dev"

cd ..

echo.
echo ========================================
echo Startup Complete!
echo ========================================
echo.
echo Services:
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo.
echo Note: Ollama must be started separately with: ollama serve
echo.
echo Press any key to close this window...
pause
