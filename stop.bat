@echo off
REM Stock Recommendation System - Stop Services (Windows)

echo ========================================
echo Stopping All Services
echo ========================================
echo.

echo [Frontend] Stopping on port 5173...
taskkill /F /IM node.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Success: Frontend stopped
) else (
    echo Info: Frontend not running
)

timeout /t 1 /nobreak

echo [Backend] Stopping on port 8000...
taskkill /F /IM python.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Success: Backend stopped
) else (
    echo Info: Backend not running
)

echo.
echo ========================================
echo All services stopped!
echo ========================================
echo.
echo To start services again, run: start.bat
echo.
pause
