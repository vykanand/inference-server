@echo off
REM ---------------------------------------------------------------------------
REM  Inference Server launcher — restore/restart
REM  Kills anything on port 8899 + stray llama-server engines, then starts fresh.
REM  Server runs in the current console (Ctrl-C to stop).
REM ---------------------------------------------------------------------------
setlocal
set PORT=8899

echo [inference-server] Stopping anything on port %PORT% ...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R ":%PORT% .*LISTENING"') do (
    echo   killing PID %%a
    taskkill /F /PID %%a >nul 2>&1
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 2 >nul

echo [inference-server] Starting on http://127.0.0.1:%PORT% ...
echo [inference-server] Press Ctrl-C to stop.
echo.
python -m app.main
endlocal
