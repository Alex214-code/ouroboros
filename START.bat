@echo off
chcp 65001 >nul 2>&1
title Ouroboros - Local AI Agent

echo ============================================
echo   Ouroboros - Self-Creating AI Agent
echo   Local Mode (Ollama + gpt-oss:20b)
echo ============================================
echo.

REM Load .env file if exists
if exist "%~dp0.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
        set "%%A=%%B"
    )
)

REM Ollama CPU optimizations (i7-10510U, 4C/8T)
set OLLAMA_NUM_THREADS=8
set OLLAMA_NUM_CTX=16384
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_NUM_PARALLEL=1

REM Check Ollama
echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo Ollama is not running. Starting...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ollama failed to start!
        echo Run manually: ollama serve
        pause
        exit /b 1
    )
)
echo Ollama OK.
echo.

REM Start dashboard in background
echo Starting dashboard (http://localhost:8765)...
start "Ouroboros Dashboard" /min python "%~dp0desktop\server.py"
timeout /t 2 /nobreak >nul

REM Start main agent
echo Starting Ouroboros agent...
echo.
python "%~dp0local_launcher.py"

pause
