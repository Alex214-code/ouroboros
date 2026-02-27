@echo off
chcp 65001 >nul 2>&1
title Ouroboros - Setup

echo ============================================
echo   Ouroboros - Initial Setup
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install Python 3.10+
    pause
    exit /b 1
)

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found! Install Git.
    pause
    exit /b 1
)

REM Check Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found. Install from https://ollama.com/
    echo After installing, run: ollama pull gpt-oss:20b
    echo.
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -q openai requests starlette uvicorn websockets duckduckgo-search
pip install -q pyautogui mss Pillow pywin32 pytesseract
echo Done.
echo.

REM Check Tesseract OCR
where tesseract >nul 2>&1
if errorlevel 1 (
    echo [INFO] Tesseract OCR not found.
    echo   For OCR support, install from:
    echo   https://github.com/UB-Mannheim/tesseract/wiki
    echo   (Add Russian language pack during installation)
    echo   Without it, read_screen/find_on_screen won't work.
    echo   Desktop control and vision analysis will still work fine.
    echo.
)

REM Pull models
echo.
echo Pulling gpt-oss:20b model (14GB, may take a while)...
ollama pull gpt-oss:20b
echo.
echo Pulling minicpm-v:8b vision model (5GB)...
echo (Used for desktop visual analysis - loads on demand, not always in RAM)
ollama pull minicpm-v:8b
echo.

REM Create .env if not exists
if not exist "%~dp0.env" (
    echo Creating .env file...
    echo.

    set /p TG_TOKEN="Enter Telegram Bot Token (from @BotFather): "
    set /p GH_TOKEN="Enter GitHub Token (ghp_...): "

    (
        echo TELEGRAM_BOT_TOKEN=%TG_TOKEN%
        echo GITHUB_USER=Alex214-code
        echo GITHUB_REPO=ouroboros
        echo GITHUB_TOKEN=%GH_TOKEN%
    ) > "%~dp0.env"

    echo .env created!
) else (
    echo .env already exists, skipping.
)

echo.
echo ============================================
echo   Setup complete!
echo   Run START.bat to launch Ouroboros.
echo ============================================
echo.
pause
