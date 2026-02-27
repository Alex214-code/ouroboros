@echo off
chcp 65001 >nul 2>&1
title Ouroboros - Stop

echo Stopping Ouroboros...

REM Kill only Python processes running OUR specific scripts
for /f "tokens=2" %%i in ('wmic process where "commandline like '%%local_launcher.py%%'" get processid 2^>nul ^| findstr [0-9]') do (
    taskkill /f /pid %%i >nul 2>&1
)

for /f "tokens=2" %%i in ('wmic process where "commandline like '%%ouroboros%%desktop_server%%'" get processid 2^>nul ^| findstr [0-9]') do (
    taskkill /f /pid %%i >nul 2>&1
)

REM Kill window by exact title (set in START.bat)
taskkill /f /fi "WINDOWTITLE eq Ouroboros - Local AI Agent" >nul 2>&1

echo Done. Ouroboros stopped.
timeout /t 2 /nobreak >nul
