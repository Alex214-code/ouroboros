@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo This script has already been run. Git repo is initialized.
echo If you need to re-initialize, do it manually.
pause
