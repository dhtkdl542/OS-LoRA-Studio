@echo off
chcp 65001 >nul
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
venv\Scripts\python.exe -m oslora.server
pause
