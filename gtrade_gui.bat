@echo off
cd /d "%~dp0"

REM Try conda environment first, then system Python
where conda >nul 2>nul
if %errorlevel% equ 0 (
    call conda activate gtrade_gpu 2>nul
    if %errorlevel% equ 0 (
        start "" python launcher.py
        exit
    )
)

REM Fallback: system Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    start "" python launcher.py
    exit
)

echo [ERROR] Python not found. Install Python 3.10+ or run setup_gpu.bat
pause
