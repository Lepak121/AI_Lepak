@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY_CMD="
if exist "C:\Users\rezer\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    set "PY_CMD=C:\Users\rezer\AppData\Local\Python\pythoncore-3.14-64\python.exe"
)
if "%PY_CMD%"=="" (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=py"
    ) else (
        set "PY_CMD=python"
    )
)

cls
echo ======================================================================
echo    Pattern Recognition System (Quantitative and Qualitative Features)
echo ======================================================================
echo.
echo Select launch mode:
echo   [1] Graphical Interface (Tkinter GUI - Recommended)
echo   [2] Console Menu and Automated Tests (CLI)
echo   [0] Exit
echo.

set /p choice="Your choice (1, 2 or 0): "

if "%choice%"=="1" (
    "%PY_CMD%" gui_app.py
) else if "%choice%"=="2" (
    "%PY_CMD%" -X utf8 main.py
) else (
    exit /b
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] An error occurred during execution.
    pause
)
