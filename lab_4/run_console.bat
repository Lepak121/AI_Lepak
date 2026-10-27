@echo off
setlocal
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

"%PY_CMD%" -X utf8 main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program exited with error code %errorlevel%
)
pause
