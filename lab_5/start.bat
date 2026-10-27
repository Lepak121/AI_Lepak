@echo off
chcp 65001 >nul
title OpenCV Vision Studio - Лабораторна робота 5

echo ========================================================
echo   Запуск OpenCV Vision Studio (Лабораторна робота 5)
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    python main.py
)

if errorlevel 1 (
    echo.
    echo Помилка запуску програми!
    pause
)
