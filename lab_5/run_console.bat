@echo off
chcp 65001 >nul
title OpenCV Vision Studio - Консольний режим (Лабораторна 5)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py --console
) else (
    python main.py --console
)

pause
