@echo off
chcp 65001 >nul
echo ========================================
echo   Wanderlust AI - Turistichniy Asistent
echo ========================================
echo.
echo Zapusk web-servera na http://localhost:5000 ...
echo.
py web_app.py
pause
