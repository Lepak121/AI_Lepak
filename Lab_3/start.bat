@echo off
setlocal EnableExtensions
chcp 1251 >nul
title Нейромережа MLP - Streamlit

echo ========================================
echo  Запуск веб-сайту нейромережі MLP
echo  California Housing - Streamlit
echo ========================================
echo.

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "UV=%USERPROFILE%\.local\bin\uv.exe"
set "PORT=8501"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

rem --- 0. Якщо сервер уже працює, просто відкриваємо браузер ---
powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', %PORT%); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo Сервер уже запущено. Відкриваю http://localhost:%PORT%
    start "" "http://localhost:%PORT%"
    timeout /t 3 >nul
    exit /b 0
)

rem --- 1. Віртуальне середовище ---
if not exist "%PY%" (
    echo [1/3] Створення віртуального середовища .venv ...
    if exist "%UV%" (
        "%UV%" venv --python 3.12 .venv
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            py -3 -m venv .venv
        ) else (
            python -m venv .venv
        )
    )
)

if not exist "%PY%" (
    echo.
    echo ПОМИЛКА: не вдалося створити .venv.
    echo Встановіть Python 3.10-3.12 або uv і запустіть start.bat ще раз.
    pause
    exit /b 1
)

rem --- 2. Залежності ---
echo [2/3] Перевірка залежностей ...
"%PY%" -c "import streamlit, sklearn, pandas, matplotlib, joblib, altair" >nul 2>nul
if errorlevel 1 (
    echo       Встановлення залежностей, перший запуск може тривати ...
    if exist "%UV%" (
        "%UV%" pip install --python "%PY%" -r requirements.txt
    ) else (
        "%PY%" -m pip install --upgrade pip
        "%PY%" -m pip install -r requirements.txt
    )
    if errorlevel 1 (
        echo.
        echo ПОМИЛКА: не вдалося встановити залежності.
        echo Перевірте підключення до Інтернету і запустіть start.bat ще раз.
        pause
        exit /b 1
    )
) else (
    echo       Залежності вже встановлено.
)

rem --- 3. Прибираємо питання Streamlit про email при першому запуску ---
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    >"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

rem --- 4. Запуск сайту ---
echo [3/3] Запуск Streamlit ...
echo.
echo   Сайт буде доступний за адресою: http://localhost:%PORT%
echo   Браузер має відкритися автоматично.
echo   Щоб зупинити сервер, натисніть Ctrl+C у цьому вікні.
echo.

"%PY%" -m streamlit run app.py --server.port %PORT%

echo.
echo Сервер Streamlit завершив роботу.
echo Якщо вище була помилка - скопіюйте її текст.
pause
