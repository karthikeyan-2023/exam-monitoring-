@echo off
echo ===================================================
echo  Exam-AI  —  Startup Script
echo ===================================================

set /p MODE="Run mode? [1] Standalone monitor.py  [2] Full API server: "

:: ─────────────────────────────────────────────────────
:: LLM service (always started first, in a new window)
:: ─────────────────────────────────────────────────────
echo.
echo [Step 1/2] Setting up LLM service …
cd services\llm
if not exist ".venv" (
    echo Creating virtual environment for LLM …
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
echo Starting LLM server on port 8003 …
start "LLM Server (port 8003)" cmd /k "call .venv\Scripts\activate.bat && set MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct && uvicorn app:app --host 127.0.0.1 --port 8003"
cd ..\..

echo Waiting 6 seconds for LLM to initialise …
timeout /t 6 >nul

:: ─────────────────────────────────────────────────────
:: Engine service
:: ─────────────────────────────────────────────────────
echo.
echo [Step 2/2] Setting up Engine …
cd services\engine
if not exist ".venv" (
    echo Creating virtual environment for Engine …
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

echo.
if "%MODE%"=="2" (
    echo ===================================================
    echo  Starting Engine API server on port 8002 …
    echo  Endpoints:
    echo    POST /camera/on        start webcam + risk loop
    echo    POST /camera/off       stop camera
    echo    GET  /video_feed       MJPEG stream
    echo    GET  /risk_events      latest risk JSON
    echo    GET  /status           server health
    echo ===================================================
    uvicorn app:app --host 127.0.0.1 --port 8002
) else (
    echo ===================================================
    echo  Starting Standalone Exam Monitor (monitor.py) …
    echo  Press Q in the video window to quit.
    echo ===================================================
    python monitor.py
)

pause
