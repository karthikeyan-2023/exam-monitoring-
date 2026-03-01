@echo off
setlocal

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║         Exam-AI + Dashboard  —  Full Startup         ║
echo ╚══════════════════════════════════════════════════════╝
echo.
echo  Services that will be started:
echo    [8003]  exam-ai\services\llm     — Qwen2.5 LLM summariser
echo    [8002]  exam-ai\services\engine  — ML vision engine (YOLO + MediaPipe)
echo    [8000]  exam-monitroing\backend  — Dashboard FastAPI API
echo    [3000]  exam-monitroing\frontend — React dashboard
echo.


:: ─────────────────────────────────────────────────
:: 1.  LLM Service  (port 8003)
:: ─────────────────────────────────────────────────
echo [1/4] Starting LLM Service (port 8003)...
cd /d "%~dp0exam-ai\services\llm"

if not exist ".venv" (
    echo     Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

start "LLM Server :8003" cmd /k "call .venv\Scripts\activate.bat & set MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct & uvicorn app:app --host 127.0.0.1 --port 8003"
cd /d "%~dp0"

echo     Waiting for LLM to initialise...
timeout /t 6 >nul


:: ─────────────────────────────────────────────────
:: 2.  ML Engine  (port 8002)
:: ─────────────────────────────────────────────────
echo [2/4] Starting ML Engine (port 8002)...
cd /d "%~dp0exam-ai\services\engine"

if not exist ".venv" (
    echo     Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

start "ML Engine :8002" cmd /k "call .venv\Scripts\activate.bat & uvicorn app:app --host 127.0.0.1 --port 8002"
cd /d "%~dp0"

echo     Waiting for Engine to initialise...
timeout /t 4 >nul


:: ─────────────────────────────────────────────────
:: 3.  Dashboard Backend  (port 8000)
:: ─────────────────────────────────────────────────
echo [3/4] Starting Dashboard Backend (port 8000)...
cd /d "%~dp0exam-monitroing\backend"

if not exist ".venv" (
    echo     Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

start "Dashboard Backend :8000" cmd /k "call .venv\Scripts\activate.bat & uvicorn server:app --host 127.0.0.1 --port 8000 --reload"
cd /d "%~dp0"

echo     Waiting for Backend to initialise...
timeout /t 4 >nul


:: ─────────────────────────────────────────────────
:: 4.  Dashboard Frontend  (port 3000)
:: ─────────────────────────────────────────────────
echo [4/4] Starting Dashboard Frontend (port 3000)...
cd /d "%~dp0exam-monitroing\frontend"

if not exist "node_modules" (
    echo     Installing npm packages (this takes a minute the first time)...
    call npm install --legacy-peer-deps
)

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║  All services started!  Opening dashboard now...     ║
echo ║                                                      ║
echo ║  Dashboard  →  http://localhost:3000                 ║
echo ║  ML Engine  →  http://127.0.0.1:8002                ║
echo ║  LLM        →  http://127.0.0.1:8003  (optional)    ║
echo ╚══════════════════════════════════════════════════════╝
echo.

npm start
pause
