# Exam-AI Monitoring System

Aim: Detect cheating or suspicious behaviour during a digital test using Object Detection (YOLOv8) and a Large Language Model (Qwen2.5) for summarized evidence interpretation.

## Architecture

The project contains two microservices:
1. **services/llm**: A FastAPI backend running a Hugging Face LLM (Qwen2.5-1.5B-Instruct).
2. **services/engine**: A computer vision module using YOLOv8 via OpenCV to detect phones and persons in webcam or video feeds.

Every 10 seconds, the Vision engine captures events (like bounding box overlaps between humans/phones) and POSTs the JSON facts to the LLM backend for risk analysis.

## Detailed Instructions to run a Demo

**Option 1: The Automated Script (Recommended)**
Double-click `run_all.bat` located at the root of the project folder. This splits the environment into two distinct Python setups, downloads packages, stars the LLM on your local port `8003`, waits, and then opens your webcam tracking window! 

**Option 2: Run Step-by-Step**

1. Ensure Python 3.10+ is installed and marked in your PATH.
   
2. Start the LLM Server:
   Open a Powershell prompt here.
   ```powershell
   cd services\llm
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   set MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
   uvicorn app:app --host 127.0.0.1 --port 8003
   ```
   Leave this window open. Your LLM runs right here locally on `127.0.0.1:8003`.

3. Start the Object Tracking Engine (in a NEW terminal):
   ```powershell
   cd services\engine
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Now to load the live monitoring window using Webcam index `0`:
   ```powershell
   python monitor.py
   ```
   (Wait up to 15 seconds to download the YOLO weights on your very first run).

## Important Files
- `services/engine/monitor.py` -> The exact file creating the vision stream.
- `services/llm/app.py` -> The fastAPI endpoint doing inference over `events:[]`.
- `run_all.bat` -> Simple starter.
