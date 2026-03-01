# 🎓 ExamGuard AI — Real-Time Examination Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/YOLO-v8-FF0000?logo=ultralytics" />
  <img src="https://img.shields.io/badge/MediaPipe-Latest-yellow" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

An AI-powered, multi-camera examination hall monitoring system that detects and flags suspicious student behaviour in real time using computer vision (YOLOv8 + MediaPipe) and a React dashboard.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Service Endpoints](#-service-endpoints)
- [Dashboard Usage](#-dashboard-usage)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **24-Camera Grid** | Live monitoring page with up to 24 cameras in configurable grids (2×2 → 4×6) |
| 🔍 **Fullscreen Camera** | Click any camera tile to open fullscreen view with individual On/Off control |
| 🤖 **AI Risk Detection** | YOLOv8 (person/phone) + MediaPipe FaceMesh (head pose) + Pose (hand-under-desk) |
| 📊 **Risk Scoring** | Combines phone use, look-away, nod/shake, hand-under-desk into a weighted risk score |
| 🗺️ **Hall Seat Map** | Interactive seat map with 4 exam centers (A1–A4), each showing 25 seats in a 5×5 grid |
| 🚨 **Live Alerts** | Real-time alert feed with risk classification badges (Normal / Suspicious / High Risk / Confirmed) |
| 🧑‍🎓 **Student Profiles** | Detailed student profiles with risk trends, event history, and identity verification |
| 📈 **Analytics Charts** | Alerts over time, event type distribution charts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser (Port 3000)                    │
│              React Frontend  — ExamGuard AI                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ /api/*  (proxy)
┌───────────────────────▼─────────────────────────────────────┐
│            Dashboard Backend  (FastAPI — Port 8000)         │
│   MongoDB  ·  Camera Registry  ·  ML Engine Proxy           │
└──────┬───────────────────────────────┬───────────────────────┘
       │                               │ HTTP proxy
       │                    ┌──────────▼──────────┐
       │                    │  ML Engine (exam-ai) │
       │                    │  FastAPI — Port 8002 │
       │                    │  YOLO + MediaPipe    │
       │                    │  /video_feed (MJPEG) │
       │                    │  /risk_events (JSON) │
       │                    └──────────┬───────────┘
       │                               │ (optional)
       │                    ┌──────────▼──────────┐
       │                    │  LLM Service         │
       │                    │  Python — Port 8003  │
       │                    └─────────────────────-┘
       │
  ┌────▼────────┐
  │  MongoDB    │
  │  Port 27017 │
  └─────────────┘
```

---

## 📁 Project Structure

```
final_project/
├── start_all.bat               ← Master launcher (starts all 4 services)
├── README.md
│
├── exam-ai/                    ← AI / ML Engine
│   ├── run_all.bat             ← Standalone ML engine launcher
│   └── services/
│       ├── engine/
│       │   ├── app.py          ← FastAPI server (port 8002)
│       │   ├── monitor.py      ← Standalone webcam monitor script
│       │   └── requirements.txt
│       └── llm/
│           ├── app.py          ← LLM summariser (port 8003)
│           └── requirements.txt
│
└── exam-monitroing/            ← Dashboard (Frontend + Backend)
    ├── backend/
    │   ├── server.py           ← FastAPI dashboard backend (port 8000)
    │   ├── requirements.txt
    │   └── .env
    └── frontend/
        ├── public/index.html
        ├── src/
        │   ├── pages/
        │   │   ├── Overview.jsx        ← Seat Map + KPI cards
        │   │   ├── LiveMonitoring.jsx  ← 24-camera grid + fullscreen
        │   │   ├── Students.jsx        ← Student table + profiles
        │   │   ├── Alerts.jsx
        │   │   └── Reports.jsx
        │   ├── mocks/data.js           ← 100 demo students (4 centers × 25)
        │   └── lib/constants.js
        ├── package.json
        └── .env
```

---

## ✅ Prerequisites

Make sure the following are installed before you start:

| Tool | Version | Download |
|---|---|---|
| **Python** | 3.10 or 3.11 | [python.org](https://python.org) |
| **Node.js** | 18+ (LTS) | [nodejs.org](https://nodejs.org) |
| **MongoDB** | Community Edition | [mongodb.com](https://www.mongodb.com/try/download/community) |
| **Git** | Latest | [git-scm.com](https://git-scm.com) |
| **Webcam** | (for Camera 1 / ML Engine) | Built-in or USB |

> ⚠️ **Windows Note:** YOLOv8 requires Microsoft C++ Build Tools. Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/examguard-ai.git
cd examguard-ai
```

### 2. Set up the ML Engine (`exam-ai`)

```bash
cd exam-ai/services/engine

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

> 📦 This installs: `ultralytics`, `opencv-python`, `mediapipe`, `fastapi`, `uvicorn`, `numpy`, `requests`

### 3. Set up the LLM Service (optional)

```bash
cd exam-ai/services/llm
pip install -r requirements.txt
```

### 4. Set up the Dashboard Backend (`exam-monitroing/backend`)

```bash
cd exam-monitroing/backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in `exam-monitroing/backend/` (or edit the existing one):

```env
MONGODB_URL=mongodb://localhost:27017
DB_NAME=examguard
SECRET_KEY=your-super-secret-key-change-this
CORS_ORIGINS=http://localhost:3000
EXAM_AI_ENGINE_URL=http://127.0.0.1:8002
```

### 5. Set up the Frontend (`exam-monitroing/frontend`)

```bash
cd exam-monitroing/frontend

# Install Node dependencies
npm install
```

The `.env` file should already exist with:

```env
REACT_APP_USE_MSW=true
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ML_STREAM_URL=http://127.0.0.1:8002
```

---

## ▶️ Running the Project

### Option A — One-click launcher (Recommended on Windows)

From the project root, simply double-click or run:

```cmd
start_all.bat
```

This opens **4 separate terminal windows** in the correct order:

| Window | Service | Port |
|---|---|---|
| 1 | LLM Summariser | 8003 |
| 2 | ML Engine (YOLO + MediaPipe) | 8002 |
| 3 | Dashboard Backend (FastAPI) | 8000 |
| 4 | React Frontend | 3000 |

Then open your browser at: **http://localhost:3000**

---

### Option B — Manual (run each service separately)

Open **4 separate terminals** and run the following:

#### Terminal 1 — LLM Service
```bash
cd exam-ai/services/llm
venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 8003 --reload
```

#### Terminal 2 — ML Engine
```bash
cd exam-ai/services/engine
python -m venv venv
venv\Scripts\activate
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

#### Terminal 3 — Dashboard Backend
```bash
cd exam-monitroing/backend
python -m venv venv
venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 4 — React Frontend
```bash
cd exam-monitroing/frontend
npm start
```

---

### Option C — Run ML engine standalone (without dashboard)

```bash
cd exam-ai/services/engine
venv\Scripts\activate
python monitor.py
```

This opens a local OpenCV window with live risk overlays. Press `Q` to quit.

---

## 🌐 Service Endpoints

### ML Engine — `http://localhost:8002`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/camera/on` | Start webcam + risk analysis |
| `POST` | `/camera/off` | Stop webcam |
| `GET` | `/video_feed` | MJPEG live video stream |
| `GET` | `/risk_events` | Latest risk scores per student |

### Dashboard Backend — `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cameras` | List all 24 cameras |
| `POST` | `/api/ml/camera/on` | Proxy: turn on ML engine camera |
| `POST` | `/api/ml/camera/off` | Proxy: turn off ML engine camera |
| `GET` | `/api/ml/status` | ML engine health check |
| `GET` | `/api/ml/risk_events` | Proxy: get latest risk events |
| `GET` | `/api/students` | Student list |
| `GET` | `/api/alerts` | Alert list |
| `GET` | `/api/stats` | Session statistics |

---

## 🖥️ Dashboard Usage

### Login
- Open **http://localhost:3000**
- Use demo credentials:
  - **Email:** `admin@examguard.com`
  - **Password:** `password123`

### Overview Page (`/`)
- KPI cards: active session, students present, alerts, risk counts
- **Hall Seat Map** — 4 exam centers (A1, A2, A3, A4), each with 25 seats
  - Click a center card to **expand** and see individual seats
  - Click any seat to see the student's risk score and details
- Alerts timeline and event distribution charts

### Live Monitoring (`/live`)
- 24-camera grid (default: 4×6 layout)
- **Click any camera tile** → opens fullscreen view
- Inside fullscreen: **Turn On** to start the stream
  - **Camera 1** = real ML engine webcam (YOLO + head pose tracking)
  - **Cameras 2–20** = demo animated stream
  - **Cameras 21–24** = offline
- Use the **grid layout dropdown** to change view: 2×2 → 4×6
- **Live Risk Feed** panel (right side) shows tracked student risk in real-time when ML engine is connected

### Students (`/students`)
- Search/filter students by name, roll number, or risk class
- Click a student row → view full profile with risk trend chart, events, and identity verification

---

## ⚙️ Configuration

### Risk Score Weights (`exam-ai/services/engine/app.py`)

```python
W_PHONE            = 0.50   # Phone detected
W_LOOK_AWAY        = 0.20   # Head turned left/right/up/down
W_NOD_SHAKE        = 0.20   # Nodding or shaking head
W_HAND_UNDER_DESK  = 0.10   # Hand below desk level
```

### Risk Thresholds

```python
TH_MILD       = 0.30   # Mildly Suspicious
TH_HIGH       = 0.60   # High Risk
TH_CONFIRMED  = 0.85   # Malpractice Confirmed
```

### Head Pose Thresholds (degrees)

```python
YAW_LOOK_THRESH   = 25    # Looking left/right
PITCH_LOOK_THRESH = 20    # Looking up/down
SUSTAIN_SEC       = 1.2   # Must sustain for this many seconds
```

---

## 🔧 Troubleshooting

### Camera not turning on
- Make sure **Terminal 2 (ML Engine)** is running on port 8002
- Check that a webcam is connected and not in use by another app
- Try visiting `http://localhost:8002/camera/on` directly in your browser

### MongoDB connection error
- Ensure MongoDB is running: `net start MongoDB` (Windows) or `mongod` (Linux/Mac)
- Check the `MONGODB_URL` in `exam-monitroing/backend/.env`

### npm install fails
- Ensure Node.js version is 18+: `node --version`
- Delete `node_modules` and run `npm install` again

### YOLO model download takes long
- On first run, YOLOv8 downloads `yolov8n.pt` (~6MB). Ensure you have internet access.

### Port already in use
- Kill the process using the port:
  ```powershell
  netstat -ano | findstr :8002
  taskkill /PID <PID> /F
  ```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with ❤️ by Karthikeyan (KKB) · ExamGuard AI</p>
#
