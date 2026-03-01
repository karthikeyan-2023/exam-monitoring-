"""
app.py  –  Exam-AI Engine  FastAPI Server  (v4 – OpenCV-only, no MediaPipe)
============================================================================
Works with Python 3.13 and any OpenCV version.

Detection on live video stream:
  ✅ YOLO person tracking with bounding boxes + ID
  ✅ OpenCV DNN face detection  → face bounding boxes (coloured by gaze)
  ✅ Mobile phone detection      → red bounding boxes + alert banner
  ✅ Head-count HUD              → top-left overlay showing people & phones
  ✅ Head pose (yaw estimate)    → derived from face-width asymmetry
  ✅ Person action labels        → Looking Left/Right, Facing Forward
  ✅ Risk scoring (10-s window)

Endpoints:
  POST /camera/on      → start webcam + analysis loop
  POST /camera/off     → stop
  GET  /video_feed     → MJPEG stream
  GET  /risk_events    → per-track risk JSON
  GET  /analytics      → real-time head_count, phone_count, per-person actions
  GET  /status         → health check

Run:
    uvicorn app:app --host 127.0.0.1 --port 8002
"""

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import time
import os
import threading
import urllib.request
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import requests as http_requests

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SEC    = 10
EVIDENCE_DIR  = "evidence"
SAVE_EVIDENCE = True

PERSON_ID = 0
PHONE_ID  = 67

USE_LLM     = False
LLM_URL     = "http://127.0.0.1:8003/summarize"
LLM_TIMEOUT = 40

W_PHONE           = 0.50
W_LOOK_AWAY       = 0.25
W_HAND_UNDER_DESK = 0.25

TH_MILD      = 0.30
TH_HIGH      = 0.60
TH_CONFIRMED = 0.85

# Gaze thresholds (face-centre offset fraction of frame width)
LOOK_AWAY_FRAC = 0.12   # if face centre deviates this much from person centre → looking away
FACE_LOOK_FRAMES = 8    # consecutive look-away frames to count as event

# ─────────────────────────────────────────────────────────────────────────────
# OpenCV DNN Face detector  (download once on first run)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_DIR    = Path(__file__).parent / "face_model"
_PROTO_PATH   = _MODEL_DIR / "deploy.prototxt"
_WEIGHTS_PATH = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

_PROTO_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"

def _download_face_model():
    _MODEL_DIR.mkdir(exist_ok=True)
    if not _PROTO_PATH.exists():
        print("⬇  Downloading face detector prototxt…")
        urllib.request.urlretrieve(_PROTO_URL, _PROTO_PATH)
    if not _WEIGHTS_PATH.exists():
        print("⬇  Downloading face detector weights (~5 MB)…")
        urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_PATH)

face_net = None
try:
    _download_face_model()
    face_net = cv2.dnn.readNetFromCaffe(str(_PROTO_PATH), str(_WEIGHTS_PATH))
    print("✅ OpenCV DNN face detector loaded.")
except Exception as e:
    print(f"⚠️  Face detector unavailable: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
CLR_PERSON = (0, 220, 80)
CLR_FACE   = (0, 200, 255)
CLR_PHONE  = (0, 50, 255)
CLR_ACTION = (0, 220, 255)
CLR_WHITE  = (255, 255, 255)
CLR_GREY   = (160, 160, 160)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def risk_class_from_score(s):
    if s < TH_MILD:      return "Normal"
    if s < TH_HIGH:      return "Mildly Suspicious"
    if s < TH_CONFIRMED: return "High-Risk"
    return "Confirmed"

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    return inter / (max(0,ax2-ax1)*max(0,ay2-ay1) + max(0,bx2-bx1)*max(0,by2-by1) - inter + 1e-6)

def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=8):
    r = min(r, (x2-x1)//3, (y2-y1)//3)
    if r < 1: r = 1
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)

def draw_label(frame, text, x, y, scale=0.48, fg=CLR_WHITE, bg=(20,20,20)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(frame, (x-2, y-th-3), (x+tw+2, y+bl), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, fg, 1, cv2.LINE_AA)

def detect_faces(frame, conf=0.65):
    """Run SSD DNN face detector. Returns list of (x1,y1,x2,y2)."""
    if face_net is None:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    dets = face_net.forward()
    faces = []
    for i in range(dets.shape[2]):
        c = float(dets[0, 0, i, 2])
        if c < conf:
            continue
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w-1,x2), min(h-1,y2)
        if x2>x1 and y2>y1:
            faces.append((x1, y1, x2, y2, c))
    return faces

def estimate_gaze(face_box, person_box):
    """
    Simple gaze: compare face centre X to person centre X.
    Returns (yaw_label, yaw_frac)  where yaw_frac is signed (-=left, +=right).
    """
    fx = (face_box[0] + face_box[2]) / 2
    px = (person_box[0] + person_box[2]) / 2
    pw = person_box[2] - person_box[0]
    if pw < 1:
        return "Forward", 0.0
    frac = (fx - px) / pw
    if frac < -LOOK_AWAY_FRAC:
        return "Looking Left", round(frac, 2)
    elif frac > LOOK_AWAY_FRAC:
        return "Looking Right", round(frac, 2)
    return "Forward", round(frac, 2)

def draw_hud(frame, head_count, phone_count, now):
    h, w = frame.shape[:2]
    time_str = time.strftime("%H:%M:%S", time.localtime(now))
    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (225, 95), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    # Labels
    cv2.putText(frame, "EXAM-AI MONITOR", (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (100, 200, 255), 1, cv2.LINE_AA)
    hc_clr = (0, 220, 80)
    cv2.putText(frame, f"HEAD COUNT : {head_count}", (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, hc_clr, 1, cv2.LINE_AA)
    ph_clr = (0, 60, 255) if phone_count > 0 else (0, 200, 70)
    cv2.putText(frame, f"PHONES DET : {phone_count}", (14, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ph_clr, 1, cv2.LINE_AA)
    cv2.putText(frame, time_str, (14, 89),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, CLR_GREY, 1, cv2.LINE_AA)
    # Phone alert banner
    if phone_count > 0:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, h-38), (w, h), (0, 0, 160), -1)
        cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"  WARNING: {phone_count} PHONE(S) DETECTED",
                    (10, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 60, 255), 2, cv2.LINE_AA)

def call_llm(student_id, seat_id, window_sec, events, risk_score, risk_class):
    if not USE_LLM:
        return None
    try:
        r = http_requests.post(LLM_URL, json={
            "student_id": student_id, "seat_id": seat_id,
            "window_sec": window_sec,
            "events": events + [{"type": "computed_risk",
                                  "score": round(risk_score, 3),
                                  "class": risk_class}]
        }, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json().get("response")
    except Exception as e:
        return f"(LLM not available) {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Per-person state tracker
# ─────────────────────────────────────────────────────────────────────────────
class PersonState:
    def __init__(self):
        self.look_away_frames = 0
        self.look_away_hits   = 0
        self.current_action   = "Forward"
        self.gaze_frac        = 0.0

    def update(self, gaze_label, gaze_frac):
        self.current_action = gaze_label
        self.gaze_frac      = gaze_frac
        if gaze_label != "Forward":
            self.look_away_frames += 1
            if self.look_away_frames >= FACE_LOOK_FRAMES:
                self.look_away_hits  += 1
                self.look_away_frames = 0
        else:
            self.look_away_frames = 0

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
camera_active = False
video_capture = None
latest_frame  = None
frame_lock    = threading.Lock()

risk_data_lock = threading.Lock()
latest_risk    = {}

analytics_lock   = threading.Lock()
latest_analytics = {
    "head_count":  0,
    "phone_count": 0,
    "persons":     {},
    "timestamp":   0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Load YOLO at startup
# ─────────────────────────────────────────────────────────────────────────────
try:
    yolo = YOLO("yolov8n.pt")
    print("✅ YOLOv8n loaded.")
except Exception as e:
    print(f"⚠️  YOLO load error: {e}")
    yolo = None

# ─────────────────────────────────────────────────────────────────────────────
# Camera capture loop
# ─────────────────────────────────────────────────────────────────────────────
def _capture_loop():
    global video_capture, latest_frame, camera_active

    # Try to open webcam
    for idx in range(3):
        video_capture = cv2.VideoCapture(idx)
        if video_capture.isOpened():
            print(f"✅ Webcam opened at index {idx}")
            break
        video_capture.release()
    else:
        print("⚠️  No webcam found (tried indices 0-2).")
        camera_active = False
        return

    phone_hits     = defaultdict(int)
    phone_conf_sum = defaultdict(float)
    person_state   = defaultdict(PersonState)
    last_evidence  = defaultdict(float)
    window_start   = time.time()
    ensure_dir(EVIDENCE_DIR)

    while camera_active:
        ret, frame = video_capture.read()
        if not ret:
            break

        now  = time.time()
        h, w = frame.shape[:2]

        # ── YOLO detection ────────────────────────────────────────────────
        persons, phones = [], []
        if yolo:
            res = yolo.track(frame, persist=True, conf=0.35, iou=0.5, verbose=False)[0]
            boxes = res.boxes
            if boxes is not None:
                for b in boxes:
                    cls  = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    tid  = int(b.id[0].item()) if b.id is not None else None

                    if cls == PERSON_ID and tid is not None:
                        persons.append((tid, (x1,y1,x2,y2), conf))
                        # Draw person box
                        px1,py1,px2,py2 = int(x1),int(y1),int(x2),int(y2)
                        cv2.rectangle(frame, (px1,py1), (px2,py2), CLR_PERSON, 2)
                        draw_label(frame, f"Person {tid}  {conf:.0%}",
                                   px1, py1-6, fg=CLR_WHITE, bg=(0, 120, 40))

                    elif cls == PHONE_ID:
                        phones.append(((x1,y1,x2,y2), conf))
                        # Draw phone box
                        phx1,phy1,phx2,phy2 = int(x1),int(y1),int(x2),int(y2)
                        draw_rounded_rect(frame, phx1,phy1, phx2,phy2, CLR_PHONE, 2)
                        draw_label(frame, f"PHONE {conf:.0%}",
                                   phx1, phy1-6, scale=0.5, fg=CLR_WHITE, bg=(0, 0, 180))

        # ── Face detection ────────────────────────────────────────────────
        detected_faces = detect_faces(frame, conf=0.60)
        face_to_person = {}   # face_idx -> tid

        for fi, (fx1,fy1,fx2,fy2,fc) in enumerate(detected_faces):
            # Assign face to overlapping person track
            fc_x = (fx1+fx2)/2; fc_y = (fy1+fy2)/2
            for tid, pb, _ in persons:
                px1,py1,px2,py2 = pb
                if px1 <= fc_x <= px2 and py1 <= fc_y <= py2:
                    face_to_person[fi] = tid
                    break
            # Draw face bounding box
            draw_rounded_rect(frame, fx1,fy1,fx2,fy2, CLR_FACE, 2, r=6)
            draw_label(frame, f"Face {fc:.0%}", fx1, fy1-5,
                       scale=0.42, fg=(0,200,255), bg=(10,50,70))

        # ── Per-person state update & action labels ───────────────────────
        person_analytics = {}
        for tid, pb, _ in persons:
            # Phone association
            has_phone = False
            for phb, phconf in phones:
                if iou(pb, phb) > 0.03:
                    phone_hits[tid]     += 1
                    phone_conf_sum[tid] += phconf
                    has_phone = True
                    break

            # Gaze estimation from face (if a face was assigned to this person)
            gaze_label, gaze_frac = "No Face detected", 0.0
            for fi, (fx1,fy1,fx2,fy2,_) in enumerate(detected_faces):
                if face_to_person.get(fi) == tid:
                    gaze_label, gaze_frac = estimate_gaze(
                        (fx1,fy1,fx2,fy2), pb)
                    break

            person_state[tid].update(gaze_label, gaze_frac)

            # Draw action label below bbox
            px1,py1,px2,py2 = int(pb[0]),int(pb[1]),int(pb[2]),int(pb[3])
            action_color = CLR_ACTION if gaze_label == "Forward" or gaze_label == "No Face detected" else (0, 60, 255)
            draw_label(frame, gaze_label, px1, py2+16, scale=0.44,
                       fg=action_color, bg=(20,20,20))
            if has_phone:
                draw_label(frame, "PHONE!", px1, py2+32,
                           scale=0.44, fg=(0,40,255), bg=(20,0,0))

            person_analytics[str(tid)] = {
                "action":    gaze_label,
                "gaze_frac": gaze_frac,
                "has_phone": has_phone,
            }

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(frame, len(persons), len(phones), now)

        # ── Window scoring ────────────────────────────────────────────────
        if now - window_start >= WINDOW_SEC:
            active_ids = set(phone_hits) | set(person_state)
            batch_risk = {}
            for tid in active_ids:
                ps  = person_state[tid]
                ph  = phone_hits[tid]
                ph_score   = clamp(ph / 2.0, 0, 1)
                look_score = clamp(ps.look_away_hits / 2.0, 0, 1)
                score = W_PHONE * ph_score + W_LOOK_AWAY * look_score
                cls   = risk_class_from_score(score)

                events = []
                if ph > 0:
                    events.append({"type": "phone_detected", "count": ph,
                                   "conf_avg": round(phone_conf_sum[tid]/max(ph,1), 3)})
                if ps.look_away_hits > 0:
                    events.append({"type": "look_away", "count": ps.look_away_hits})

                if SAVE_EVIDENCE and cls in ("High-Risk", "Confirmed"):
                    if (time.time() - last_evidence[tid]) > (WINDOW_SEC * 1.5):
                        ts   = time.strftime("%Y%m%d_%H%M%S")
                        path = os.path.join(EVIDENCE_DIR, f"tid_{tid}_{cls}_{ts}.jpg")
                        with frame_lock:
                            cf = latest_frame.copy() if latest_frame is not None else frame
                        cv2.imwrite(path, cf)
                        last_evidence[tid] = time.time()
                        events.append({"type": "evidence_saved", "path": path})

                batch_risk[str(tid)] = {
                    "risk_score": round(score, 3),
                    "risk_class": cls,
                    "events":     events,
                    "timestamp":  round(now, 1),
                }
                call_llm(f"TRACK_{tid}", f"SEAT_{tid}", WINDOW_SEC, events, score, cls)
                ps.look_away_hits = 0

            with risk_data_lock:
                latest_risk.update(batch_risk)

            phone_hits.clear()
            phone_conf_sum.clear()
            window_start = now

        # ── Update live analytics ─────────────────────────────────────────
        with analytics_lock:
            latest_analytics.update({
                "head_count":  len(persons),
                "phone_count": len(phones),
                "persons":     person_analytics,
                "timestamp":   round(now, 2),
            })

        # Share annotated frame
        with frame_lock:
            latest_frame = frame.copy()

        time.sleep(0.01)

    if video_capture:
        video_capture.release()
        video_capture = None

# ─────────────────────────────────────────────────────────────────────────────
# MJPEG generator
# ─────────────────────────────────────────────────────────────────────────────
def _gen_frames():
    while camera_active:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Exam-AI Engine", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return {
        "camera_active": camera_active,
        "model_loaded":  yolo is not None,
        "face_detector": face_net is not None,
    }

@app.post("/camera/on")
def camera_on():
    global camera_active
    if camera_active:
        return {"status": "Already running"}
    camera_active = True
    threading.Thread(target=_capture_loop, daemon=True).start()
    return {"status": "Camera turned ON"}

@app.post("/camera/off")
def camera_off():
    global camera_active
    camera_active = False
    return {"status": "Camera turned OFF"}

@app.get("/video_feed")
def video_feed():
    if not camera_active:
        return Response(
            content="Camera is off — POST /camera/on first",
            status_code=503,
        )
    return StreamingResponse(
        _gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/risk_events")
def risk_events():
    with risk_data_lock:
        return {"students": dict(latest_risk), "window_sec": WINDOW_SEC}

@app.get("/analytics")
def analytics():
    """
    Real-time per-frame analytics:
      head_count  – persons in current frame
      phone_count – phones in current frame
      persons     – per-track: action, gaze_frac, has_phone
    """
    with analytics_lock:
        return dict(latest_analytics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
