"""
monitor.py  –  Exam-AI Standalone Webcam Monitor
=================================================
Combines both detection approaches:
  • Script-1 style : FaceMesh + solvePnP  → accurate yaw / pitch / roll
  • Script-2 style : MediaPipe Pose       → hand-under-desk wrist check

Run:
    python monitor.py
Press Q in the window to quit.
"""

import os
import time
import math
from collections import deque, defaultdict

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import mediapipe as mp

# =========================
# SETTINGS  (tune freely)
# =========================
WINDOW_SEC      = 10          # risk-scoring window (seconds)
EVIDENCE_DIR    = "evidence"
SAVE_EVIDENCE   = True
SHOW_VIDEO      = True

PERSON_ID = 0                 # COCO class id
PHONE_ID  = 67                # "cell phone"

# LLM (optional — set False to skip and run vision only)
USE_LLM    = False
LLM_URL    = "http://127.0.0.1:8003/summarize"
LLM_TIMEOUT = 40

# ─── Risk weights ──────────────────────────────────────────────────────────────
W_PHONE           = 0.50
W_LOOK_AWAY       = 0.20
W_NOD_SHAKE       = 0.20
W_HAND_UNDER_DESK = 0.10

TH_MILD      = 0.30
TH_HIGH      = 0.60
TH_CONFIRMED = 0.85

# ─── Head-pose thresholds (degrees) ───────────────────────────────────────────
YAW_LOOK_THRESH   = 25        # left/right look-away
PITCH_LOOK_THRESH = 20        # up/down look-away
SUSTAIN_SEC       = 1.2       # seconds before a "look-away" is counted

# ─── Nod / shake detection ────────────────────────────────────────────────────
ACTION_WIN_SEC    = 3.0
NOD_SWINGS_MIN    = 2
SHAKE_SWINGS_MIN  = 2
NOD_PITCH_PEAK    = 12
SHAKE_YAW_PEAK    = 14

# ─── Hand-under-desk (Pose landmarks) ─────────────────────────────────────────
WRIST_BELOW_DESK_FRAC = 0.72   # desk line = y1 + frac*(y2-y1) of person bbox
MIN_POSE_VIS          = 0.5    # landmark visibility threshold

# =========================
# FACE MESH CONSTANTS
# =========================
FACE_LANDMARKS = {
    "nose_tip":        1,
    "chin":          152,
    "left_eye_outer": 33,
    "right_eye_outer":263,
    "left_mouth":     61,
    "right_mouth":   291,
}

MODEL_POINTS_3D = np.array([
    ( 0.0,    0.0,    0.0),    # nose tip
    ( 0.0,  -63.6,  -12.5),   # chin
    (-43.3,  32.7,  -26.0),   # left eye outer
    ( 43.3,  32.7,  -26.0),   # right eye outer
    (-28.9, -28.9,  -24.1),   # left mouth
    ( 28.9, -28.9,  -24.1),   # right mouth
], dtype=np.float64)

# =========================
# HELPERS
# =========================
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
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

def call_llm(student_id, seat_id, window_sec, events, risk_score, risk_class):
    if not USE_LLM:
        return None
    payload = {
        "student_id": student_id,
        "seat_id":    seat_id,
        "window_sec": window_sec,
        "events": events + [{"type": "computed_risk", "score": round(risk_score, 3), "class": risk_class}],
    }
    try:
        r = requests.post(LLM_URL, json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json().get("response")
    except Exception as e:
        return f"(LLM not available) {e}"

# =========================
# HEAD POSE  (FaceMesh + solvePnP)
# =========================
def rotation_to_euler(R):
    """Returns pitch, yaw, roll in degrees from rotation matrix R."""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy >= 1e-6:
        x = math.atan2( R[2,1], R[2,2])   # pitch
        y = math.atan2(-R[2,0], sy)        # yaw
        z = math.atan2( R[1,0], R[0,0])   # roll
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

def estimate_head_pose(face_landmarks, frame_w, frame_h):
    """Returns (yaw, pitch, roll) in degrees, or None on failure."""
    pts_2d = []
    for k in ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth"]:
        lm = face_landmarks.landmark[FACE_LANDMARKS[k]]
        pts_2d.append((lm.x * frame_w, lm.y * frame_h))
    pts_2d = np.array(pts_2d, dtype=np.float64)

    focal = frame_w
    cx, cy = frame_w / 2, frame_h / 2
    cam_mat = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist    = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, pts_2d, cam_mat, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_to_euler(R)
    return float(yaw), float(pitch), float(roll)

# =========================
# HEAD-ACTION STATE
# =========================
class HeadActionState:
    def __init__(self):
        self.yaw_hist   = deque()
        self.pitch_hist = deque()
        self.look_right_start = None
        self.look_left_start  = None
        self.look_down_start  = None
        self.look_up_start    = None
        self.look_away_hits = 0
        self.nod_hits       = 0
        self.shake_hits     = 0

    def _trim(self, now):
        while self.yaw_hist   and now - self.yaw_hist[0][0]   > ACTION_WIN_SEC: self.yaw_hist.popleft()
        while self.pitch_hist and now - self.pitch_hist[0][0] > ACTION_WIN_SEC: self.pitch_hist.popleft()

    def update(self, now, yaw, pitch):
        self.yaw_hist.append((now, yaw))
        self.pitch_hist.append((now, pitch))
        self._trim(now)
        self._sustained(now, yaw, pitch)
        self._detect_nod_shake()

    # ── sustained look-away ────────────────────────────────────────────────────
    def _sustained(self, now, yaw, pitch):
        def _check(val, thresh, attr_name):
            start = getattr(self, attr_name)
            if val > thresh if "right" in attr_name or "down" in attr_name else val < -thresh:
                if start is None:
                    setattr(self, attr_name, now)
                elif (now - start) >= SUSTAIN_SEC:
                    self.look_away_hits += 1
                    setattr(self, attr_name, now + 999)
            else:
                setattr(self, attr_name, None)

        # right / left
        if yaw > YAW_LOOK_THRESH:
            if self.look_right_start is None: self.look_right_start = now
            elif (now - self.look_right_start) >= SUSTAIN_SEC:
                self.look_away_hits += 1; self.look_right_start = now + 999
        else:
            self.look_right_start = None

        if yaw < -YAW_LOOK_THRESH:
            if self.look_left_start is None: self.look_left_start = now
            elif (now - self.look_left_start) >= SUSTAIN_SEC:
                self.look_away_hits += 1; self.look_left_start = now + 999
        else:
            self.look_left_start = None

        # down / up
        if pitch > PITCH_LOOK_THRESH:
            if self.look_down_start is None: self.look_down_start = now
            elif (now - self.look_down_start) >= SUSTAIN_SEC:
                self.look_away_hits += 1; self.look_down_start = now + 999
        else:
            self.look_down_start = None

        if pitch < -PITCH_LOOK_THRESH:
            if self.look_up_start is None: self.look_up_start = now
            elif (now - self.look_up_start) >= SUSTAIN_SEC:
                self.look_away_hits += 1; self.look_up_start = now + 999
        else:
            self.look_up_start = None

    # ── nod / shake ────────────────────────────────────────────────────────────
    @staticmethod
    def _count_swings(values, peak_thresh):
        if len(values) < 6:
            return 0
        swings = 0; last_sign = 0; last_peak = 0.0
        for v in values:
            sign = 1 if v > 0 else (-1 if v < 0 else 0)
            if abs(v) > abs(last_peak): last_peak = v
            if sign != 0 and last_sign != 0 and sign != last_sign:
                if abs(last_peak) >= peak_thresh: swings += 1
                last_peak = v
            if sign != 0: last_sign = sign
        return swings

    def _detect_nod_shake(self):
        yaw_vals   = [v for _, v in self.yaw_hist]
        pitch_vals = [v for _, v in self.pitch_hist]
        if len(yaw_vals) < 6 or len(pitch_vals) < 6:
            return
        yaw_z   = [v - float(np.mean(yaw_vals))   for v in yaw_vals]
        pitch_z = [v - float(np.mean(pitch_vals)) for v in pitch_vals]

        if self._count_swings(yaw_z,   SHAKE_YAW_PEAK) >= SHAKE_SWINGS_MIN:
            self.shake_hits += 1; self.yaw_hist.clear()
        if self._count_swings(pitch_z, NOD_PITCH_PEAK)  >= NOD_SWINGS_MIN:
            self.nod_hits   += 1; self.pitch_hist.clear()

# =========================
# MAIN
# =========================
def main():
    ensure_dir(EVIDENCE_DIR)

    print("Loading models …")
    yolo = YOLO("yolov8n.pt")

    # FaceMesh – for accurate head pose
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # Pose – for hand-under-desk wrist check
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try camera index 1 or 2.")

    print("✅ Exam monitoring running. Press Q to quit.")
    print(f"   Window: {WINDOW_SEC}s | LLM: {'ON' if USE_LLM else 'OFF'}")

    window_start      = time.time()
    phone_hits        = defaultdict(int)
    phone_conf_sum    = defaultdict(float)
    hand_under_hits   = defaultdict(int)
    head_state        = defaultdict(HeadActionState)
    last_evidence_t   = defaultdict(float)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        h, w = frame.shape[:2]

        # ── YOLO tracking ─────────────────────────────────────────────────────
        res   = yolo.track(frame, persist=True, conf=0.35, iou=0.5, verbose=False)[0]
        boxes = res.boxes

        persons, phones = [], []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls  = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                tid  = int(b.id[0].item()) if b.id is not None else None

                if cls == PERSON_ID and tid is not None:
                    persons.append((tid, (x1, y1, x2, y2), conf))
                elif cls == PHONE_ID:
                    phones.append(((x1, y1, x2, y2), conf))

        # ── FaceMesh on full frame ─────────────────────────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_out = mp_face.process(rgb)

        faces = []  # (cx, cy, yaw, pitch, roll)
        if face_out.multi_face_landmarks:
            for flm in face_out.multi_face_landmarks:
                pose = estimate_head_pose(flm, w, h)
                if pose is None:
                    continue
                yaw, pitch, roll = pose
                nose = flm.landmark[FACE_LANDMARKS["nose_tip"]]
                cx, cy = int(nose.x * w), int(nose.y * h)
                faces.append((cx, cy, yaw, pitch, roll))

        # Assign each face to the person whose bbox contains the nose point
        face_assigned = {}
        for tid, pb, _ in persons:
            x1, y1, x2, y2 = pb
            for (cx, cy, yaw, pitch, roll) in faces:
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    face_assigned[tid] = (yaw, pitch, roll)
                    break

        # ── Per-person processing ──────────────────────────────────────────────
        for tid, pb, _ in persons:
            x1, y1, x2, y2 = pb
            xi1 = int(clamp(x1, 0, w-1)); yi1 = int(clamp(y1, 0, h-1))
            xi2 = int(clamp(x2, 0, w-1)); yi2 = int(clamp(y2, 0, h-1))

            # Phone overlap
            for phb, phconf in phones:
                if iou(pb, phb) > 0.03:
                    phone_hits[tid]     += 1
                    phone_conf_sum[tid] += phconf
                    break

            # Head pose (FaceMesh)
            if tid in face_assigned:
                yaw, pitch, roll = face_assigned[tid]
                head_state[tid].update(now, yaw, pitch)
                if SHOW_VIDEO:
                    cv2.putText(frame, f"yaw:{yaw:.0f} p:{pitch:.0f}",
                                (xi1, yi1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)

            # Hand-under-desk (Pose on person crop)
            if xi2 > xi1 and yi2 > yi1:
                crop = frame[yi1:yi2, xi1:xi2]
                if crop.size > 0:
                    pose_out = mp_pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    if pose_out.pose_landmarks:
                        lm = pose_out.pose_landmarks.landmark
                        desk_y = yi1 + int(WRIST_BELOW_DESK_FRAC * (yi2 - yi1))
                        lw_vis = lm[15].visibility; rw_vis = lm[16].visibility
                        under  = False
                        if lw_vis >= MIN_POSE_VIS:
                            abs_y = yi1 + int(lm[15].y * (yi2 - yi1))
                            if abs_y > desk_y: under = True
                        if rw_vis >= MIN_POSE_VIS:
                            abs_y = yi1 + int(lm[16].y * (yi2 - yi1))
                            if abs_y > desk_y: under = True
                        if under:
                            hand_under_hits[tid] += 1
                        if SHOW_VIDEO:
                            cv2.line(frame, (xi1, desk_y), (xi2, desk_y), (200, 200, 200), 1)

            # Draw person bbox
            if SHOW_VIDEO:
                cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}", (xi1, yi1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw phones
        if SHOW_VIDEO:
            for bb, conf in phones:
                px1, py1, px2, py2 = map(int, bb)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(frame, f"phone {conf:.2f}", (px1, py1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        # ── Window scoring ─────────────────────────────────────────────────────
        if now - window_start >= WINDOW_SEC:
            active_ids = set(phone_hits) | set(head_state) | set(hand_under_hits)

            for tid in active_ids:
                hs = head_state[tid]

                ph  = phone_hits[tid]
                ph_score = clamp(ph / 2.0, 0, 1)

                look_score     = clamp(hs.look_away_hits / 2.0, 0, 1)
                nodshake_score = clamp((hs.nod_hits + hs.shake_hits) / 2.0, 0, 1)
                hu_score       = clamp(hand_under_hits[tid] / 8.0, 0, 1)

                risk_score = (W_PHONE           * ph_score
                            + W_LOOK_AWAY       * look_score
                            + W_NOD_SHAKE       * nodshake_score
                            + W_HAND_UNDER_DESK * hu_score)
                risk_cls = risk_class_from_score(risk_score)

                events = []
                if ph > 0:
                    conf_avg = phone_conf_sum[tid] / max(ph, 1)
                    events.append({"type": "phone_detected_near_student",
                                   "count": ph, "confidence_avg": round(conf_avg, 3)})
                if hs.look_away_hits > 0:
                    events.append({"type": "sustained_look_away",
                                   "count": hs.look_away_hits,
                                   "yaw_thresh": YAW_LOOK_THRESH,
                                   "pitch_thresh": PITCH_LOOK_THRESH,
                                   "sustain_sec": SUSTAIN_SEC})
                if hs.nod_hits > 0:
                    events.append({"type": "head_nod_detected",
                                   "count": hs.nod_hits, "window_sec": ACTION_WIN_SEC})
                if hs.shake_hits > 0:
                    events.append({"type": "head_shake_detected",
                                   "count": hs.shake_hits, "window_sec": ACTION_WIN_SEC})
                if hand_under_hits[tid] > 0:
                    events.append({"type": "hand_under_desk",
                                   "count": hand_under_hits[tid],
                                   "desk_line_frac": WRIST_BELOW_DESK_FRAC})

                # Save evidence frame for high-risk students
                if SAVE_EVIDENCE and risk_cls in ("High-Risk", "Confirmed"):
                    last = last_evidence_t[tid]
                    if (time.time() - last) > (WINDOW_SEC * 1.5):
                        ts   = time.strftime("%Y%m%d_%H%M%S")
                        path = os.path.join(EVIDENCE_DIR, f"tid_{tid}_{risk_cls}_{ts}.jpg")
                        cv2.imwrite(path, frame)
                        last_evidence_t[tid] = time.time()
                        events.append({"type": "evidence_saved", "path": path})

                # Console report
                print("\n==============================")
                print(f"Student : TRACK_{tid}   Seat: WEBCAM_SEAT_{tid}")
                print(f"Risk    : {risk_score:.3f}  Class: {risk_cls}")
                print("Events  :", events if events else "[none]")

                llm_text = call_llm(f"TRACK_{tid}", f"WEBCAM_SEAT_{tid}",
                                    WINDOW_SEC, events, risk_score, risk_cls)
                if llm_text:
                    print("\nLLM Summary:\n", llm_text)

                # Reset per-window hit counters (keep history in deques)
                hs.look_away_hits = 0
                hs.nod_hits       = 0
                hs.shake_hits     = 0

            # Reset window accumulators
            phone_hits.clear()
            phone_conf_sum.clear()
            hand_under_hits.clear()
            window_start = now

        # Display
        if SHOW_VIDEO:
            cv2.imshow("Exam Monitor (press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitor stopped.")


if __name__ == "__main__":
    main()
