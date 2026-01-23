import tkinter as tk
from tkinter import messagebox
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
from queue import Queue
import logging
import csv
from datetime import datetime
from pathlib import Path

from src.utils.storage import get_employees_root
from src.detection.yolo_face_detector import YOLOv8FaceDetector
from src.tracking.bytetrack.byte_tracker import BYTETracker
from src.recognition.arc_face_recognition import ArcFaceONNX
from src.alignment.face_alignment import align_face


# ================== LOGGING ==================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("LIVE")


# ================== CONSTANTS ==================
BG_MAIN = "#0b0f1a"
ACCENT = "#00e5ff"

EMB_BUFFER_SIZE = 1
RECOGNITION_COOLDOWN = 1.2
SIM_THRESHOLD = 0.60
FAST_LOCK_THRESHOLD = 0.72

ATTENDANCE_COOLDOWN = 2  # seconds


# ================== BYTETRACK CONFIG ==================
class ByteTrackArgs:
    track_thresh = 0.4
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False


# ================== ATTENDANCE HELPERS ==================
def get_attendance_file() -> Path:
    base = Path.home() / "Documents" / "YOAR_AttendaceSystem"
    base.mkdir(parents=True, exist_ok=True)
    return base / "attendance.csv"


def write_attendance(emp_id: str, emp_name: str):
    file = get_attendance_file()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    found = False

    if file.exists():
        with open(file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["empId"] == emp_id:
                    row["lastSeen"] = now
                    found = True
                rows.append(row)

    if not found:
        rows.append({
            "empId": emp_id,
            "empName": emp_name,
            "lastSeen": now
        })

    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["empId", "empName", "lastSeen"]
        )
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"📝 Attendance marked → {emp_id} @ {now}")


# ================== LIVE RTSP UI ==================
class LiveRTSPUI:

    def __init__(self, parent):
        log.info("🚀 Initializing Live RTSP UI")

        self.parent = parent
        self.root = tk.Toplevel(parent)
        self.root.title("Live RTSP Recognition")
        self.root.state("zoomed")
        self.root.configure(bg=BG_MAIN)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.running = False

        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)

        self.frame_count = 0
        self.detect_every_n = 15
        self.target_fps = 15

        log.info("🧠 Loading models")
        self.face_detector = YOLOv8FaceDetector(
            "models/detection/yolov8n-faceL.pt",
            conf_thres=0.5
        )
        self.tracker = BYTETracker(ByteTrackArgs(), frame_rate=self.target_fps)
        self.arcface = ArcFaceONNX("models/recognition/arcface_R100.onnx")

        self.known_embeddings = self.load_registered_faces()

        self.identity_cache = {}
        self.attendance_cache = {}  # emp_key -> timestamp

        self.rtsp_var = tk.StringVar(
            value="rtsp://admin:password@192.168.1.35:554/cam/realmonitor?channel=1&subtype=1"
        )

        self.build_ui()

    # ================== UI ==================
    def build_ui(self):
        top = tk.Frame(self.root, bg=BG_MAIN)
        top.pack(fill="x", padx=20, pady=15)

        tk.Entry(top, textvariable=self.rtsp_var, width=90).pack(side="left", padx=10)
        tk.Button(top, text="Connect", bg=ACCENT, fg="#000",
                  command=self.start_stream).pack(side="left")

        self.video_label = tk.Label(self.root, bg="#000")
        self.video_label.pack(fill="both", expand=True, padx=20, pady=20)

    # ================== START STREAM ==================
    def start_stream(self):
        if self.running:
            return

        url = self.rtsp_var.get().strip()
        log.info(f"📡 Connecting RTSP: {url}")

        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            messagebox.showerror("Error", "RTSP open failed")
            return

        self.running = True
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.inference_loop, daemon=True).start()
        self.update_ui()

    # ================== CAPTURE LOOP ==================
    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            if self.frame_queue.full():
                self.frame_queue.get_nowait()

            self.frame_queue.put(frame)

    # ================== INFERENCE LOOP ==================
    def inference_loop(self):
        last_detections = []

        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.005)
                continue

            frame = self.frame_queue.get()
            self.frame_count += 1
            h, w = frame.shape[:2]

            if self.frame_count % self.detect_every_n == 0:
                det_in = cv2.resize(frame, (416, 416))
                last_detections = self.face_detector.detect(det_in)

            track_inputs = []
            landmark_inputs = []
            sx, sy = w / 416, h / 416

            for det in last_detections:
                x1, y1, x2, y2 = map(int, (
                    det.bbox[0]*sx, det.bbox[1]*sy,
                    det.bbox[2]*sx, det.bbox[3]*sy
                ))

                if (x2 - x1) * (y2 - y1) < 3600:
                    continue

                track_inputs.append([x1, y1, x2, y2, det.confidence])

                lm = det.landmarks.copy()
                lm[:, 0] *= sx
                lm[:, 1] *= sy
                landmark_inputs.append((x1, y1, x2, y2, lm))

            tracks = []
            if track_inputs:
                tracks = self.tracker.update(
                    np.array(track_inputs, dtype=np.float32),
                    (h, w), (h, w)
                )

            for t in tracks:
                self.process_track(t, frame, landmark_inputs)

            now = time.time()
            self.identity_cache = {
                k: v for k, v in self.identity_cache.items()
                if now - v["last_seen_track"] < 2.0
            }

            if self.result_queue.full():
                self.result_queue.get_nowait()

            self.result_queue.put(frame)

    # ================== TRACK PROCESS ==================
    def process_track(self, t, frame, landmark_inputs):
        x1, y1, x2, y2 = map(int, t.tlbr)
        tid = t.track_id

        cache = self.identity_cache.setdefault(tid, {
            "embeddings": [],
            "locked": False,
            "last_attempt": 0,
            "last_seen_track": time.time()
        })

        cache["last_seen_track"] = time.time()

        landmarks = None
        for bx1, by1, _, _, lm in landmark_inputs:
            if abs(bx1 - x1) < 40 and abs(by1 - y1) < 40:
                landmarks = lm
                break

        if landmarks is not None and not cache["locked"]:
            if time.time() - cache["last_attempt"] >= RECOGNITION_COOLDOWN:
                self.run_arcface(cache, frame, x1, y1, x2, y2, landmarks)

        label = "UNKNOWN"
        color = (0, 0, 255)
        if cache.get("locked"):
            label = f"{cache['emp_id']} ({cache['sim']:.2f})"
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ================== ARCFACE ==================
    def run_arcface(self, cache, frame, x1, y1, x2, y2, landmarks):
        cache["last_attempt"] = time.time()

        pad = 20
        fx1, fy1 = max(0, x1 - pad), max(0, y1 - pad)
        fx2, fy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)

        roi = frame[fy1:fy2, fx1:fx2]
        if roi.size == 0:
            return

        lm = landmarks.copy()
        lm[:, 0] -= fx1
        lm[:, 1] -= fy1

        try:
            aligned = align_face(roi, lm)
        except Exception:
            return

        aligned = cv2.resize(aligned, (112, 112))
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        emb = self.arcface.get_embedding(aligned)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return

        emb = emb / norm
        cache["embeddings"].append(emb)

        count = len(cache["embeddings"])

        if count >= 2:
            mean_fast = np.mean(cache["embeddings"], axis=0)
            mean_fast /= np.linalg.norm(mean_fast)
            best_id, best_sim = self.match_embedding(mean_fast)

            if best_sim >= FAST_LOCK_THRESHOLD:
                self.lock_identity(cache, best_id, best_sim)
                return

        if count < EMB_BUFFER_SIZE:
            return

        mean_emb = np.mean(cache["embeddings"], axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        best_id, best_sim = self.match_embedding(mean_emb)

        if best_sim >= SIM_THRESHOLD:
            self.lock_identity(cache, best_id, best_sim)
        else:
            cache["embeddings"].clear()

    # ================== MATCH & LOCK ==================
    def match_embedding(self, emb):
        best_id, best_sim = None, 0.0
        for emp, lst in self.known_embeddings.items():
            for ref in lst:
                sim = float(np.dot(emb, ref))
                if sim > best_sim:
                    best_sim, best_id = sim, emp
        return best_id, best_sim

    def lock_identity(self, cache, emp_key, sim):
        emp_id, emp_name = emp_key.split("_", 1)

        cache.update({
            "emp_id": emp_key,
            "sim": sim,
            "locked": True
        })

        now = time.time()
        last = self.attendance_cache.get(emp_key, 0)

        if now - last >= ATTENDANCE_COOLDOWN:
            write_attendance(emp_id, emp_name)
            self.attendance_cache[emp_key] = now

        log.info(f"🔒 LOCKED → {emp_key} ({sim:.2f})")

    # ================== UI UPDATE ==================
    def update_ui(self):
        if not self.result_queue.empty():
            frame = self.result_queue.get()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        if self.running:
            self.root.after(30, self.update_ui)

    # ================== CLOSE ==================
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    # ================== LOAD EMBEDDINGS ==================
    def load_registered_faces(self):
        db = {}
        base = get_employees_root()

        if not base.exists():
            return db

        for emp_dir in base.iterdir():
            emb_path = emp_dir / "face_data" / "embeddings.npy"
            if not emb_path.exists():
                continue

            data = np.load(emb_path, allow_pickle=True).item()
            db[emp_dir.name] = []

            for v in data.values():
                if isinstance(v, np.ndarray):
                    v = v / np.linalg.norm(v)
                    db[emp_dir.name].append(v)

            log.info(f"✅ {emp_dir.name}: {len(db[emp_dir.name])} embeddings")

        return db
