import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
from datetime import datetime

from src.camera.opencv_camera import OpenCVCamera
from src.alignment.face_alignment import align_face
from src.utils.storage import get_employee_face_dir


BG_MAIN = "#0b0f1a"
FG_TITLE = "#e8f1ff"
FG_SUB = "#8fa2c7"
ACCENT = "#00e5ff"
SUCCESS = "#00c853"


class RegistrationUI:
    def __init__(self, parent, camera_index, face_detector, arcface):
        self.parent = parent
        self.camera_index = camera_index
        self.face_detector = face_detector
        self.arcface = arcface

        self.camera = None
        self.running = False

        self.current_frame = None          # UI frame
        self.current_raw_frame = None      # RAW frame (truth)

        self.ready_to_capture = False
        self.stable_frames = 0
        self.required_stable_frames = 8

        self.required_angles = ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]
        self.current_angle_index = 0
        self.captured_faces = []

        self.root = tk.Toplevel(parent)
        self.root.title("FaceAttend – Employee Registration")
        self.root.state("zoomed")
        self.root.configure(bg=BG_MAIN)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.emp_id_var = tk.StringVar()
        self.emp_name_var = tk.StringVar()

        self.build_ui()

    # ================= UI =================
    def build_ui(self):
        container = tk.Frame(self.root, bg=BG_MAIN)
        container.pack(fill="both", expand=True, padx=30, pady=30)

        self.left_panel = tk.Frame(container, bg=BG_MAIN, width=350)
        self.left_panel.pack(side="left", fill="y")

        tk.Label(
            self.left_panel, text="Employee Registration",
            font=("Segoe UI", 20, "bold"),
            fg=FG_TITLE, bg=BG_MAIN
        ).pack(anchor="w", pady=(0, 30))

        tk.Label(self.left_panel, text="Employee ID",
                 fg=FG_SUB, bg=BG_MAIN).pack(anchor="w")
        self.emp_id_entry = tk.Entry(self.left_panel, textvariable=self.emp_id_var, width=28)
        self.emp_id_entry.pack(anchor="w", pady=(5, 20))

        tk.Label(self.left_panel, text="Employee Name",
                 fg=FG_SUB, bg=BG_MAIN).pack(anchor="w")
        self.emp_name_entry = tk.Entry(self.left_panel, textvariable=self.emp_name_var, width=28)
        self.emp_name_entry.pack(anchor="w", pady=(5, 30))

        self.start_btn = tk.Button(
            self.left_panel, text="Start Face Registration",
            bg=ACCENT, fg="#000", width=22, height=2,
            command=self.on_start_clicked
        )
        self.start_btn.pack(anchor="w", pady=10)

        self.capture_btn = tk.Button(
            self.left_panel, text="Capture Face",
            bg=SUCCESS, fg="#000", width=22, height=2,
            state="disabled", command=self.capture_face
        )
        self.capture_btn.pack(anchor="w", pady=10)

        self.status_label = tk.Label(
            self.left_panel, text="Waiting for input...",
            fg=FG_SUB, bg=BG_MAIN
        )
        self.status_label.pack(anchor="w", pady=(10, 0))

        self.right_panel = tk.Frame(container, bg="#000")
        self.right_panel.pack(side="right", fill="both", expand=True)

        self.video_label = tk.Label(self.right_panel, bg="#000")
        self.video_label.pack(fill="both", expand=True)

    # ================= START =================
    def on_start_clicked(self):
        if not self.emp_id_var.get().strip() or not self.emp_name_var.get().strip():
            messagebox.showerror("Validation Error", "Employee ID and Name are required")
            return

        self.start_btn.config(state="disabled")
        self.emp_id_entry.config(state="disabled")
        self.emp_name_entry.config(state="disabled")

        threading.Thread(target=self.camera_loop, daemon=True).start()
        self.root.after(0, self.update_ui_frame)

    # ================= CAMERA LOOP =================
    def camera_loop(self):
        self.camera = OpenCVCamera(self.camera_index)
        self.running = True

        while self.running:
            raw = self.camera.read()
            if raw is None:
                continue

            ui = raw.copy()
            detections = self.face_detector.detect(raw)
            ui = self.validate_and_draw(ui, detections)
            ui = self.resize_to_widget(ui)

            self.current_raw_frame = raw
            self.current_frame = ui

        self.camera.release()

    # ================= VALIDATION =================
    def validate_and_draw(self, frame, detections):
        if len(detections) != 1:
            self.stable_frames = 0
            self.ready_to_capture = False
            self.capture_btn.config(state="disabled")
            return frame

        det = detections[0]
        self.stable_frames += 1

        if self.stable_frames >= self.required_stable_frames:
            self.ready_to_capture = True
            self.capture_btn.config(state="normal")
        else:
            self.ready_to_capture = False
            self.capture_btn.config(state="disabled")

        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0) if self.ready_to_capture else (0, 255, 255), 2)

        for (x, y) in det.landmarks:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        return frame

    # ================= CAPTURE =================
    def capture_face(self):
        raw = self.current_raw_frame
        if raw is None or not self.ready_to_capture:
            return

        det = self.face_detector.detect(raw)[0]
        aligned = align_face(raw, det.landmarks)

        self.captured_faces.append({
            "angle": self.required_angles[self.current_angle_index],
            "aligned_face": aligned
        })

        self.current_angle_index += 1
        self.ready_to_capture = False
        self.capture_btn.config(state="disabled")

        if self.current_angle_index >= len(self.required_angles):
            self.finish_registration()

    # ================= FINISH =================
    def finish_registration(self):
        self.running = False

        emp_id = self.emp_id_var.get()
        emp_name = self.emp_name_var.get()

        face_dir = get_employee_face_dir(emp_id, emp_name)

        meta = {
            "emp_id": emp_id,
            "emp_name": emp_name,
            "created_at": datetime.now().isoformat(),
            "samples": []
        }

        for item in self.captured_faces:
            name = item["angle"].lower()
            path = face_dir / f"{name}.png"
            cv2.imwrite(str(path), item["aligned_face"])
            meta["samples"].append({"angle": name, "file": path.name})

        with open(face_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)

        messagebox.showinfo("Done", f"{emp_name} registered successfully")
        self.on_close()

    # ================= UI =================
    def resize_to_widget(self, frame):
        h, w = frame.shape[:2]
        ww, wh = self.video_label.winfo_width(), self.video_label.winfo_height()
        if ww < 10 or wh < 10:
            return frame
        s = min(ww / w, wh / h)
        r = cv2.resize(frame, (int(w * s), int(h * s)))
        canvas = np.zeros((wh, ww, 3), dtype=np.uint8)
        y, x = (wh - r.shape[0]) // 2, (ww - r.shape[1]) // 2
        canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
        return canvas

    def update_ui_frame(self):
        if self.current_frame is not None:
            rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
        if self.root.winfo_exists():
            self.root.after(30, self.update_ui_frame)

    def on_close(self):
        self.running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
