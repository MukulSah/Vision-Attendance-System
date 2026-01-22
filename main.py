# main.py
# Modern Dashboard UI for CCTV Face Attendance System (Scrollable - Fixed)

import tkinter as tk
from tkinter import messagebox

BG_MAIN = "#0b0f1a"
BG_CARD = "#0f1629"
FG_TITLE = "#e8f1ff"
FG_SUB = "#8fa2c7"
ACCENT = "#00e5ff"
SUCCESS = "#00c853"
WARNING = "#ffab00"


class MainMenuUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("FaceAttend – CCTV Attendance System")
        self.root.state("zoomed")
        self.root.configure(bg=BG_MAIN)

        self.build_scrollable_root()
        self.build_ui()

    # ================= SCROLL ROOT =================
    def build_scrollable_root(self):
        self.canvas = tk.Canvas(self.root, bg=BG_MAIN, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=BG_MAIN)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")

        self.scrollable_frame.bind("<Configure>", self._resize_scroll_region)
        self.canvas.bind("<Configure>", self._resize_canvas_width)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _resize_scroll_region(self, event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_canvas_width(self, event):
            self.canvas.itemconfig(self.canvas_window, width=event.width)

    # ================= ROOT BUILD =================
    def build_ui(self):
        self.build_header()
        self.build_hero_section()
        self.build_cards_section()
        self.build_system_overview()

    # ================= HEADER =================
    def build_header(self):
        header = tk.Frame(self.scrollable_frame, bg=BG_MAIN)
        header.pack(fill="x", padx=30, pady=(20, 10))

        left = tk.Frame(header, bg=BG_MAIN)
        left.pack(side="left")

        tk.Label(
            left, text="FaceAttend",
            font=("Segoe UI", 18, "bold"),
            fg=FG_TITLE, bg=BG_MAIN
        ).pack(side="left")

        tk.Label(
            left, text="CCTV Attendance System",
            font=("Segoe UI", 10),
            fg=FG_SUB, bg=BG_MAIN
        ).pack(side="left", padx=(10, 0))

        tk.Label(
            header, text="● SYSTEM ONLINE",
            font=("Segoe UI", 10, "bold"),
            fg=SUCCESS, bg=BG_MAIN
        ).pack(side="right")

    # ================= HERO =================
    def build_hero_section(self):
        hero = tk.Frame(self.scrollable_frame, bg=BG_MAIN)
        hero.pack(fill="both", expand=True, pady=(40, 30))
        hero_inner = tk.Frame(hero, bg=BG_MAIN)
        hero_inner.pack(expand=True)

        tk.Label(
            hero, text=" Secure Face Recognition ",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT, bg="#0a1a2a"
        ).pack(pady=(0, 10))

        tk.Label(
            hero, text="CCTV Face\nAttendance System",
            font=("Segoe UI", 34, "bold"),
            fg=ACCENT, bg=BG_MAIN, justify="center"
        ).pack()

        tk.Label(
            hero,
            text="Advanced facial recognition system for seamless employee attendance tracking.\n"
                 "Register employees, monitor live feeds, and manage attendance records.",
            font=("Segoe UI", 12),
            fg=FG_SUB, bg=BG_MAIN, justify="center"
        ).pack(pady=(10, 0))

    # ================= BUTTON =================
    def create_glow_button(self, parent, text, command):
        btn = tk.Label(
            parent, text=text,
            fg=ACCENT, bg=BG_CARD,
            font=("Segoe UI", 11, "bold"),
            bd=1, relief="solid",
            padx=25, pady=10,
            cursor="hand2"
        )

        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT, fg=BG_MAIN))
        btn.bind("<Leave>", lambda e: btn.config(bg=BG_CARD, fg=ACCENT))
        btn.bind("<Button-1>", lambda e: command())

        return btn

    # ================= CARDS =================
    def build_cards_section(self):
        container = tk.Frame(self.scrollable_frame, bg=BG_MAIN)
        container.pack(fill="both", expand=True, pady=30)

        row = tk.Frame(container, bg=BG_MAIN)
        row.pack(expand=True)

        self.create_card(row, "Registration",
                         "Register new employees with face capture",
                         self.open_registration).pack(side="left", padx=15)

        self.create_card(row, "Live Recognition",
                         "Real-time face detection via RTSP stream",
                         self.open_live_recognition).pack(side="left", padx=15)

        self.create_card(row, "Attendance View",
                         "View registered employees and records",
                         self.open_attendance_view).pack(side="left", padx=15)

    def create_card(self, parent, title, desc, command):
        card = tk.Frame(parent, bg=BG_CARD, width=280, height=220)
        card.pack_propagate(False)

        tk.Label(card, text=title,
                 font=("Segoe UI", 16, "bold"),
                 fg=FG_TITLE, bg=BG_CARD).pack(pady=(25, 10))

        tk.Label(card, text=desc,
                 font=("Segoe UI", 10),
                 fg=FG_SUB, bg=BG_CARD,
                 wraplength=240, justify="center").pack(pady=(0, 25))

        self.create_glow_button(card, "Open Module", command).pack()

        return card

    # ================= SYSTEM OVERVIEW =================
    def build_system_overview(self):
        wrapper = tk.Frame(self.scrollable_frame, bg=BG_MAIN)
        wrapper.pack(fill="both", expand=True, padx=60, pady=(40, 60))


        tk.Label(
            wrapper, text="System Overview",
            font=("Segoe UI", 16, "bold"),
            fg=FG_TITLE, bg=BG_MAIN
        ).pack(anchor="w", pady=(0, 15))

        panel = tk.Frame(wrapper, bg=BG_CARD)
        panel.pack(fill="both", expand=True, ipady=20)


        self.create_stat(panel, "0", "Registered", ACCENT).pack(side="left", expand=True)
        self.create_stat(panel, "0", "Today's Attendance", SUCCESS).pack(side="left", expand=True)
        self.create_stat(panel, "0", "Active Cameras", WARNING).pack(side="left", expand=True)
        self.create_stat(panel, "0%", "Avg Confidence", FG_TITLE).pack(side="left", expand=True)

    def create_stat(self, parent, value, label, color):
        frame = tk.Frame(parent, bg=BG_CARD, width=200, height=90)
        frame.pack_propagate(False)

        tk.Label(frame, text=value,
                 font=("Segoe UI", 22, "bold"),
                 fg=color, bg=BG_CARD).pack(pady=(10, 0))

        tk.Label(frame, text=label,
                 font=("Segoe UI", 10),
                 fg=FG_SUB, bg=BG_CARD).pack()

        return frame

    # ================= ACTIONS =================
    def open_registration(self):
        #self.root.withdraw()
        try:
            from src.ui.register_ui import RegistrationUI
            from src.detection.yolo_face_detector import YOLOv8FaceDetector
            from src.recognition.arc_face_recognition import ArcFaceONNX

            model_path = "models/detection/yolov8n-faceL.pt"
            face_detector = YOLOv8FaceDetector(model_path, conf_thres=0.5)

            arcface_path = "models/recognition/arcface_R100.onnx"
            arcface = ArcFaceONNX(arcface_path)

            # 2. Open Registration as child window
            #self.root.attributes("-disabled", True)

            ui = RegistrationUI(parent=self.root,camera_index=0, face_detector=face_detector,arcface=arcface)
            ui.run()

        except Exception as e:
            messagebox.showerror("Registration Error", str(e))
        #finally:
            #self.root.deiconify()

    def open_live_recognition(self):
        try:
            from src.ui.live_rtsp_ui import LiveRTSPUI
            ui = LiveRTSPUI(parent=self.root)
            ui.run()
        except Exception as e:
            messagebox.showerror("Live Recognition Error", str(e))

    def open_attendance_view(self):
        messagebox.showinfo("Info", "Attendance View module coming next")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MainMenuUI().run()
