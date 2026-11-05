"""Giao diện thử nghiệm: đăng ký tên, chụp vân tay bằng camera có hướng dẫn, xác thực.

    python -m demo                     # webcam mặc định
    python -m demo --camera 1
    python -m demo --camera http://192.168.1.5:8080/video   # điện thoại làm camera (IP Webcam)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
from PIL import Image, ImageTk

from fpsys.config import Config
from fpsys.io import load_image
from fpsys.types import CaptureError
from fpsys.viz import grid

from .camera import Camera
from .guidance import Guidance, Guide
from .session import FINGERS, HINTS, DemoSession, finger_label

log = logging.getLogger(__name__)

VIDEO_W, VIDEO_H = 760, 570
COLORS = {"ready": "#1a9e3f", "ok": "#5aa02c", "warn": "#d97706", "err": "#c0392b",
          "info": "#2c5aa0"}


def _to_photo(img: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


class Analyzer(threading.Thread):
    """Chạy Guide.analyze trên khung hình mới nhất, liên tục, trong luồng nền."""

    def __init__(self, app: App):
        super().__init__(daemon=True)
        self.app = app
        self.result: Guidance | None = None
        self.paused = False
        self._stop = threading.Event()

    def run(self) -> None:
        last = -1
        while not self._stop.is_set():
            cam = self.app.camera
            if cam is None or self.paused:
                time.sleep(0.05)
                continue
            fid, frame = cam.latest()
            if frame is None or fid == last:
                time.sleep(0.01)
                continue
            last = fid
            try:
                with self.app.model_lock:
                    self.result = self.app.guide.analyze(frame)
            except Exception as e:  # không để luồng nền chết im lặng
                log.exception("Lỗi phân tích khung hình")
                self.app.events.put(("log", f"Lỗi phân tích: {e}"))
                time.sleep(0.5)

    def stop(self) -> None:
        self._stop.set()


class App:
    def __init__(self, root: tk.Tk, cfg: Config, camera_source: str = "0"):
        self.root = root
        self.cfg = cfg
        self.session = DemoSession(cfg)
        self.guide = Guide(cfg)
        self.model_lock = threading.Lock()
        self.events: queue.Queue = queue.Queue()
        self.camera: Camera | None = None
        self.busy = False
        self.mode = "idle"  # idle | enroll | verify
        self.samples: list = []
        self._photo = None
        self._preview = None
        self._last_steps: dict = {}

        root.title("fpsys — thử nghiệm vân tay không tiếp xúc")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build(camera_source)
        self.analyzer = Analyzer(self)
        self.analyzer.start()
        self.refresh_users()
        self.open_camera()
        self.root.after(30, self._tick)

    # ------------------------------------------------------------------ UI
    def _build(self, camera_source: str) -> None:
        style = ttk.Style()
        style.configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=8)
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Label không có ảnh hiểu width/height theo ký tự → gắn ảnh trống để tính theo pixel
        self._blank_video = ImageTk.PhotoImage(Image.new("RGB", (VIDEO_W, VIDEO_H), "#111"))
        self.video = tk.Label(left, bg="#111", image=self._blank_video)
        self.video.grid(row=0, column=0, sticky="nsew")
        self.msg = tk.Label(left, text="Đang khởi động…", font=("Segoe UI", 15, "bold"),
                            fg="white", bg=COLORS["info"], wraplength=VIDEO_W, pady=8)
        self.msg.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.metrics = ttk.Label(left, text="", font=("Consolas", 9))
        self.metrics.grid(row=2, column=0, sticky="w")
        bar = ttk.Frame(left)
        bar.grid(row=3, column=0, sticky="ew", pady=6)
        self.capture_btn = ttk.Button(bar, text="📷  Chụp", style="Big.TButton",
                                      command=self.capture)
        self.capture_btn.pack(side="left")
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Tự chụp khi sẵn sàng", variable=self.auto_var).pack(
            side="left", padx=10)
        ttk.Button(bar, text="Mở ảnh từ file…", command=self.open_file).pack(side="left")
        ttk.Button(bar, text="Xem 9 bước", command=self.show_steps).pack(side="left", padx=6)

        # camera
        cam = ttk.LabelFrame(right, text="Camera", padding=6)
        cam.pack(fill="x")
        self.src_var = tk.StringVar(value=camera_source)
        ttk.Entry(cam, textvariable=self.src_var, width=34).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(cam, text="Mở", command=self.open_camera).grid(row=0, column=1)
        ttk.Label(cam, wraplength=330, foreground="#555", text=(
            "Webcam laptop thường không đủ nét để thấy vân. Nên dùng điện thoại làm camera: "
            "cài app IP Webcam (Android) rồi nhập http://<ip-điện-thoại>:8080/video")).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        tabs = ttk.Notebook(right)
        tabs.pack(fill="x", pady=8)
        tabs.bind("<<NotebookTabChanged>>", lambda e: self._cancel_mode())

        # đăng ký
        t1 = ttk.Frame(tabs, padding=8)
        tabs.add(t1, text="Đăng ký")
        ttk.Label(t1, text="Họ tên").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(t1, textvariable=self.name_var, width=26).grid(row=0, column=1, sticky="w")
        ttk.Label(t1, text="Ngón").grid(row=1, column=0, sticky="w", pady=4)
        self.finger_var = tk.StringVar(value=finger_label("R2"))
        ttk.Combobox(t1, textvariable=self.finger_var, state="readonly", width=24,
                     values=[finger_label(f) for f in FINGERS]).grid(row=1, column=1, sticky="w")
        ttk.Label(t1, text="Số mẫu").grid(row=2, column=0, sticky="w")
        self.n_var = tk.IntVar(value=self.cfg.guide.enroll_samples)
        ttk.Spinbox(t1, from_=max(2, self.cfg.enroll.min_templates), to=8,
                    textvariable=self.n_var, width=5).grid(row=2, column=1, sticky="w")
        ttk.Button(t1, text="Bắt đầu đăng ký", command=self.start_enroll).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.enroll_status = ttk.Label(t1, text="", font=("Segoe UI", 10, "bold"))
        self.enroll_status.grid(row=4, column=0, columnspan=2, sticky="w")

        # xác thực
        t2 = ttk.Frame(tabs, padding=8)
        tabs.add(t2, text="Xác thực")
        ttk.Label(t2, text="Người dùng").grid(row=0, column=0, sticky="w")
        self.user_var = tk.StringVar()
        self.user_box = ttk.Combobox(t2, textvariable=self.user_var, state="readonly",
                                     width=24)
        self.user_box.grid(row=0, column=1, sticky="w")
        self.user_box.bind("<<ComboboxSelected>>", lambda e: self._update_verify_fingers())
        ttk.Label(t2, text="Ngón").grid(row=1, column=0, sticky="w", pady=4)
        self.vfinger_var = tk.StringVar()
        self.vfinger_box = ttk.Combobox(t2, textvariable=self.vfinger_var, state="readonly",
                                        width=24)
        self.vfinger_box.grid(row=1, column=1, sticky="w")
        ttk.Button(t2, text="Bắt đầu xác thực", command=self.start_verify).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.verify_result = tk.Label(t2, text="", font=("Segoe UI", 16, "bold"), width=22)
        self.verify_result.grid(row=3, column=0, columnspan=2, pady=4)
        self.verify_detail = ttk.Label(t2, text="", font=("Consolas", 9))
        self.verify_detail.grid(row=4, column=0, columnspan=2, sticky="w")

        # người dùng
        t3 = ttk.Frame(tabs, padding=8)
        tabs.add(t3, text="Người dùng")
        self.tree = ttk.Treeview(t3, columns=("sid", "fingers"), height=6)
        self.tree.heading("#0", text="Tên")
        self.tree.heading("sid", text="Mã")
        self.tree.heading("fingers", text="Ngón đã đăng ký")
        self.tree.column("#0", width=130)
        self.tree.column("sid", width=50)
        self.tree.column("fingers", width=140)
        self.tree.pack(fill="x")
        ttk.Button(t3, text="Xóa người dùng đã chọn", command=self.delete_user).pack(
            anchor="w", pady=4)

        # thu dữ liệu
        dc = ttk.LabelFrame(right, text="Thu dữ liệu đánh giá", padding=6)
        dc.pack(fill="x")
        self.save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dc, variable=self.save_var, text=(
            "Lưu ảnh vào data/own (đặt tên theo quy ước)")).grid(row=0, column=0, columnspan=4,
                                                                sticky="w")
        ttk.Label(dc, text="Phiên").grid(row=1, column=0, sticky="w")
        self.sess_var = tk.StringVar(value="s1")
        ttk.Combobox(dc, textvariable=self.sess_var, values=["s1", "s2"], width=4,
                     state="readonly").grid(row=1, column=1, sticky="w")
        ttk.Label(dc, text="Điều kiện").grid(row=1, column=2, sticky="w", padx=(8, 0))
        self.cond_var = tk.StringVar(value="c")
        ttk.Combobox(dc, textvariable=self.cond_var, values=["c", "u"], width=4,
                     state="readonly").grid(row=1, column=3, sticky="w")
        ttk.Label(dc, foreground="#555", wraplength=330, text=(
            "Ảnh vân tay là dữ liệu sinh trắc học: chỉ lưu khi người được chụp đồng ý. "
            "c = có kiểm soát (nền tối, đúng khoảng cách), u = tự nhiên.")).grid(
            row=2, column=0, columnspan=4, sticky="w")

        # kết quả
        res = ttk.LabelFrame(right, text="Ảnh vừa chụp", padding=6)
        res.pack(fill="both", expand=True, pady=(8, 0))
        self._blank_preview = ImageTk.PhotoImage(Image.new("RGB", (330, 220), "#222"))
        self.preview = tk.Label(res, bg="#222", image=self._blank_preview)
        self.preview.pack()
        self.log = ScrolledText(res, height=8, width=46, font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=(6, 0))

    # ---------------------------------------------------------------- helpers
    def _log(self, text: str) -> None:
        self.log.insert("end", time.strftime("%H:%M:%S ") + text + "\n")
        self.log.see("end")

    def _set_msg(self, text: str, kind: str) -> None:
        self.msg.configure(text=text, bg=COLORS[kind])

    def _finger(self, var: tk.StringVar) -> str:
        return var.get().split(" ")[0]

    def refresh_users(self) -> None:
        users = self.session.users()
        self.user_box.configure(values=users)
        if users and self.user_var.get() not in users:
            self.user_var.set(users[0])
        self._update_verify_fingers()
        self.tree.delete(*self.tree.get_children())
        for u in users:
            fingers = self.session.enrolled_fingers(u)
            self.tree.insert("", "end", text=u, values=(self.session.subject_id(u),
                                                         ", ".join(fingers)))

    def _update_verify_fingers(self) -> None:
        u = self.user_var.get()
        fingers = self.session.enrolled_fingers(u) if u else []
        self.vfinger_box.configure(values=[finger_label(f) for f in fingers])
        self.vfinger_var.set(finger_label(fingers[0]) if fingers else "")

    def _cancel_mode(self) -> None:
        if self.mode != "idle" and not self.busy:
            self.mode = "idle"
            self.samples = []
            self.enroll_status.configure(text="")

    # ---------------------------------------------------------------- camera
    def open_camera(self) -> None:
        if self.camera is not None:
            self.camera.stop()
        self.guide.reset()
        self.camera = Camera(self.src_var.get())
        self.camera.start()
        self._log(f"Mở camera {self.src_var.get()!r}")

    def _tick(self) -> None:
        cam = self.camera
        g = self.analyzer.result
        if cam is not None and cam.error:
            self._set_msg(cam.error + " — kiểm tra nguồn camera rồi bấm Mở.", "err")
        elif cam is not None:
            _, frame = cam.latest()
            if frame is not None:
                view = frame.copy()
                if g is not None and g.box is not None:
                    x1, y1, x2, y2 = (int(v) for v in g.box)
                    color = (60, 200, 60) if g.ok else (0, 140, 255)
                    t = max(2, frame.shape[1] // 300)
                    cv2.rectangle(view, (x1, y1), (x2, y2), color, t)
                self._photo = _to_photo(view, VIDEO_W, VIDEO_H)
                self.video.configure(image=self._photo)
                if self.busy:
                    self._set_msg("Đang xử lý ảnh…", "info")
                elif g is None:
                    self._set_msg("Đang nạp model YOLO…", "info")
                else:
                    kind = "ready" if g.ready else ("ok" if g.ok else "warn")
                    prefix = {"enroll": f"[Đăng ký {len(self.samples) + 1}/{self.n_var.get()}] ",
                              "verify": "[Xác thực] "}.get(self.mode, "")
                    self._set_msg(prefix + g.message, kind)
                    res = cam.resolution or (0, 0)
                    self.metrics.configure(text=f"camera {res[0]}×{res[1]}  |  " + "  ".join(
                        f"{k}={v}" for k, v in g.metrics.items()) + f"  |  {g.elapsed_ms:.0f} ms")
                    if g.ready and self.auto_var.get() and self.mode != "idle" and not self.busy:
                        self.capture()
        self._drain_events()
        self.root.after(30, self._tick)

    # ---------------------------------------------------------------- actions
    def start_enroll(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Thiếu tên", "Nhập họ tên trước khi đăng ký.")
            return
        finger = self._finger(self.finger_var)
        if name in self.session.users() and finger in self.session.enrolled_fingers(name):
            if not messagebox.askyesno("Đã đăng ký",
                                       f"{name} đã đăng ký {finger}. Đăng ký lại (ghi đè)?"):
                return
        self.mode, self.samples = "enroll", []
        self.guide.reset()
        self.enroll_status.configure(text=f"Chụp mẫu 1/{self.n_var.get()}…", foreground="")
        self._log(f"Bắt đầu đăng ký {name} / {finger}")

    def start_verify(self) -> None:
        if not self.user_var.get() or not self.vfinger_var.get():
            messagebox.showwarning("Chưa chọn", "Chọn người dùng và ngón đã đăng ký.")
            return
        self.mode = "verify"
        self.guide.reset()
        self.verify_result.configure(text="Đưa ngón tay vào…", fg="#333")
        self.verify_detail.configure(text="")

    def capture(self, image: np.ndarray | None = None, source: str = "camera") -> None:
        if self.busy:
            return
        if image is None:
            if self.camera is None:
                return
            _, frame = self.camera.latest()
            if frame is None:
                return
            image = frame.copy()
        self.busy = True
        self.guide.reset()
        threading.Thread(target=self._process, args=(image, source), daemon=True).start()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if path:
            self.capture(load_image(path), source=path)

    # ---------------------------------------------------------------- worker
    def _process(self, image: np.ndarray, source: str) -> None:
        try:
            with self.model_lock:
                tr = self.session.extract(image)
            self.events.put(("trace", (image, tr, source)))
        except Exception as e:
            log.exception("Lỗi xử lý ảnh")
            self.events.put(("fatal", str(e)))

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                self._log(payload)
            elif kind == "fatal":
                self.busy = False
                self._log(f"LỖI: {payload}")
            elif kind == "trace":
                self._handle_trace(*payload)

    def _save_if_enabled(self, image: np.ndarray, name: str | None, finger: str | None) -> None:
        if not self.save_var.get():
            return
        if not name or not finger:
            self._log("Không lưu ảnh: chưa có tên/ngón (nhập ở tab Đăng ký hoặc Xác thực).")
            return
        p = self.session.save_image(image, name, finger, self.sess_var.get(), self.cond_var.get())
        self._log(f"Đã lưu {p.name}")

    def _handle_trace(self, image: np.ndarray, tr, source: str) -> None:
        self.busy = False
        self._last_steps = tr.steps
        best = next((tr.steps[k] for k in ("09_minutiae", "03_norm", "01_crop") if k in tr.steps),
                    image)
        self._preview = _to_photo(best, 330, 220)
        self.preview.configure(image=self._preview)
        name = finger = None
        if self.mode == "enroll":
            name, finger = self.name_var.get().strip(), self._finger(self.finger_var)
        elif self.mode == "verify":
            name, finger = self.user_var.get(), self._finger(self.vfinger_var)
        elif self.name_var.get().strip():
            name, finger = self.name_var.get().strip(), self._finger(self.finger_var)
        if source == "camera":
            self._save_if_enabled(image, name, finger)

        if tr.error is not None:
            hint = HINTS.get(tr.error.code, str(tr.error))
            self._log(f"✗ {tr.error.code.value}: {tr.error.detail}")
            self._set_msg(f"Chụp lại: {hint}", "err")
            if self.mode == "verify":
                self.verify_result.configure(text="Chụp lại", fg=COLORS["warn"])
                self.verify_detail.configure(text=hint)
            return
        t = tr.template
        self._log(f"✓ {len(t.minutiae)} minutiae, chu kỳ vân {t.ridge_period:.1f} px, "
                  f"{tr.timing_ms.get('total', 0):.0f} ms")
        if self.mode == "enroll":
            self.samples.append(t)
            n = self.n_var.get()
            if len(self.samples) < n:
                self.enroll_status.configure(text=f"Đã có {len(self.samples)}/{n} mẫu — chụp "
                                                  f"mẫu tiếp theo (nhấc tay ra rồi đặt lại).")
                return
            try:
                kept = self.session.enroll(name, finger, self.samples)
                self.enroll_status.configure(
                    text=f"✓ Đăng ký xong {name} / {finger}: giữ {len(kept)}/{n} mẫu",
                    foreground=COLORS["ready"])
                self._log(f"Đăng ký xong {name}/{finger} ({len(kept)}/{n} mẫu nhất quán)")
            except CaptureError as e:
                self.enroll_status.configure(text="✗ " + HINTS.get(e.code, str(e)),
                                             foreground=COLORS["err"])
                self._log(f"✗ Đăng ký thất bại: {e}")
            self.mode, self.samples = "idle", []
            self.refresh_users()
        elif self.mode == "verify":
            res = self.session.verify(name, finger, t)
            self.verify_result.configure(text="✓ CHẤP NHẬN" if res.accept else "✗ TỪ CHỐI",
                                         fg=COLORS["ready"] if res.accept else COLORS["err"])
            self.verify_detail.configure(text=f"điểm {res.score:.3f} / ngưỡng {res.threshold:.3f}"
                                              f"\ntừng mẫu: "
                                              + ", ".join(f"{s:.3f}" for s in res.per_template))
            self._log(f"Xác thực {name}/{finger}: {'CHẤP NHẬN' if res.accept else 'TỪ CHỐI'} "
                      f"(điểm {res.score:.3f}, ngưỡng {res.threshold:.3f})")
            self.mode = "idle"

    # ---------------------------------------------------------------- misc
    def show_steps(self) -> None:
        if not self._last_steps:
            messagebox.showinfo("Chưa có ảnh", "Chụp một ảnh trước.")
            return
        win = tk.Toplevel(self.root)
        win.title("9 bước xử lý")
        photo = _to_photo(grid(self._last_steps), 1100, 900)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo  # giữ tham chiếu
        lbl.pack()

    def delete_user(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0], "text")
        if messagebox.askyesno("Xóa", f"Xóa {name} và toàn bộ mẫu đã đăng ký? "
                                      "(Ảnh đã lưu trong data/own không bị xóa.)"):
            self.session.delete(name)
            self.refresh_users()
            self._log(f"Đã xóa {name}")

    def close(self) -> None:
        self.analyzer.stop()
        if self.camera is not None:
            self.camera.stop()
        self.root.destroy()
