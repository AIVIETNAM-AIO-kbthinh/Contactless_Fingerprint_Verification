"""Đọc camera trong luồng nền (webcam theo chỉ số, hoặc URL luồng video của điện thoại)."""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


def parse_source(source: str) -> int | str:
    s = source.strip()
    return int(s) if s.isdigit() else s


class Camera(threading.Thread):
    """Luôn giữ khung hình mới nhất (tránh trễ do bộ đệm, nhất là với luồng mạng)."""

    def __init__(self, source: str, width: int = 1920, height: int = 1080):
        super().__init__(daemon=True)
        self.source = parse_source(source)
        self.width, self.height = width, height
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._stop = threading.Event()
        self.error: str | None = None
        self.resolution: tuple[int, int] | None = None

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"Không mở được camera: {self.source!r}"
            return
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        fails = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 50:
                    self.error = "Mất tín hiệu camera"
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self.resolution = (frame.shape[1], frame.shape[0])
        cap.release()

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, self._frame

    def stop(self) -> None:
        self._stop.set()
