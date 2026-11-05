"""Hướng dẫn trực tiếp khi chụp: mỗi khung hình → một lời nhắc cho người dùng.

Các ngưỡng phơi sáng / độ nét dùng chung với pipeline (cfg.quality, cfg.normalize) để khung hình
được báo "sẵn sàng" cũng qua được extract().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from fpsys import detect as det_mod
from fpsys.config import Config
from fpsys.normalize import estimate_period

# Thứ tự ưu tiên: lỗi đứng trước được báo trước.
MESSAGES = {
    "NO_FINGER": "Không thấy đầu ngón tay. Giơ MỘT ngón, mặt bụng ngón hướng vào camera.",
    "MULTI": "Thấy nhiều ngón tay. Chỉ giơ MỘT ngón, gập các ngón còn lại.",
    "OFF_CENTER": "Đầu ngón bị cắt ở mép khung. Đưa đầu ngón vào giữa khung hình.",
    "TOO_CLOSE": "Quá gần. Lùi ngón tay ra xa camera một chút.",
    "TOO_FAR": "Quá xa. Đưa ngón tay lại gần camera hơn.",
    "DARK": "Thiếu sáng. Bật thêm đèn hoặc đèn flash.",
    "BRIGHT": "Quá sáng. Giảm đèn hoặc lùi xa nguồn sáng.",
    "GLARE": "Da bị cháy sáng. Giảm đèn/flash hoặc nghiêng nhẹ ngón tay.",
    "MOVING": "Giữ yên tay.",
    "BLURRY": "Chưa thấy rõ vân. Giữ yên, chạm để lấy nét vào đầu ngón.",
    "OK": "Tốt, giữ nguyên tư thế…",
    "READY": "Sẵn sàng. Bấm Chụp (hoặc chờ tự chụp).",
}


@dataclass
class Guidance:
    code: str  # một khóa của MESSAGES
    message: str
    ok: bool  # khung hình hiện tại đạt mọi điều kiện
    ready: bool  # đạt liên tiếp đủ ready_frames lần
    box: tuple[float, float, float, float] | None = None  # box đầu ngón (tọa độ khung hình)
    metrics: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


def _inner(frame: np.ndarray, box: tuple[float, float, float, float], frac: float = 0.6
           ) -> np.ndarray:
    """Vùng giữa box (bỏ viền, nơi thường là nền hoặc mép ngón)."""
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * frac, (y2 - y1) * frac
    H, W = frame.shape[:2]
    a, b = int(max(0, cx - w / 2)), int(min(W, cx + w / 2))
    c, d = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
    return frame[c:d, a:b]


class Guide:
    """Có trạng thái (để kiểm tra rung và đếm số khung đạt liên tiếp)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_center: tuple[float, float] | None = None
        self.ok_count = 0

    def reset(self) -> None:
        self.prev_center = None
        self.ok_count = 0

    def _result(self, code: str, box=None, metrics=None, t0: float = 0.0) -> Guidance:
        ok = code == "OK"
        self.ok_count = self.ok_count + 1 if ok else 0
        ready = ok and self.ok_count >= self.cfg.guide.ready_frames
        if ready:
            code = "READY"
        return Guidance(code, MESSAGES[code], ok, ready, box, metrics or {},
                        (time.perf_counter() - t0) * 1000)

    def analyze(self, frame: np.ndarray) -> Guidance:
        t0 = time.perf_counter()
        cfg, gc, q = self.cfg, self.cfg.guide, self.cfg.quality
        boxes = det_mod._run_model(frame, cfg)  # không dùng TTA để giữ tốc độ khung hình
        if len(boxes) == 0:
            self.prev_center = None
            return self._result("NO_FINGER", t0=t0)
        i = det_mod.select_box(boxes, cfg.detect.tie_conf_delta)
        x1, y1, x2, y2, conf = (float(v) for v in boxes[i])
        box = (x1, y1, x2, y2)
        H, W = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        m: dict = {"conf": round(conf, 3), "box_w": round(bw), "n_boxes": len(boxes)}
        strong = int(np.sum(boxes[:, 4] >= conf - cfg.detect.tie_conf_delta))
        if strong > 1:
            return self._result("MULTI", box, m, t0)
        margin = gc.edge_margin_frac * min(W, H)
        if x1 < margin or y1 < margin or x2 > W - margin:
            # đáy box chạm biên dưới là bình thường (thân ngón đi ra khỏi khung)
            return self._result("OFF_CENTER", box, m, t0)
        if bh > gc.max_box_frac * H or bw > gc.max_box_frac * W:
            return self._result("TOO_CLOSE", box, m, t0)
        if bw < gc.min_tip_px:
            return self._result("TOO_FAR", box, m, t0)

        roi = _inner(frame, box)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        # cùng định nghĩa với normalize(): pixel có kênh lớn nhất ≥ sat_level
        sat = float((roi.max(axis=2) >= cfg.normalize.sat_level).mean())
        m.update(brightness=round(brightness, 1), sat_ratio=round(sat, 3))
        if brightness < q.min_brightness:
            return self._result("DARK", box, m, t0)
        if brightness > q.max_brightness:
            return self._result("BRIGHT", box, m, t0)
        if sat > q.max_sat_ratio:
            return self._result("GLARE", box, m, t0)

        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        moving = self.prev_center is not None and np.hypot(
            center[0] - self.prev_center[0], center[1] - self.prev_center[1]) > \
            gc.stable_frac * bw
        self.prev_center = center
        if moving:
            return self._result("MOVING", box, m, t0)

        nc = cfg.normalize
        est = estimate_period(gray, np.ones(gray.shape, bool), nc.period_block, nc.period_min,
                              nc.period_max, max_blocks=60,
                              min_cycles=nc.period_min_cycles)
        if est is not None:
            m.update(band_energy=round(est.band_energy, 3), period=round(est.period, 1),
                     ridges_across=round(bw / est.period, 1))
        if est is None or est.band_energy < q.min_band_energy or \
                bw / est.period < gc.min_ridges_across:
            return self._result("BLURRY", box, m, t0)
        return self._result("OK", box, m, t0)
