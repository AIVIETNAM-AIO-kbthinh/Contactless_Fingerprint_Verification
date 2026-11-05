"""Phát hiện đầu ngón tay bằng YOLO11n và crop trên ảnh gốc độ phân giải đầy đủ."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .config import Config, resolve_path
from .types import CaptureError, ErrorCode

log = logging.getLogger(__name__)


@dataclass
class Detection:
    crop: np.ndarray  # BGR, cắt từ ảnh gốc theo crop_xyxy
    box_xyxy: tuple[float, float, float, float]  # box YOLO (chưa pad), hệ tọa độ ảnh gốc
    conf: float
    n_boxes: int
    rotated_deg: int  # 0, 90 hoặc 270: góc xoay ảnh đã dùng khi tìm thấy box (TTA)
    crop_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)  # vùng crop sau pad, hệ tọa độ ảnh gốc

    def box_in_crop(self) -> tuple[float, float, float, float]:
        """Box YOLO (chưa pad) trong hệ tọa độ crop."""
        x0, y0 = self.crop_xyxy[:2]
        x1, y1, x2, y2 = self.box_xyxy
        return (x1 - x0, y1 - y0, x2 - x0, y2 - y0)


@lru_cache(maxsize=4)
def _load_model(weights: str) -> Any:
    from ultralytics import YOLO  # import muộn: ultralytics nặng

    path = resolve_path(weights)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy weights YOLO: {path}")
    log.info("Nạp YOLO từ %s", path)
    return YOLO(str(path))


def _run_model(image: np.ndarray, cfg: Config) -> np.ndarray:
    """Chạy YOLO trên ảnh BGR, trả mảng (N, 5): x1, y1, x2, y2, conf theo tọa độ ảnh đầu vào."""
    model = _load_model(cfg.detect.weights)
    results = model.predict(image, imgsz=cfg.detect.imgsz, conf=cfg.detect.conf, verbose=False,
                            device="cpu")
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.zeros((0, 5), np.float32)
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)[:, None]
    return np.concatenate([xyxy, conf], axis=1)


def select_box(boxes: np.ndarray, tie_conf_delta: float) -> int:
    """Trong các box có conf ≥ conf_max − delta, chọn box diện tích lớn nhất."""
    conf = boxes[:, 4]
    cand = np.flatnonzero(conf >= conf.max() - tie_conf_delta)
    area = (boxes[cand, 2] - boxes[cand, 0]) * (boxes[cand, 3] - boxes[cand, 1])
    return int(cand[np.argmax(area)])


def unrotate_boxes(boxes: np.ndarray, rotated_deg: int, orig_hw: tuple[int, int]) -> np.ndarray:
    """Đổi box tìm được trên ảnh đã xoay (cv2.rotate) về tọa độ ảnh gốc.

    rotated_deg = 90: ảnh xoay 90° theo chiều kim đồng hồ, điểm gốc (x, y) → (H − y, x).
    rotated_deg = 270: xoay ngược chiều kim đồng hồ, (x, y) → (y, W − x).
    """
    h, w = orig_hw
    out = boxes.copy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if rotated_deg == 0:
        return out
    if rotated_deg == 90:
        out[:, 0], out[:, 2] = y1, y2
        out[:, 1], out[:, 3] = h - x2, h - x1
    elif rotated_deg == 270:
        out[:, 0], out[:, 2] = w - y2, w - y1
        out[:, 1], out[:, 3] = x1, x2
    else:
        raise ValueError(rotated_deg)
    return out


_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _otsu_boxes(image: np.ndarray) -> np.ndarray:
    """Ablation otsu_crop: phân đoạn da Otsu trên kênh Cr toàn ảnh, box của thành phần lớn nhất."""
    h, w = image.shape[:2]
    s = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA)
    cr = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)[:, :, 1], (5, 5), 0)
    _, m = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros((0, 5), np.float32)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.01 * m.size:
        return np.zeros((0, 5), np.float32)
    x, y, bw, bh = stats[i, :4]
    return np.array([[x / s, y / s, (x + bw) / s, (y + bh) / s, 1.0]], np.float32)


def detect(image: np.ndarray, cfg: Config) -> Detection:
    """Tìm đầu ngón, chọn box, nới `pad`, crop trên ảnh gốc. Không thấy → NO_FINGER."""
    h, w = image.shape[:2]
    rotated = 0
    if cfg.detect.method == "otsu":
        boxes = _otsu_boxes(image)
    elif cfg.detect.method == "yolo":
        boxes = _run_model(image, cfg)
        if len(boxes) == 0 and cfg.detect.tta_rotate:
            for deg in (90, 270):
                rb = _run_model(cv2.rotate(image, _ROTATIONS[deg]), cfg)
                if len(rb):
                    boxes, rotated = unrotate_boxes(rb, deg, (h, w)), deg
                    log.debug("TTA: tìm thấy đầu ngón trên ảnh xoay %d°", deg)
                    break
    else:
        raise ValueError(f"detect.method không hợp lệ: {cfg.detect.method!r}")
    if len(boxes) == 0:
        raise CaptureError(ErrorCode.NO_FINGER, "không phát hiện đầu ngón tay")

    i = select_box(boxes, cfg.detect.tie_conf_delta)
    x1, y1, x2, y2, conf = (float(v) for v in boxes[i])
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    bw, bh = x2 - x1, y2 - y1
    pad = cfg.detect.pad
    cx1 = int(max(0, np.floor(x1 - pad * bw)))
    cy1 = int(max(0, np.floor(y1 - pad * bh)))
    cx2 = int(min(w, np.ceil(x2 + pad * bw)))
    cy2 = int(min(h, np.ceil(y2 + pad * bh)))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        raise CaptureError(ErrorCode.NO_FINGER, "box đầu ngón quá nhỏ")
    crop = image[cy1:cy2, cx1:cx2].copy()
    return Detection(crop=crop, box_xyxy=(x1, y1, x2, y2), conf=conf, n_boxes=len(boxes),
                     rotated_deg=rotated, crop_xyxy=(cx1, cy1, cx2, cy2))
