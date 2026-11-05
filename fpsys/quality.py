"""Đánh giá chất lượng ảnh sau normalize (mục 6.5)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .normalize import Normalized
from .types import ErrorCode


@dataclass
class QualityReport:
    sharpness: float  # = band_energy: năng lượng dải tần vân
    brightness: float
    sat_ratio: float
    laplacian_var: float  # chỉ để so sánh trong báo cáo, không dùng để chặn
    ok: bool
    code: ErrorCode | None


def laplacian_variance(gray: np.ndarray, mask: np.ndarray, width: int = 512) -> float:
    """Phương sai Laplacian trên vùng mask, sau khi resize về rộng `width` px."""
    h, w = gray.shape[:2]
    s = width / float(w)
    size = (width, max(1, round(h * s)))
    g = cv2.resize(gray, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not m.any():
        return 0.0
    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    return float(lap[m].var())


def assess(norm: Normalized, cfg: Config) -> QualityReport:
    """Kiểm tra phơi sáng rồi độ nét (band_energy). Ảnh tiếp xúc bỏ qua kiểm tra phơi sáng."""
    q = cfg.quality
    code: ErrorCode | None = None
    if cfg.pipeline.source != "contact" and (
            not q.min_brightness <= norm.brightness <= q.max_brightness
            or norm.sat_ratio > q.max_sat_ratio):
        code = ErrorCode.BAD_EXPOSURE
    elif norm.band_energy < q.min_band_energy:
        code = ErrorCode.BLURRY
    return QualityReport(sharpness=float(norm.band_energy), brightness=float(norm.brightness),
                         sat_ratio=float(norm.sat_ratio),
                         laplacian_var=laplacian_variance(norm.gray, norm.mask),
                         ok=code is None, code=code)
