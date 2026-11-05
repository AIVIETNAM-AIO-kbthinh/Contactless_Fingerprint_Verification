"""Bộ sinh dữ liệu tổng hợp có đáp án cho test (không cần ảnh thật)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from fpsys.types import Minutia, Template


def sine_ridges(shape: tuple[int, int], period: float, angle: float, phase: float = 0.0
                ) -> np.ndarray:
    """cos(2π·(x cos a + y sin a)/period): vân song song, pháp tuyến theo góc `angle` (rad)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.cos(2 * np.pi * (x * math.cos(angle) + y * math.sin(angle)) / period + phase)


def dislocation_phase(shape: tuple[int, int], period: float, n_points: int, seed: int,
                      margin: int = 40, min_sep: float = 28.0) -> tuple[np.ndarray, np.ndarray]:
    """Trường pha dạng vòm + các điểm lệch pha (mỗi điểm sinh một minutia). Trả (pha, các điểm)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w * 0.5 + rng.uniform(-20, 20), h * 1.8
    phi = 2 * np.pi * np.hypot(x - cx, y - cy) / period
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n_points and tries < 10000:
        tries += 1
        p = (rng.uniform(margin, w - margin), rng.uniform(margin, h - margin))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep for q in pts):
            pts.append(p)
    for i, (px, py) in enumerate(pts):
        phi += (1 if i % 2 == 0 else -1) * np.arctan2(y - py, x - px)
    return phi, np.array(pts)


def synthetic_fingerprint(shape: tuple[int, int] = (400, 320), period: float = 9.0,
                          n_points: int = 24, seed: int = 0, noise: float = 8.0,
                          dark_ridges: bool = True) -> np.ndarray:
    """Ảnh xám uint8 giống vân tay tiếp xúc (vân tối nếu dark_ridges)."""
    phi, _ = dislocation_phase(shape, period, n_points, seed)
    rng = np.random.default_rng(seed + 1000)
    sign = -1.0 if dark_ridges else 1.0
    img = 128 + sign * 90 * np.cos(phi) + rng.normal(0, noise, shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def finger_mask(shape: tuple[int, int], center: tuple[float, float], half_len: float,
                half_width: float, angle_deg: float) -> np.ndarray:
    """Mask hình ngón: thân chữ nhật + chóp bo tròn. angle_deg: hướng gốc→đỉnh (độ, hệ ảnh)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    a = math.radians(angle_deg)
    u = (x - center[0]) * math.cos(a) + (y - center[1]) * math.sin(a)  # dọc trục, + về phía đỉnh
    v = -(x - center[0]) * math.sin(a) + (y - center[1]) * math.cos(a)
    cap0 = half_len - half_width
    body = (u >= -half_len) & (u <= cap0) & (np.abs(v) <= half_width)
    cap = (u > cap0) & ((u - cap0) ** 2 + v ** 2 <= half_width ** 2)
    return body | cap


def synthetic_finger_photo(shape: tuple[int, int] = (480, 360), period: float = 14.0,
                           seed: int = 0, background: tuple[int, int, int] = (25, 25, 25),
                           skin: tuple[int, int, int] = (120, 150, 205),
                           angle_deg: float = -90.0) -> tuple[np.ndarray, np.ndarray]:
    """Ảnh màu BGR một đầu ngón (vân sáng) trên nền; trả (ảnh, mask đáp án)."""
    h, w = shape
    rng = np.random.default_rng(seed)
    mask = finger_mask(shape, (w / 2, h * 0.62), h * 0.55, w * 0.30, angle_deg)
    phi, _ = dislocation_phase(shape, period, 30, seed, margin=50, min_sep=32)
    # bóng sáng tần số thấp (flash, màu da không đều) như ảnh thật
    low = cv2.GaussianBlur(rng.normal(0, 1, shape), (0, 0), 25)
    shading = 1.0 + 0.15 * low / max(np.abs(low).max(), 1e-9)
    tex = (0.78 + 0.22 * np.cos(phi)) * shading
    img = np.empty((h, w, 3), np.float64)
    for c in range(3):
        img[:, :, c] = np.where(mask, skin[c] * tex, background[c])
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask


def random_template(n: int = 60, shape: tuple[int, int] = (400, 300), seed: int = 0,
                    min_sep: float = 12.0, margin: float = 20.0) -> Template:
    rng = np.random.default_rng(seed)
    h, w = shape
    pts: list[tuple[float, float]] = []
    while len(pts) < n:
        p = (rng.uniform(margin, w - margin), rng.uniform(margin, h - margin))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep for q in pts):
            pts.append(p)
    mins = [Minutia(x=float(x), y=float(y), theta=float(rng.uniform(0, 2 * math.pi)),
                    kind="ending" if rng.random() < 0.5 else "bifurcation", quality=1.0)
            for x, y in pts]
    return Template(mins, shape, 9.0, meta={"cfg_hash": "test"})


def transform_template(t: Template, angle_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0,
                       scale: float = 1.0) -> Template:
    """Xoay quanh tâm ảnh, phóng `scale`, rồi dịch (dx, dy)."""
    h, w = t.shape
    cx, cy = w / 2, h / 2
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for m in t.minutiae:
        x, y = m.x - cx, m.y - cy
        out.append(Minutia(x=cx + scale * (c * x - s * y) + dx, y=cy + scale * (s * x + c * y) + dy,
                           theta=(m.theta + a) % (2 * math.pi), kind=m.kind, quality=m.quality))
    return Template(out, t.shape, t.ridge_period, meta=dict(t.meta))


def perturb_template(t: Template, seed: int, add_frac: float = 0.2, drop_frac: float = 0.2,
                     jitter: float = 3.0) -> Template:
    rng = np.random.default_rng(seed)
    h, w = t.shape
    n = len(t.minutiae)
    keep = rng.permutation(n)[: n - int(round(drop_frac * n))]
    out = []
    for i in sorted(keep):
        m = t.minutiae[i]
        out.append(Minutia(x=m.x + rng.uniform(-jitter, jitter),
                           y=m.y + rng.uniform(-jitter, jitter), theta=m.theta, kind=m.kind,
                           quality=m.quality))
    for _ in range(int(round(add_frac * n))):
        out.append(Minutia(x=float(rng.uniform(0, w)), y=float(rng.uniform(0, h)),
                           theta=float(rng.uniform(0, 2 * math.pi)), kind="ending", quality=1.0))
    return Template(out, t.shape, t.ridge_period, meta=dict(t.meta))


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
