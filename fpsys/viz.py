"""Vẽ ảnh trung gian và minutiae (debug, báo cáo)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .types import Minutia

STEP_NAMES = ("01_crop", "02_mask", "03_norm", "04_flat", "05_enhanced", "06_orientation",
              "07_binary", "08_skeleton", "09_minutiae")


def _bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)
                 ) -> np.ndarray:
    out = _bgr(image)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    t = max(1, round(max(out.shape[:2]) / 300))
    tint = out.copy()
    tint[~mask] = (tint[~mask] * 0.4).astype(np.uint8)
    cv2.drawContours(tint, cnts, -1, color, t)
    return tint


def signed_to_u8(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    vals = img[mask] if mask is not None and mask.any() else img.ravel()
    a = float(np.percentile(np.abs(vals), 99)) or 1.0
    out = np.clip(127.5 + 127.5 * img / a, 0, 255).astype(np.uint8)
    if mask is not None:
        out[~mask] = 0
    return out


def orientation_view(base: np.ndarray, theta: np.ndarray, mask: np.ndarray, step: int = 12
                     ) -> np.ndarray:
    out = _bgr(base)
    h, w = theta.shape
    r = step * 0.45
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not mask[y, x]:
                continue
            t = float(theta[y, x])
            dx, dy = r * math.cos(t), r * math.sin(t)
            cv2.line(out, (round(x - dx), round(y - dy)), (round(x + dx), round(y + dy)),
                     (0, 0, 255), 1, cv2.LINE_AA)
    return out


def draw_minutiae(base: np.ndarray, minutiae: list[Minutia], length: float = 12.0
                  ) -> np.ndarray:
    """Ending: đỏ, bifurcation: xanh dương; đoạn thẳng chỉ hướng θ."""
    out = _bgr(base)
    for m in minutiae:
        color = (0, 0, 255) if m.kind == "ending" else (255, 128, 0)
        p = (round(m.x), round(m.y))
        cv2.circle(out, p, 4, color, 1, cv2.LINE_AA)
        q = (round(m.x + length * math.cos(m.theta)), round(m.y + length * math.sin(m.theta)))
        cv2.line(out, p, q, color, 1, cv2.LINE_AA)
    return out


def _tile(img: np.ndarray | None, title: str, size: int) -> np.ndarray:
    tile = np.full((size + 28, size, 3), 30, np.uint8)
    if img is not None:
        im = _bgr(img)
        h, w = im.shape[:2]
        s = size / max(h, w)
        im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))),
                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        y0, x0 = 28 + (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
        tile[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    else:
        cv2.putText(tile, "-", (size // 2 - 8, size // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(tile, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                cv2.LINE_AA)
    return tile


def grid(steps: dict[str, np.ndarray], size: int = 360, footer: str = "") -> np.ndarray:
    """Lưới 3×3 ảnh 9 bước của pipeline."""
    tiles = [_tile(steps.get(n), n, size) for n in STEP_NAMES]
    rows = [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, 9, 3)]
    out = np.concatenate(rows, axis=0)
    if footer:
        bar = np.full((32, out.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, footer, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 1,
                    cv2.LINE_AA)
        out = np.concatenate([out, bar], axis=0)
    return out
