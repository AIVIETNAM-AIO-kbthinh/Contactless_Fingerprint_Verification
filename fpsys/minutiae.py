"""Trích minutiae bằng Crossing Number, tính góc theo quy ước mục 5, lọc minutiae giả (mục 6.8)."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .config import Config, MinutiaeCfg
from .enhance import Enhanced
from .skeleton import RING, SkeletonResult, crossing_number
from .types import CaptureError, ErrorCode, Minutia

log = logging.getLogger(__name__)

TWO_PI = 2.0 * math.pi


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Chênh lệch góc vòng tròn, giá trị tuyệt đối trong [0, π]."""
    d = np.mod(np.asarray(a) - np.asarray(b) + math.pi, TWO_PI) - math.pi
    return np.abs(d)


def _neighbors(sk: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = sk.shape
    return [(y + dy, x + dx) for dy, dx in RING
            if 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]]


def trace(sk: np.ndarray, start: tuple[int, int], visited: set[tuple[int, int]],
          max_steps: int) -> tuple[tuple[int, int], int]:
    """Dò theo skeleton từ `start` tối đa `max_steps` bước; dừng ở điểm cụt hoặc chỗ rẽ nhánh."""
    cur = start
    visited = set(visited) | {start}
    steps = 0
    while steps < max_steps:
        nbrs = [p for p in _neighbors(sk, *cur) if p not in visited]
        if not nbrs:
            break
        if len(nbrs) > 1:
            adjacent = all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
                           for a in nbrs for b in nbrs)
            if not adjacent:
                break
            # bậc thang: ưu tiên lân cận 4
            nbrs.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
            visited.update(nbrs[1:])
        cur = nbrs[0]
        visited.add(cur)
        steps += 1
    return cur, steps


def _branch_starts(sk: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    """Mỗi nhóm lân cận liên tiếp trên vòng 8 điểm là một nhánh; chọn một pixel đại diện."""
    h, w = sk.shape
    vals = [0 <= y + dy < h and 0 <= x + dx < w and bool(sk[y + dy, x + dx]) for dy, dx in RING]
    if all(vals):
        return []
    start = next(i for i in range(8) if not vals[i])
    groups: list[list[int]] = []
    for k in range(1, 9):
        i = (start + k) % 8
        if vals[i]:
            if vals[(i - 1) % 8] and groups:
                groups[-1].append(i)
            else:
                groups.append([i])
    out = []
    for g in groups:
        g4 = [i for i in g if i % 2 == 0]  # chỉ số chẵn là lân cận 4 (E, N, W, S)
        i = g4[0] if g4 else g[0]
        out.append((y + RING[i][0], x + RING[i][1]))
    return out


def _ending_angle(sk: np.ndarray, y: int, x: int, trace_px: int) -> float | None:
    (ty, tx), steps = trace(sk, (y, x), set(), trace_px)
    if steps < 2:
        return None
    return math.atan2(y - ty, x - tx) % TWO_PI


def _bifurcation_angle(sk: np.ndarray, y: int, x: int, trace_px: int, ridge_theta: float
                       ) -> float | None:
    starts = _branch_starts(sk, y, x)
    if len(starts) != 3:
        return None
    blocked = {(y, x), *starts}
    dirs = []
    for s in starts:
        (ty, tx), _ = trace(sk, s, blocked, trace_px)
        v = np.array([tx - x, ty - y], np.float64)
        n = np.linalg.norm(v)
        if n < 1e-9:
            return None
        dirs.append(v / n)
    d = np.array(dirs)
    cos = d @ d.T
    single = int(np.argmin(cos.sum(axis=1) - 1.0))  # nhánh khác hướng nhất với hai nhánh còn lại
    ds = d[single]
    t = float(ridge_theta)
    if ds[0] * math.cos(t) + ds[1] * math.sin(t) < 0:
        t += math.pi
    return t % TWO_PI


def _dedup(points: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """Giữ một điểm trong mỗi cụm điểm cùng loại sát nhau (rẽ nhánh biểu diễn bởi 2 pixel)."""
    keep: list[int] = []
    for i, p in enumerate(points):
        if all(np.hypot(*(p - points[j])) > radius for j in keep):
            keep.append(i)
    return np.array(keep, dtype=np.int64)


def filter_false(xy: np.ndarray, theta: np.ndarray, is_end: np.ndarray, mc: MinutiaeCfg
                 ) -> np.ndarray:
    """Cờ giữ lại (bool) sau các luật lọc minutiae giả. Các luật xét trên tập ban đầu."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, bool)
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(D, np.inf)
    E, B = is_end, ~is_end
    opposite = angle_diff(theta[:, None], theta[None, :] + math.pi) <= math.radians(
        mc.break_angle_deg)
    bad = (D < mc.break_px) & E[:, None] & E[None, :] & opposite
    bad |= (D < mc.bridge_px) & B[:, None] & B[None, :]
    bad |= (D < mc.spur_px) & ((E[:, None] & B[None, :]) | (B[:, None] & E[None, :]))
    remove = bad.any(axis=1)
    near = D < mc.cluster_px
    dense = (near.sum(axis=1) + 1) > mc.cluster_max
    if dense.any():
        remove |= dense | near[dense].any(axis=0)
    return ~remove


def find_minutiae(skel: np.ndarray, orientation: np.ndarray, coherence: np.ndarray,
                  valid_mask: np.ndarray, ridge_period: float, mc: MinutiaeCfg
                  ) -> list[Minutia]:
    """Skeleton → danh sách Minutia (chưa kiểm tra số lượng tối thiểu)."""
    cn = crossing_number(skel)
    inner = valid_mask.astype(np.uint8)
    if mc.border_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mc.border_px + 1,) * 2)
        inner = cv2.erode(inner, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    inner = inner > 0
    cand: list[tuple[int, int, bool]] = []
    for is_end, val in ((True, 1), (False, 3)):
        pts = np.argwhere((cn == val) & inner)
        if len(pts):
            pts = pts[_dedup(pts.astype(np.float64))]
        cand += [(int(y), int(x), is_end) for y, x in pts]

    ys, xs, ths, ends = [], [], [], []
    for y, x, is_end in cand:
        if is_end:
            th = _ending_angle(skel, y, x, mc.trace_px)
        else:
            th = _bifurcation_angle(skel, y, x, mc.trace_px, float(orientation[y, x]))
        if th is None:
            continue
        ys.append(y)
        xs.append(x)
        ths.append(th)
        ends.append(is_end)
    if not ys:
        return []
    xy = np.stack([np.array(xs, np.float64), np.array(ys, np.float64)], axis=1)
    theta = np.array(ths)
    is_end = np.array(ends)
    keep = filter_false(xy, theta, is_end, mc)

    r = max(2, int(round(ridge_period)))
    qmap = cv2.blur(coherence.astype(np.float32), (2 * r + 1, 2 * r + 1))
    shift = mc.shift_frac * ridge_period
    out: list[Minutia] = []
    for i in np.flatnonzero(keep):
        t = float(theta[i])
        sgn = 1.0 if is_end[i] else -1.0  # ending tiến về đầu vân, bifurcation lùi về 2 nhánh
        x = float(xy[i, 0] + sgn * shift * math.cos(t))
        y = float(xy[i, 1] + sgn * shift * math.sin(t))
        q = float(np.clip(qmap[int(xy[i, 1]), int(xy[i, 0])], 0.0, 1.0))
        out.append(Minutia(x=x, y=y, theta=t % TWO_PI,
                           kind="ending" if is_end[i] else "bifurcation", quality=q))
    return out


def extract_minutiae(sk: SkeletonResult, enh: Enhanced, cfg: Config) -> list[Minutia]:
    """Skeleton + Enhanced → minutiae. Ít hơn min_minutiae → CaptureError(TOO_FEW_MINUTIAE)."""
    mins = find_minutiae(sk.skeleton, enh.orientation, enh.coherence, enh.valid_mask,
                         enh.ridge_period, cfg.minutiae)
    if len(mins) < cfg.minutiae.min_minutiae:
        raise CaptureError(ErrorCode.TOO_FEW_MINUTIAE,
                           f"{len(mins)} < {cfg.minutiae.min_minutiae}")
    return mins
