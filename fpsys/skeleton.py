"""Nhị phân hóa, làm mảnh và cắt nhánh gai (mục 6.7)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize as _sk_skeletonize
from skimage.morphology import thin as _sk_thin

from .config import Config, SkeletonCfg
from .enhance import Enhanced

# 8 lân cận theo thứ tự vòng tròn (dy, dx): E, NE, N, NW, W, SW, S, SE
RING = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))


@dataclass
class SkeletonResult:
    binary: np.ndarray  # bool, vân = True
    skeleton: np.ndarray  # bool, rộng 1 px


def remove_small_components(m: np.ndarray, min_px: int) -> np.ndarray:
    """Bỏ thành phần liên thông (8-lân cận) có ít hơn `min_px` pixel."""
    if min_px <= 1 or not m.any():
        return m.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


def fill_small_holes(m: np.ndarray, max_px: int) -> np.ndarray:
    """Lấp các lỗ (thành phần nền 4-liên thông không chạm biên) có tối đa `max_px` pixel."""
    if max_px <= 0:
        return m.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    h, w = m.shape
    x, y, bw, bh = (stats[:, i] for i in range(4))
    touches = (x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)
    fill = np.zeros(n, bool)
    fill[1:] = (stats[1:, cv2.CC_STAT_AREA] <= max_px) & ~touches[1:]
    return m | fill[labels]


def neighbor_count(sk: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT
                        ).round().astype(np.int32)


def crossing_number(sk: np.ndarray) -> np.ndarray:
    """CN = ½·Σ|Pᵢ − Pᵢ₊₁| trên vòng 8 lân cận (0 tại pixel không thuộc skeleton)."""
    p = np.pad(sk.astype(np.int8), 1)
    h, w = sk.shape
    ring = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING]
    cn = np.zeros((h, w), np.int32)
    for i in range(8):
        cn += np.abs(ring[i] - ring[(i + 1) % 8])
    return np.where(sk, cn // 2, 0)


def _cn_at(sk: np.ndarray, y: int, x: int) -> int:
    h, w = sk.shape
    vals = [bool(sk[y + dy, x + dx]) if 0 <= y + dy < h and 0 <= x + dx < w else False
            for dy, dx in RING]
    return sum(vals[i] != vals[(i + 1) % 8] for i in range(8)) // 2


def _unvisited_neighbors(sk: np.ndarray, y: int, x: int, visited: set[tuple[int, int]]
                         ) -> list[tuple[int, int]]:
    h, w = sk.shape
    out = []
    for dy, dx in RING:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] and (ny, nx) not in visited:
            out.append((ny, nx))
    return out


def prune_spurs(sk: np.ndarray, max_len: int, n_iter: int = 2) -> np.ndarray:
    """Từ mỗi điểm kết thúc dò tối đa `max_len` px; gặp điểm rẽ nhánh trước thì xóa đoạn đã dò."""
    sk = sk.copy()
    for _ in range(n_iter):
        ends = np.argwhere(sk & (neighbor_count(sk) == 1))
        for y0, x0 in ends:
            if not sk[y0, x0]:
                continue
            path = [(int(y0), int(x0))]
            visited = set(path)
            while len(path) <= max_len:
                y, x = path[-1]
                nbrs = _unvisited_neighbors(sk, y, x, visited)
                if any(_cn_at(sk, ny, nx) >= 3 for ny, nx in nbrs):
                    for py, px in path:
                        sk[py, px] = False
                    break
                if len(nbrs) != 1:
                    break
                path.append(nbrs[0])
                visited.add(nbrs[0])
        sk = _sk_thin(sk)
    return sk


def skeleton_from_binary(binary: np.ndarray, sc: SkeletonCfg) -> np.ndarray:
    """Ảnh nhị phân vân → skeleton 1 px đã làm sạch."""
    b = remove_small_components(binary, sc.min_component_px)
    b = fill_small_holes(b, sc.max_hole_px)
    sk = _sk_thin(_sk_skeletonize(b))
    sk = prune_spurs(sk, sc.spur_len_px, n_iter=2)
    return remove_small_components(sk, sc.min_segment_px)


def skeletonize(enh: Enhanced, cfg: Config) -> SkeletonResult:
    """Enhanced → nhị phân (đáp ứng Gabor < 0 trong valid_mask) → skeleton."""
    binary = (enh.image < 0) & enh.valid_mask
    return SkeletonResult(binary=binary, skeleton=skeleton_from_binary(binary, cfg.skeleton))
