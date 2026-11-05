"""Chuẩn hóa crop đầu ngón: mask da, phơi sáng, xoay, duỗi phẳng, chu kỳ vân, tỷ lệ (mục 6.4)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .config import Config
from .detect import Detection
from .io import to_gray
from .types import CaptureError, ErrorCode

log = logging.getLogger(__name__)


@dataclass
class Normalized:
    gray: np.ndarray  # uint8, đã xoay (đỉnh ngón hướng lên), duỗi phẳng, chuẩn hóa tỷ lệ
    mask: np.ndarray  # bool, vùng da có thể chứa vân (đã loại pixel bão hòa)
    angle_deg: float  # góc đã xoay (độ, ngược chiều kim đồng hồ theo quy ước cv2)
    scale: float
    ridge_period: float  # chu kỳ vân sau resize (px)
    period_raw: float  # chu kỳ vân trước resize (px)
    band_energy: float
    sat_ratio: float
    brightness: float
    crop_mask: np.ndarray | None = None  # mask da trên crop (để debug)


@dataclass
class PeriodEstimate:
    period: float
    band_energy: float
    n_blocks: int


# ---------------------------------------------------------------------------
# Mask da


def _largest_component(m: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(m, dtype=bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == i


def _ellipse(radius: int) -> np.ndarray:
    k = 2 * max(1, int(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_mask(m: np.ndarray, erode_px: int, morph_px: int = 2) -> np.ndarray:
    """Thành phần lớn nhất → lấp lỗ → mở/đóng → co `erode_px` (biên ảnh không bị co)."""
    m = _largest_component(m)
    if not m.any():
        return m
    m = ndimage.binary_fill_holes(m)
    u8 = m.astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, _ellipse(morph_px))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, _ellipse(morph_px))
    m = _largest_component(u8 > 0)
    m = ndimage.binary_fill_holes(m)
    if erode_px > 0:
        m = cv2.erode(m.astype(np.uint8), _ellipse(erode_px)) > 0
    return m


def skin_mask(crop: np.ndarray, cfg: Config,
              rect: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Mask da (bool, kích thước crop). `rect` = box YOLO chưa pad trong hệ tọa độ crop."""
    nc = cfg.normalize
    h, w = crop.shape[:2]
    s = min(1.0, nc.mask_work_width / float(w))
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    method = nc.skin_method
    if method == "grabcut":
        if rect is None:
            x1, y1, x2, y2 = 0.05 * sw, 0.05 * sh, 0.95 * sw, 0.95 * sh
        else:
            x1, y1, x2, y2 = (rect[0] * sw / w, rect[1] * sh / h, rect[2] * sw / w,
                              rect[3] * sh / h)
        # grabCut cần một ít nền ngoài hình chữ nhật
        x1, y1 = int(max(1, round(x1))), int(max(1, round(y1)))
        x2, y2 = int(min(sw - 2, round(x2))), int(min(sh - 2, round(y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            x1, y1, x2, y2 = 1, 1, sw - 2, sh - 2
        gc = np.zeros((sh, sw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(cfg.seed)
        try:
            cv2.grabCut(small, gc, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, nc.grabcut_iters,
                        cv2.GC_INIT_WITH_RECT)
            m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        except cv2.error as e:  # ảnh đồng nhất làm GMM suy biến
            log.warning("grabCut lỗi (%s), dùng toàn bộ hình chữ nhật", e)
            m = np.zeros((sh, sw), bool)
            m[y1:y2, x1:x2] = True
    elif method == "otsu_cr":
        cr = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)[:, :, 1], (5, 5), 0)
        _, t = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = t > 0
    elif method == "fixed_ycrcb":
        ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        m = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127)) > 0
    else:
        raise ValueError(f"normalize.skin_method không hợp lệ: {method!r}")
    m = clean_mask(m, nc.mask_erode_px)
    if (sw, sh) != (w, h):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return m


def contact_mask(gray: np.ndarray, block: int, std_frac: float) -> np.ndarray:
    """Ảnh cảm biến tiếp xúc: mask từ ngưỡng độ lệch chuẩn cục bộ theo khối."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (block, block))
    sq = cv2.blur(g * g, (block, block))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = std_frac * float(np.percentile(std, 95))
    m = std > thr
    return clean_mask(m, erode_px=max(1, block // 4), morph_px=max(2, block // 4))


# ---------------------------------------------------------------------------
# Xoay


def _rotate(img: np.ndarray, angle_deg: float, interp: int) -> tuple[np.ndarray, np.ndarray]:
    """Xoay ảnh quanh tâm, nới khung để không mất pixel. Trả (ảnh, ma trận affine 2×3)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    out = cv2.warpAffine(img, M, (nw, nh), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return out, M


def _principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trục chính (vector đơn vị, hệ x-phải y-xuống) và tâm của mask bằng PCA."""
    ys, xs = np.nonzero(mask)
    step = max(1, len(xs) // 50000)
    pts = np.stack([xs[::step], ys[::step]]).astype(np.float64)
    center = pts.mean(axis=1)
    cov = np.cov(pts)
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    return v, center


def _end_touches(mask: np.ndarray, v: np.ndarray, center: np.ndarray) -> tuple[bool, bool]:
    """Đầu dương / âm (theo trục v) của mask có chạm biên crop không."""
    h, w = mask.shape
    band = max(2, round(0.01 * min(h, w)))
    border = np.zeros_like(mask)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True
    ys, xs = np.nonzero(mask)
    t = (xs - center[0]) * v[0] + (ys - center[1]) * v[1]
    by, bx = np.nonzero(mask & border)
    if len(bx) == 0:
        return False, False
    bt = (bx - center[0]) * v[0] + (by - center[1]) * v[1]
    min_count = max(3, int(0.02 * math.sqrt(mask.sum())))
    pos = int(np.sum(bt > 0.5 * t.max())) >= min_count
    neg = int(np.sum(bt < 0.5 * t.min())) >= min_count
    return pos, neg


def _bottom_is_narrower(mask: np.ndarray, frac: float) -> bool:
    """Quy tắc bề rộng: so bề rộng trung bình đoạn `frac` cuối ở hai đầu theo phương dọc."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 4:
        return False
    y0, y1 = rows[0], rows[-1]
    n = max(1, int(round(frac * (y1 - y0 + 1))))
    widths = mask.sum(axis=1).astype(np.float64)
    top = widths[y0:y0 + n].mean()
    bottom = widths[y1 - n + 1:y1 + 1].mean()
    return bottom < top


def orientation_angle(mask: np.ndarray, frac: float) -> float:
    """Góc xoay (độ, quy ước cv2.getRotationMatrix2D) đưa trục ngón thẳng đứng, đỉnh lên trên."""
    v, center = _principal_axis(mask)
    a = math.degrees(math.atan2(v[1], v[0]))
    # Sau khi xoay angle = a − 90, vector v chỉ xuống dưới: đầu dương nằm ở đáy ảnh.
    angle = a - 90.0
    pos_touch, neg_touch = _end_touches(mask, v, center)
    if pos_touch != neg_touch:
        tip_positive = neg_touch  # gốc ngón chạm biên, đỉnh là đầu còn lại
    else:
        rot, _ = _rotate(mask.astype(np.uint8), angle, cv2.INTER_NEAREST)
        tip_positive = _bottom_is_narrower(rot > 0, frac)
    if tip_positive:
        angle += 180.0
    return (angle + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Duỗi phẳng


def unwarp(gray: np.ndarray, geo_mask: np.ndarray, tex_mask: np.ndarray, max_angle_deg: float,
           smooth_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duỗi phẳng mô hình hình trụ trục đứng: x' = r·arcsin((x − c)/r) (cv2.remap ánh xạ ngược)."""
    h, w = geo_mask.shape
    has = geo_mask.any(axis=1)
    left = np.argmax(geo_mask, axis=1).astype(np.float64)
    right = (w - 1 - np.argmax(geo_mask[:, ::-1], axis=1)).astype(np.float64)
    c = (left + right) / 2.0
    r = (right - left + 1) / 2.0
    valid = has & (r >= 4)
    yv = np.flatnonzero(valid)
    if len(yv) < 3:
        return gray, geo_mask, tex_mask
    k = max(1, int(smooth_rows) | 1)
    c_s = np.interp(np.arange(h), yv, ndimage.median_filter(c[yv], size=k, mode="nearest"))
    r_s = np.interp(np.arange(h), yv, ndimage.median_filter(r[yv], size=k, mode="nearest"))
    th = math.radians(max_angle_deg)
    half = int(math.ceil(r_s[valid].max() * th)) + 2
    xo = np.arange(-half, half + 1, dtype=np.float64)[None, :]
    rr = r_s[:, None]
    inside = (np.abs(xo) <= rr * th) & valid[:, None]
    map_x = (c_s[:, None] + rr * np.sin(xo / rr)).astype(np.float32)
    map_x[~inside] = -1.0
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], xo.shape[1], axis=1)
    g = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                  borderValue=0)
    gm = cv2.remap(geo_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    tm = cv2.remap(tex_mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return g, gm & inside, tm & inside


# ---------------------------------------------------------------------------
# Chu kỳ vân


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n)
    return np.outer(w, w)


def _parabolic(lm: np.ndarray, c: np.ndarray, rp: np.ndarray) -> np.ndarray:
    den = lm - 2.0 * c + rp
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.abs(den) > 1e-12, 0.5 * (lm - rp) / den, 0.0)
    return np.clip(d, -0.5, 0.5)


def estimate_period(gray: np.ndarray, mask: np.ndarray, block: int, pmin: float, pmax: float,
                    max_blocks: int = 300, min_cycles: float = 3.0) -> PeriodEstimate | None:
    """Chu kỳ vân (px của ảnh đầu vào) và band_energy, ước lượng đa tỷ lệ.

    Khối FFT cần chứa ≥ `min_cycles` chu kỳ thì đỉnh phổ mới rõ; vùng tần số thấp hơn bị năng
    lượng bóng sáng lấn át. Vì vậy ở mỗi tỷ lệ s = 1, ½, ¼… chỉ tìm chu kỳ trong
    [pmin, block / min_cycles] (px của tỷ lệ đó), rồi chọn tỷ lệ có band_energy cao nhất.
    Chu kỳ trả về nằm trong [pmin, pmax] px của ảnh đầu vào.
    """
    best: tuple[float, PeriodEstimate] | None = None
    fallback: tuple[float, PeriodEstimate] | None = None
    s = 1.0
    g, m = gray, mask
    while True:
        last = pmin / (s / 2) > pmax or min(g.shape) // 2 < 2 * block
        hi = min(block / min_cycles, pmax * s)
        if hi > pmin:
            e = _estimate_period_single(g, m, block, pmin, hi, max_blocks)
            if e is not None:
                # đỉnh sát biên khoảng thường là đỉnh thật nằm ngoài khoảng tràn vào; các khoảng
                # chồng nhau nên tỷ lệ kế bên sẽ thấy nó ở giữa khoảng
                edge = (e.period >= 0.92 * hi and not last) or (s < 1 and e.period <= 1.1 * pmin)
                cand = (s, e)
                if not edge and (best is None or e.band_energy > best[1].band_energy):
                    best = cand
                if fallback is None or e.band_energy > fallback[1].band_energy:
                    fallback = cand
        if last:
            break
        g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
        s *= 0.5
    best = best or fallback
    if best is None:
        return None
    s, e = best
    return PeriodEstimate(e.period / s, e.band_energy, e.n_blocks)


def _estimate_period_single(gray: np.ndarray, mask: np.ndarray, block: int, pmin: float,
                            pmax: float, max_blocks: int = 300) -> PeriodEstimate | None:
    """Trung vị trên các khối nằm hoàn toàn trong mask (FFT 2D), ở một tỷ lệ."""
    h, w = gray.shape
    if h < block or w < block:
        return None
    step = max(1, block // 2)
    ii = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, h - block + 1, step)
    xs = np.arange(0, w - block + 1, step)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    cnt = ii[Y + block, X + block] - ii[Y, X + block] - ii[Y + block, X] + ii[Y, X]
    inside = cnt == block * block
    if not inside.any():
        return None
    by, bx = Y[inside], X[inside]
    # ưu tiên khối gần tâm mask
    my, mx = np.nonzero(mask)
    cy, cx = my.mean(), mx.mean()
    d = (by + block / 2 - cy) ** 2 + (bx + block / 2 - cx) ** 2
    order = np.argsort(d, kind="stable")[:max_blocks]
    by, bx = by[order], bx[order]

    g = gray.astype(np.float32)
    idx_y = by[:, None] + np.arange(block)[None, :]
    idx_x = bx[:, None] + np.arange(block)[None, :]
    blocks = g[idx_y[:, :, None], idx_x[:, None, :]]  # (N, b, b)
    std = blocks.std(axis=(1, 2))
    blocks = blocks - blocks.mean(axis=(1, 2), keepdims=True)
    blocks *= _hann2d(block)[None].astype(np.float32)
    P = np.abs(np.fft.fftshift(np.fft.fft2(blocks), axes=(1, 2))) ** 2
    u = np.arange(block) - block // 2
    R = np.hypot(u[None, :], u[:, None])
    band = (R >= block / pmax) & (R <= block / pmin)
    Pb = np.where(band[None], P, 0.0)
    flat = Pb.reshape(len(P), -1).argmax(axis=1)
    iy, ix = np.unravel_index(flat, (block, block))
    iy = np.clip(iy, 1, block - 2)
    ix = np.clip(ix, 1, block - 2)
    n = np.arange(len(P))
    lp = np.log(P + 1e-12)
    dy = _parabolic(lp[n, iy - 1, ix], lp[n, iy, ix], lp[n, iy + 1, ix])
    dx = _parabolic(lp[n, iy, ix - 1], lp[n, iy, ix], lp[n, iy, ix + 1])
    fy = (iy - block // 2) + dy
    fx = (ix - block // 2) + dx
    r_peak = np.hypot(fx, fy)
    with np.errstate(divide="ignore"):
        period = np.where(r_peak > 0, block / r_peak, np.inf)
    ring = np.abs(R[None] - r_peak[:, None, None]) <= 1.0
    total = (P * (R[None] > 0)).sum(axis=(1, 2))
    be = np.where(total > 0, (P * ring).sum(axis=(1, 2)) / np.maximum(total, 1e-12), 0.0)
    ok = (std > 1.0) & (period >= pmin) & (period <= pmax)
    if not ok.any():
        return None
    return PeriodEstimate(float(np.median(period[ok])), float(np.median(be[ok])), int(ok.sum()))


# ---------------------------------------------------------------------------


def _crop_to(mask: np.ndarray, *arrays: np.ndarray, margin: int = 4) -> list[np.ndarray]:
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin + 1)
    return [a[y0:y1, x0:x1] for a in (mask, *arrays)]


def _period_failure(cfg: Config, brightness: float, sat_ratio: float, why: str) -> CaptureError:
    q = cfg.quality
    if cfg.pipeline.source != "contact" and (
            not q.min_brightness <= brightness <= q.max_brightness
            or sat_ratio > q.max_sat_ratio):
        return CaptureError(ErrorCode.BAD_EXPOSURE,
                            f"brightness={brightness:.1f}, sat_ratio={sat_ratio:.3f}")
    return CaptureError(ErrorCode.BLURRY, why)


def _resize(gray: np.ndarray, masks: list[np.ndarray], scale: float
            ) -> tuple[np.ndarray, list[np.ndarray]]:
    if abs(scale - 1.0) < 1e-6:
        return gray, masks
    h, w = gray.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    g = cv2.resize(gray, size, interpolation=interp)
    ms = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
          for m in masks]
    return g, ms


def normalize(crop: np.ndarray, cfg: Config, det: Detection | None = None) -> Normalized:
    """Crop BGR (hoặc xám với source=contact) → Normalized."""
    nc = cfg.normalize
    source = cfg.pipeline.source

    if source == "contact":
        gray = to_gray(crop, "gray")
        mask = contact_mask(gray, nc.contact_block, nc.contact_std_frac)
        if mask.sum() < 0.05 * mask.size:
            raise CaptureError(ErrorCode.LOW_QUALITY, "vùng vân quá nhỏ")
        brightness = float(gray[mask].mean())
        sat_ratio = float((gray[mask] >= nc.sat_level).mean())
        est = estimate_period(gray, mask, nc.period_block, nc.period_min, nc.period_max,
                              min_cycles=nc.period_min_cycles)
        if est is None:
            raise _period_failure(cfg, brightness, sat_ratio, "không ước lượng được chu kỳ vân")
        return Normalized(gray=gray, mask=mask, angle_deg=0.0, scale=1.0,
                          ridge_period=est.period, period_raw=est.period,
                          band_energy=est.band_energy, sat_ratio=sat_ratio,
                          brightness=brightness, crop_mask=mask)

    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    rect = det.box_in_crop() if det is not None else None
    geo = skin_mask(crop, cfg, rect)
    if geo.sum() < 0.02 * geo.size:
        raise CaptureError(ErrorCode.LOW_QUALITY, "mask da rỗng")

    # 2. Phơi sáng: loại đốm bão hòa khỏi mask
    lum = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(lum[geo].mean())
    sat = crop.max(axis=2) >= nc.sat_level
    sat_ratio = float(sat[geo].mean())
    if nc.sat_dilate_px > 0 and sat.any():
        sat = cv2.dilate(sat.astype(np.uint8), _ellipse(nc.sat_dilate_px)) > 0
    tex = geo & ~sat

    gray = to_gray(crop, cfg.enhance.channel)

    # 3. Xoay: trục ngón thẳng đứng, đỉnh lên trên
    angle = orientation_angle(geo, nc.tip_end_frac)
    g_r, _ = _rotate(gray, angle, cv2.INTER_LINEAR)
    geo_r = _rotate(geo.astype(np.uint8), angle, cv2.INTER_NEAREST)[0] > 0
    tex_r = _rotate(tex.astype(np.uint8), angle, cv2.INTER_NEAREST)[0] > 0
    geo_r, g_r, tex_r = _crop_to(geo_r, g_r, tex_r)

    # 4. Duỗi phẳng
    if nc.unwarp:
        g_r, geo_r, tex_r = unwarp(g_r, geo_r, tex_r, nc.unwarp_max_angle_deg,
                                   nc.unwarp_smooth_rows)
        if geo_r.any():
            geo_r, g_r, tex_r = _crop_to(geo_r, g_r, tex_r)

    # 5. Chu kỳ vân
    est = estimate_period(g_r, tex_r, nc.period_block, nc.period_min, nc.period_max,
                          min_cycles=nc.period_min_cycles)
    if est is None:
        raise _period_failure(cfg, brightness, sat_ratio, "không ước lượng được chu kỳ vân")

    # 6. Chuẩn hóa tỷ lệ
    scale = 1.0
    if nc.scale_norm:
        scale = float(np.clip(nc.target_period / est.period, nc.scale_min, nc.scale_max))
    g_s, (tex_s,) = _resize(g_r, [tex_r], scale)
    return Normalized(gray=g_s, mask=tex_s, angle_deg=float(angle), scale=scale,
                      ridge_period=est.period * scale, period_raw=est.period,
                      band_energy=est.band_energy, sat_ratio=sat_ratio, brightness=brightness,
                      crop_mask=tex)
