"""Tăng cường vân theo Hong–Wan–Jain (1998) + làm phẳng ánh sáng (mục 6.6)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .config import Config
from .normalize import Normalized
from .types import CaptureError, ErrorCode

log = logging.getLogger(__name__)


@dataclass
class Enhanced:
    image: np.ndarray  # float32, đáp ứng Gabor; vân (sau đảo cực tính) có giá trị < 0
    orientation: np.ndarray  # float32, hướng vân theo pixel, radian trong [0, π)
    frequency: np.ndarray  # float32, tần số vân theo pixel (chu kỳ / px)
    valid_mask: np.ndarray  # bool
    coherence: np.ndarray  # float32, 0..1
    flat: np.ndarray  # uint8, sau làm phẳng ánh sáng (trước CLAHE)
    ridge_period: float = 9.0


def _fill_outside(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    if mask.any():
        out[~mask] = np.uint8(round(float(img[mask].mean())))
    return out


def _stretch(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = img[mask] if mask.any() else img.ravel()
    lo, hi = np.percentile(vals, [1, 99])
    out = (img - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def flatten_illumination(gray: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """gray / blur(gray) với normalized convolution trong mask, kéo về [0, 255]."""
    g = gray.astype(np.float32)
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(g * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    local = num / (den + 1e-6)
    flat = g / (local + 1e-3)
    return _fill_outside(_stretch(flat, mask), mask)


def local_normalize(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Chuẩn hóa mean/variance cục bộ (cửa sổ Gaussian cỡ khối) trong mask → float32 ~N(0, 1)."""
    g = img.astype(np.float32)
    m = mask.astype(np.float32)
    den = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
    mean = cv2.GaussianBlur(g * m, (0, 0), sigma) / den
    sq = cv2.GaussianBlur(g * g * m, (0, 0), sigma) / den
    std = np.sqrt(np.maximum(sq - mean * mean, 1e-2))
    out = (g - mean) / std
    out[~mask] = 0.0
    return out.astype(np.float32)


def orientation_field(img: np.ndarray, mask: np.ndarray | None, sigma: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Hướng vân θ ∈ [0, π) và độ nhất quán (coherence) từ tensor gradient đã làm mượt."""
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.ones_like(g) if mask is None else mask.astype(np.float32)
    gxx = cv2.GaussianBlur(gx * gx * m, (0, 0), sigma)
    gyy = cv2.GaussianBlur(gy * gy * m, (0, 0), sigma)
    gxy = cv2.GaussianBlur(gx * gy * m, (0, 0), sigma)
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2
    theta = np.mod(theta, np.pi).astype(np.float32)
    coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)
    return theta, np.clip(coh, 0, 1).astype(np.float32)


def ridge_frequency(img: np.ndarray, mask: np.ndarray, theta: np.ndarray, block: int,
                    pmin: float, pmax: float, default_period: float) -> np.ndarray:
    """Tần số vân theo khối (x-signature trong cửa sổ xoay theo hướng vân), nội suy ra pixel."""
    h, w = img.shape
    cy = np.arange(block // 2, h, block, dtype=np.float64)
    cx = np.arange(block // 2, w, block, dtype=np.float64)
    CY, CX = np.meshgrid(cy, cx, indexing="ij")
    iy = np.clip(CY.astype(int), 0, h - 1)
    ix = np.clip(CX.astype(int), 0, w - 1)
    th = theta[iy, ix].astype(np.float64).ravel()
    L, W = 2 * block, block
    u = np.arange(L) - L / 2 + 0.5  # dọc pháp tuyến vân
    v = np.arange(W) - W / 2 + 0.5  # dọc theo vân
    nx, ny = -np.sin(th), np.cos(th)
    tx, ty = np.cos(th), np.sin(th)
    X = CX.ravel()[:, None, None] + u[None, :, None] * nx[:, None, None] \
        + v[None, None, :] * tx[:, None, None]
    Y = CY.ravel()[:, None, None] + u[None, :, None] * ny[:, None, None] \
        + v[None, None, :] * ty[:, None, None]
    samp = ndimage.map_coordinates(img, [Y, X], order=1, cval=0.0)
    msamp = ndimage.map_coordinates(mask.astype(np.float32), [Y, X], order=0, cval=0.0)
    sig = ndimage.gaussian_filter1d(samp.mean(axis=2), 1.0, axis=1)
    mfrac = msamp.mean(axis=(1, 2))
    peaks = (sig[:, 1:-1] > sig[:, :-2]) & (sig[:, 1:-1] >= sig[:, 2:])
    lam = np.full(len(sig), np.nan)
    in_mask = mask[iy, ix].ravel()
    for i in np.flatnonzero((mfrac >= 0.8) & in_mask):
        pos = np.flatnonzero(peaks[i])
        if len(pos) >= 2:
            lam[i] = float(np.median(np.diff(pos)))
    lam[(lam < pmin) | (lam > pmax)] = np.nan
    lam = lam.reshape(CY.shape)
    ok = np.isfinite(lam)
    med = float(np.median(lam[ok])) if ok.any() else default_period
    lam = np.where(ok, lam, med)
    lam = ndimage.median_filter(lam, size=3, mode="nearest")
    freq = (1.0 / lam).astype(np.float32)
    return cv2.resize(freq, (w, h), interpolation=cv2.INTER_LINEAR)


def gabor_filter(img: np.ndarray, mask: np.ndarray, theta: np.ndarray, freq: np.ndarray,
                 n_angles: int, gabor_k: float, pmin: float, pmax: float,
                 gamma: float = 1.0) -> np.ndarray:
    """Lọc Gabor định hướng: mỗi pixel lấy đáp ứng của kernel đúng hướng / bước sóng khối.

    `gamma` < 1 kéo dài kernel dọc theo vân (σ dọc vân = σ / gamma): nối liền vân bị đứt do
    nhiễu kết cấu da mà không làm nhòe theo phương vuông góc.
    """
    out = np.zeros(img.shape, np.float32)
    lam_q = np.clip(np.rint(1.0 / np.maximum(freq, 1e-6)), pmin, pmax).astype(np.int32)
    k_map = np.rint(theta / (np.pi / n_angles)).astype(np.int32) % n_angles
    h, w = img.shape
    for lam in np.unique(lam_q[mask]):
        sel_l = mask & (lam_q == lam)
        sigma = gabor_k * float(lam)
        half = int(math.ceil(3 * sigma / min(gamma, 1.0)))
        ksize = 2 * half + 1
        for k in np.unique(k_map[sel_l]):
            sel = sel_l & (k_map == k)
            ys, xs = np.nonzero(sel)
            y0, y1 = max(0, ys.min() - half), min(h, ys.max() + half + 1)
            x0, x1 = max(0, xs.min() - half), min(w, xs.max() + half + 1)
            kern = cv2.getGaborKernel((ksize, ksize), sigma, k * np.pi / n_angles + np.pi / 2,
                                      float(lam), gamma, 0.0, ktype=cv2.CV_32F)
            kern -= kern.mean()
            resp = cv2.filter2D(img[y0:y1, x0:x1], cv2.CV_32F, kern,
                                borderType=cv2.BORDER_CONSTANT)
            sub = sel[y0:y1, x0:x1]
            out[y0:y1, x0:x1][sub] = resp[sub]
    return out


def _preprocess(norm: Normalized, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Làm phẳng ánh sáng → CLAHE → đảo cực tính. Trả (flat, ảnh uint8 vân tối)."""
    ec = cfg.enhance
    mask = norm.mask
    if ec.flatten_sigma > 0:
        flat = flatten_illumination(norm.gray, mask, ec.flatten_sigma
                                    * cfg.normalize.target_period)
    else:
        flat = _fill_outside(norm.gray, mask)
    img = flat
    if ec.clahe.clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=ec.clahe.clip_limit,
                                tileGridSize=(ec.clahe.tile, ec.clahe.tile))
        img = _fill_outside(clahe.apply(img), mask)
    if ec.invert and cfg.pipeline.source != "contact":
        img = 255 - img
    return flat, img


def _finish(image: np.ndarray, theta: np.ndarray, coh: np.ndarray, freq: np.ndarray,
            flat: np.ndarray, norm: Normalized, cfg: Config) -> Enhanced:
    valid = norm.mask & (coh > cfg.enhance.min_coherence)
    frac = float(valid.sum()) / valid.size
    if frac < cfg.enhance.min_valid_area:
        raise CaptureError(ErrorCode.LOW_QUALITY, f"vùng vân hợp lệ {frac:.2f} < "
                                                  f"{cfg.enhance.min_valid_area}")
    return Enhanced(image=image, orientation=theta, frequency=freq, valid_mask=valid,
                    coherence=coh, flat=flat, ridge_period=float(norm.ridge_period))


def enhance(norm: Normalized, cfg: Config) -> Enhanced:
    """Normalized → Enhanced. Vùng vân hợp lệ quá nhỏ → CaptureError(LOW_QUALITY)."""
    if cfg.enhance.method == "reference":
        return enhance_reference(norm, cfg)
    if cfg.enhance.method != "hong":
        raise ValueError(f"enhance.method không hợp lệ: {cfg.enhance.method!r}")
    ec, nc = cfg.enhance, cfg.normalize
    mask = norm.mask
    flat, img = _preprocess(norm, cfg)
    nimg = local_normalize(img, mask, float(ec.block))
    theta, coh = orientation_field(nimg, mask, ec.orient_sigma)
    pmin, pmax = nc.period_min, nc.period_max
    freq = ridge_frequency(nimg, mask, theta, ec.freq_block, pmin, pmax, norm.ridge_period)
    image = gabor_filter(nimg, mask, theta, freq, ec.n_angles, ec.gabor_k, pmin, pmax,
                         ec.gabor_gamma)
    return _finish(image, theta, coh, freq, flat, norm, cfg)


def enhance_reference(norm: Normalized, cfg: Config) -> Enhanced:
    """Mốc so sánh cho ablation: dùng thư viện `fingerprint-enhancer` (không thuộc luồng chính)."""
    try:
        import fingerprint_enhancer  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError("Chưa cài fingerprint-enhancer (pip install fingerprint-enhancer)") \
            from e
    fn = getattr(fingerprint_enhancer, "enhance_fingerprint", None) or \
        fingerprint_enhancer.enhance_Fingerprint
    flat, img = _preprocess(norm, cfg)
    out = np.asarray(fn(img))
    ridge = out > (out.max() / 2 if out.max() > 0 else 0)
    if ridge.shape != img.shape:
        ridge = cv2.resize(ridge.astype(np.uint8), img.shape[::-1],
                           interpolation=cv2.INTER_NEAREST) > 0
    image = np.where(ridge, -1.0, 1.0).astype(np.float32)
    image[~norm.mask] = 0.0
    nimg = local_normalize(img, norm.mask, float(cfg.enhance.block))
    theta, coh = orientation_field(nimg, norm.mask, cfg.enhance.orient_sigma)
    freq = np.full(img.shape, 1.0 / max(norm.ridge_period, 1e-6), np.float32)
    return _finish(image, theta, coh, freq, flat, norm, cfg)
