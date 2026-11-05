import numpy as np
import pytest

from fpsys.config import load_config
from fpsys.enhance import enhance, gabor_filter, orientation_field
from fpsys.normalize import Normalized
from fpsys.types import CaptureError, ErrorCode


def _curved_ridges(shape=(360, 300), period=9.0):
    """Vân cong (cung tròn tâm ngoài ảnh): trả (pha, hướng vân thật)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w * 0.3, h * 1.5
    r = np.hypot(x - cx, y - cy)
    phase = 2 * np.pi * r / period
    normal = np.arctan2(y - cy, x - cx)
    ridge_dir = np.mod(normal + np.pi / 2, np.pi)
    return phase, ridge_dir


def _noisy_norm(seed=0, noise=25.0):
    rng = np.random.default_rng(seed)
    phase, ridge_dir = _curved_ridges()
    h, w = phase.shape
    clean = np.cos(phase)  # vân sáng = +1 (ảnh không tiếp xúc)
    illum = np.linspace(0.25, 1.0, w)[None, :]  # gradient sáng tuyến tính
    img = illum * (130 + 60 * clean) + rng.normal(0, noise, (h, w))
    gray = np.clip(img, 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[10:-10, 10:-10] = True
    norm = Normalized(gray=gray, mask=mask, angle_deg=0.0, scale=1.0, ridge_period=9.0,
                      period_raw=9.0, band_energy=0.5, sat_ratio=0.0, brightness=float(img.mean()))
    return norm, clean, ridge_dir


def _inner(mask, m=25):
    out = np.zeros_like(mask)
    out[m:-m, m:-m] = True
    return out & mask


def test_orientation_and_gabor_quality():
    cfg = load_config(overrides=["pipeline.source=contactless_crop"])
    norm, clean, ridge_dir = _noisy_norm()
    enh = enhance(norm, cfg)
    sel = _inner(enh.valid_mask)
    err = np.abs(np.mod(enh.orientation - ridge_dir + np.pi / 2, np.pi) - np.pi / 2)
    assert np.degrees(err[sel].mean()) < 10.0
    # invert=true → vân thành tối → đáp ứng Gabor âm trên vân: tương quan với −clean
    corr = np.corrcoef(enh.image[sel], -clean[sel])[0, 1]
    assert corr > 0.8, corr


def test_flatten_improves_correlation():
    norm, clean, _ = _noisy_norm(seed=1, noise=30.0)
    corrs = {}
    for fs in (2.5, 0.0):
        cfg = load_config(overrides=["pipeline.source=contactless_crop",
                                     f"enhance.flatten_sigma={fs}"])
        enh = enhance(norm, cfg)
        sel = _inner(norm.mask)
        corrs[fs] = np.corrcoef(enh.image[sel], -clean[sel])[0, 1]
    assert corrs[0.0] < corrs[2.5], corrs


def test_orientation_field_straight_ridges():
    y, x = np.mgrid[0:128, 0:128].astype(np.float64)
    ang = np.radians(30)  # pháp tuyến 30° → hướng vân 120°
    img = np.cos(2 * np.pi * (x * np.cos(ang) + y * np.sin(ang)) / 9.0)
    theta, coh = orientation_field(img.astype(np.float32), None, 5.0)
    c = theta[32:96, 32:96]
    assert np.allclose(np.degrees(c), 120.0, atol=2.0)
    assert coh[32:96, 32:96].mean() > 0.9


def test_low_quality_on_noise():
    cfg = load_config(overrides=["pipeline.source=contactless_crop"])
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, (200, 200)).astype(np.uint8)
    mask = np.ones(gray.shape, bool)
    norm = Normalized(gray=gray, mask=mask, angle_deg=0, scale=1, ridge_period=9,
                      period_raw=9, band_energy=0.5, sat_ratio=0, brightness=128)
    with pytest.raises(CaptureError) as ei:
        enhance(norm, cfg)
    assert ei.value.code is ErrorCode.LOW_QUALITY


def test_gabor_gamma_elongated_kernel_denoises_along_ridges():
    """gamma < 1 (kernel dài theo vân) khôi phục vân tốt hơn khi nhiễu lớn hơn tín hiệu."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:160, 0:160].astype(np.float64)
    ang = np.radians(30)  # pháp tuyến 30° → hướng vân 120°
    clean = np.cos(2 * np.pi * (x * np.cos(ang) + y * np.sin(ang)) / 9.0)
    img = (clean + rng.normal(0, 2.0, clean.shape)).astype(np.float32)
    mask = np.ones(img.shape, bool)
    theta = np.full(img.shape, np.radians(120), np.float32)
    freq = np.full(img.shape, 1 / 9.0, np.float32)
    sel = np.zeros_like(mask)
    sel[40:120, 40:120] = True
    corr = {}
    for gamma in (1.0, 0.5):
        out = gabor_filter(img, mask, theta, freq, 16, 0.65, 4, 48, gamma)
        corr[gamma] = abs(np.corrcoef(out[sel], clean[sel])[0, 1])
    assert corr[0.5] > corr[1.0] > 0.5, corr
