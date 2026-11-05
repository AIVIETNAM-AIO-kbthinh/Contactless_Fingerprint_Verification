import math

import cv2
import numpy as np
import pytest
from synth import finger_mask, sine_ridges, synthetic_finger_photo

from fpsys.config import load_config
from fpsys.normalize import (
    _principal_axis,
    _rotate,
    estimate_period,
    normalize,
    orientation_angle,
    skin_mask,
    unwarp,
)


@pytest.mark.parametrize("period", [6.0, 9.0, 14.0])
def test_period_estimate_and_scale(period, cfg):
    rng = np.random.default_rng(int(period))
    for _ in range(3):
        angle = rng.uniform(0, np.pi)
        img = (128 + 80 * sine_ridges((320, 320), period, angle)).astype(np.uint8)
        mask = np.ones(img.shape, bool)
        est = estimate_period(img, mask, 64, 4, 30)
        assert est is not None
        assert abs(est.period - period) / period < 0.10
        scale = cfg.normalize.target_period / est.period
        small = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        est2 = estimate_period(small, np.ones(small.shape, bool), 64, 4, 30)
        assert 8.0 <= est2.period <= 10.0


@pytest.mark.parametrize("period", [22.0, 34.0])
def test_period_estimate_close_up(period):
    """Ảnh chụp cận: vân to hơn khối FFT có thể phân giải → ước lượng đa tỷ lệ."""
    rng = np.random.default_rng(int(period))
    for _ in range(3):
        img = (128 + 80 * sine_ridges((640, 640), period, rng.uniform(0, np.pi))).astype(np.uint8)
        est = estimate_period(img, np.ones(img.shape, bool), 64, 4, 48)
        assert est is not None and abs(est.period - period) / period < 0.10, est


def test_normalize_scales_synthetic_finger_photo():
    cfg = load_config(overrides=["pipeline.source=contactless_crop", "normalize.unwarp=false"])
    img, _ = synthetic_finger_photo(period=14.0)
    norm = normalize(img, cfg)
    assert abs(norm.period_raw - 14.0) / 14.0 < 0.10
    est = estimate_period(norm.gray, norm.mask, 64, 4, 30)
    assert 8.0 <= est.period <= 10.0
    assert 8.0 <= norm.ridge_period <= 10.0


def _tip_up_and_vertical(mask: np.ndarray, angle: float, tip: tuple[float, float]) -> None:
    rot, M = _rotate(mask.astype(np.uint8), angle, cv2.INTER_NEAREST)
    rot = rot > 0
    v, _ = _principal_axis(rot)
    assert abs(abs(v[1]) - 1.0) < 1e-2  # trục chính thẳng đứng (lệch < ~8°)
    tx, ty = M @ np.array([tip[0], tip[1], 1.0])
    rows = np.flatnonzero(rot.any(axis=1))
    assert ty - rows[0] < 0.05 * (rows[-1] - rows[0])  # chóp ngón nằm ở mép trên


@pytest.mark.parametrize("tip_dir", [-60.0, 120.0])  # nghiêng 30° so với trục đứng, 2 chiều
def test_rotation_base_touches_border(tip_dir):
    h, w = 400, 400
    a = math.radians(tip_dir)
    # đặt tâm sao cho gốc ngón bị biên ảnh cắt ngang
    c = (w / 2 - 80 * math.cos(a), h / 2 - 80 * math.sin(a))
    m = finger_mask((h, w), c, 240, 60, tip_dir)
    assert m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
    tip = (c[0] + 240 * math.cos(a), c[1] + 240 * math.sin(a))
    _tip_up_and_vertical(m, orientation_angle(m, 0.15), tip)


@pytest.mark.parametrize("tip_dir", [-60.0, 120.0])
def test_rotation_close_up_width_rule(tip_dir):
    """Ảnh cận: cả hai đầu cùng chạm biên → quy tắc bề rộng."""
    big = finger_mask((800, 800), (400, 400), 300, 90, tip_dir)
    ys, xs = np.nonzero(big)
    # cắt sát theo trục để cả gốc lẫn chóp chạm biên
    y0, x0 = ys.min() + 15, xs.min()
    m = big[y0:ys.max() - 15, x0:xs.max() + 1]
    assert m[0].any() and m[-1].any()
    a = math.radians(tip_dir)
    tip = (400 + 300 * math.cos(a) - x0, 400 + 300 * math.sin(a) - y0)
    _tip_up_and_vertical(m, orientation_angle(m, 0.15), tip)


def _ridge_spacing_cv(row: np.ndarray, valid: np.ndarray) -> float:
    idx = np.flatnonzero(valid)
    seg = row[idx[0]:idx[-1] + 1].astype(np.float64)
    mins = [i for i in range(1, len(seg) - 1) if seg[i] < seg[i - 1] and seg[i] <= seg[i + 1]]
    d = np.diff(mins)
    return float(d.std() / d.mean())


def test_unwarp_cylinder():
    h, w, R, P = 120, 400, 180.0, 12.0
    c = w / 2
    x = np.arange(w, dtype=np.float64)
    inside = np.abs(x - c) < R * math.sin(math.radians(75))
    s = R * np.arcsin(np.clip((x - c) / R, -1, 1))  # độ dài cung trên mặt trụ
    row = 128 + 100 * np.cos(2 * np.pi * s / P)
    gray = np.repeat(np.where(inside, row, 0)[None, :], h, axis=0).astype(np.uint8)
    mask = np.repeat((np.abs(x - c) < R)[None, :], h, axis=0)
    # chỉ xét vùng nằm trong ảnh duỗi phẳng (|x − c|/R ≤ sin 65°)
    sel = np.abs(x - c) <= R * math.sin(math.radians(65))
    before = _ridge_spacing_cv(gray[h // 2], sel)
    g2, _, m2 = unwarp(gray, mask, mask, 65, 31)
    after = _ridge_spacing_cv(g2[h // 2], m2[h // 2])
    assert before > 0.20
    assert after < 0.10


def test_grabcut_near_skin_background():
    cfg = load_config(overrides=["normalize.mask_erode_px=0"])
    rng = np.random.default_rng(3)
    img, gt = synthetic_finger_photo(shape=(480, 360), period=12.0, seed=3,
                                     background=(150, 175, 205), skin=(110, 140, 200))
    # nền gỗ: sọc mờ
    yy, xx = np.mgrid[0:480, 0:360]
    wood = (10 * np.sin(xx / 7.0 + 3 * np.sin(yy / 40.0)))[..., None]
    img = np.where(gt[..., None], img, np.clip(img + wood + rng.normal(0, 4, img.shape), 0, 255)
                   ).astype(np.uint8)
    ys, xs = np.nonzero(gt)
    rect = (xs.min() - 5.0, ys.min() - 5.0, xs.max() + 5.0, ys.max() + 5.0)
    m = skin_mask(img, cfg, rect)
    iou = (m & gt).sum() / (m | gt).sum()
    assert iou >= 0.9
