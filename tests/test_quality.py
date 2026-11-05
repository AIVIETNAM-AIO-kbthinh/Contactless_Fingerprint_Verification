import cv2
import numpy as np
import pytest
from synth import sine_ridges, synthetic_finger_photo

from fpsys.config import load_config
from fpsys.normalize import estimate_period, normalize
from fpsys.quality import assess
from fpsys.types import CaptureError, ErrorCode


def test_band_energy_decreases_with_blur():
    rng = np.random.default_rng(0)
    shape = (320, 320)
    ridges = 40 * sine_ridges(shape, 9.0, 0.7)
    low = cv2.GaussianBlur(rng.normal(0, 1, shape), (0, 0), 20)
    low = 40 * low / np.abs(low).max()
    img = 128 + ridges + low + rng.normal(0, 3, shape)
    mask = np.ones(shape, bool)
    energies = []
    for sigma in [0, 1.0, 1.5, 2.0, 2.5, 3.0]:
        b = cv2.GaussianBlur(img, (0, 0), sigma) if sigma else img
        est = estimate_period(np.clip(b, 0, 255).astype(np.uint8), mask, 64, 4, 30)
        energies.append(est.band_energy if est else 0.0)
    assert all(a > b for a, b in zip(energies, energies[1:])), energies


def test_saturated_image_bad_exposure():
    cfg = load_config(overrides=["pipeline.source=contactless_crop"])
    img, gt = synthetic_finger_photo(period=12.0)
    ys, xs = np.nonzero(gt)
    idx = np.argsort(ys, kind="stable")[: int(0.30 * len(ys))]  # 30% pixel ngón (phía trên)
    img[ys[idx], xs[idx]] = 255
    try:
        q = assess(normalize(img, cfg), cfg)
        code = q.code
    except CaptureError as e:
        code = e.code
    assert code is ErrorCode.BAD_EXPOSURE


def test_blurry_detected():
    cfg = load_config(overrides=["pipeline.source=contactless_crop"])
    img, _ = synthetic_finger_photo(period=12.0)
    sharp = assess(normalize(img, cfg), cfg)
    assert sharp.ok and sharp.code is None
    blurred = cv2.GaussianBlur(img, (0, 0), 6)
    try:
        q = assess(normalize(blurred, cfg), cfg)
        code = q.code
    except CaptureError as e:
        code = e.code
    assert code is ErrorCode.BLURRY


@pytest.mark.parametrize("value", [10, 245])
def test_dark_or_bright_bad_exposure(value):
    cfg = load_config(overrides=["pipeline.source=contactless_crop", "quality.max_sat_ratio=1.0",
                                 "normalize.sat_level=256"])
    img, gt = synthetic_finger_photo(period=12.0, skin=(value,) * 3)
    if value > 128:  # đẩy vùng da lên gần trắng (độ sáng trung bình > max_brightness)
        img[gt] = np.clip(img[gt].astype(np.int32) + 60, 0, 255).astype(np.uint8)
    try:
        code = assess(normalize(img, cfg), cfg).code
    except CaptureError as e:
        code = e.code
    assert code is ErrorCode.BAD_EXPOSURE
