import numpy as np
import pytest
from synth import synthetic_finger_photo, synthetic_fingerprint, to_bgr

from fpsys import detect as det_mod
from fpsys import extract, match, select_enrollment, verify
from fpsys.config import cfg_hash, load_config
from fpsys.pipeline import run
from fpsys.types import CaptureError, ErrorCode
from fpsys.viz import STEP_NAMES, grid


@pytest.fixture
def contact_cfg():
    return load_config(overrides=["pipeline.source=contact"])


def test_extract_contact_synthetic(contact_cfg, tmp_path):
    img = to_bgr(synthetic_fingerprint(seed=5))
    t = extract(img, contact_cfg, debug_dir=str(tmp_path / "dbg"))
    assert len(t.minutiae) >= contact_cfg.minutiae.min_minutiae
    assert t.meta["cfg_hash"] == cfg_hash(contact_cfg)
    assert set(t.meta["timing_ms"]) >= {"detect", "normalize", "enhance", "minutiae", "total"}
    assert 7 <= t.ridge_period <= 11
    written = sorted(p.stem for p in (tmp_path / "dbg").glob("*.png"))
    assert set(written) == set(STEP_NAMES)


def test_enroll_then_verify_same_image(contact_cfg):
    imgs = [to_bgr(synthetic_fingerprint(seed=5, noise=n)) for n in (6, 8, 10)]
    ts = [extract(im, contact_cfg) for im in imgs]
    refs = select_enrollment(ts, contact_cfg)
    assert verify(extract(imgs[0], contact_cfg), refs, contact_cfg).accept
    other = extract(to_bgr(synthetic_fingerprint(seed=9)), contact_cfg)
    assert match(other, ts[0], contact_cfg) < match(ts[1], ts[0], contact_cfg)


def test_contactless_crop_synthetic_photo():
    cfg = load_config(overrides=["pipeline.source=contactless_crop"])
    img, _ = synthetic_finger_photo(period=13.0, seed=4)
    tr = run(img, cfg, collect=True)
    assert tr.error is None, tr.error
    assert len(tr.template.minutiae) >= 12
    assert set(tr.steps) == set(STEP_NAMES)
    out = grid(tr.steps)
    assert out.shape[0] > 900 and out.shape[1] == 3 * 360


def test_contactless_no_finger(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(det_mod, "_run_model", lambda im, c: np.zeros((0, 5), np.float32))
    with pytest.raises(CaptureError) as ei:
        extract(np.full((200, 200, 3), 90, np.uint8), cfg)
    assert ei.value.code is ErrorCode.NO_FINGER


def test_contactless_full_with_mock_yolo(monkeypatch):
    cfg = load_config()
    photo, gt = synthetic_finger_photo(shape=(480, 360), period=13.0, seed=6)
    canvas = np.full((900, 700, 3), 25, np.uint8)
    canvas[200:680, 150:510] = photo
    ys, xs = np.nonzero(gt)
    box = [150 + xs.min(), 200 + ys.min(), 150 + xs.max() + 1, 200 + ys.max() + 1, 0.9]
    monkeypatch.setattr(det_mod, "_run_model", lambda im, c: np.array([box], np.float32))
    t = extract(canvas, cfg)
    assert t.meta["conf"] == pytest.approx(0.9)
    assert len(t.minutiae) >= 12
