import importlib.util

import numpy as np
import pytest

from fpsys import detect as det_mod
from fpsys.config import load_config, resolve_path
from fpsys.types import CaptureError, ErrorCode

EMPTY = np.zeros((0, 5), np.float32)


def test_uniform_image_no_finger_mock(monkeypatch, cfg):
    calls = []

    def fake(image, cfg):
        calls.append(image.shape)
        return EMPTY

    monkeypatch.setattr(det_mod, "_run_model", fake)
    with pytest.raises(CaptureError) as ei:
        det_mod.detect(np.full((300, 200, 3), 128, np.uint8), cfg)
    assert ei.value.code is ErrorCode.NO_FINGER
    assert calls == [(300, 200, 3), (200, 300, 3), (200, 300, 3)]  # gốc + TTA 90° + 270°


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None
                    or not resolve_path("models/best.pt").exists(),
                    reason="cần ultralytics và models/best.pt")
def test_uniform_image_no_finger_real_yolo(cfg):
    with pytest.raises(CaptureError) as ei:
        det_mod.detect(np.full((640, 480, 3), 128, np.uint8), cfg)
    assert ei.value.code is ErrorCode.NO_FINGER


def test_select_box_tie_conf_and_area():
    boxes = np.array([[0, 0, 10, 10, 0.90],     # conf cao nhất, nhỏ
                      [0, 0, 50, 50, 0.85],     # trong ngưỡng tie, lớn nhất → chọn
                      [0, 0, 200, 200, 0.50]],  # lớn hơn nhưng conf thấp
                     np.float32)
    assert det_mod.select_box(boxes, 0.10) == 1
    assert det_mod.select_box(boxes, 0.01) == 0
    assert det_mod.select_box(boxes, 0.50) == 2


def _bright_rect_detector(fail_first_rotated: bool = False):
    """YOLO giả: chỉ 'thấy' ngón khi ảnh dọc (h > w); box = vùng pixel sáng."""
    state = {"rotated_calls": 0}

    def fake(image, cfg):
        if image.shape[0] <= image.shape[1]:
            return EMPTY
        state["rotated_calls"] += 1
        if fail_first_rotated and state["rotated_calls"] == 1:
            return EMPTY
        ys, xs = np.nonzero(image[:, :, 0] > 200)
        return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9]], np.float32)

    return fake


@pytest.mark.parametrize("fail_first,expected_deg", [(False, 90), (True, 270)])
def test_tta_maps_box_back_to_original(monkeypatch, fail_first, expected_deg):
    cfg = load_config(overrides=["detect.pad=0"])
    img = np.zeros((300, 500, 3), np.uint8)
    x1, y1, x2, y2 = 320, 60, 450, 140
    img[y1:y2, x1:x2] = 255
    monkeypatch.setattr(det_mod, "_run_model", _bright_rect_detector(fail_first))
    d = det_mod.detect(img, cfg)
    assert d.rotated_deg == expected_deg
    assert np.allclose(d.box_xyxy, (x1, y1, x2, y2), atol=1.0)
    assert d.crop.shape[:2] == (y2 - y1, x2 - x1)
    assert (d.crop > 200).all()


def test_pad_and_clamp(monkeypatch):
    cfg = load_config(overrides=["detect.pad=0.1"])
    monkeypatch.setattr(det_mod, "_run_model",
                        lambda im, c: np.array([[100, 0, 200, 100, 0.9]], np.float32))
    d = det_mod.detect(np.zeros((400, 600, 3), np.uint8), cfg)
    assert d.crop_xyxy == (90, 0, 210, 110)
    assert d.box_in_crop() == (10, 0, 110, 100)
