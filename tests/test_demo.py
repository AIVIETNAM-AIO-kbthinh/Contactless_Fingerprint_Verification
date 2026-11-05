"""Test logic của giao diện demo (không mở cửa sổ): hướng dẫn chụp và phiên đăng ký/xác thực."""

import numpy as np
import pytest
from synth import synthetic_finger_photo, synthetic_fingerprint, to_bgr

from demo.camera import parse_source
from demo.guidance import Guide
from demo.session import DemoSession, finger_label
from fpsys import detect as det_mod
from fpsys.config import load_config
from fpsys.types import CaptureError, ErrorCode


def _frame(scale_photo=(900, 700), period=12.0, canvas=(1080, 1440), offset=(80, 370)):
    photo, gt = synthetic_finger_photo(shape=scale_photo, period=period, seed=2)
    frame = np.full((*canvas, 3), 25, np.uint8)
    oy, ox = offset
    frame[oy:oy + photo.shape[0], ox:ox + photo.shape[1]] = photo
    ys, xs = np.nonzero(gt)
    # box đầu ngón (phần trên ~40% của ngón) như YOLO trả về
    y_tip = ys.min() + int(0.45 * (ys.max() - ys.min()))
    box = [ox + xs.min(), oy + ys.min(), ox + xs.max() + 1, oy + y_tip, 0.9]
    return frame, box


def _mock(monkeypatch, boxes):
    monkeypatch.setattr(det_mod, "_run_model", lambda im, c: np.array(boxes, np.float32)
                        .reshape(-1, 5))


@pytest.fixture
def guide():
    return Guide(load_config())


def test_no_finger(monkeypatch, guide):
    _mock(monkeypatch, [])
    assert guide.analyze(np.zeros((480, 640, 3), np.uint8)).code == "NO_FINGER"


def test_multi_finger(monkeypatch, guide):
    frame, box = _frame()
    other = [box[0] + 300, box[1], box[2] + 300, box[3], 0.88]
    _mock(monkeypatch, [box, other])
    assert guide.analyze(frame).code == "MULTI"


def test_too_far_and_off_center(monkeypatch, guide):
    frame, box = _frame()
    _mock(monkeypatch, [[600, 300, 700, 420, 0.9]])  # box rộng 100 px < min_tip_px
    assert guide.analyze(frame).code == "TOO_FAR"
    _mock(monkeypatch, [[0, 300, 400, 600, 0.9]])  # chạm mép trái
    assert guide.analyze(frame).code == "OFF_CENTER"
    _mock(monkeypatch, [[100, 30, 1300, 1080, 0.9]])
    assert guide.analyze(frame).code == "TOO_CLOSE"


def test_dark_and_glare(monkeypatch, guide):
    frame, box = _frame()
    _mock(monkeypatch, [box])
    assert guide.analyze((frame * 0.15).astype(np.uint8)).code == "DARK"
    bright = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    bright[y1:y2, x1:x2, 2] = 255  # kênh đỏ cháy (cùng luật với pipeline)
    guide.reset()
    assert guide.analyze(bright).code == "GLARE"


def test_blurry_when_ridges_not_resolved(monkeypatch, guide):
    import cv2

    frame, box = _frame()
    _mock(monkeypatch, [box])
    assert guide.analyze(cv2.GaussianBlur(frame, (0, 0), 7)).code == "BLURRY"


def test_moving_then_ready(monkeypatch, guide):
    frame, box = _frame()
    _mock(monkeypatch, [box])
    codes = [guide.analyze(frame).code for _ in range(guide.cfg.guide.ready_frames)]
    assert codes[:-1] == ["OK"] * (len(codes) - 1) and codes[-1] == "READY"
    shifted = [box[0] + 80, box[1], box[2] + 80, box[3], 0.9]
    _mock(monkeypatch, [shifted])
    g = guide.analyze(frame)
    assert g.code == "MOVING" and not g.ready


def test_parse_source():
    assert parse_source("0") == 0 and parse_source(" 2 ") == 2
    assert parse_source("http://192.168.1.5:8080/video") == "http://192.168.1.5:8080/video"


def test_session_enroll_verify_save(tmp_path):
    cfg = load_config(overrides=["pipeline.source=contact"])
    s = DemoSession(cfg, data_dir=tmp_path / "demo", save_dir=tmp_path / "own")
    imgs = [to_bgr(synthetic_fingerprint(seed=5, noise=n)) for n in (6, 8, 10)]
    ts = [s.extract(im).template for im in imgs]
    assert all(t is not None for t in ts)
    s.enroll("Nguyễn Văn A", "R2", ts)
    assert s.users() == ["Nguyễn Văn A"] and s.enrolled_fingers("Nguyễn Văn A") == ["R2"]
    assert s.subject_id("Nguyễn Văn A") == "S01"
    assert s.subject_id("Trần B", create=True) == "S02"
    assert s.verify("Nguyễn Văn A", "R2", s.extract(imgs[0]).template).accept
    other = s.extract(to_bgr(synthetic_fingerprint(seed=9))).template
    assert not s.verify("Nguyễn Văn A", "R2", other).accept
    p1 = s.save_image(imgs[0], "Nguyễn Văn A", "R2", "s1", "c")
    p2 = s.save_image(imgs[1], "Nguyễn Văn A", "R2", "s1", "c")
    assert (p1.name, p2.name) == ("S01_R2_s1_c_01.jpg", "S01_R2_s1_c_02.jpg")
    s.delete("Nguyễn Văn A")
    assert s.users() == ["Trần B"] and s.store.get("Nguyễn Văn A", "R2") == []


def test_session_subject_ids_skip_existing_files(tmp_path):
    own = tmp_path / "own"
    own.mkdir()
    (own / "S07_R2_s1_c_01.jpg").write_bytes(b"x")
    s = DemoSession(load_config(), data_dir=tmp_path / "demo", save_dir=own)
    assert s.subject_id("X", create=True) == "S08"
    with pytest.raises(KeyError):
        s.subject_id("Y")


def test_session_inconsistent_enroll(tmp_path):
    cfg = load_config(overrides=["pipeline.source=contact"])
    s = DemoSession(cfg, data_dir=tmp_path / "demo", save_dir=tmp_path / "own")
    ts = [s.extract(to_bgr(synthetic_fingerprint(seed=k))).template for k in (1, 2, 3)]
    with pytest.raises(CaptureError) as ei:
        s.enroll("A", "R2", ts)
    assert ei.value.code is ErrorCode.INCONSISTENT_ENROLL


def test_finger_label():
    assert finger_label("R2") == "R2 (ngón trỏ phải)"
    assert finger_label("L1") == "L1 (ngón cái trái)"
