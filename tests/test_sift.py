import cv2
import numpy as np
import pytest
from synth import random_template, synthetic_fingerprint

from fpsys.config import apply_overrides, default_config
from fpsys.match_sift import extract_sift, match_sift
from fpsys.matcher import match
from fpsys.types import Template, TemplateMismatchError


def _sift(img, cfg):
    return extract_sift(img, None, cfg.sift)


def _warp(img):
    h, w = img.shape
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.03), (w, h))


def test_cung_ngon_diem_cao_hon_khac_ngon():
    cfg = default_config()
    a = synthetic_fingerprint(seed=1)
    other = synthetic_fingerprint(seed=2)
    sa, sb, sc = _sift(a, cfg), _sift(_warp(a), cfg), _sift(other, cfg)
    assert match_sift(sa, sb, cfg.sift) > 2 * match_sift(sa, sc, cfg.sift)


def test_doi_xung_va_nam_trong_0_1():
    cfg = default_config()
    a = _sift(synthetic_fingerprint(seed=1), cfg)
    b = _sift(_warp(synthetic_fingerprint(seed=1)), cfg)
    s = match_sift(a, b, cfg.sift)
    assert s == match_sift(b, a, cfg.sift)
    assert 0 <= s <= 1


def test_anh_khong_co_dac_trung():
    cfg = default_config()
    flat = _sift(np.full((200, 200), 128, np.uint8), cfg)
    assert match_sift(flat, flat, cfg.sift) == 0


def test_template_giu_du_lieu_sift_va_bao_loi_khi_thieu():
    cfg = apply_overrides(default_config(), ["matcher.method=sift"])
    t = random_template(seed=0)
    t.sift = _sift(synthetic_fingerprint(seed=1), cfg)
    t.meta["cfg_hash"] = "x"
    back = Template.from_dict(t.to_dict())
    assert back.sift == t.sift
    assert match(back, t, cfg) > 0.5  # cùng ảnh
    old = random_template(seed=1)
    old.meta["cfg_hash"] = "x"
    assert Template.from_dict(old.to_dict()).sift is None
    with pytest.raises(TemplateMismatchError):
        match(old, t, cfg)
