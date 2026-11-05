"""Test chung cho hai matcher (MCC, Hough) qua match() — tiêu chí M5."""

import time

import numpy as np
import pytest
from synth import perturb_template, random_template, transform_template

from fpsys.config import load_config
from fpsys.match_mcc import build_cylinders, get_cylinders
from fpsys.matcher import match
from fpsys.types import Minutia, Template, TemplateMismatchError

METHODS = ["mcc", "hough"]


def _cfg(method):
    return load_config(overrides=[f"matcher.method={method}"])


@pytest.mark.parametrize("method", METHODS)
def test_symmetric(method):
    cfg = _cfg(method)
    for seed in range(3):
        a = random_template(seed=seed)
        b = perturb_template(transform_template(a, 10, 5, -5), seed=seed + 10)
        c = random_template(seed=seed + 100)
        assert abs(match(a, b, cfg) - match(b, a, cfg)) <= 1e-6
        assert abs(match(a, c, cfg) - match(c, a, cfg)) <= 1e-6


@pytest.mark.parametrize("method,expected", [("mcc", 0.95), ("hough", 1.0)])
def test_self_match(method, expected):
    cfg = _cfg(method)
    for seed in range(3):
        a = random_template(seed=seed)
        s = match(a, a, cfg)
        assert s >= expected - 1e-9, s


@pytest.mark.parametrize("method", METHODS)
def test_rotation_translation(method):
    cfg = _cfg(method)
    for seed in range(3):
        a = random_template(seed=seed)
        s = match(a, transform_template(a, 15, 20, -10), cfg)
        assert s >= 0.9, s


@pytest.mark.parametrize("method,expected", [("mcc", 0.7), ("hough", 0.6)])
def test_rotation_translation_scale(method, expected):
    cfg = _cfg(method)
    for seed in range(3):
        a = random_template(seed=seed)
        s = match(a, transform_template(a, 15, 20, -10, scale=1.05), cfg)
        assert s >= expected, s


@pytest.mark.parametrize("method", METHODS)
def test_noisy(method):
    cfg = _cfg(method)
    for seed in range(3):
        a = random_template(seed=seed)
        b = perturb_template(transform_template(a, 15, 20, -10), seed=seed + 50)
        s = match(a, b, cfg)
        assert s >= 0.5, s


@pytest.mark.parametrize("method", METHODS)
def test_random_impostor(method):
    cfg = _cfg(method)
    for seed in range(5):
        a = random_template(seed=seed)
        b = random_template(seed=seed + 1000)
        s = match(a, b, cfg)
        assert s < 0.2, s


def test_cfg_hash_mismatch():
    a = random_template(seed=1)
    b = random_template(seed=2)
    b.meta["cfg_hash"] = "other"
    for m in METHODS:
        with pytest.raises(TemplateMismatchError):
            match(a, b, _cfg(m))


def test_too_few_minutiae():
    a = random_template(seed=1)
    tiny = Template([Minutia(1, 1, 0, "ending", 1), Minutia(5, 5, 1, "ending", 1)], (50, 50), 9,
                    meta={"cfg_hash": "test"})
    assert match(a, tiny, _cfg("mcc")) == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_speed_60_points(method):
    """< 50 ms cho hai template 60 điểm; cylinder MCC được tính sẵn khi nạp template (mục 6.9.1)."""
    cfg = _cfg(method)
    match(random_template(seed=1), random_template(seed=2), cfg)  # khởi động
    times = []
    for seed in range(5):
        a, b = random_template(seed=10 + seed), random_template(seed=50 + seed)
        if method == "mcc":
            get_cylinders(a, cfg.matcher.mcc)
            get_cylinders(b, cfg.matcher.mcc)
        t0 = time.perf_counter()
        match(a, b, cfg)
        times.append((time.perf_counter() - t0) * 1000)
    dt = float(np.median(times))
    assert dt < 50, f"{dt:.1f} ms"


def test_mcc_cylinder_build_time():
    mc = _cfg("mcc").matcher.mcc
    times = []
    for seed in range(5):
        t = random_template(seed=seed)
        t0 = time.perf_counter()
        build_cylinders(t, mc)
        times.append((time.perf_counter() - t0) * 1000)
    assert float(np.median(times)) < 100, times  # một lần cho mỗi template
