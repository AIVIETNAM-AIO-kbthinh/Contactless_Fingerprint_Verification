import math

import cv2
import numpy as np
import pytest
from synth import synthetic_fingerprint

from fpsys.config import load_config
from fpsys.enhance import orientation_field
from fpsys.minutiae import angle_diff, filter_false, find_minutiae
from fpsys.skeleton import skeleton_from_binary


def _orient(img: np.ndarray):
    return orientation_field(cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0), None, 5.0)


def test_one_ending_one_bifurcation():
    cfg = load_config(overrides=["minutiae.shift_frac=0", "minutiae.min_minutiae=0"])
    h, w = 120, 160
    canvas = np.zeros((h, w), np.uint8)
    cv2.line(canvas, (30, 60), (80, 60), 255, 1, lineType=8)
    cv2.line(canvas, (80, 60), (159, 20), 255, 1, lineType=8)
    cv2.line(canvas, (80, 60), (159, 100), 255, 1, lineType=8)
    sk = skeleton_from_binary(canvas > 0, cfg.skeleton)
    theta, coh = _orient(canvas)
    mins = find_minutiae(sk, theta, np.ones_like(coh), np.ones((h, w), bool), 9.0,
                         cfg.minutiae)
    kinds = sorted(m.kind for m in mins)
    assert kinds == ["bifurcation", "ending"], mins
    end = next(m for m in mins if m.kind == "ending")
    bif = next(m for m in mins if m.kind == "bifurcation")
    assert math.hypot(end.x - 30, end.y - 60) <= 2
    assert math.hypot(bif.x - 80, bif.y - 60) <= 2
    assert angle_diff(end.theta, math.pi) < math.radians(15)  # thân vân ở bên phải → θ = π
    assert angle_diff(bif.theta, math.pi) < math.radians(15)  # nhánh đơn ở bên trái → θ = π


@pytest.mark.parametrize("flip", [False, True])
def test_duality_ending_vs_valley_bifurcation(flip):
    cfg = load_config(overrides=["minutiae.min_minutiae=0"])
    h, w = 130, 170
    binary = np.zeros((h, w), bool)
    for k in range(12):
        y = 12 + 9 * k
        if y + 2 >= h:
            break
        x_end = 85 if k == 6 else w
        binary[y - 2:y + 2, :x_end] = True
    if flip:
        binary = binary[:, ::-1].copy()
    theta, coh = _orient(binary.astype(np.float32) * 255)
    valid = np.ones((h, w), bool)
    ones = np.ones((h, w), np.float32)
    ridge_sk = skeleton_from_binary(binary, cfg.skeleton)
    valley_sk = skeleton_from_binary(~binary, cfg.skeleton)
    ridge = find_minutiae(ridge_sk, theta, ones, valid, 9.0, cfg.minutiae)
    valley = find_minutiae(valley_sk, theta, ones, valid, 9.0, cfg.minutiae)
    tip = (w - 85, 66) if flip else (85, 66)
    end = min((m for m in ridge if m.kind == "ending"),
              key=lambda m: math.hypot(m.x - tip[0], m.y - tip[1]))
    bif = min((m for m in valley if m.kind == "bifurcation"),
              key=lambda m: math.hypot(m.x - tip[0], m.y - tip[1]))
    assert math.hypot(end.x - bif.x, end.y - bif.y) <= 3.0, (end, bif)
    assert angle_diff(end.theta, bif.theta) <= math.radians(20), (end, bif)
    assert angle_diff(end.theta, math.pi if flip else 0.0) <= math.radians(20)


def test_filter_rules(cfg):
    mc = cfg.minutiae
    xy = np.array([[0, 0], [6, 0],          # hai ending đối nhau (vân đứt) → bỏ
                   [100, 0], [105, 0],      # hai bifurcation gần (cầu nối) → bỏ
                   [200, 0], [205, 0],      # ending + bifurcation gần (gai) → bỏ
                   [300, 0]], np.float64)   # cô lập → giữ
    theta = np.array([0, math.pi, 0, 1, 0, 0, 0], np.float64)
    is_end = np.array([True, True, False, False, True, False, True])
    keep = filter_false(xy, theta, is_end, mc)
    assert keep.tolist() == [False, False, False, False, False, False, True]


def test_cluster_removed(cfg):
    rng = np.random.default_rng(0)
    xy = np.concatenate([rng.uniform(0, 6, (6, 2)), [[100, 100]]])
    theta = rng.uniform(0, 2 * math.pi, 7)
    keep = filter_false(xy, theta, np.ones(7, bool), cfg.minutiae)
    assert keep.tolist() == [False] * 6 + [True]


def test_synthetic_fingerprint_minutiae_count(cfg):
    img = synthetic_fingerprint(seed=2, n_points=24)
    sk = skeleton_from_binary(img < 128, cfg.skeleton)
    theta, coh = _orient(255 - img)
    mins = find_minutiae(sk, theta, coh, np.ones(img.shape, bool), 9.0, cfg.minutiae)
    assert 12 <= len(mins) <= 60
    assert all(0 <= m.theta < 2 * math.pi for m in mins)
