"""Đối sánh Hough có scale (so sánh / ablation, mục 6.9.2)."""

from __future__ import annotations

import math

import numpy as np

from .config import HoughCfg
from .types import Template

TWO_PI = 2.0 * math.pi


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.mod(a + math.pi, TWO_PI) - math.pi


def _arrays(t: Template) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.array([[m.x, m.y] for m in t.minutiae], np.float64).reshape(-1, 2)
    th = np.array([m.theta for m in t.minutiae], np.float64)
    q = np.array([m.quality for m in t.minutiae], np.float64)
    if not np.any(q > 0):
        q = np.ones_like(q)
    return xy, th, q


def _vote(pxy: np.ndarray, pth: np.ndarray, pq: np.ndarray, rxy: np.ndarray, rth: np.ndarray,
          rq: np.ndarray, hc: HoughCfg) -> list[tuple[float, float, float, float]]:
    """Bỏ phiếu (Δθ, s, Δx, Δy); trả `top_k` phép biến đổi (s, Δθ, Δx, Δy) nhiều phiếu nhất."""
    dth = _wrap(rth[None, :] - pth[:, None])  # (np, nr)
    ok = np.abs(dth) <= math.radians(hc.max_rotation_deg)
    pi, ri = np.nonzero(ok)
    if len(pi) == 0:
        return []
    d = dth[pi, ri]
    c, s_ = np.cos(d), np.sin(d)
    w = pq[pi] * rq[ri]
    keys, vals = [], []
    for si, s in enumerate(hc.scales):
        px = s * (c * pxy[pi, 0] - s_ * pxy[pi, 1])
        py = s * (s_ * pxy[pi, 0] + c * pxy[pi, 1])
        dx = rxy[ri, 0] - px
        dy = rxy[ri, 1] - py
        keys.append(np.stack([np.full(len(d), si), np.rint(np.degrees(d) / hc.theta_step_deg),
                              np.rint(dx / hc.xy_step_px), np.rint(dy / hc.xy_step_px)],
                             axis=1).astype(np.int64))
        vals.append(np.stack([d, dx, dy, w], axis=1))
    K = np.concatenate(keys)
    V = np.concatenate(vals)
    uniq, inv = np.unique(K, axis=0, return_inverse=True)
    inv = inv.ravel()
    acc = np.bincount(inv, weights=V[:, 3], minlength=len(uniq))
    out = []
    for b in np.argsort(-acc, kind="stable")[:hc.top_k]:
        sel = inv == b
        ws = V[sel, 3]
        wsum = max(ws.sum(), 1e-12)
        mean_d = math.atan2(float((np.sin(V[sel, 0]) * ws).sum()),
                            float((np.cos(V[sel, 0]) * ws).sum()))
        out.append((float(hc.scales[int(uniq[b, 0])]), mean_d,
                    float((V[sel, 1] * ws).sum() / wsum), float((V[sel, 2] * ws).sum() / wsum)))
    return out


def _pair_count(pxy: np.ndarray, pth: np.ndarray, rxy: np.ndarray, rth: np.ndarray,
                tf: tuple[float, float, float, float], hc: HoughCfg) -> int:
    """Ghép cặp một-một tham lam theo khoảng cách tăng dần sau khi biến đổi probe."""
    s, d, dx, dy = tf
    c, sn = math.cos(d), math.sin(d)
    x = s * (c * pxy[:, 0] - sn * pxy[:, 1]) + dx
    y = s * (sn * pxy[:, 0] + c * pxy[:, 1]) + dy
    t = pth + d
    D = np.hypot(x[:, None] - rxy[None, :, 0], y[:, None] - rxy[None, :, 1])
    A = np.abs(_wrap(t[:, None] - rth[None, :]))
    cand = (D <= hc.match_dist_px) & (A <= math.radians(hc.match_angle_deg))
    pi, ri = np.nonzero(cand)
    order = np.argsort(D[pi, ri], kind="stable")
    used_p = np.zeros(len(pxy), bool)
    used_r = np.zeros(len(rxy), bool)
    m = 0
    for k in order:
        a, b = pi[k], ri[k]
        if not used_p[a] and not used_r[b]:
            used_p[a] = used_r[b] = True
            m += 1
    return m


def _one_way(p: Template, r: Template, hc: HoughCfg) -> float:
    pxy, pth, pq = _arrays(p)
    rxy, rth, rq = _arrays(r)
    best = 0.0
    for tf in _vote(pxy, pth, pq, rxy, rth, rq, hc):
        m = _pair_count(pxy, pth, rxy, rth, tf, hc)
        best = max(best, m * m / float(len(pxy) * len(rxy)))
    return min(1.0, best)


def match_hough(probe: Template, ref: Template, hc: HoughCfg) -> float:
    """Điểm m² / (n_probe × n_ref), lớn nhất trong top_k phép biến đổi, trung bình hai chiều."""
    if not probe.minutiae or not ref.minutiae:
        return 0.0
    return 0.5 * (_one_way(probe, ref, hc) + _one_way(ref, probe, hc))
