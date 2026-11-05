"""Minutia Cylinder-Code + LSS-R (Cappelli, Ferrara, Maltoni, TPAMI 2010) — matcher mặc định."""

from __future__ import annotations

import hashlib
import math
from dataclasses import astuple, dataclass

import cv2
import numpy as np
from scipy.special import erf

from .config import MccCfg
from .types import Template

TWO_PI = 2.0 * math.pi


def _wrap(a: np.ndarray) -> np.ndarray:
    """Đưa góc về [−π, π)."""
    return np.mod(a + math.pi, TWO_PI) - math.pi


def _z(v: np.ndarray | float, mu: float, tau: float) -> np.ndarray:
    """Sigmoid Z(v; μ, τ) = 1 / (1 + e^{−τ(v − μ)})."""
    return 1.0 / (1.0 + np.exp(np.clip(-tau * (np.asarray(v, np.float64) - mu), -60, 60)))


@dataclass
class Cylinders:
    values: np.ndarray  # (n, ns·ns·nd) float32, 0 tại ô không hợp lệ
    cell_mask: np.ndarray  # (n, ns·ns) float32, 1 = ô hợp lệ
    valid: np.ndarray  # (n,) bool, cylinder hợp lệ
    xy: np.ndarray  # (n, 2)
    theta: np.ndarray  # (n,)
    n_circle: int  # số ô nằm trong hình tròn bán kính R
    mask_nd: np.ndarray | None = None  # (n, ns·ns·nd) float32, cell_mask lặp theo chiều góc


def _hull_contains(points: np.ndarray, xy: np.ndarray, offset: float) -> np.ndarray:
    """Điểm nằm trong bao lồi của `xy` nới thêm `offset` px (raster 1 px + distance transform)."""
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
    lo = np.floor(np.minimum(points.min(axis=0), hull.min(axis=0))) - 2
    hi = np.ceil(np.maximum(points.max(axis=0), hull.max(axis=0))) + 2
    w, h = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1
    canvas = np.full((h, w), 255, np.uint8)
    poly = np.rint(hull - lo).astype(np.int32)
    if len(poly) >= 3:
        cv2.fillPoly(canvas, [poly], 0)
    else:
        cv2.polylines(canvas, [poly], False, 0, 1)
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)  # khoảng cách tới bao lồi
    ij = np.rint(points - lo).astype(np.int64)
    return dist[ij[:, 1], ij[:, 0]] <= offset


def build_cylinders(t: Template, mc: MccCfg) -> Cylinders:
    """Dựng cylinder cho mọi minutia của template."""
    n = len(t.minutiae)
    ns, nd, R = mc.ns, mc.nd, float(mc.radius)
    xy = np.array([[m.x, m.y] for m in t.minutiae], np.float64).reshape(n, 2)
    th = np.array([m.theta for m in t.minutiae], np.float64)
    ds, dd = 2.0 * R / ns, TWO_PI / nd
    idx = np.arange(ns) - (ns - 1) / 2.0
    gi, gj = np.meshgrid(idx, idx, indexing="ij")
    off = ds * np.stack([gi.ravel(), gj.ravel()], axis=1)  # (C, 2) trong hệ tọa độ của minutia
    C = off.shape[0]
    in_circle = np.hypot(off[:, 0], off[:, 1]) <= R
    n_circle = int(in_circle.sum())
    offc = off[in_circle]  # chỉ tính các ô trong hình tròn
    c, s = np.cos(th), np.sin(th)
    # p = m + R(θ)·off  → cylinder quay theo hướng minutia
    cx = xy[:, 0, None] + c[:, None] * offc[None, :, 0] - s[:, None] * offc[None, :, 1]
    cy = xy[:, 1, None] + s[:, None] * offc[None, :, 0] + c[:, None] * offc[None, :, 1]
    Cc = offc.shape[0]
    cell_c = np.zeros((n, Cc), bool)
    if n >= 1:
        pts = np.stack([cx.ravel(), cy.ravel()], axis=1)
        cell_c = _hull_contains(pts, xy, mc.hull_offset).reshape(n, Cc)

    # đóng góp không gian (Gaussian σ_S, cắt ở 3σ_S, bỏ chính minutia gốc)
    sig_s = mc.sigma_s
    dx = (cx[:, :, None] - xy[None, None, :, 0]).astype(np.float32)
    dy = (cy[:, :, None] - xy[None, None, :, 1]).astype(np.float32)
    d2 = dx * dx + dy * dy  # (n, Cc, n)
    near = d2 <= (3 * sig_s) ** 2
    eye = np.eye(n, dtype=bool)
    near &= ~eye[:, None, :]
    cs = np.zeros_like(d2)
    np.exp(d2 * np.float32(-0.5 / sig_s ** 2), out=cs, where=near)
    cs *= np.float32(1.0 / (sig_s * math.sqrt(TWO_PI)))
    # đóng góp hướng
    dth = _wrap(th[:, None] - th[None, :])  # dφ(θ_m, θ_t), (n, n)
    phi_k = -math.pi + (np.arange(nd) + 0.5) * dd
    alpha = _wrap(phi_k[None, None, :] - dth[:, :, None])  # (n, n, nd)
    k = 1.0 / (mc.sigma_d * math.sqrt(2.0))
    cd = (0.5 * (erf((alpha + dd / 2) * k) - erf((alpha - dd / 2) * k))).astype(np.float32)
    v = np.einsum("ict,itk->ick", cs, cd)
    psi_c = _z(v, mc.mu_psi, mc.tau_psi).astype(np.float32) * cell_c[:, :, None]

    psi = np.zeros((n, C, nd), np.float32)
    psi[:, in_circle] = psi_c
    cell_mask = np.zeros((n, C), np.float32)
    cell_mask[:, in_circle] = cell_c

    dist_m = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    n_neigh = ((dist_m <= R + 3 * sig_s) & ~eye).sum(axis=1)
    valid = (cell_c.sum(axis=1) >= mc.min_valid_cells * n_circle) & (
        n_neigh >= mc.min_neighbors)
    return Cylinders(values=psi.reshape(n, -1), cell_mask=cell_mask, valid=valid, xy=xy,
                     theta=th, n_circle=n_circle, mask_nd=np.repeat(cell_mask, nd, axis=1))


def _fingerprint(t: Template) -> str:
    arr = np.array([[m.x, m.y, m.theta] for m in t.minutiae], np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def get_cylinders(t: Template, mc: MccCfg) -> Cylinders:
    """Cylinder được tính một lần và cache trên đối tượng template (không thuộc to_dict)."""
    key = (astuple(mc), _fingerprint(t))
    cached = t.__dict__.get("_mcc_cache")
    if cached is not None and cached[0] == key:
        return cached[1]
    cyl = build_cylinders(t, mc)
    t.__dict__["_mcc_cache"] = (key, cyl)
    return cyl


def local_similarity(a: Cylinders, b: Cylinders, mc: MccCfg) -> np.ndarray:
    """Ma trận độ tương đồng cục bộ n_a × n_b: 1 − ‖a − b‖ / (‖a‖ + ‖b‖) trên ô hợp lệ chung."""
    A, B = a.values, b.values
    MA, MB = a.mask_nd, b.mask_nd
    ab = (A @ B.T).astype(np.float64)
    na2 = ((A * A) @ MB.T).astype(np.float64)
    nb2 = (MA @ (B * B).T).astype(np.float64)
    diff = np.sqrt(np.maximum(na2 + nb2 - 2.0 * ab, 0.0))
    den = np.sqrt(na2) + np.sqrt(nb2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(den > 0, 1.0 - diff / den, 0.0)
    common = a.cell_mask @ b.cell_mask.T
    sim[common < mc.min_common_cells * a.n_circle] = 0.0
    dtheta = np.abs(_wrap(a.theta[:, None] - b.theta[None, :]))
    sim[dtheta > math.radians(mc.max_dir_diff_deg)] = 0.0
    sim[~a.valid, :] = 0.0
    sim[:, ~b.valid] = 0.0
    return np.clip(sim, 0.0, 1.0)


def _greedy_pairs(sim: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Chọn tối đa `n_max` cặp một-một có độ tương đồng cao nhất (> 0)."""
    flat = sim.ravel()
    order = np.argsort(-flat, kind="stable")
    used_r = np.zeros(sim.shape[0], bool)
    used_c = np.zeros(sim.shape[1], bool)
    rows, cols = [], []
    for f in order:
        if flat[f] <= 0 or len(rows) >= n_max:
            break
        r, c = divmod(int(f), sim.shape[1])
        if used_r[r] or used_c[c]:
            continue
        used_r[r] = used_c[c] = True
        rows.append(r)
        cols.append(c)
    return np.array(rows, np.int64), np.array(cols, np.int64)


def lss_r(sim: np.ndarray, a: Cylinders, b: Cylinders, mc: MccCfg) -> float:
    """Gộp điểm toàn cục bằng Local Similarity Sort with Relaxation."""
    na, nb = sim.shape
    if na == 0 or nb == 0:
        return 0.0
    n_p = mc.min_np + int(round(float(_z(min(na, nb), mc.mu_p, mc.tau_p))
                                * (mc.max_np - mc.min_np)))
    rows, cols = _greedy_pairs(sim, min(na, nb))
    n_r = len(rows)
    if n_r == 0:
        return 0.0
    lam0 = sim[rows, cols]
    lam = lam0.copy()
    if n_r > 1:
        rc = mc.relax
        pa, pb = a.xy[rows], b.xy[cols]
        ta, tb = a.theta[rows], b.theta[cols]
        da = np.hypot(pa[:, None, 0] - pa[None, :, 0], pa[:, None, 1] - pa[None, :, 1])
        db = np.hypot(pb[:, None, 0] - pb[None, :, 0], pb[:, None, 1] - pb[None, :, 1])
        # dung sai tương đối cho sai số tỷ lệ giữa hai lần chụp (scale_tol = 0: như bài báo)
        d1 = np.maximum(np.abs(da - db) - rc.scale_tol * 0.5 * (da + db), 0.0)
        d2 = np.abs(_wrap(_wrap(ta[:, None] - ta[None, :]) - _wrap(tb[:, None] - tb[None, :])))
        ra = _wrap(ta[:, None] - np.arctan2(pa[None, :, 1] - pa[:, None, 1],
                                            pa[None, :, 0] - pa[:, None, 0]))
        rb = _wrap(tb[:, None] - np.arctan2(pb[None, :, 1] - pb[:, None, 1],
                                            pb[None, :, 0] - pb[:, None, 0]))
        d3 = np.abs(_wrap(ra - rb))
        rho = (_z(d1, rc.mu1, rc.tau1) * _z(d2, math.radians(rc.mu2_deg), rc.tau2)
               * _z(d3, math.radians(rc.mu3_deg), rc.tau3))
        np.fill_diagonal(rho, 0.0)
        for _ in range(rc.n_iter):
            if rc.support == "weighted":
                # trung bình của ρ·λ_k có trọng số λ_k: cặp rác (λ nhỏ) không làm loãng hỗ trợ
                support = (rho @ (lam * lam)) / np.maximum(lam.sum() - lam, 1e-12)
            elif rc.support == "mean":  # như bài báo: Σ ρ·λ_k / (n_R − 1)
                support = (rho @ lam) / (n_r - 1)
            else:
                raise ValueError(f"matcher.mcc.relax.support không hợp lệ: {rc.support!r}")
            lam = rc.w * lam + (1.0 - rc.w) * support
    eff = lam / np.maximum(lam0, 1e-12)
    top = np.argsort(-eff, kind="stable")[:min(n_p, n_r)]
    return float(np.clip(lam[top].sum() / n_p, 0.0, 1.0))


def match_mcc(probe: Template, ref: Template, mc: MccCfg) -> float:
    """Điểm MCC trong [0, 1], đối xứng (trung bình hai chiều của LSS-R)."""
    a, b = get_cylinders(probe, mc), get_cylinders(ref, mc)
    if not a.valid.any() or not b.valid.any():
        return 0.0
    sim = local_similarity(a, b, mc)
    return 0.5 * (lss_r(sim, a, b, mc) + lss_r(sim.T.copy(), b, a, mc))
