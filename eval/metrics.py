"""FAR/FRR, EER, FRR@FAR, CI bootstrap theo subject, FTA, đường DET, histogram (mục 8.3).

Quy ước: chấp nhận khi score ≥ t. FAR(t) = tỷ lệ impostor ≥ t; FRR(t) = tỷ lệ genuine < t.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RateCurve:
    """FAR/FRR trên lưới ngưỡng cố định, hỗ trợ trọng số (cho bootstrap) mà không sắp xếp lại."""

    thresholds: np.ndarray
    gen_sorted: np.ndarray
    imp_sorted: np.ndarray
    gen_order: np.ndarray
    imp_order: np.ndarray
    gen_idx: np.ndarray  # số genuine < t
    imp_idx: np.ndarray  # số impostor < t

    @classmethod
    def build(cls, gen: np.ndarray, imp: np.ndarray) -> RateCurve:
        gen, imp = np.asarray(gen, np.float64), np.asarray(imp, np.float64)
        thr = np.unique(np.concatenate([gen, imp]))
        thr = np.concatenate([thr, [np.inf]])
        go, io = np.argsort(gen, kind="stable"), np.argsort(imp, kind="stable")
        gs, is_ = gen[go], imp[io]
        return cls(thr, gs, is_, go, io, np.searchsorted(gs, thr, "left"),
                   np.searchsorted(is_, thr, "left"))

    def rates(self, w_gen: np.ndarray | None = None, w_imp: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """(FAR, FRR) tại mỗi ngưỡng."""
        if w_gen is None:
            cg = np.arange(len(self.gen_sorted) + 1, dtype=np.float64)
        else:
            cg = np.concatenate([[0.0], np.cumsum(np.asarray(w_gen, np.float64)[self.gen_order])])
        if w_imp is None:
            ci = np.arange(len(self.imp_sorted) + 1, dtype=np.float64)
        else:
            ci = np.concatenate([[0.0], np.cumsum(np.asarray(w_imp, np.float64)[self.imp_order])])
        wg, wi = max(cg[-1], 1e-12), max(ci[-1], 1e-12)
        frr = cg[self.gen_idx] / wg
        far = (ci[-1] - ci[self.imp_idx]) / wi
        return far, frr


def _eer_from_rates(thr: np.ndarray, far: np.ndarray, frr: np.ndarray) -> tuple[float, float]:
    """EER bằng nội suy tuyến tính tại điểm FAR = FRR. Trả (eer, ngưỡng)."""
    d = far - frr  # giảm dần theo ngưỡng
    i = int(np.argmax(d <= 0)) if np.any(d <= 0) else len(d) - 1
    if i == 0:
        return float((far[0] + frr[0]) / 2), float(thr[0])
    d0, d1 = d[i - 1], d[i]
    a = d0 / (d0 - d1) if d0 != d1 else 0.5
    eer = far[i - 1] + a * (far[i] - far[i - 1])
    eer2 = frr[i - 1] + a * (frr[i] - frr[i - 1])
    t0, t1 = thr[i - 1], thr[i]
    t = t0 + a * (t1 - t0) if np.isfinite(t1) else t0
    return float((eer + eer2) / 2), float(t)


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """(EER, ngưỡng EER)."""
    if len(gen) == 0 or len(imp) == 0:
        return float("nan"), float("nan")
    rc = RateCurve.build(gen, imp)
    far, frr = rc.rates()
    return _eer_from_rates(rc.thresholds, far, frr)


def rates_at(gen: np.ndarray, imp: np.ndarray, t: float) -> tuple[float, float]:
    """(FAR, FRR) tại ngưỡng t."""
    gen, imp = np.asarray(gen), np.asarray(imp)
    far = float(np.mean(imp >= t)) if len(imp) else float("nan")
    frr = float(np.mean(gen < t)) if len(gen) else float("nan")
    return far, frr


def threshold_at_far(gen: np.ndarray, imp: np.ndarray, target: float) -> tuple[float, float, float]:
    """Ngưỡng nhỏ nhất có FAR ≤ target. Trả (ngưỡng, FAR, FRR)."""
    rc = RateCurve.build(gen, imp)
    far, frr = rc.rates()
    i = int(np.argmax(far <= target))
    t = float(rc.thresholds[i])
    if not np.isfinite(t):  # mọi impostor đều bằng điểm genuine cao nhất
        t = float(np.nextafter(rc.thresholds[-2], np.inf)) if len(rc.thresholds) > 1 else 1.0
    return t, float(far[i]), float(frr[i])


def bootstrap_eer_ci(gen: np.ndarray, imp: np.ndarray, gen_subj: list[str], imp_subj: list[str],
                     n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """CI (1 − α) của EER: lấy mẫu lại subject (không lấy mẫu lại cặp)."""
    if len(gen) == 0 or len(imp) == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    subjects = sorted(set(gen_subj) | set(imp_subj))
    index = {s: i for i, s in enumerate(subjects)}
    gi = np.array([index[s] for s in gen_subj])
    ii = np.array([index[s] for s in imp_subj])
    rc = RateCurve.build(gen, imp)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(subjects), len(subjects)),
                             minlength=len(subjects)).astype(np.float64)
        wg, wi = counts[gi], counts[ii]
        if wg.sum() == 0 or wi.sum() == 0:
            continue
        far, frr = rc.rates(wg, wi)
        vals.append(_eer_from_rates(rc.thresholds, far, frr)[0])
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fta_stats(errors: dict[str, str | None]) -> dict[str, Any]:
    """errors: ảnh → mã lỗi (None nếu thành công)."""
    n = len(errors)
    codes = Counter(c for c in errors.values() if c)
    n_fail = sum(codes.values())
    return {"n_images": n, "n_fail": n_fail, "fta": n_fail / n if n else 0.0,
            "by_code": {k: v / n for k, v in sorted(codes.items())},
            "count_by_code": dict(sorted(codes.items()))}


def compute_metrics(gen: np.ndarray, imp: np.ndarray, gen_subj: list[str], imp_subj: list[str],
                    imp_same_subject: np.ndarray, fta: float, target_far: float, n_boot: int,
                    seed: int, operating_threshold: float | None = None) -> dict[str, Any]:
    gen, imp = np.asarray(gen, np.float64), np.asarray(imp, np.float64)
    out: dict[str, Any] = {"n_genuine": int(len(gen)), "n_impostor": int(len(imp))}
    if len(gen) == 0 or len(imp) == 0:
        out["error"] = "thiếu điểm genuine hoặc impostor"
        return out
    e, t_eer = eer(gen, imp)
    lo, hi = bootstrap_eer_ci(gen, imp, gen_subj, imp_subj, n_boot, seed)
    t_far, far_at, frr_at = threshold_at_far(gen, imp, target_far)
    same = np.asarray(imp_same_subject, bool)
    e_same = eer(gen, imp[same])[0] if same.any() else None
    out.update({
        "eer": e, "eer_ci95": [lo, hi], "threshold_eer": t_eer,
        "target_far": target_far, "threshold_at_far": t_far,
        "far_at_threshold_at_far": far_at, "frr_at_far": frr_at,
        "fta": fta, "frr_general_at_far": fta + (1 - fta) * frr_at,
        "eer_same_subject": e_same, "n_impostor_same_subject": int(same.sum()),
        "genuine_mean": float(gen.mean()), "impostor_mean": float(imp.mean()),
    })
    if operating_threshold is not None:
        far_op, frr_op = rates_at(gen, imp, operating_threshold)
        out["operating"] = {"threshold": operating_threshold, "far": far_op, "frr": frr_op,
                            "frr_general": fta + (1 - fta) * frr_op}
    return out


def plot_det(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path) -> None:
    """Đường DET (trục lệch chuẩn hóa) cho một hoặc nhiều tập điểm {nhãn: (gen, imp)}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    ticks = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6])
    eps = 1e-4
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for label, (gen, imp) in curves.items():
        if len(gen) == 0 or len(imp) == 0:
            continue
        rc = RateCurve.build(gen, imp)
        far, frr = rc.rates()
        e, _ = eer(gen, imp)
        ax.plot(norm.ppf(np.clip(far, eps, 1 - eps)), norm.ppf(np.clip(frr, eps, 1 - eps)),
                label=f"{label} (EER {100 * e:.2f}%)")
    t = norm.ppf(ticks)
    ax.set_xticks(t, [f"{100 * v:g}" for v in ticks])
    ax.set_yticks(t, [f"{100 * v:g}" for v in ticks])
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(t[0], t[-1])
    ax.plot(t, t, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.grid(True, color="0.9")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_hist(gen: np.ndarray, imp: np.ndarray, path: str | Path,
              threshold: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(0, max(1e-6, float(np.max(np.concatenate([gen, imp])))), 50)
    ax.hist(imp, bins=bins, density=True, alpha=0.6, label=f"impostor (n={len(imp)})")
    ax.hist(gen, bins=bins, density=True, alpha=0.6, label=f"genuine (n={len(gen)})")
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1, label=f"ngưỡng {threshold:.3f}")
    ax.set_xlabel("score")
    ax.set_ylabel("mật độ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
