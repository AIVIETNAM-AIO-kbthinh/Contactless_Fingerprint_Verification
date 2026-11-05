"""Chạy đánh giá: trích template (có cache), chấm điểm cặp, tính metrics, ablation, đo thời gian.

Ví dụ:
    python -m eval.run_eval --dataset own --split dev --cond c
    python -m eval.run_eval --dataset own --split test --cond c --dev-run own_dev_c_<hash>_mcc
    python -m eval.run_eval --dataset fvc
    python -m eval.run_eval --dataset own --split dev --cond c --ablation
    python -m eval.run_eval --dataset own --split dev --timing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fpsys.config import (
    Config,
    apply_overrides,
    cfg_hash,
    dump_config,
    extract_hash,
    load_config,
    resolve_path,
)
from fpsys.decision import fuse_scores, select_enrollment
from fpsys.io import load_image
from fpsys.matcher import match
from fpsys.pipeline import extract
from fpsys.types import CaptureError, ErrorCode, Template

from . import metrics as M
from .datasets import DATASET_SOURCE, Record, load_dataset
from .make_pairs import Pair, fvc_pairs, own_pairs, own_sets
from .splits import load_or_create_split

log = logging.getLogger("eval")

ABLATIONS: list[tuple[str, list[str]]] = [
    ("baseline", []),
    ("hough", ["matcher.method=hough"]),
    ("no_unwarp", ["normalize.unwarp=false"]),
    ("no_flatten", ["enhance.flatten_sigma=0"]),
    ("otsu_mask", ["normalize.skin_method=otsu_cr"]),
    ("no_clahe", ["enhance.clahe.clip_limit=0"]),
    ("no_scale_norm", ["normalize.scale_norm=false"]),
    ("green", ["enhance.channel=green"]),
    ("no_invert", ["enhance.invert=false"]),
    ("ref_enhancer", ["enhance.method=reference"]),
    ("otsu_crop", ["detect.method=otsu"]),
    ("mean_top2", ["verify.fusion=mean_top2"]),
]


# ---------------------------------------------------------------------------
# Trích template có cache


@dataclass
class Extracted:
    template: Template | None
    error: str | None  # mã lỗi CaptureError
    detail: str = ""
    timing_ms: dict[str, float] = field(default_factory=dict)


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _extract_one(path: str, cfg: Config) -> Extracted:
    try:
        t = extract(load_image(path), cfg)
        return Extracted(t, None, "", dict(t.meta.get("timing_ms", {})))
    except CaptureError as e:
        return Extracted(None, e.code.value, e.detail)


def _cache_file(cache_dir: Path, path: str, cfg: Config) -> Path:
    return cache_dir / f"{_file_hash(path)}_{extract_hash(cfg)}.json"


def _load_cached(p: Path) -> Extracted | None:
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("template") is not None:
        t = Template.from_dict(d["template"])
        return Extracted(t, None, "", dict(t.meta.get("timing_ms", {})))
    return Extracted(None, d["error"], d.get("detail", ""))


def _save_cached(p: Path, ex: Extracted) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {"template": ex.template.to_dict() if ex.template else None, "error": ex.error,
         "detail": ex.detail}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)


_WORKER_CFG: Config | None = None


def _init_worker(cfg: Config) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _worker(path: str) -> tuple[str, Extracted]:
    assert _WORKER_CFG is not None
    return path, _extract_one(path, _WORKER_CFG)


def extract_all(paths: list[str], cfg: Config, use_cache: bool = True, workers: int = 1
                ) -> dict[str, Extracted]:
    """Gọi extract() mỗi ảnh đúng một lần; CaptureError được ghi nhận, không dừng chương trình."""
    cache_dir = resolve_path("data/cache")
    out: dict[str, Extracted] = {}
    todo = []
    for p in paths:
        cached = _load_cached(_cache_file(cache_dir, p, cfg)) if use_cache else None
        if cached is not None:
            out[p] = cached
        else:
            todo.append(p)
    log.info("Trích template: %d ảnh (%d từ cache)", len(paths), len(paths) - len(todo))
    t0 = time.perf_counter()

    def done(p: str, ex: Extracted, k: int) -> None:
        out[p] = ex
        if use_cache:
            _save_cached(_cache_file(cache_dir, p, cfg), ex)
        if k % 20 == 0 or k == len(todo):
            log.info("  %d/%d (%.1f s)", k, len(todo), time.perf_counter() - t0)

    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) as ex:
            for k, (p, res) in enumerate(ex.map(_worker, todo, chunksize=2), 1):
                done(p, res, k)
    else:
        for k, p in enumerate(todo, 1):
            done(p, _extract_one(p, cfg), k)
    return out


# ---------------------------------------------------------------------------
# Chấm điểm


def _score_pairs(pairs: list[Pair], ex: dict[str, Extracted], cfg: Config
                 ) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for p in pairs:
        a, b = ex[p.probe.path].template, ex[p.ref.path].template
        if a is None or b is None:
            continue
        key = (p.probe.path, p.ref.path)
        if key not in scores:
            scores[key] = match(a, b, cfg)
    return scores


def _template_rows(pairs: list[Pair], scores: dict[tuple[str, str], float]) -> list[dict]:
    rows = []
    for p in pairs:
        s = scores.get((p.probe.path, p.ref.path))
        if s is None:
            continue
        rows.append({"probe": p.probe.name, "ref": p.ref.name, "label": p.label,
                     "same_subject": int(p.same_subject), "score": s,
                     "subject": p.probe.subject})
    return rows


def _verify_rows(records: list[Record], cond: str, subjects: set[str] | None,
                 ex: dict[str, Extracted], scores: dict[tuple[str, str], float], cfg: Config
                 ) -> tuple[list[dict], dict[str, str]]:
    """Điểm mức hệ thống: probe so với tập enroll (đã qua select_enrollment) của mỗi identity."""
    enroll, probes = own_sets(records, cond, subjects)
    by_id: dict[str, list[Record]] = {}
    for r in enroll:
        if ex[r.path].template is not None:
            by_id.setdefault(r.identity, []).append(r)
    enrolled: dict[str, list[Record]] = {}
    fte: dict[str, str] = {}
    for ident, rs in sorted(by_id.items()):
        ts = [ex[r.path].template for r in rs]
        try:
            kept = select_enrollment(ts, cfg)  # type: ignore[arg-type]
        except CaptureError as e:
            fte[ident] = e.code.value
            continue
        kept_ids = {id(t) for t in kept}
        enrolled[ident] = [r for r, t in zip(rs, ts) if id(t) in kept_ids]
    all_ids = {r.identity for r in enroll}
    for ident in sorted(all_ids - set(by_id)):
        fte[ident] = ErrorCode.INCONSISTENT_ENROLL.value  # không còn mẫu enroll hợp lệ nào
    rows = []
    for p in probes:
        if ex[p.path].template is None:
            continue
        for ident, refs in enrolled.items():
            per = [scores[(p.path, r.path)] for r in refs if (p.path, r.path) in scores]
            if not per:
                continue
            genuine = ident == p.identity
            subj = ident.rsplit("_", 1)[0]
            rows.append({"probe": p.name, "ref": ident, "label": int(genuine),
                         "same_subject": int((not genuine) and subj == p.subject),
                         "score": fuse_scores(per, cfg.verify.fusion), "subject": p.subject})
    return rows, fte


def _metrics_from_rows(rows: list[dict], fta: float, cfg: Config,
                       operating_threshold: float | None) -> dict[str, Any]:
    gen = np.array([r["score"] for r in rows if r["label"] == 1])
    imp = np.array([r["score"] for r in rows if r["label"] == 0])
    return M.compute_metrics(
        gen, imp, [r["subject"] for r in rows if r["label"] == 1],
        [r["subject"] for r in rows if r["label"] == 0],
        np.array([r["same_subject"] for r in rows if r["label"] == 0], bool), fta,
        cfg.eval.target_far, cfg.eval.n_bootstrap, cfg.seed, operating_threshold)


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe", "ref", "label", "same_subject", "score"])
        for r in rows:
            w.writerow([r["probe"], r["ref"], r["label"], r["same_subject"], f"{r['score']:.6f}"])


def _select_records(dataset: str, records: list[Record], split: str, cfg: Config
                    ) -> set[str] | None:
    if dataset == "fvc" or split == "all":
        return None
    sp = load_or_create_split(dataset, [r.subject for r in records], cfg.eval.dev_fraction,
                              cfg.seed)
    return set(sp[split])


def evaluate(dataset: str, records: list[Record], cfg: Config, split: str, cond: str,
             run_dir: Path, use_cache: bool = True, workers: int = 1,
             operating_threshold: float | None = None) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    subjects = _select_records(dataset, records, split, cfg)
    if dataset == "fvc":
        pairs = fvc_pairs(records)
    else:
        pairs = own_pairs(records, cond, subjects)
    if not pairs:
        raise RuntimeError("Không có cặp nào để đánh giá (kiểm tra split/cond/dữ liệu)")
    paths = sorted({p.probe.path for p in pairs} | {p.ref.path for p in pairs})
    t0 = time.perf_counter()
    ex = extract_all(paths, cfg, use_cache, workers)
    t_extract = time.perf_counter() - t0
    errors = {p: ex[p].error for p in paths}
    fta = M.fta_stats(errors)
    with open(run_dir / "fta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "code", "detail"])
        for p in paths:
            if ex[p].error:
                w.writerow([Path(p).name, ex[p].error, ex[p].detail])

    t0 = time.perf_counter()
    scores = _score_pairs(pairs, ex, cfg)
    t_match = time.perf_counter() - t0
    trows = _template_rows(pairs, scores)
    _write_csv(trows, run_dir / "scores.csv")

    timings = [e.timing_ms.get("total") for e in ex.values() if e.timing_ms.get("total")]
    result: dict[str, Any] = {
        "dataset": dataset, "split": split, "cond": cond if dataset != "fvc" else None,
        "cfg_hash": cfg_hash(cfg), "matcher": cfg.matcher.method, "fta": fta,
        "extract_mean_ms": float(np.mean(timings)) if timings else None,
        "extract_wall_s": round(t_extract, 1),
        "match_mean_ms": 1000 * t_match / max(1, len(scores)),
    }
    result["template_level"] = _metrics_from_rows(trows, fta["fta"], cfg, operating_threshold)
    primary = "template_level"
    vrows: list[dict] = []
    if dataset != "fvc":
        vrows, fte = _verify_rows(records, cond, subjects, ex, scores, cfg)
        _write_csv(vrows, run_dir / "scores_verify.csv")
        result["fte"] = fte
        result["verify_level"] = _metrics_from_rows(vrows, fta["fta"], cfg, operating_threshold)
        primary = "verify_level"
    result["primary"] = primary
    pm = result[primary]
    result["recommended_threshold"] = pm.get("threshold_at_far")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    dump_config(cfg, run_dir / "config.yaml")

    rows = vrows if primary == "verify_level" else trows
    gen = np.array([r["score"] for r in rows if r["label"] == 1])
    imp = np.array([r["score"] for r in rows if r["label"] == 0])
    if len(gen) and len(imp):
        curves = {primary.replace("_level", ""): (gen, imp)}
        if primary == "verify_level":
            tg = np.array([r["score"] for r in trows if r["label"] == 1])
            ti = np.array([r["score"] for r in trows if r["label"] == 0])
            curves["template"] = (tg, ti)
        M.plot_det(curves, run_dir / "det.png")
        M.plot_hist(gen, imp, run_dir / "hist.png",
                    operating_threshold if operating_threshold is not None
                    else pm.get("threshold_at_far"))
    log.info("%s: EER=%s FTA=%.3f → %s", primary, pm.get("eer"), fta["fta"], run_dir)
    return result


# ---------------------------------------------------------------------------
# Ablation & thời gian


def run_ablation(dataset: str, records: list[Record], base: Config, cond: str, use_cache: bool,
                 workers: int) -> Path:
    import importlib.util

    out_rows = []
    for name, ov in ABLATIONS:
        if name == "ref_enhancer" and importlib.util.find_spec("fingerprint_enhancer") is None:
            log.warning("Bỏ qua ablation %s: chưa cài fingerprint-enhancer", name)
            out_rows.append({"name": name, "note": "skipped: fingerprint-enhancer chưa cài"})
            continue
        cfg = apply_overrides(base, ov)
        run_dir = resolve_path(f"results/ablation/{dataset}_dev_{cond}_{name}")
        res = evaluate(dataset, records, cfg, "dev", cond, run_dir, use_cache, workers)
        pm = res[res["primary"]]
        out_rows.append({"name": name, "eer": pm.get("eer"),
                         "ci_low": pm.get("eer_ci95", [None, None])[0],
                         "ci_high": pm.get("eer_ci95", [None, None])[1],
                         "frr_at_far": pm.get("frr_at_far"), "fta": res["fta"]["fta"],
                         "extract_mean_ms": res["extract_mean_ms"],
                         "match_mean_ms": res["match_mean_ms"], "note": " ".join(ov)})
    path = resolve_path("results/ablation.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["name", "eer", "ci_low", "ci_high", "frr_at_far", "fta", "extract_mean_ms",
            "match_mean_ms", "note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in keys})
    log.info("Đã ghi %s", path)
    return path


def run_timing(records: list[Record], cfg: Config, n: int, run_dir: Path, seed: int
               ) -> dict[str, Any]:
    paths = [r.path for r in records][:n]
    ex = extract_all(paths, cfg, use_cache=False, workers=1)
    steps: dict[str, list[float]] = {}
    for e in ex.values():
        for k, v in e.timing_ms.items():
            steps.setdefault(k, []).append(v)
    temps = [e.template for e in ex.values() if e.template is not None]
    rng = np.random.default_rng(seed)
    match_ms = []
    for _ in range(min(100, len(temps) * (len(temps) - 1) // 2)):
        i, j = rng.choice(len(temps), 2, replace=False)
        a = Template.from_dict(temps[i].to_dict())  # bỏ cache cylinder để đo đủ
        b = Template.from_dict(temps[j].to_dict())
        t0 = time.perf_counter()
        match(a, b, cfg)
        match_ms.append(1000 * (time.perf_counter() - t0))
    out = {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "n": len(v)}
           for k, v in steps.items()}
    if match_ms:
        out["match"] = {"mean": float(np.mean(match_ms)), "p95": float(np.percentile(match_ms, 95)),
                        "n": len(match_ms)}
    out["n_images"] = len(paths)
    out["n_fail"] = sum(1 for e in ex.values() if e.error)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"{'bước':<12}{'mean ms':>10}{'p95 ms':>10}")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:<12}{v['mean']:>10.1f}{v['p95']:>10.1f}")
    return out


def _dev_threshold(run: str) -> float:
    p = Path(run)
    if not p.exists():
        p = resolve_path(f"results/{run}")
    if p.is_dir():
        p = p / "metrics.json"
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("split") != "dev":
        raise ValueError(f"{p} không phải kết quả tập dev")
    return float(m["recommended_threshold"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Đánh giá fpsys")
    ap.add_argument("--dataset", required=True, choices=["own", "fvc", "ridgebase"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--cond", default="c", choices=["c", "u"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ablation", action="store_true", help="chạy bảng ablation trên tập dev")
    ap.add_argument("--timing", action="store_true", help="đo thời gian trên 50 ảnh")
    ap.add_argument("--n-timing", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=None, help="ngưỡng vận hành (từ dev)")
    ap.add_argument("--dev-run", default=None, help="thư mục/metrics.json của lần chạy dev")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    overrides = [f"pipeline.source={DATASET_SOURCE[args.dataset]}", *args.set]
    cfg = load_config(args.config, overrides)
    records = load_dataset(args.dataset, args.data_dir)
    if not records:
        raise SystemExit("Không có ảnh nào")

    if args.timing:
        subjects = _select_records(args.dataset, records, args.split, cfg)
        recs = [r for r in records if subjects is None or r.subject in subjects]
        name = args.run_name or f"timing_{args.dataset}_{cfg_hash(cfg)}"
        run_timing(recs, cfg, args.n_timing, resolve_path(f"results/{name}"), cfg.seed)
        return 0
    if args.ablation:
        run_ablation(args.dataset, records, cfg, args.cond, not args.no_cache, args.workers)
        return 0

    thr = args.threshold
    if thr is None and args.dev_run:
        thr = _dev_threshold(args.dev_run)
    if args.split == "test" and thr is None:
        log.warning("Chạy tập test mà không có ngưỡng từ dev (--dev-run/--threshold): chỉ báo "
                    "EER, không báo FAR/FRR tại ngưỡng vận hành")
    split = "all" if args.dataset == "fvc" else args.split
    cond = args.cond
    name = args.run_name or (f"{args.dataset}_{split}" + (f"_{cond}" if args.dataset != "fvc"
                                                          else "")
                             + f"_{cfg_hash(cfg)}_{cfg.matcher.method}")
    res = evaluate(args.dataset, records, cfg, split, cond, resolve_path(f"results/{name}"),
                   not args.no_cache, args.workers, thr)
    pm = res[res["primary"]]
    print(json.dumps({"run": name, "eer": pm.get("eer"), "eer_ci95": pm.get("eer_ci95"),
                      "frr_at_far": pm.get("frr_at_far"), "fta": res["fta"]["fta"],
                      "threshold_at_far": pm.get("threshold_at_far"),
                      "operating": pm.get("operating")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
