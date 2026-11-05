"""EER trên một thư mục có labels.csv (nguoi, anh, ngon, do_tin_cay): mọi cặp ảnh hợp lệ.

    python -m eval.labeled "data/own/Vân tay test" results/own_vantaytest [KEY=VALUE ...]

Ba cách tính:
  A = mọi cặp;
  B = impostor chỉ giữa hai người khác nhau (không bị ảnh hưởng bởi lỗi nhãn trong một người);
  C = chỉ ảnh có do_tin_cay = "cao".
Ảnh có cột `ngon` trống không được ghép cặp nhưng vẫn tính vào FTA.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from fpsys.config import load_config
from fpsys.matcher import match

from . import metrics as M
from .run_eval import extract_all

log = logging.getLogger("eval.labeled")


def _metrics(rows: list[dict], sel: Callable[[dict], bool], fta: float, cfg: Any
             ) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray]]:
    rs = [r for r in rows if sel(r)]
    g = [r for r in rs if r["label"] == 1]
    i = [r for r in rs if r["label"] == 0]
    gen = np.array([r["score"] for r in g])
    imp = np.array([r["score"] for r in i])
    # bootstrap theo ngón (identity), không theo người: chỉ có vài người
    m = M.compute_metrics(gen, imp, [r["ia"] for r in g], [r["ia"] for r in i],
                          np.array([r["same_person"] for r in i], bool), fta,
                          cfg.eval.target_far, cfg.eval.n_bootstrap, cfg.seed)
    m["n_fingers_with_genuine"] = len({r["ia"] for r in g})
    return m, (gen, imp)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING)
    root, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(None, ["pipeline.source=contactless", *argv[2:]])
    with open(root / "labels.csv", encoding="utf-8-sig") as f:
        lab = list(csv.DictReader(f))
    allp = {str(root / r["nguoi"] / r["anh"]): r for r in lab}

    ex = extract_all(sorted(allp), cfg, True, 1)
    fta_all = M.fta_stats({p: e.error for p, e in ex.items()})
    labeled = [(p, r) for p, r in allp.items() if r["ngon"]]
    fta_lab = M.fta_stats({p: ex[p].error for p, _ in labeled})
    ok = [(p, r) for p, r in labeled if ex[p].template is not None]

    t0 = time.perf_counter()
    rows = []
    for (pa, ra), (pb, rb) in itertools.combinations(ok, 2):
        ia, ib = f"{ra['nguoi']}|{ra['ngon']}", f"{rb['nguoi']}|{rb['ngon']}"
        rows.append({"a": ra["anh"], "b": rb["anh"], "ia": ia, "ib": ib,
                     "label": int(ia == ib), "same_person": int(ra["nguoi"] == rb["nguoi"]),
                     "hi": int(ra["do_tin_cay"] == "cao" and rb["do_tin_cay"] == "cao"),
                     "score": match(ex[pa].template, ex[pb].template, cfg)})
    t_match = time.perf_counter() - t0
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    variants: list[tuple[str, Callable[[dict], bool]]] = [
        ("A_all_pairs", lambda r: True),
        ("B_impostor_cross_person", lambda r: r["label"] == 1 or not r["same_person"]),
        ("C_high_conf_only", lambda r: r["hi"] == 1),
    ]
    res, curves = {}, {}
    for name, sel in variants:
        res[name], curves[name] = _metrics(rows, sel, fta_lab["fta"], cfg)
    M.plot_det(curves, out / "det.png")
    M.plot_hist(*curves["A_all_pairs"], out / "hist.png")

    gen = sorted((r for r in rows if r["label"]), key=lambda r: r["score"])
    imp = sorted((r for r in rows if not r["label"]), key=lambda r: -r["score"])
    summary = {
        "overrides": argv[2:], "fta_all": fta_all, "fta_labeled": fta_lab,
        "n_ok_labeled": len(ok), "match_ms": 1000 * t_match / max(1, len(rows)),
        "results": res,
        "lowest_genuine": [(r["a"], r["b"], r["ia"], round(r["score"], 4)) for r in gen[:8]],
        "highest_impostor": [(r["a"], r["b"], r["ia"], r["ib"], round(r["score"], 4))
                             for r in imp[:8]],
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    for name, m in res.items():
        print(f"{name}: EER {m['eer']:.3f} CI95 [{m['eer_ci95'][0]:.2f}, "
              f"{m['eer_ci95'][1]:.2f}] genuine {m['n_genuine']} impostor {m['n_impostor']}")
    print(f"FTA {fta_all['fta']:.3f} ({fta_all['count_by_code']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
