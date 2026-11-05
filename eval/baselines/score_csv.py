"""Tính metrics cho scores.csv của baseline ngoài (cùng metrics.py với fpsys).

    python -m eval.baselines.score_csv data/enhanced/scores_sourceafis.csv --out results/sourceafis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fpsys.config import load_config

from .. import metrics as M
from ..datasets import OWN_RE


def _subject(path: str) -> str:
    m = OWN_RE.match(Path(path).stem + ".jpg")
    return m.group(1).upper() if m else Path(path).stem.split("_")[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    with open(args.scores, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gen = [r for r in rows if r["label"] == "1"]
    imp = [r for r in rows if r["label"] == "0"]
    m = M.compute_metrics(np.array([float(r["score"]) for r in gen]),
                          np.array([float(r["score"]) for r in imp]),
                          [_subject(r["probe"]) for r in gen], [_subject(r["probe"]) for r in imp],
                          np.array([r["same_subject"] == "1" for r in imp]), 0.0,
                          cfg.eval.target_far, cfg.eval.n_bootstrap, cfg.seed)
    print(json.dumps(m, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        g = np.array([float(r["score"]) for r in gen])
        i = np.array([float(r["score"]) for r in imp])
        M.plot_det({"baseline": (g, i)}, out / "det.png")
        M.plot_hist(g, i, out / "hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
