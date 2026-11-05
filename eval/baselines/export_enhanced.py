"""Xuất ảnh Enhanced (~500 ppi, vân tối) + pairs.csv cho baseline ngoài (SourceAFIS/NBIS).

    python -m eval.baselines.export_enhanced --dataset own --split dev --cond c --out data/enhanced
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from fpsys.config import load_config, resolve_path
from fpsys.detect import detect
from fpsys.enhance import enhance
from fpsys.io import load_image
from fpsys.normalize import normalize
from fpsys.types import CaptureError

from ..datasets import DATASET_SOURCE, load_dataset
from ..make_pairs import fvc_pairs, own_pairs
from ..splits import load_or_create_split

log = logging.getLogger(__name__)


def enhanced_png(path: str, cfg) -> np.ndarray:
    img = load_image(path)
    det = detect(img, cfg) if cfg.pipeline.source == "contactless" else None
    norm = normalize(det.crop if det else img, cfg, det)
    enh = enhance(norm, cfg)
    a = float(np.percentile(np.abs(enh.image[enh.valid_mask]), 99)) or 1.0
    out = np.clip(127.5 + 127.5 * enh.image / a, 0, 255).astype(np.uint8)  # vân (âm) → tối
    out[~enh.valid_mask] = 255
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["own", "fvc", "ridgebase"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--cond", default="c", choices=["c", "u"])
    ap.add_argument("--out", default="data/enhanced")
    ap.add_argument("--set", nargs="+", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(None, [f"pipeline.source={DATASET_SOURCE[args.dataset]}", *args.set])
    recs = load_dataset(args.dataset, args.data_dir)
    if args.dataset == "fvc":
        pairs = fvc_pairs(recs)
    else:
        subjects = None
        if args.split != "all":
            sp = load_or_create_split(args.dataset, [r.subject for r in recs],
                                      cfg.eval.dev_fraction, cfg.seed)
            subjects = set(sp[args.split])
        pairs = own_pairs(recs, args.cond, subjects)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok: dict[str, str] = {}
    for p in sorted({q.probe.path for q in pairs} | {q.ref.path for q in pairs}):
        try:
            dst = out / (Path(p).stem + ".png")
            cv2.imwrite(str(dst), enhanced_png(p, cfg))
            ok[p] = str(dst)
        except CaptureError as e:
            log.info("%s: %s", Path(p).name, e.code.value)
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe", "ref", "label", "same_subject"])
        for q in pairs:
            if q.probe.path in ok and q.ref.path in ok:
                w.writerow([ok[q.probe.path], ok[q.ref.path], q.label, int(q.same_subject)])
    log.info("Xuất %d ảnh, pairs.csv tại %s", len(ok), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
