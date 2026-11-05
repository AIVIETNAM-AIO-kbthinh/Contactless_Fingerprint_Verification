"""Chia subject thành dev/test cố định (có seed), ghi ra data/splits/{dataset}.json."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from fpsys.config import load_config, resolve_path

from .datasets import load_dataset

log = logging.getLogger(__name__)


def make_split(subjects: list[str], dev_fraction: float, seed: int) -> dict[str, list[str]]:
    subs = sorted(set(subjects))
    if not subs:
        raise ValueError("Không có subject nào")
    rng = np.random.default_rng(seed)
    perm = [subs[i] for i in rng.permutation(len(subs))]
    n_dev = int(round(dev_fraction * len(subs)))
    if len(subs) >= 2:
        n_dev = min(max(n_dev, 1), len(subs) - 1)
    return {"dev": sorted(perm[:n_dev]), "test": sorted(perm[n_dev:])}


def split_path(dataset: str) -> Path:
    return resolve_path(f"data/splits/{dataset}.json")


def load_or_create_split(dataset: str, subjects: list[str], dev_fraction: float, seed: int,
                         path: Path | None = None) -> dict[str, list[str]]:
    """Đọc file chia tập nếu đã có (cố định), không thì tạo mới. Subject lạ → lỗi rõ ràng."""
    path = path or split_path(dataset)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        known = set(split["dev"]) | set(split["test"])
        unknown = sorted(set(subjects) - known)
        if unknown:
            raise ValueError(f"Subject chưa có trong {path}: {unknown}. Xóa file để chia lại "
                             "(chỉ làm khi chưa báo cáo kết quả test).")
        return split
    split = make_split(subjects, dev_fraction, seed)
    split.update({"dataset": dataset, "seed": seed, "dev_fraction": dev_fraction})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    log.info("Đã ghi %s (%d dev, %d test)", path, len(split["dev"]), len(split["test"]))
    return split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chia subject dev/test cố định")
    ap.add_argument("--dataset", required=True, choices=["own", "ridgebase"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    recs = load_dataset(args.dataset, args.data_dir)
    split = load_or_create_split(args.dataset, [r.subject for r in recs], cfg.eval.dev_fraction,
                                 cfg.seed)
    print(json.dumps({"dev": split["dev"], "test": split["test"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
