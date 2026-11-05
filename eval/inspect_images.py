"""Kiểm tra nhanh pipeline trên một thư mục ảnh thật (không cần nhãn): lưới 9 bước + bảng tóm tắt.

Phục vụ các bước kiểm tra bằng mắt của M1 (crop), M2 (ngưỡng quality), M3 (invert/channel), M6.

    python -m eval.inspect_images data/own --out results/real_check [--set enhance.invert=false]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from fpsys.config import load_config
from fpsys.detect import detect
from fpsys.io import load_image
from fpsys.normalize import normalize
from fpsys.pipeline import run
from fpsys.quality import laplacian_variance
from fpsys.types import CaptureError
from fpsys.viz import grid

from .datasets import IMAGE_EXTS

FIELDS = ["image", "status", "detail", "conf", "n_boxes", "rotated_deg", "crop_w", "crop_h",
          "angle_deg", "period_raw", "scale", "band_energy", "brightness", "sat_ratio",
          "laplacian_var", "n_minutiae", "total_ms"]


def _measure(img: np.ndarray, cfg) -> dict:
    """Đo các đại lượng quality ngay cả khi pipeline dừng sớm (để hiệu chỉnh ngưỡng)."""
    out: dict = {}
    try:
        det = detect(img, cfg) if cfg.pipeline.source == "contactless" else None
        crop = det.crop if det else img
        if det:
            out.update(conf=round(det.conf, 3), n_boxes=det.n_boxes, rotated_deg=det.rotated_deg)
        out.update(crop_w=crop.shape[1], crop_h=crop.shape[0])
        n = normalize(crop, cfg, det)
        out.update(angle_deg=round(n.angle_deg, 1), period_raw=round(n.period_raw, 2),
                   scale=round(n.scale, 3), band_energy=round(n.band_energy, 3),
                   brightness=round(n.brightness, 1), sat_ratio=round(n.sat_ratio, 4),
                   laplacian_var=round(laplacian_variance(n.gray, n.mask), 1))
    except CaptureError:
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/real_check")
    ap.add_argument("--set", nargs="+", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(None, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = []
    for k, p in enumerate(paths, 1):
        img = load_image(p)
        tr = run(img, cfg, collect=True)
        row = {"image": p.name, **_measure(img, cfg),
               "total_ms": tr.timing_ms.get("total")}
        if tr.error is not None:
            row.update(status=tr.error.code.value, detail=tr.error.detail)
            footer = f"{p.name[:40]}  {tr.error.code.value}: {tr.error.detail}"
        else:
            row.update(status="OK", n_minutiae=len(tr.template.minutiae))
            footer = (f"{p.name[:40]}  {len(tr.template.minutiae)} minutiae, period "
                      f"{tr.template.ridge_period:.1f}, {tr.timing_ms['total']:.0f} ms")
        cv2.imwrite(str(out / f"{k:02d}_{p.stem[:24]}.jpg"), grid(tr.steps, footer=footer),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(row)
        print(f"{k:2d} {p.name[:28]} {row['status']:<16} {row.get('detail', '')}")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    stats = {}
    for key in ("band_energy", "brightness", "sat_ratio", "period_raw", "total_ms",
                "n_minutiae"):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            stats[key] = {"min": min(vals), "median": float(np.median(vals)), "max": max(vals)}
    status: dict[str, int] = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
    (out / "stats.json").write_text(json.dumps({"status": status, "stats": stats}, indent=2),
                                    encoding="utf-8")
    print(json.dumps({"status": status, "stats": stats}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
