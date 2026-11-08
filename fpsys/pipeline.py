"""extract(): ghép các bước từ ảnh tới Template (mục 6.11)."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import viz
from .config import Config, cfg_hash, default_config
from .detect import Detection, detect
from .enhance import enhance
from .match_sift import extract_sift
from .minutiae import extract_minutiae
from .normalize import normalize
from .quality import assess
from .skeleton import skeletonize
from .types import CaptureError, Template

log = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    template: Template | None = None
    error: CaptureError | None = None
    steps: dict[str, np.ndarray] = field(default_factory=dict)  # ảnh trung gian (khi collect)
    timing_ms: dict[str, float] = field(default_factory=dict)


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if hasattr(v, "value"):  # Enum
            v = v.value
        elif isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


def run(image: np.ndarray, cfg: Config | None = None, collect: bool = False) -> PipelineTrace:
    """Chạy pipeline; CaptureError ghi vào trace.error. collect=True giữ ảnh trung gian."""
    cfg = cfg or default_config()
    tr = PipelineTrace()
    t_all = time.perf_counter()
    t = t_all

    def tick(name: str) -> None:
        nonlocal t
        now = time.perf_counter()
        tr.timing_ms[name] = round((now - t) * 1000.0, 2)
        t = now

    try:
        det: Detection | None = None
        if cfg.pipeline.source == "contactless":
            det = detect(image, cfg)
            crop = det.crop
        elif cfg.pipeline.source in ("contactless_crop", "contact"):
            crop = image
        else:
            raise ValueError(f"pipeline.source không hợp lệ: {cfg.pipeline.source!r}")
        tick("detect")
        if collect:
            tr.steps["01_crop"] = crop
        norm = normalize(crop, cfg, det)
        tick("normalize")
        if collect:
            if norm.crop_mask is not None and norm.crop_mask.shape == crop.shape[:2]:
                tr.steps["02_mask"] = viz.overlay_mask(crop, norm.crop_mask)
            tr.steps["03_norm"] = viz.overlay_mask(norm.gray, norm.mask)
        q = assess(norm, cfg)
        tick("quality")
        if not q.ok:
            assert q.code is not None
            raise CaptureError(q.code, f"sharpness={q.sharpness:.3f}, brightness="
                                       f"{q.brightness:.1f}, sat_ratio={q.sat_ratio:.3f}")
        enh = enhance(norm, cfg)
        tick("enhance")
        if collect:
            tr.steps["04_flat"] = enh.flat
            tr.steps["05_enhanced"] = viz.signed_to_u8(enh.image, enh.valid_mask)
            tr.steps["06_orientation"] = viz.orientation_view(enh.flat, enh.orientation,
                                                              enh.valid_mask)
        sk = skeletonize(enh, cfg)
        tick("skeleton")
        if collect:
            tr.steps["07_binary"] = (sk.binary * 255).astype(np.uint8)
            tr.steps["08_skeleton"] = (sk.skeleton * 255).astype(np.uint8)
        mins = extract_minutiae(sk, enh, cfg)
        tick("minutiae")
        if collect:
            tr.steps["09_minutiae"] = viz.draw_minutiae(
                viz.overlay_mask(norm.gray, enh.valid_mask), mins)
        emb = None
        if cfg.fusion.enabled:
            from .embed import embed

            emb = embed(norm, cfg)
            tick("embed")
        tr.timing_ms["total"] = round((time.perf_counter() - t_all) * 1000.0, 2)
        meta: dict[str, Any] = {
            "cfg_hash": cfg_hash(cfg),
            "source": cfg.pipeline.source,
            "conf": det.conf if det else None,
            "n_boxes": det.n_boxes if det else None,
            "rotated_deg": det.rotated_deg if det else None,
            "quality": _jsonable(asdict(q)),
            "scale": norm.scale,
            "angle_deg": norm.angle_deg,
            "period_raw": norm.period_raw,
            "n_minutiae": len(mins),
            "timing_ms": dict(tr.timing_ms),
        }
        sift = extract_sift(enh.flat, norm.mask, cfg.sift) if cfg.sift.enabled else None
        tr.template = Template(mins, (int(norm.gray.shape[0]), int(norm.gray.shape[1])),
                               float(norm.ridge_period), embedding=emb, meta=meta, sift=sift)
    except CaptureError as e:
        tr.error = e
        tr.timing_ms["total"] = round((time.perf_counter() - t_all) * 1000.0, 2)
        log.info("CaptureError %s", e)
    return tr


def save_steps(steps: dict[str, np.ndarray], debug_dir: str | Path) -> None:
    d = Path(debug_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, img in steps.items():
        cv2.imwrite(str(d / f"{name}.png"), img)


def extract(image: np.ndarray, cfg: Config | None = None,
            debug_dir: str | None = None) -> Template:
    """Ảnh BGR uint8 → Template. Loại ảnh theo cfg.pipeline.source. Lỗi chụp → CaptureError."""
    tr = run(image, cfg, collect=debug_dir is not None)
    if debug_dir is not None:
        save_steps(tr.steps, debug_dir)
    if tr.error is not None:
        raise tr.error
    assert tr.template is not None
    return tr.template
