"""match(): kiểm tra tương thích template, chọn method theo config."""

from __future__ import annotations

from .config import Config, default_config
from .match_hough import match_hough
from .match_mcc import match_mcc
from .match_sift import match_sift
from .types import Template, TemplateMismatchError


def check_compatible(a: Template, b: Template) -> None:
    """Raise TemplateMismatchError nếu hai template khác version hoặc cfg_hash."""
    if a.version != b.version:
        raise TemplateMismatchError(f"version khác nhau: {a.version} vs {b.version}")
    ha, hb = a.meta.get("cfg_hash"), b.meta.get("cfg_hash")
    if ha != hb:
        raise TemplateMismatchError(f"cfg_hash khác nhau: {ha} vs {hb} (cần enroll lại)")


def match(probe: Template, ref: Template, cfg: Config | None = None) -> float:
    """Điểm tương đồng trong [0, 1], đối xứng: match(a, b) == match(b, a)."""
    cfg = cfg or default_config()
    check_compatible(probe, ref)
    if cfg.matcher.method == "sift":
        if probe.sift is None or ref.sift is None:
            raise TemplateMismatchError("template không có dữ liệu SIFT (cần enroll lại)")
        return float(min(1.0, max(0.0, match_sift(probe.sift, ref.sift, cfg.sift))))
    if len(probe.minutiae) < 3 or len(ref.minutiae) < 3:
        return 0.0
    method = cfg.matcher.method
    if method == "mcc":
        score = match_mcc(probe, ref, cfg.matcher.mcc)
    elif method == "hough":
        score = match_hough(probe, ref, cfg.matcher.hough)
    else:
        raise ValueError(f"matcher.method không hợp lệ: {method!r}")
    return float(min(1.0, max(0.0, score)))
