"""Quyết định enroll / verify (mục 6.10). Backend dùng chung logic này."""

from __future__ import annotations

import logging

import numpy as np

from .config import Config, default_config
from .matcher import match
from .types import CaptureError, ErrorCode, Template, VerifyResult

log = logging.getLogger(__name__)


def consistency_matrix(templates: list[Template], cfg: Config) -> np.ndarray:
    n = len(templates)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = match(templates[i], templates[j], cfg)
    return S


def select_enrollment(templates: list[Template], cfg: Config | None = None) -> list[Template]:
    """Lọc các mẫu enroll không nhất quán. Còn ít hơn enroll.min_templates → INCONSISTENT_ENROLL."""
    cfg = cfg or default_config()
    ec = cfg.enroll
    if len(templates) < ec.min_templates:
        raise CaptureError(ErrorCode.INCONSISTENT_ENROLL,
                           f"cần ít nhất {ec.min_templates} mẫu, nhận {len(templates)}")
    if len(templates) == 1:
        return list(templates)
    S = consistency_matrix(templates, cfg)
    med = np.nanmedian(S, axis=1)
    kept = []
    for i, t in enumerate(templates):
        if med[i] >= ec.min_consistency:
            kept.append(t)
        else:
            log.info("Loại mẫu enroll #%d: trung vị điểm %.3f < %.3f", i, med[i],
                     ec.min_consistency)
    if len(kept) < ec.min_templates:
        raise CaptureError(ErrorCode.INCONSISTENT_ENROLL,
                           f"chỉ {len(kept)} mẫu nhất quán (trung vị: "
                           f"{', '.join(f'{m:.3f}' for m in med)})")
    return kept


def fuse_scores(scores: list[float], fusion: str) -> float:
    if not scores:
        return 0.0
    s = sorted(scores, reverse=True)
    if fusion == "max":
        return float(s[0])
    if fusion == "mean_top2":
        return float(np.mean(s[:2]))
    raise ValueError(f"verify.fusion không hợp lệ: {fusion!r}")


def verify(probe: Template, refs: list[Template], cfg: Config | None = None) -> VerifyResult:
    """Gộp điểm với mọi mẫu đã enroll theo cfg.verify.fusion, so với cfg.verify.threshold."""
    cfg = cfg or default_config()
    if not refs:
        raise ValueError("Không có mẫu enroll để so khớp")
    per = [match(probe, r, cfg) for r in refs]
    score = fuse_scores(per, cfg.verify.fusion)
    thr = float(cfg.verify.threshold)
    return VerifyResult(score=score, threshold=thr, accept=score >= thr, per_template=per)
