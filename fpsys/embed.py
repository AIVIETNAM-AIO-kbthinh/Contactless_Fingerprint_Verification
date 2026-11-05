"""M8 (tùy chọn): embedding học sâu. Chưa triển khai — chỉ làm khi người dùng yêu cầu."""

from __future__ import annotations

import numpy as np

from .config import Config
from .normalize import Normalized


def embed(norm: Normalized, cfg: Config) -> list[float]:
    raise NotImplementedError("fusion (M8) chưa được triển khai; đặt fusion.enabled=false")


def embedding_score(a: list[float], b: list[float]) -> float:
    """Điểm embedding = (cos + 1) / 2."""
    va, vb = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cos = float(va @ vb / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))
    return (cos + 1.0) / 2.0
