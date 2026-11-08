"""So khớp SIFT trên ảnh đã làm phẳng sáng: tùy chọn cho ảnh chụp không chạm.

Minutiae trên ảnh điện thoại kém ổn định vì vân chỉ chênh vài mức xám. SIFT so hình dạng vân cục
bộ nên chịu được điều đó hơn. Điểm là số cặp khớp hình học (RANSAC), đưa về [0, 1].
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .config import SiftCfg


def extract_sift(gray: np.ndarray, mask: np.ndarray | None, cfg: SiftCfg) -> dict:
    """Điểm đặc trưng trong vùng ngón; trả dict JSON được để lưu vào Template."""
    m = None if mask is None else (mask > 0).astype(np.uint8) * 255
    sift = cv2.SIFT_create(nfeatures=cfg.nfeatures, contrastThreshold=cfg.contrast)
    kps, desc = sift.detectAndCompute(gray, m)
    pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
    d = np.zeros((0, 128), np.uint8) if desc is None else np.clip(desc, 0, 255).astype(np.uint8)
    return {"pts": base64.b64encode(pts.tobytes()).decode(),
            "desc": base64.b64encode(d.tobytes()).decode()}


def _decode(s: dict) -> tuple[np.ndarray, np.ndarray]:
    pts = np.frombuffer(base64.b64decode(s["pts"]), np.float32).reshape(-1, 2)
    desc = np.frombuffer(base64.b64decode(s["desc"]), np.uint8).reshape(-1, 128)
    return pts, desc.astype(np.float32)


def _inliers(pa: np.ndarray, da: np.ndarray, pb: np.ndarray, db: np.ndarray, cfg: SiftCfg) -> int:
    if len(pa) < 4 or len(pb) < 4:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.ratio * p[1].distance]
    if len(good) < 4:
        return 0
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    _, flags = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=cfg.ransac_px)
    return 0 if flags is None else int(flags.sum())


def match_sift(a: dict, b: dict, cfg: SiftCfg) -> float:
    """Điểm [0, 1]; lấy min hai chiều nên đối xứng."""
    pa, da = _decode(a)
    pb, db = _decode(b)
    n = min(_inliers(pa, da, pb, db, cfg), _inliers(pb, db, pa, da, cfg))
    return n / (n + cfg.k)
