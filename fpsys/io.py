"""Đọc ảnh có xử lý xoay EXIF."""

from __future__ import annotations

import io as _io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # ảnh HEIC/HEIF của iPhone (tùy chọn: pip install pillow-heif)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


def load_image(src: str | Path | bytes) -> np.ndarray:
    """Đọc ảnh từ đường dẫn hoặc bytes → BGR uint8, đã áp xoay EXIF.

    Không dùng cv2.imdecode vì nó bỏ qua EXIF orientation (ảnh điện thoại bị nằm ngang).
    """
    if isinstance(src, (bytes, bytearray, memoryview)):
        img = Image.open(_io.BytesIO(bytes(src)))
    else:
        img = Image.open(src)
    with img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_gray(image: np.ndarray, channel: str = "gray") -> np.ndarray:
    """BGR/xám → ảnh xám uint8 theo `channel` ("gray", "green" hoặc "blue")."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if channel == "green":
        return np.ascontiguousarray(image[:, :, 1])
    if channel == "blue":
        return np.ascontiguousarray(image[:, :, 0])
    if channel != "gray":
        raise ValueError(f"enhance.channel không hợp lệ: {channel!r}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
