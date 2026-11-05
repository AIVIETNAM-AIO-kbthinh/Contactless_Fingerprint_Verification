"""Hợp đồng dữ liệu của fpsys (mục 5 của plan). Không đổi chữ ký nếu chưa thống nhất với backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ErrorCode(str, Enum):
    NO_FINGER = "NO_FINGER"  # YOLO không thấy đầu ngón (kể cả sau TTA)
    BLURRY = "BLURRY"  # năng lượng dải tần vân dưới ngưỡng / không ước lượng được chu kỳ
    BAD_EXPOSURE = "BAD_EXPOSURE"  # quá tối / quá sáng / quá nhiều pixel bão hòa
    LOW_QUALITY = "LOW_QUALITY"  # vùng vân hợp lệ quá nhỏ sau enhance
    TOO_FEW_MINUTIAE = "TOO_FEW_MINUTIAE"
    INCONSISTENT_ENROLL = "INCONSISTENT_ENROLL"  # các mẫu enroll không khớp nhau


class CaptureError(Exception):
    def __init__(self, code: ErrorCode, detail: str = ""):
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


class TemplateMismatchError(ValueError):
    """Hai template sinh từ cấu hình trích xuất khác nhau (cfg_hash khác)."""


@dataclass
class Minutia:
    x: float  # px, hệ tọa độ ảnh đã chuẩn hóa, gốc trên-trái, y hướng xuống
    y: float
    theta: float  # radian, [0, 2π), atan2(dy, dx) trong hệ tọa độ ảnh
    kind: Literal["ending", "bifurcation"]
    quality: float  # 0..1

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "theta": self.theta, "kind": self.kind,
                "quality": self.quality}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Minutia:
        return cls(x=float(d["x"]), y=float(d["y"]), theta=float(d["theta"]),
                   kind=d["kind"], quality=float(d["quality"]))


@dataclass
class Template:
    minutiae: list[Minutia]
    shape: tuple[int, int]  # (h, w) ảnh đã chuẩn hóa
    ridge_period: float  # px sau chuẩn hóa, kỳ vọng ≈ 9
    version: str = "fp-v1"
    embedding: list[float] | None = None  # M8; None khi không bật fusion
    meta: dict = field(default_factory=dict)  # cfg_hash, conf YOLO, quality, scale, angle...

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "ridge_period": float(self.ridge_period),
            "minutiae": [m.to_dict() for m in self.minutiae],
            "embedding": None if self.embedding is None else [float(v) for v in self.embedding],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Template:
        emb = d.get("embedding")
        return cls(
            minutiae=[Minutia.from_dict(m) for m in d["minutiae"]],
            shape=(int(d["shape"][0]), int(d["shape"][1])),
            ridge_period=float(d["ridge_period"]),
            version=d.get("version", "fp-v1"),
            embedding=None if emb is None else [float(v) for v in emb],
            meta=dict(d.get("meta", {})),
        )


@dataclass
class VerifyResult:
    score: float
    threshold: float
    accept: bool
    per_template: list[float]  # điểm với từng mẫu đã enroll
