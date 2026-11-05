"""Đọc ảnh của các bộ dữ liệu thành bản ghi (path, subject, finger, session, cond, sample)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fpsys.config import resolve_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".heif"}

# Dữ liệu tự thu: {subject}_{finger}_{session}_{cond}_{sample}.jpg, ví dụ S03_R2_s1_c_04.jpg
OWN_RE = re.compile(r"^(S\d+)_([LR]\d)_(s\d+)_([cu])_(\d+)\.(jpe?g|png)$", re.IGNORECASE)
# FVC2002: 101_1.tif (ngón 101, lần lấy mẫu 1)
FVC_RE = re.compile(r"^(\d+)_(\d+)\.(tif|tiff|bmp|png)$", re.IGNORECASE)

DEFAULT_ROOTS = {"own": "data/own", "fvc": "data/fvc2002_db1b", "ridgebase": "data/ridgebase"}
# pipeline.source phù hợp với từng bộ dữ liệu
DATASET_SOURCE = {"own": "contactless", "fvc": "contact", "ridgebase": "contactless_crop"}


@dataclass(frozen=True)
class Record:
    path: str
    subject: str
    finger: str
    session: str
    cond: str
    sample: int

    @property
    def identity(self) -> str:
        """Một ngón của một người: đơn vị enroll/verify."""
        return f"{self.subject}_{self.finger}"

    @property
    def name(self) -> str:
        return Path(self.path).name


def parse_own_name(name: str) -> tuple[str, str, str, str, int]:
    m = OWN_RE.match(name)
    if not m:
        raise ValueError(f"Tên file không đúng định dạng S03_R2_s1_c_04.jpg: {name!r}")
    subj, finger, session, cond, sample = m.group(1, 2, 3, 4, 5)
    return subj.upper(), finger.upper(), session.lower(), cond.lower(), int(sample)


def _image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục dữ liệu: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_own(root: str | Path) -> list[Record]:
    """Dữ liệu tự thu. Tên file sai định dạng → ValueError liệt kê tất cả (không bỏ qua im lặng)."""
    records, bad = [], []
    for p in _image_files(Path(root)):
        try:
            subj, finger, session, cond, sample = parse_own_name(p.name)
        except ValueError:
            bad.append(str(p))
            continue
        records.append(Record(str(p), subj, finger, session, cond, sample))
    if bad:
        raise ValueError("Tên file không hợp lệ (cần {subject}_{finger}_{session}_{cond}_"
                         "{sample}.jpg):\n  " + "\n  ".join(bad))
    return records


def load_fvc(root: str | Path) -> list[Record]:
    """FVC2002 DB1_B: mỗi ngón (101..110) là một identity, 8 lần lấy mẫu."""
    records, bad = [], []
    for p in _image_files(Path(root)):
        m = FVC_RE.match(p.name)
        if not m:
            bad.append(str(p))
            continue
        records.append(Record(str(p), m.group(1), "F", "s1", "c", int(m.group(2))))
    if bad:
        raise ValueError("Tên file FVC không hợp lệ (cần 101_1.tif):\n  " + "\n  ".join(bad))
    return records


def load_ridgebase(root: str | Path) -> list[Record]:
    raise NotImplementedError(
        "Parser RidgeBase chưa viết: cần xem cấu trúc thư mục thực tế trong data/ridgebase/ "
        "và thống nhất với người dùng (giao thức/chia tập chính thức) trước.")


def load_dataset(name: str, root: str | Path | None = None) -> list[Record]:
    loaders = {"own": load_own, "fvc": load_fvc, "ridgebase": load_ridgebase}
    if name not in loaders:
        raise ValueError(f"Bộ dữ liệu không hỗ trợ: {name!r}")
    return loaders[name](resolve_path(str(root or DEFAULT_ROOTS[name])))
