"""Logic đăng ký / xác thực của demo, tách khỏi giao diện để test được."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from fpsys.config import Config, resolve_path
from fpsys.pipeline import PipelineTrace, run
from fpsys.store import TemplateStore
from fpsys.types import ErrorCode, Template, VerifyResult

FINGERS = ["R1", "R2", "R3", "R4", "R5", "L1", "L2", "L3", "L4", "L5"]
FINGER_NAMES = {"1": "cái", "2": "trỏ", "3": "giữa", "4": "áp út", "5": "út"}

HINTS = {
    ErrorCode.NO_FINGER: "Không thấy đầu ngón tay. Giơ một ngón, mặt bụng ngón hướng vào camera.",
    ErrorCode.BLURRY: "Ảnh mờ, không thấy vân. Giữ yên tay, đưa gần hơn và lấy nét vào đầu ngón.",
    ErrorCode.BAD_EXPOSURE: "Ánh sáng không phù hợp (quá tối, quá sáng hoặc da bị cháy sáng). "
                            "Chỉnh đèn rồi chụp lại.",
    ErrorCode.LOW_QUALITY: "Vùng vân rõ quá nhỏ. Đưa đầu ngón vào giữa khung, gần hơn.",
    ErrorCode.TOO_FEW_MINUTIAE: "Trích được quá ít điểm đặc trưng. Chụp lại, giữ ngón thẳng, "
                                "rõ nét.",
    ErrorCode.INCONSISTENT_ENROLL: "Các mẫu không khớp nhau. Đăng ký lại và dùng cùng một ngón.",
}


def finger_label(code: str) -> str:
    side = "phải" if code[0] == "R" else "trái"
    return f"{code} (ngón {FINGER_NAMES.get(code[1], code[1])} {side})"


class DemoSession:
    """Danh sách người dùng (tên → mã S01, S02…), kho template, lưu ảnh theo quy ước đặt tên."""

    def __init__(self, cfg: Config, data_dir: str | Path | None = None,
                 save_dir: str | Path | None = None):
        self.cfg = cfg
        self.data_dir = Path(data_dir) if data_dir else resolve_path(cfg.guide.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "users.json"
        self.store = TemplateStore(self.data_dir / "store.json", cfg)
        self.save_dir = Path(save_dir) if save_dir else resolve_path("data/own")

    # --- người dùng -------------------------------------------------------
    def _load(self) -> dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save(self, reg: dict[str, dict]) -> None:
        self.registry_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    def users(self) -> list[str]:
        return sorted(self._load())

    def subject_id(self, name: str, create: bool = False) -> str:
        """Mã subject dùng trong tên file (S01, S02…); tạo mới nếu create=True."""
        name = name.strip()
        if not name:
            raise ValueError("Tên trống")
        reg = self._load()
        if name in reg:
            return reg[name]["subject"]
        if not create:
            raise KeyError(f"Chưa đăng ký: {name}")
        used = {int(v["subject"][1:]) for v in reg.values()}
        if self.save_dir.exists():  # tránh trùng mã với ảnh đã có trong data/own
            for p in self.save_dir.iterdir():
                m = re.match(r"S(\d+)_", p.name)
                if m:
                    used.add(int(m.group(1)))
        sid = f"S{(max(used) + 1) if used else 1:02d}"
        reg[name] = {"subject": sid, "created": datetime.now().isoformat(timespec="seconds"),
                     "fingers": []}
        self._save(reg)
        return sid

    def enrolled_fingers(self, name: str) -> list[str]:
        return list(self._load().get(name, {}).get("fingers", []))

    def delete(self, name: str) -> None:
        reg = self._load()
        if name in reg:
            del reg[name]
            self._save(reg)
        self.store.delete(name)

    # --- xử lý ảnh --------------------------------------------------------
    def extract(self, image: np.ndarray) -> PipelineTrace:
        return run(image, self.cfg, collect=True)

    def enroll(self, name: str, finger: str, templates: list[Template]) -> list[Template]:
        """Raise CaptureError(INCONSISTENT_ENROLL) nếu các mẫu không nhất quán."""
        self.subject_id(name, create=True)
        kept = self.store.enroll(name, finger, templates)
        reg = self._load()
        fingers = set(reg[name].get("fingers", [])) | {finger}
        reg[name]["fingers"] = sorted(fingers)
        self._save(reg)
        return kept

    def verify(self, name: str, finger: str, probe: Template) -> VerifyResult:
        return self.store.verify(name, finger, probe)

    def save_image(self, image: np.ndarray, name: str, finger: str, session: str = "s1",
                   cond: str = "c") -> Path:
        """Lưu ảnh vào data/own theo quy ước {subject}_{finger}_{session}_{cond}_{sample}.jpg."""
        sid = self.subject_id(name, create=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{sid}_{finger}_{session}_{cond}_"
        nums = [int(p.stem[len(prefix):]) for p in self.save_dir.glob(prefix + "*.jpg")
                if p.stem[len(prefix):].isdigit()]
        path = self.save_dir / f"{prefix}{(max(nums) + 1) if nums else 1:02d}.jpg"
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 97])
        if not ok:
            raise OSError("Không mã hóa được ảnh JPEG")
        path.write_bytes(buf.tobytes())
        return path
