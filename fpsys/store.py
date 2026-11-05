"""Lưu template cục bộ dạng JSON (thử nghiệm, CLI, eval). Backend thật thay bằng SQLite mã hóa."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from .config import Config, default_config, resolve_path
from .decision import select_enrollment
from .decision import verify as _verify
from .types import Template, VerifyResult

log = logging.getLogger(__name__)


class TemplateStore:
    """File JSON dạng {user_id: {finger: [template_dict, ...]}}."""

    def __init__(self, path: str | Path | None = None, cfg: Config | None = None):
        self.cfg = cfg or default_config()
        self.path = Path(path) if path is not None else resolve_path(self.cfg.store.path)

    def _load(self) -> dict[str, dict[str, list[dict]]]:
        if not self.path.exists():
            return {}
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: dict[str, dict[str, list[dict]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def enroll(self, user_id: str, finger: str, templates: list[Template]) -> list[Template]:
        """Lọc mẫu không nhất quán (select_enrollment) rồi lưu, thay thế mẫu cũ của ngón này."""
        kept = select_enrollment(templates, self.cfg)
        data = self._load()
        data.setdefault(user_id, {})[finger] = [t.to_dict() for t in kept]
        self._save(data)
        log.info("Enroll %s/%s: lưu %d/%d mẫu", user_id, finger, len(kept), len(templates))
        return kept

    def get(self, user_id: str, finger: str) -> list[Template]:
        data = self._load()
        return [Template.from_dict(d) for d in data.get(user_id, {}).get(finger, [])]

    def delete(self, user_id: str) -> bool:
        data = self._load()
        if user_id not in data:
            return False
        del data[user_id]
        self._save(data)
        return True

    def verify(self, user_id: str, finger: str, probe: Template) -> VerifyResult:
        refs = self.get(user_id, finger)
        if not refs:
            raise KeyError(f"Chưa enroll: {user_id}/{finger}")
        return _verify(probe, refs, self.cfg)
