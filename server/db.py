"""Kho template SQLite, mã hóa Fernet. Cùng giao diện với fpsys.store.TemplateStore."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet

from fpsys import Config, Template, VerifyResult, select_enrollment
from fpsys import verify as _verify

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS templates (
    name TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    finger TEXT NOT NULL,
    idx INTEGER NOT NULL,
    blob BLOB NOT NULL,
    PRIMARY KEY (name, finger, idx)
);
"""


class SqliteStore:
    """Bảng users + templates; mỗi template là JSON của Template.to_dict() đã mã hóa."""

    def __init__(self, path: str | Path, key: str | bytes, cfg: Config):
        self.cfg = cfg
        self.fernet = Fernet(key)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        # mỗi lần gọi một kết nối: FastAPI chạy endpoint `def` trên nhiều thread
        c = sqlite3.connect(self.path)
        c.execute("PRAGMA foreign_keys = ON")
        return c

    def enroll(self, name: str, finger: str, templates: list[Template]) -> list[Template]:
        """Lọc mẫu không nhất quán (select_enrollment) rồi lưu, thay thế mẫu cũ của ngón này."""
        kept = select_enrollment(templates, self.cfg)
        blobs = [self.fernet.encrypt(json.dumps(t.to_dict()).encode()) for t in kept]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (name, now))
            c.execute("DELETE FROM templates WHERE name = ? AND finger = ?", (name, finger))
            c.executemany("INSERT INTO templates VALUES (?, ?, ?, ?)",
                          [(name, finger, i, b) for i, b in enumerate(blobs)])
        return kept

    def get(self, name: str, finger: str) -> list[Template]:
        with self._conn() as c:
            rows = c.execute("SELECT blob FROM templates WHERE name = ? AND finger = ? "
                             "ORDER BY idx", (name, finger)).fetchall()
        return [Template.from_dict(json.loads(self.fernet.decrypt(b))) for (b,) in rows]

    def users(self) -> list[dict]:
        with self._conn() as c:
            names = [n for (n,) in c.execute("SELECT name FROM users ORDER BY name")]
            pairs = c.execute("SELECT DISTINCT name, finger FROM templates "
                              "ORDER BY finger").fetchall()
        return [{"name": n, "fingers": [f for m, f in pairs if m == n]} for n in names]

    def delete(self, name: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM users WHERE name = ?", (name,)).rowcount > 0

    def verify(self, name: str, finger: str, probe: Template) -> VerifyResult:
        refs = self.get(name, finger)
        if not refs:
            raise KeyError(f"Chưa đăng ký: {name}/{finger}")
        return _verify(probe, refs, self.cfg)
