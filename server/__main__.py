"""python -m server [--host 0.0.0.0] [--port 8000] [--db data/server.db] [--set a.b=c]

Cần biến môi trường FPSYS_API_KEY và FPSYS_DB_KEY (khóa Fernet).
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from fpsys import load_config

from .app import create_app
from .db import SqliteStore


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default="data/server.db")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=c")
    args = ap.parse_args()
    api_key, db_key = os.environ.get("FPSYS_API_KEY"), os.environ.get("FPSYS_DB_KEY")
    if not api_key or not db_key:
        # không tự sinh khóa: khóa DB mới sẽ không giải mã được template đã lưu
        sys.exit("Thiếu FPSYS_API_KEY hoặc FPSYS_DB_KEY. Tạo khóa DB: python -c \"from "
                 "cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    store = SqliteStore(args.db, db_key, load_config(overrides=args.set))
    uvicorn.run(create_app(store, api_key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
