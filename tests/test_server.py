"""Test backend FastAPI bằng TestClient, ảnh tổng hợp, không cần YOLO thật."""

import sqlite3

import cv2
import numpy as np
import pytest
from synth import synthetic_fingerprint, to_bgr

pytest.importorskip("fastapi")
from cryptography.fernet import Fernet  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fpsys import detect as det_mod  # noqa: E402
from fpsys.config import load_config  # noqa: E402
from server.app import create_app  # noqa: E402
from server.db import SqliteStore  # noqa: E402

KEY = {"X-API-Key": "k"}


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _client(tmp_path, source="contact"):
    cfg = load_config(overrides=[f"pipeline.source={source}"])
    store = SqliteStore(tmp_path / "s.db", Fernet.generate_key(), cfg)
    return TestClient(create_app(store, "k")), store


def _finger(seed=5, noise=6):
    return ("f.png", _png(to_bgr(synthetic_fingerprint(seed=seed, noise=noise))), "image/png")


def test_api_key_required(tmp_path):
    c, _ = _client(tmp_path)
    assert c.get("/users").status_code == 401
    assert c.get("/users", headers={"X-API-Key": "sai"}).status_code == 401
    assert c.get("/users", headers=KEY).json() == []


def test_enroll_verify_delete(tmp_path):
    c, store = _client(tmp_path)
    url = "/users/Nguyễn Văn A/fingers/R2"
    imgs = [("images", _finger(noise=n)) for n in (6, 8, 10)]
    r = c.post(f"{url}/enroll", files=imgs, headers=KEY)
    assert r.status_code == 200 and r.json()["total"] == 3
    assert c.get("/users", headers=KEY).json() == [{"name": "Nguyễn Văn A", "fingers": ["R2"]}]

    assert c.post(f"{url}/verify", files={"image": _finger()}, headers=KEY).json()["accept"]
    other = c.post(f"{url}/verify", files={"image": _finger(seed=9)}, headers=KEY).json()
    assert not other["accept"]

    blobs = sqlite3.connect(store.path).execute("SELECT blob FROM templates").fetchall()
    assert blobs and all(b"minutiae" not in b for (b,) in blobs)  # đã mã hóa

    assert c.delete("/users/Nguyễn Văn A", headers=KEY).status_code == 204
    assert c.post(f"{url}/verify", files={"image": _finger()}, headers=KEY).status_code == 404


def test_inconsistent_enroll_and_bad_input(tmp_path):
    c, _ = _client(tmp_path)
    imgs = [("images", _finger(seed=k)) for k in (1, 2, 3)]
    r = c.post("/users/A/fingers/R2/enroll", files=imgs, headers=KEY)
    assert r.status_code == 422 and r.json()["code"] == "INCONSISTENT_ENROLL"
    assert c.post("/users/A/fingers/X9/verify", files={"image": _finger()},
                  headers=KEY).status_code == 400
    bad = {"image": ("x.jpg", b"not an image", "image/jpeg")}
    assert c.post("/check", files=bad, headers=KEY).status_code == 400


def test_check_no_finger(tmp_path, monkeypatch):
    monkeypatch.setattr(det_mod, "_run_model", lambda im, cfg: np.zeros((0, 5), np.float32))
    c, _ = _client(tmp_path, source="contactless")
    blank = {"image": ("b.png", _png(np.full((480, 640, 3), 30, np.uint8)), "image/png")}
    r = c.post("/check", files=blank, headers=KEY).json()
    assert r["ok"] is False and r["code"] == "NO_FINGER" and r["message"]
