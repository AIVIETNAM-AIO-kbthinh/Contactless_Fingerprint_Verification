"""API FastAPI cho app điện thoại: kiểm tra ảnh, đăng ký, xác thực, quản lý người dùng."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError

import fpsys
from demo.session import FINGERS, HINTS

from .db import SqliteStore

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def create_app(store: SqliteStore, api_key: str) -> FastAPI:
    cfg = store.cfg

    def require_key(x_api_key: Annotated[str, Header()] = "") -> None:
        if not secrets.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(401, "Sai hoặc thiếu X-API-Key")

    app = FastAPI(title="fpsys", dependencies=[Depends(require_key)])

    @app.exception_handler(fpsys.CaptureError)
    def capture_error(_req, e: fpsys.CaptureError) -> JSONResponse:
        return JSONResponse({"code": e.code.value, "message": HINTS.get(e.code, str(e))}, 422)

    @app.exception_handler(fpsys.TemplateMismatchError)
    def mismatch(_req, e: fpsys.TemplateMismatchError) -> JSONResponse:
        return JSONResponse({"code": "TEMPLATE_MISMATCH",
                             "message": "Mẫu cũ không còn tương thích, cần đăng ký lại."}, 409)

    def extract(f: UploadFile) -> fpsys.Template:
        data = f.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Ảnh lớn hơn 20 MB")
        try:
            img = fpsys.load_image(data)
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(400, "Không đọc được ảnh") from e
        return fpsys.extract(img, cfg)

    def check_finger(finger: str) -> None:
        if finger not in FINGERS:
            raise HTTPException(400, f"Mã ngón không hợp lệ, dùng một trong {FINGERS}")

    @app.post("/check")
    def check(image: Annotated[UploadFile, File()]) -> dict:
        try:
            t = extract(image)
        except fpsys.CaptureError as e:
            return {"ok": False, "code": e.code.value, "message": HINTS.get(e.code, str(e))}
        return {"ok": True, "n_minutiae": len(t.minutiae)}

    @app.post("/users/{name}/fingers/{finger}/enroll")
    def enroll(name: str, finger: str, images: Annotated[list[UploadFile], File()]) -> dict:
        check_finger(finger)
        kept = store.enroll(name, finger, [extract(f) for f in images])
        return {"kept": len(kept), "total": len(images)}

    @app.post("/users/{name}/fingers/{finger}/verify")
    def verify(name: str, finger: str, image: Annotated[UploadFile, File()]) -> dict:
        check_finger(finger)
        if not store.get(name, finger):
            raise HTTPException(404, f"{name} chưa đăng ký ngón {finger}")
        r = store.verify(name, finger, extract(image))
        return {"score": r.score, "threshold": r.threshold, "accept": r.accept}

    @app.get("/users")
    def users() -> list[dict]:
        return store.users()

    @app.delete("/users/{name}", status_code=204)
    def delete(name: str) -> Response:
        if not store.delete(name):
            raise HTTPException(404, f"Không có người dùng {name}")
        return Response(status_code=204)

    return app
