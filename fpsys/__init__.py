"""fpsys — lõi xác thực vân tay không tiếp xúc.

API công khai cho backend: load_config, extract, match, select_enrollment, verify.
"""

from __future__ import annotations

from .config import Config, cfg_hash, load_config
from .decision import select_enrollment, verify
from .io import load_image
from .matcher import match
from .pipeline import extract
from .types import (
    CaptureError,
    ErrorCode,
    Minutia,
    Template,
    TemplateMismatchError,
    VerifyResult,
)

__all__ = [
    "CaptureError",
    "Config",
    "ErrorCode",
    "Minutia",
    "Template",
    "TemplateMismatchError",
    "VerifyResult",
    "cfg_hash",
    "extract",
    "load_config",
    "load_image",
    "match",
    "select_enrollment",
    "verify",
]
