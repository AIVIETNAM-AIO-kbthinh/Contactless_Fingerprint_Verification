"""CLI: python -m fpsys {extract,enroll,verify,match,viz} ...

Mã thoát: 0 chấp nhận / thành công, 1 từ chối, 2 CaptureError.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from .config import Config, load_config
from .decision import verify as _verify
from .io import load_image
from .matcher import match
from .pipeline import extract, run, save_steps
from .store import TemplateStore
from .types import CaptureError, Template
from .viz import grid

EXIT_OK, EXIT_REJECT, EXIT_CAPTURE = 0, 1, 2


def _cfg(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.set)


def _extract_path(path: str, cfg: Config, debug_dir: str | None = None) -> Template:
    return extract(load_image(path), cfg, debug_dir=debug_dir)


def _capture_fail(path: str, e: CaptureError) -> int:
    print(f"{path}: {e.code.value} {e.detail}")
    return EXIT_CAPTURE


def cmd_extract(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    try:
        t = _extract_path(args.image, cfg, args.debug_dir)
    except CaptureError as e:
        return _capture_fail(args.image, e)
    if args.out:
        Path(args.out).write_text(json.dumps(t.to_dict()), encoding="utf-8")
    info = {"n_minutiae": len(t.minutiae), "shape": t.shape,
            "ridge_period": round(t.ridge_period, 2), **{k: t.meta[k] for k in
                                                         ("cfg_hash", "conf", "scale",
                                                          "angle_deg", "timing_ms")}}
    print(json.dumps(info, ensure_ascii=False))
    return EXIT_OK


def cmd_enroll(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    templates = []
    for p in args.images:
        try:
            templates.append(_extract_path(p, cfg))
        except CaptureError as e:
            _capture_fail(p, e)
    store = TemplateStore(args.store, cfg)
    try:
        kept = store.enroll(args.user, args.finger, templates)
    except CaptureError as e:
        print(f"enroll thất bại: {e.code.value} {e.detail}")
        return EXIT_CAPTURE
    print(f"Đã enroll {args.user}/{args.finger}: {len(kept)}/{len(args.images)} mẫu")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    try:
        probe = _extract_path(args.image, cfg)
    except CaptureError as e:
        return _capture_fail(args.image, e)
    refs = TemplateStore(args.store, cfg).get(args.user, args.finger)
    if not refs:
        print(f"Chưa enroll {args.user}/{args.finger}")
        return EXIT_REJECT
    res = _verify(probe, refs, cfg)
    print(json.dumps({"accept": res.accept, "score": round(res.score, 4),
                      "threshold": res.threshold,
                      "per_template": [round(s, 4) for s in res.per_template]}))
    return EXIT_OK if res.accept else EXIT_REJECT


def cmd_match(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ts = []
    for p in (args.image_a, args.image_b):
        try:
            ts.append(_extract_path(p, cfg))
        except CaptureError as e:
            return _capture_fail(p, e)
    score = match(ts[0], ts[1], cfg)
    print(f"{score:.4f}")
    return EXIT_OK


def cmd_viz(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    tr = run(load_image(args.image), cfg, collect=True)
    if tr.error is not None:
        footer = f"{tr.error.code.value}: {tr.error.detail}"
    else:
        assert tr.template is not None
        footer = (f"{len(tr.template.minutiae)} minutiae | period {tr.template.ridge_period:.1f}"
                  f" | {tr.timing_ms.get('total', 0):.0f} ms")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid(tr.steps, footer=footer))
    if args.debug_dir:
        save_steps(tr.steps, args.debug_dir)
    print(f"{out}: {footer}")
    return EXIT_CAPTURE if tr.error is not None else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML (mặc định configs/default.yaml)")
    common.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="ghi đè config, ví dụ enhance.invert=false")
    common.add_argument("-v", "--verbose", action="store_true")

    p = argparse.ArgumentParser(prog="fpsys", description="Xác thực vân tay không tiếp xúc")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("extract", parents=[common], help="trích template từ một ảnh")
    s.add_argument("image")
    s.add_argument("--debug-dir", default=None)
    s.add_argument("--out", default=None, help="ghi template JSON")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("enroll", parents=[common], help="enroll một ngón từ nhiều ảnh")
    s.add_argument("--user", required=True)
    s.add_argument("--finger", required=True)
    s.add_argument("--store", default=None)
    s.add_argument("images", nargs="+")
    s.set_defaults(func=cmd_enroll)

    s = sub.add_parser("verify", parents=[common], help="xác thực một ảnh với mẫu đã enroll")
    s.add_argument("--user", required=True)
    s.add_argument("--finger", required=True)
    s.add_argument("--store", default=None)
    s.add_argument("image")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("match", parents=[common], help="in điểm so khớp hai ảnh")
    s.add_argument("image_a")
    s.add_argument("image_b")
    s.set_defaults(func=cmd_match)

    s = sub.add_parser("viz", parents=[common], help="lưới 3×3 ảnh 9 bước")
    s.add_argument("image")
    s.add_argument("--out", required=True)
    s.add_argument("--debug-dir", default=None)
    s.set_defaults(func=cmd_viz)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    return int(args.func(args))
