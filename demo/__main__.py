import argparse
import logging
import sys
import tkinter as tk

from fpsys.config import load_config

from .app import App


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demo",
                                 description="Giao diện thử nghiệm vân tay không tiếp xúc")
    ap.add_argument("--camera", default="0",
                    help="chỉ số webcam (0, 1…) hoặc URL luồng video, ví dụ "
                         "http://192.168.1.5:8080/video")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if sys.platform == "win32":
        try:  # chữ nét trên màn hình độ phân giải cao
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    App(root, load_config(args.config, args.set), args.camera)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
