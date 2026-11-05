import cv2
import numpy as np
from synth import synthetic_fingerprint

from fpsys.cli import main

CONTACT = ["--set", "pipeline.source=contact"]


def _write(tmp_path, name, img):
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


def test_cli_workflow(tmp_path, capsys):
    imgs = [_write(tmp_path, f"a{i}.png", synthetic_fingerprint(seed=5, noise=6 + 2 * i))
            for i in range(3)]
    other = _write(tmp_path, "b.png", synthetic_fingerprint(seed=11))
    store = str(tmp_path / "store.json")

    assert main(["extract", imgs[0], "--out", str(tmp_path / "t.json"), *CONTACT]) == 0
    assert (tmp_path / "t.json").exists()
    assert main(["enroll", "--user", "U01", "--finger", "R2", "--store", store, *imgs,
                 *CONTACT]) == 0
    assert main(["verify", "--user", "U01", "--finger", "R2", "--store", store, imgs[0],
                 *CONTACT]) == 0
    assert main(["verify", "--user", "U01", "--finger", "R2", "--store", store, other,
                 *CONTACT]) == 1
    assert main(["match", imgs[0], imgs[1], *CONTACT]) == 0
    out = tmp_path / "viz.png"
    assert main(["viz", imgs[0], "--out", str(out), *CONTACT]) == 0
    assert cv2.imread(str(out)) is not None
    capsys.readouterr()


def test_cli_capture_error_exit_code(tmp_path, capsys):
    flat = _write(tmp_path, "flat.png", np.full((300, 300), 128, np.uint8))
    assert main(["extract", flat, *CONTACT]) == 2
    assert "LOW_QUALITY" in capsys.readouterr().out
