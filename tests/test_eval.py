"""Test cho eval/: datasets, splits, make_pairs, metrics, run_eval (dữ liệu tổng hợp)."""

import json

import cv2
import numpy as np
import pytest
from synth import synthetic_fingerprint

from eval import metrics as M
from eval.datasets import load_fvc, load_own, parse_own_name
from eval.make_pairs import fvc_pairs, own_pairs
from eval.splits import load_or_create_split, make_split


def test_parse_own_name():
    assert parse_own_name("S03_R2_s1_c_04.jpg") == ("S03", "R2", "s1", "c", 4)
    with pytest.raises(ValueError):
        parse_own_name("S03_R2_s1_x_04.jpg")


def test_load_own_rejects_bad_names(tmp_path):
    (tmp_path / "S01_R2_s1_c_01.jpg").write_bytes(b"x")
    (tmp_path / "bad.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="bad.jpg"):
        load_own(tmp_path)


def _own_records(tmp_path, n_subj=4):
    for s in range(1, n_subj + 1):
        for f in ("R2", "L2"):
            for sess, cond, k in (("s1", "c", 3), ("s2", "c", 2), ("s2", "u", 1)):
                for i in range(1, k + 1):
                    (tmp_path / f"S{s:02d}_{f}_{sess}_{cond}_{i:02d}.jpg").write_bytes(b"x")
    return load_own(tmp_path)


def test_own_pairs_counts(tmp_path):
    recs = _own_records(tmp_path)
    pairs = own_pairs(recs, "c")
    n_id = 8
    gen = [p for p in pairs if p.label == 1]
    imp = [p for p in pairs if p.label == 0]
    assert len(gen) == n_id * 2 * 3
    assert len(imp) == n_id * 2 * (n_id - 1) * 3
    same = [p for p in imp if p.same_subject]
    assert len(same) == n_id * 2 * 3  # mỗi probe × 3 mẫu enroll của ngón còn lại
    assert all(p.probe.session == "s2" and p.ref.session == "s1" and p.ref.cond == "c"
               for p in pairs)
    assert len(own_pairs(recs, "u", {"S01", "S02"})) == 4 * 1 * 4 * 3


def test_fvc_pairs(tmp_path):
    for f in range(101, 111):
        for i in range(1, 9):
            (tmp_path / f"{f}_{i}.tif").write_bytes(b"x")
    pairs = fvc_pairs(load_fvc(tmp_path))
    assert sum(p.label for p in pairs) == 10 * 28
    assert sum(1 - p.label for p in pairs) == 45


def test_split_fixed(tmp_path):
    subs = [f"S{i:02d}" for i in range(20)]
    a = make_split(subs, 0.4, 42)
    assert a == make_split(subs, 0.4, 42)
    assert len(a["dev"]) == 8 and not set(a["dev"]) & set(a["test"])
    p = tmp_path / "own.json"
    s1 = load_or_create_split("own", subs, 0.4, 42, p)
    s2 = load_or_create_split("own", subs, 0.4, 7, p)  # đọc lại file, không chia lại
    assert s1["dev"] == s2["dev"]
    with pytest.raises(ValueError):
        load_or_create_split("own", subs + ["S99"], 0.4, 42, p)


def test_eer_simple():
    gen = np.array([0.6, 0.7, 0.8, 0.9])
    imp = np.array([0.1, 0.2, 0.3, 0.65])
    e, t = M.eer(gen, imp)
    assert e == pytest.approx(0.25, abs=1e-9)
    assert 0.6 <= t <= 0.65
    assert M.eer(np.array([0.9, 0.8]), np.array([0.1, 0.2]))[0] == pytest.approx(0.0)


def test_eer_gaussian_and_threshold_at_far():
    rng = np.random.default_rng(0)
    gen = rng.normal(2.0, 1.0, 20000)
    imp = rng.normal(0.0, 1.0, 20000)
    e, _ = M.eer(gen, imp)
    assert e == pytest.approx(0.1587, abs=0.01)  # Φ(−1)
    t, far, frr = M.threshold_at_far(gen, imp, 0.01)
    assert far <= 0.01
    assert t == pytest.approx(2.326, abs=0.06)
    assert frr == pytest.approx(0.6273, abs=0.03)  # Φ(0.326)


def test_bootstrap_ci_contains_eer():
    rng = np.random.default_rng(1)
    subj = [f"S{i}" for i in range(20)]
    gen_s = [subj[i % 20] for i in range(400)]
    imp_s = [subj[i % 20] for i in range(4000)]
    gen = rng.normal(1.5, 1, 400)
    imp = rng.normal(0, 1, 4000)
    e, _ = M.eer(gen, imp)
    lo, hi = M.bootstrap_eer_ci(gen, imp, gen_s, imp_s, 200, 0)
    assert lo <= e <= hi and hi - lo < 0.2


def test_fta_stats():
    s = M.fta_stats({"a": None, "b": "BLURRY", "c": "BLURRY", "d": "NO_FINGER"})
    assert s["fta"] == 0.75 and s["by_code"]["BLURRY"] == 0.5


def test_run_eval_fvc_synthetic(tmp_path, monkeypatch):
    """Chạy run_eval đầu-cuối trên bộ 'FVC' tổng hợp nhỏ (3 ngón × 3 mẫu)."""
    import eval.run_eval as R

    data = tmp_path / "fvc"
    data.mkdir()
    for f in range(101, 104):
        for i in range(1, 4):
            img = synthetic_fingerprint(seed=f, noise=5 + 2 * i)
            cv2.imwrite(str(data / f"{f}_{i}.png"), img)
    monkeypatch.setattr(R, "resolve_path", lambda p: tmp_path / p)
    rc = R.main(["--dataset", "fvc", "--data-dir", str(data), "--run-name", "fvc_test",
                 "--set", "eval.n_bootstrap=20"])
    assert rc == 0
    out = tmp_path / "results" / "fvc_test"
    for name in ("scores.csv", "metrics.json", "det.png", "hist.png", "fta.csv", "config.yaml"):
        assert (out / name).exists(), name
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    tl = m["template_level"]
    assert tl["n_genuine"] == 9 and tl["n_impostor"] == 3
    assert tl["eer"] == pytest.approx(0.0)
    assert "threshold_at_far" in tl and "eer_ci95" in tl
