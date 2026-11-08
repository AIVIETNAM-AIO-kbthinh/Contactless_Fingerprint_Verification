import pytest
from synth import perturb_template, random_template, transform_template

from fpsys.config import apply_overrides, load_config
from fpsys.decision import fuse_scores, select_enrollment, verify
from fpsys.types import CaptureError, ErrorCode


def _genuine_set(seed=0, n=3):
    base = random_template(seed=seed)
    return [perturb_template(transform_template(base, 5 * i, 3 * i, -2 * i), seed=seed + i,
                             add_frac=0.1, drop_frac=0.1, jitter=2.0) for i in range(n)]


def test_select_enrollment_removes_outlier(cfg):
    genuine = _genuine_set()
    outlier = random_template(seed=999)
    templates = genuine[:2] + [outlier] + genuine[2:]
    kept = select_enrollment(templates, cfg)
    assert len(kept) == 3
    assert all(k is not outlier for k in kept)


def test_select_enrollment_inconsistent(cfg):
    with pytest.raises(CaptureError) as ei:
        select_enrollment([random_template(seed=s) for s in range(3)], cfg)
    assert ei.value.code is ErrorCode.INCONSISTENT_ENROLL
    with pytest.raises(CaptureError):
        select_enrollment([random_template(seed=0)], cfg)


def test_select_enrollment_max_giu_mau_chi_khop_mot_mau(cfg):
    a, b = _genuine_set(n=2)
    outliers = [random_template(seed=s) for s in (901, 902)]
    with pytest.raises(CaptureError):
        select_enrollment([a, b, *outliers], cfg)  # median: a, b đều bị 2 mẫu lạ kéo xuống
    kept = select_enrollment([a, b, *outliers], apply_overrides(cfg, ["enroll.consistency=max"]))
    assert kept == [a, b]


def test_select_enrollment_consistency_la(cfg):
    with pytest.raises(ValueError):
        select_enrollment(_genuine_set(), apply_overrides(cfg, ["enroll.consistency=x"]))


def test_verify_accepts_genuine_rejects_impostor(cfg):
    refs = _genuine_set()
    probe = perturb_template(refs[0], seed=77, add_frac=0.1, drop_frac=0.1, jitter=2.0)
    res = verify(probe, refs, cfg)
    assert res.accept and len(res.per_template) == 3
    assert res.score == max(res.per_template)
    assert not verify(random_template(seed=555), refs, cfg).accept


def test_fusion_modes():
    assert fuse_scores([0.1, 0.5, 0.3], "max") == 0.5
    assert fuse_scores([0.1, 0.5, 0.3], "mean_top2") == pytest.approx(0.4)
    assert fuse_scores([0.2], "mean_top2") == pytest.approx(0.2)
    cfg = load_config(overrides=["verify.fusion=mean_top2"])
    refs = _genuine_set()
    res = verify(refs[0], refs, cfg)
    assert res.score == pytest.approx(sum(sorted(res.per_template)[-2:]) / 2)
