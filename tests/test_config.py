import pytest

from fpsys.config import Config, apply_overrides, cfg_hash, load_config


def test_default_yaml_matches_dataclass_defaults():
    assert load_config() == Config()


def test_overrides_cast_types():
    cfg = load_config(overrides=["enhance.invert=false", "enhance.n_angles=8",
                                 "verify.threshold=0.3", "matcher.method=hough",
                                 "matcher.hough.scales=[1.0, 1.1]", "enhance.clahe.tile=4",
                                 "normalize.target_period=10"])
    assert cfg.enhance.invert is False
    assert cfg.enhance.n_angles == 8 and isinstance(cfg.enhance.n_angles, int)
    assert cfg.verify.threshold == pytest.approx(0.3)
    assert cfg.matcher.method == "hough"
    assert cfg.matcher.hough.scales == [1.0, 1.1]
    assert cfg.enhance.clahe.tile == 4
    assert isinstance(cfg.normalize.target_period, float)


def test_override_errors():
    cfg = Config()
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["enhance.nope=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["enhance.invert=maybe"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["enhance.invert"])
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["enhance.clahe=1"])


def test_overrides_do_not_mutate_input():
    cfg = Config()
    apply_overrides(cfg, ["enhance.invert=false"])
    assert cfg.enhance.invert is True


def test_cfg_hash_sensitivity():
    base = cfg_hash(Config())
    assert len(base) == 8
    assert cfg_hash(load_config(overrides=["enhance.invert=false"])) != base
    assert cfg_hash(load_config(overrides=["verify.threshold=0.5"])) == base
    assert cfg_hash(load_config(overrides=["matcher.method=hough"])) == base
    assert cfg_hash(load_config(overrides=["fusion.model=x.onnx"])) == base
    assert cfg_hash(load_config(overrides=["fusion.enabled=true"])) != base
