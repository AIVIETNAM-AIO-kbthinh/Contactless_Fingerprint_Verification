import pytest
from synth import perturb_template, random_template

from fpsys.store import TemplateStore


def test_enroll_get_verify_delete(tmp_path, cfg):
    store = TemplateStore(tmp_path / "store.json", cfg)
    base = random_template(seed=3)
    ts = [perturb_template(base, seed=i, add_frac=0.05, drop_frac=0.05, jitter=1.5)
          for i in range(3)]
    kept = store.enroll("U01", "R2", ts)
    assert len(kept) == 3
    got = store.get("U01", "R2")
    assert got == kept
    assert store.verify("U01", "R2", base).accept
    assert not store.verify("U01", "R2", random_template(seed=42)).accept
    assert store.get("U01", "L2") == []
    with pytest.raises(KeyError):
        store.verify("U02", "R2", base)
    assert store.delete("U01") and not store.delete("U01")
    assert store.get("U01", "R2") == []
