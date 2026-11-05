import numpy as np
from synth import synthetic_fingerprint

from fpsys.config import load_config
from fpsys.skeleton import crossing_number, prune_spurs, skeleton_from_binary


def _has_2x2(sk: np.ndarray) -> bool:
    s = sk.astype(np.uint8)
    blk = s[:-1, :-1] + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:]
    return bool((blk == 4).any())


def test_skeleton_is_one_pixel_wide(cfg):
    for seed in range(3):
        img = synthetic_fingerprint(seed=seed, noise=20)
        binary = img < 128
        sk = skeleton_from_binary(binary, cfg.skeleton)
        assert sk.any()
        assert not _has_2x2(sk)


def test_spur_is_pruned():
    sk = np.zeros((100, 120), bool)
    sk[50, 10:91] = True  # đường thẳng
    sk[45:50, 50] = True  # gai 5 px
    out = prune_spurs(sk, 10)
    assert not out[45:50, 50].any()
    assert out[50, 10:91].all()
    cn = crossing_number(out)
    assert (cn == 3).sum() == 0 and (cn == 1).sum() == 2


def test_small_segments_removed():
    cfg = load_config()
    b = np.zeros((100, 100), bool)
    b[20:24, 10:90] = True  # vân dài
    b[60:63, 40:48] = True  # mảnh nhỏ (< min_component_px)
    sk = skeleton_from_binary(b, cfg.skeleton)
    assert sk[15:30].any() and not sk[55:70].any()
