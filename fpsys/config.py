"""Cấu hình: YAML → dataclass lồng nhau, ghi đè dạng "a.b=c", cfg_hash()."""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

# Các mục ảnh hưởng tới đặc trưng: đổi chúng thì phải enroll lại.
HASHED_SECTIONS = ("pipeline", "normalize", "enhance", "skeleton", "minutiae")


@dataclass
class PipelineCfg:
    source: str = "contactless"  # contactless | contactless_crop | contact


@dataclass
class DetectCfg:
    method: str = "yolo"  # yolo | otsu
    weights: str = "models/best.pt"
    conf: float = 0.5
    pad: float = 0.10
    imgsz: int = 640
    tie_conf_delta: float = 0.10
    tta_rotate: bool = True


@dataclass
class NormalizeCfg:
    skin_method: str = "grabcut"  # grabcut | otsu_cr | fixed_ycrcb
    mask_work_width: int = 384
    grabcut_iters: int = 5
    mask_erode_px: int = 6
    sat_level: int = 250
    sat_dilate_px: int = 5
    tip_end_frac: float = 0.15
    unwarp: bool = False
    unwarp_max_angle_deg: float = 65.0
    unwarp_smooth_rows: int = 31
    target_period: float = 9.0
    period_block: int = 64
    period_min: float = 4.0
    period_max: float = 48.0
    period_min_cycles: float = 3.0
    scale_norm: bool = True
    scale_min: float = 0.25
    scale_max: float = 3.0
    contact_block: int = 16
    contact_std_frac: float = 0.25


@dataclass
class QualityCfg:
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    max_sat_ratio: float = 0.15
    min_band_energy: float = 0.20


@dataclass
class ClaheCfg:
    clip_limit: float = 2.0
    tile: int = 8


@dataclass
class EnhanceCfg:
    method: str = "hong"  # hong | reference
    channel: str = "gray"  # gray | green | blue
    invert: bool = True
    flatten_sigma: float = 2.5
    clahe: ClaheCfg = field(default_factory=ClaheCfg)
    block: int = 16
    orient_sigma: float = 14.0
    freq_block: int = 32
    gabor_k: float = 0.65
    gabor_gamma: float = 1.0  # tỷ lệ σ vuông góc / σ dọc vân; < 1 = kernel dài theo vân
    n_angles: int = 16
    min_coherence: float = 0.3
    min_valid_area: float = 0.15


@dataclass
class SkeletonCfg:
    min_component_px: int = 30
    max_hole_px: int = 20
    spur_len_px: int = 10
    min_segment_px: int = 15


@dataclass
class MinutiaeCfg:
    border_px: int = 12
    trace_px: int = 10
    shift_frac: float = 0.5
    break_px: float = 10.0
    break_angle_deg: float = 45.0
    bridge_px: float = 8.0
    spur_px: float = 8.0
    cluster_px: float = 10.0
    cluster_max: int = 4
    min_minutiae: int = 12


@dataclass
class RelaxCfg:
    n_iter: int = 5
    w: float = 0.5
    mu1: float = 5.0
    tau1: float = -1.6
    scale_tol: float = 0.06
    support: str = "weighted"  # weighted | mean (mean = như bài báo)
    mu2_deg: float = 15.0
    tau2: float = -30.0
    mu3_deg: float = 15.0
    tau3: float = -30.0


@dataclass
class MccCfg:
    radius: float = 70.0
    ns: int = 16
    nd: int = 6
    sigma_s: float = 9.333
    sigma_d: float = 0.698
    mu_psi: float = 0.01
    tau_psi: float = 400.0
    hull_offset: float = 50.0
    min_valid_cells: float = 0.75
    min_neighbors: int = 2
    min_common_cells: float = 0.6
    max_dir_diff_deg: float = 90.0
    min_np: int = 4
    max_np: int = 12
    mu_p: float = 20.0
    tau_p: float = 0.4
    relax: RelaxCfg = field(default_factory=RelaxCfg)


@dataclass
class HoughCfg:
    theta_step_deg: float = 10.0
    xy_step_px: float = 8.0
    scales: list[float] = field(default_factory=lambda: [0.90, 0.95, 1.00, 1.05, 1.10])
    max_rotation_deg: float = 45.0
    top_k: int = 3
    match_dist_px: float = 15.0
    match_angle_deg: float = 20.0


@dataclass
class MatcherCfg:
    method: str = "mcc"  # mcc | hough
    mcc: MccCfg = field(default_factory=MccCfg)
    hough: HoughCfg = field(default_factory=HoughCfg)


@dataclass
class EnrollCfg:
    min_templates: int = 2
    min_consistency: float = 0.15


@dataclass
class VerifyCfg:
    threshold: float = 0.10
    fusion: str = "max"  # max | mean_top2


@dataclass
class FusionCfg:
    enabled: bool = False
    model: str = "models/embed.onnx"
    w: float = 0.5


@dataclass
class StoreCfg:
    path: str = "data/store.json"


@dataclass
class EvalCfg:
    dev_fraction: float = 0.4
    target_far: float = 0.01
    n_bootstrap: int = 1000


@dataclass
class GuideCfg:
    """Hướng dẫn trực tiếp khi chụp (demo/). Không ảnh hưởng đặc trưng, không thuộc cfg_hash."""

    min_tip_px: int = 300
    max_box_frac: float = 0.85
    edge_margin_frac: float = 0.02
    min_ridges_across: float = 20.0
    stable_frac: float = 0.08
    ready_frames: int = 4
    enroll_samples: int = 3
    data_dir: str = "data/demo"


@dataclass
class Config:
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    detect: DetectCfg = field(default_factory=DetectCfg)
    normalize: NormalizeCfg = field(default_factory=NormalizeCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    enhance: EnhanceCfg = field(default_factory=EnhanceCfg)
    skeleton: SkeletonCfg = field(default_factory=SkeletonCfg)
    minutiae: MinutiaeCfg = field(default_factory=MinutiaeCfg)
    matcher: MatcherCfg = field(default_factory=MatcherCfg)
    enroll: EnrollCfg = field(default_factory=EnrollCfg)
    verify: VerifyCfg = field(default_factory=VerifyCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    store: StoreCfg = field(default_factory=StoreCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    guide: GuideCfg = field(default_factory=GuideCfg)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cast(value: Any, default: Any, key: str) -> Any:
    """Ép `value` (chuỗi từ CLI hoặc giá trị YAML) theo kiểu của giá trị mặc định."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{key}: không phải bool: {value!r}")
    if isinstance(default, int):
        if isinstance(value, bool):
            raise ValueError(f"{key}: cần int, nhận bool")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{key}: cần int, nhận {value!r}")
        return int(float(value)) if isinstance(value, str) else int(value)
    if isinstance(default, float):
        if isinstance(value, bool):
            raise ValueError(f"{key}: cần float, nhận bool")
        return float(value)
    if isinstance(default, list):
        if isinstance(value, str):
            value = yaml.safe_load(value)
        if not isinstance(value, (list, tuple)):
            value = [value]
        elem = default[0] if default else None
        return [_cast(v, elem, key) if elem is not None else v for v in value]
    if isinstance(default, str):
        return str(value)
    if default is None and isinstance(value, str):
        return yaml.safe_load(value)
    return value


def _build(cls: type, data: dict[str, Any] | None, prefix: str = "") -> Any:
    """Dựng dataclass `cls` từ dict, kiểm tra khóa lạ, ép kiểu theo giá trị mặc định."""
    obj = cls()
    if data is None:
        return obj
    if not isinstance(data, dict):
        raise ValueError(f"{prefix or 'config'}: cần một mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Khóa config không tồn tại: {prefix}{key}")
        default = getattr(obj, key)
        ftype = hints[key]
        if dataclasses.is_dataclass(ftype):
            setattr(obj, key, _build(ftype, value, f"{prefix}{key}."))
        else:
            setattr(obj, key, _cast(value, default, f"{prefix}{key}"))
    return obj


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Trả bản sao `cfg` đã áp các ghi đè dạng "a.b.c=value"."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Ghi đè không hợp lệ (cần a.b=c): {item!r}")
        path, raw = item.split("=", 1)
        parts = path.strip().split(".")
        node: Any = cfg
        for p in parts[:-1]:
            if not dataclasses.is_dataclass(node) or not hasattr(node, p):
                raise KeyError(f"Khóa config không tồn tại: {path}")
            node = getattr(node, p)
        leaf = parts[-1]
        if not dataclasses.is_dataclass(node) or leaf not in {f.name for f in
                                                             dataclasses.fields(node)}:
            raise KeyError(f"Khóa config không tồn tại: {path}")
        current = getattr(node, leaf)
        if dataclasses.is_dataclass(current):
            raise KeyError(f"{path} là một mục, không phải giá trị")
        setattr(node, leaf, _cast(raw.strip(), current, path))
    return cfg


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    """Đọc YAML (mặc định configs/default.yaml) rồi áp `overrides` ("a.b=c")."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif path:
        raise FileNotFoundError(path)
    cfg = _build(Config, data)
    return apply_overrides(cfg, overrides)


@functools.lru_cache(maxsize=1)
def default_config() -> Config:
    """Config mặc định dùng chung khi hàm công khai nhận cfg=None (không được sửa đối tượng này)."""
    return load_config()


def _digest(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def cfg_hash(cfg: Config) -> str:
    """SHA-1 (8 ký tự) của các mục config ảnh hưởng tới đặc trưng."""
    d = asdict(cfg)
    payload: dict[str, Any] = {k: d[k] for k in HASHED_SECTIONS}
    if cfg.fusion.enabled:
        payload["fusion.model"] = cfg.fusion.model
    return _digest(payload)


def extract_hash(cfg: Config) -> str:
    """cfg_hash + mục detect: khóa cache cho eval (otsu_crop đổi crop nhưng không đổi cfg_hash)."""
    return _digest({"cfg": cfg_hash(cfg), "detect": asdict(cfg.detect)})


def resolve_path(p: str) -> Path:
    """Đường dẫn tương đối trong config được hiểu theo gốc repo."""
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def dump_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
