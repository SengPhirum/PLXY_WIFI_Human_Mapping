"""Typed experiment configuration loaded from YAML.

Everything downstream (simulator, preprocessing, training, server) reads a
single :class:`Config` object so that room geometry, link layout, and signal
parameters stay consistent across the whole pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class Room:
    name: str = "room"
    width: float = 5.0
    depth: float = 6.0
    height: float = 2.8


@dataclass
class Grid:
    cols: int = 4
    rows: int = 4

    @property
    def n_zones(self) -> int:
        return self.cols * self.rows


@dataclass
class Node:
    id: str
    pos: tuple[float, float, float]


@dataclass
class Links:
    tx: Node = field(default_factory=lambda: Node("TX0", (2.5, 0.15, 1.2)))
    rx: list[Node] = field(default_factory=list)

    @property
    def n_links(self) -> int:
        return len(self.rx)


@dataclass
class Signal:
    carrier_freq_hz: float = 2.412e9
    bandwidth_hz: float = 20e6
    n_subcarriers: int = 52
    sample_rate_hz: float = 100.0
    noise_std: float = 0.05
    packet_loss: float = 0.02


@dataclass
class Preprocess:
    hampel_window: int = 11
    hampel_sigmas: float = 3.0
    lowpass_cutoff_hz: float = 10.0
    window_size: int = 100
    window_step: int = 50
    n_pca_components: int = 20


@dataclass
class ModelCfg:
    type: str = "rf"
    task: str = "both"


@dataclass
class Tracking:
    process_noise: float = 0.35
    measurement_noise: float = 0.45


@dataclass
class Paths:
    data_dir: str = "data"
    models_dir: str = "data/models"
    datasets_dir: str = "data/datasets"
    sessions_dir: str = "data/sessions"


@dataclass
class Server:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    room: Room = field(default_factory=Room)
    grid: Grid = field(default_factory=Grid)
    links: Links = field(default_factory=Links)
    signal: Signal = field(default_factory=Signal)
    preprocess: Preprocess = field(default_factory=Preprocess)
    model: ModelCfg = field(default_factory=ModelCfg)
    tracking: Tracking = field(default_factory=Tracking)
    paths: Paths = field(default_factory=Paths)
    server: Server = field(default_factory=Server)

    # ---- geometry helpers -------------------------------------------------

    def zone_of(self, x: float, y: float) -> int:
        """Row-major zone id of a point, clamped to the room."""
        cx = min(max(x, 0.0), self.room.width - 1e-9)
        cy = min(max(y, 0.0), self.room.depth - 1e-9)
        col = int(cx / self.room.width * self.grid.cols)
        row = int(cy / self.room.depth * self.grid.rows)
        return row * self.grid.cols + col

    def zone_center(self, zone: int) -> tuple[float, float]:
        row, col = divmod(zone, self.grid.cols)
        cw = self.room.width / self.grid.cols
        ch = self.room.depth / self.grid.rows
        return (col + 0.5) * cw, (row + 0.5) * ch

    def resolve(self, rel: str) -> Path:
        """Resolve a path from config relative to the project root."""
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _node(d: dict[str, Any]) -> Node:
    return Node(id=d["id"], pos=tuple(float(v) for v in d["pos"]))


def _fill(cls: type, data: dict[str, Any]) -> Any:
    """Build a flat dataclass from a dict, keeping defaults for missing keys."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()
    if "room" in raw:
        cfg.room = _fill(Room, raw["room"])
    if "grid" in raw:
        cfg.grid = _fill(Grid, raw["grid"])
    if "links" in raw:
        cfg.links = Links(
            tx=_node(raw["links"]["tx"]),
            rx=[_node(r) for r in raw["links"]["rx"]],
        )
    for key, cls, attr in [
        ("signal", Signal, "signal"),
        ("preprocess", Preprocess, "preprocess"),
        ("model", ModelCfg, "model"),
        ("tracking", Tracking, "tracking"),
        ("paths", Paths, "paths"),
        ("server", Server, "server"),
    ]:
        if key in raw:
            setattr(cfg, attr, _fill(cls, raw[key]))
    # YAML "5.0e6"-style floats can parse as str in some emitters; coerce.
    for f_ in dataclasses.fields(cfg.signal):
        v = getattr(cfg.signal, f_.name)
        if isinstance(v, str):
            setattr(cfg.signal, f_.name, float(v))
    return cfg
