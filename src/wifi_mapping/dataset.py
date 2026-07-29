"""Dataset generation, storage, and assembly into model-ready arrays.

Storage format — one ``.npz`` file per recording session:

    csi   : complex64 (n_packets, n_links, n_subcarriers)  raw CSI stream
    xy    : float32   (n_packets, 2)                        ground-truth position
    seg   : int32     (n_packets,)  contiguous-recording segment id
    meta  : JSON string — {"session", "day", "person", "room", "kind", ...}

Sessions are the atomic unit for leakage-aware splitting (eval.splits).
Sliding windows never cross segment boundaries (a segment is one contiguous
capture — one standing spot or one walk), so no window mixes two positions.
Window labels are derived from ground truth: mean (x, y) over the window for
regression, zone of that mean point for classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .preprocess.pipeline import PreprocessPipeline, windows_from_stream
from .simulate import CsiSimulator, WalkGenerator


# --------------------------------------------------------------- generation

def generate_session(cfg: Config, session_id: int, day: int, person: int,
                     packets_per_zone: int = 300, walk_seconds: float = 60.0,
                     spots_per_zone: int = 4,
                     seed: int | None = None) -> dict[str, Any]:
    """Simulate one collection session following the plan's protocol (§9):
    person standing at several random spots inside every grid zone, then
    continuous walking. CSI fingerprints vary strongly *within* a zone
    (multipath decorrelates over ~λ/2 ≈ 6 cm), so each session must cover
    multiple spots per zone or models cannot generalize across sessions.
    Per-day environment perturbation makes cross-day testing honest.
    """
    rng = np.random.default_rng(seed)
    # Environment changes day to day but is fixed within a day.
    env_cfg = _perturb_environment(cfg, day, rng)
    sim = CsiSimulator(env_cfg, seed=int(rng.integers(2 ** 31)))

    csi_parts: list[np.ndarray] = []
    xy_parts: list[np.ndarray] = []
    seg_parts: list[np.ndarray] = []
    seg_id = 0

    # Standing captures per zone (people fidget differently → jitter varies;
    # breathing/sway is on the order of a centimetre).
    jitter = 0.008 + 0.010 * (person % 5) / 4
    per_spot = max(cfg.preprocess.window_size, packets_per_zone // spots_per_zone)
    cw = cfg.room.width / cfg.grid.cols
    ch = cfg.room.depth / cfg.grid.rows
    for zone in range(cfg.grid.n_zones):
        zx, zy = cfg.zone_center(zone)
        for _ in range(spots_per_zone):
            px = zx + rng.uniform(-0.4, 0.4) * cw
            py = zy + rng.uniform(-0.4, 0.4) * ch
            csi_parts.append(sim.record_static(px, py, per_spot, jitter=jitter))
            xy_parts.append(np.tile([px, py], (per_spot, 1)))
            seg_parts.append(np.full(per_spot, seg_id))
            seg_id += 1

    # Walking capture (speed varies by person).
    n_walk = int(walk_seconds * cfg.signal.sample_rate_hz)
    if n_walk > 0:
        walker = WalkGenerator(cfg, speed=0.6 + 0.1 * (person % 4),
                               seed=int(rng.integers(2 ** 31)))
        path = walker.trajectory(n_walk, 1.0 / cfg.signal.sample_rate_hz)
        csi_parts.append(sim.record_path(path))
        xy_parts.append(path)
        seg_parts.append(np.full(n_walk, seg_id))

    return {
        "csi": np.concatenate(csi_parts).astype(np.complex64),
        "xy": np.concatenate(xy_parts).astype(np.float32),
        "seg": np.concatenate(seg_parts).astype(np.int32),
        "meta": {"session": session_id, "day": day, "person": person,
                 "room": cfg.room.name, "kind": "simulated"},
    }


def _perturb_environment(cfg: Config, day: int, rng: np.random.Generator) -> Config:
    """Clone the config with small day-specific device-position offsets."""
    import copy
    env = copy.deepcopy(cfg)
    # A few millimetres of re-mounting error: enough to shift multipath
    # phases noticeably (λ ≈ 12.5 cm at 2.4 GHz) without erasing the
    # fingerprint entirely — matching reported cross-day degradation.
    day_rng = np.random.default_rng(1000 + day)
    for node in [env.links.tx, *env.links.rx]:
        node.pos = tuple(
            p + day_rng.normal(0, 0.004) if i < 2 else p
            for i, p in enumerate(node.pos)
        )
    return env


# ------------------------------------------------------------------ storage

def save_session(session: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, csi=session["csi"], xy=session["xy"],
                        seg=session["seg"], meta=json.dumps(session["meta"]))


def load_session(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {"csi": z["csi"], "xy": z["xy"], "seg": z["seg"],
                "meta": json.loads(str(z["meta"]))}


def load_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    files = sorted(dataset_dir.glob("session_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no session_*.npz files in {dataset_dir} — "
            "run scripts/generate_dataset.py first"
        )
    return [load_session(f) for f in files]


# ----------------------------------------------------------------- assembly

def sessions_to_arrays(cfg: Config, sessions: list[dict[str, Any]],
                       pipeline: PreprocessPipeline, fit: bool = False,
                       rssi: bool = False) -> dict[str, np.ndarray]:
    """Turn raw sessions into (X, zone labels, xy labels, group metadata).

    With ``fit=True`` (training data only) the pipeline's normalizer and PCA
    are fitted on the concatenated training recordings.
    """
    from .models.rssi import rssi_window_features

    p = cfg.preprocess
    if fit and not rssi:
        # Fit the amplitude normalizer on all training recordings at once.
        amps = [pipeline.clean(s["csi"])[0] for s in sessions]
        pipeline.fit_normalizer(np.concatenate(amps))

    X_parts, zone_parts, xy_parts, meta_parts = [], [], [], []
    for s in sessions:
        # Process each contiguous capture segment separately so that
        # filtering transients and sliding windows never cross the position
        # jump between two captures.
        seg = s.get("seg", np.zeros(s["csi"].shape[0], dtype=np.int32))
        for seg_id in np.unique(seg):
            m = seg == seg_id
            csi, xy = s["csi"][m], s["xy"][m]
            if rssi:
                feats = rssi_window_features(csi, p.window_size, p.window_step)
            else:
                feats = pipeline.stream_to_features(csi)
            if feats.size == 0:
                continue
            spans = windows_from_stream(csi.shape[0], p.window_size, p.window_step)
            w_xy = np.stack([xy[a:b].mean(axis=0) for a, b in spans])
            w_zone = np.array([cfg.zone_of(x, y) for x, y in w_xy])
            X_parts.append(feats)
            xy_parts.append(w_xy)
            zone_parts.append(w_zone)
            meta_parts.extend([s["meta"]] * len(feats))

    X = np.concatenate(X_parts)
    if fit and not rssi:
        pipeline.fit_pca(X)
    X = pipeline.project(X) if not rssi else X
    return {
        "X": X,
        "zone": np.concatenate(zone_parts),
        "xy": np.concatenate(xy_parts),
        "meta": meta_parts,
    }
