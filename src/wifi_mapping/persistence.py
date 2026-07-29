"""Save/load trained model bundles.

A bundle keeps everything inference needs in one file: the fitted
preprocessing state (normalizer + PCA), the zone classifier and/or xy
regressor, and the config snapshot used at training time — so the live
server can never mix a model with the wrong normalization.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config
from .preprocess import PreprocessPipeline


def save_bundle(path: Path, cfg: Config, pipeline: PreprocessPipeline,
                models: dict[str, Any], extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "config": asdict(cfg),
            "pipeline_state": pipeline.state_dict(),
            "models": models,          # e.g. {"zone": clf, "xy": reg}
            "extra": extra or {},
        }, f)


def load_bundle(path: Path, cfg: Config) -> tuple[PreprocessPipeline, dict[str, Any], dict]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    pipeline = PreprocessPipeline(cfg)
    pipeline.load_state_dict(bundle["pipeline_state"])
    return pipeline, bundle["models"], bundle.get("extra", {})
