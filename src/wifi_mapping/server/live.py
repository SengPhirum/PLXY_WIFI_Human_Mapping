"""Streaming inference: CSI frames in, smoothed positions out.

:class:`LivePredictor` is transport-agnostic — the same object serves the
simulator-driven demo and real ESP32 frames (via SessionRecorder-style
alignment). It keeps a ring buffer of recent frames; every ``window_step``
new frames it preprocesses the latest window, predicts zone and coordinates,
and Kalman-smooths the coordinate track.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from ..config import Config
from ..preprocess import PreprocessPipeline
from ..preprocess.pipeline import extract_window_features, spectral_band_features
from ..tracking import KalmanTracker2D


class LivePredictor:
    """Streaming inference for localization and/or pose.

    ``models`` may contain any of: "zone" (classifier), "xy" (regressor),
    "posture" (classifier) and "joints" (regressor). Pose models consume
    the extended feature vector (window statistics + motion spectrum), so
    the two bundles are not interchangeable — ``pose_mode`` is set from the
    presence of the pose heads.
    """

    def __init__(self, cfg: Config, pipeline: PreprocessPipeline,
                 models: dict[str, Any]):
        self.cfg = cfg
        self.pipeline = pipeline
        self.models = models
        self.pose_mode = "posture" in models or "joints" in models
        p = cfg.preprocess
        self.buffer: deque[np.ndarray] = deque(maxlen=p.window_size)
        self._since_last = 0
        dt = p.window_step / cfg.signal.sample_rate_hz
        self.tracker = KalmanTracker2D(cfg.tracking.process_noise,
                                       cfg.tracking.measurement_noise, dt=dt)

    def reset(self) -> None:
        self.buffer.clear()
        self._since_last = 0
        self.tracker.reset()
        self._prev_joints = None

    def push_frame(self, frame: np.ndarray) -> dict | None:
        """Feed one CSI frame (n_links, n_subcarriers); returns a prediction
        dict every ``window_step`` frames once the buffer is full, else None.
        """
        self.buffer.append(frame)
        self._since_last += 1
        p = self.cfg.preprocess
        if len(self.buffer) < p.window_size or self._since_last < p.window_step:
            return None
        self._since_last = 0
        return self._predict()

    def _predict(self) -> dict:
        t0 = time.perf_counter()
        csi = np.stack(self.buffer)
        amp, phase = self.pipeline.clean(csi)
        amp_n = self.pipeline.normalize(amp)
        feats = extract_window_features(amp_n, phase)
        if self.pose_mode:
            feats = np.concatenate([
                feats, spectral_band_features(amp, self.cfg.signal.sample_rate_hz)])
        feats = self.pipeline.project(feats[None, :])

        out: dict[str, Any] = {"t": time.time()}
        if self.pose_mode:
            self._add_pose(out, feats)
        if "xy" in self.models:
            raw = self.models["xy"].predict(feats)[0]
            raw_x = float(np.clip(raw[0], 0, self.cfg.room.width))
            raw_y = float(np.clip(raw[1], 0, self.cfg.room.depth))
            sx, sy = self.tracker.update(raw_x, raw_y)
            sx = float(np.clip(sx, 0, self.cfg.room.width))
            sy = float(np.clip(sy, 0, self.cfg.room.depth))
            out.update(x=sx, y=sy, raw_x=raw_x, raw_y=raw_y,
                       zone_from_xy=self.cfg.zone_of(sx, sy))
        if "zone" in self.models:
            out["zone"] = int(self.models["zone"].predict(feats)[0])
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return out

    def _add_pose(self, out: dict, feats: np.ndarray) -> None:
        """Posture + fused skeleton, with temporal smoothing of the joints."""
        from ..models.pose import fuse_pose
        from ..simulate.body_model import N_JOINTS, POSTURES

        proba = np.zeros(len(POSTURES))
        if "posture" in self.models:
            clf = self.models["posture"]
            p = clf.predict_proba(feats)[0]
            proba[clf.classes_] = p
            out["posture"] = POSTURES[int(proba.argmax())]
            out["posture_confidence"] = float(proba.max())
            out["posture_proba"] = {POSTURES[i]: round(float(v), 3)
                                    for i, v in enumerate(proba)}
        if "joints" in self.models:
            raw = self.models["joints"].predict(feats)[0]
            joints = fuse_pose(raw, proba) if proba.any() else raw.reshape(N_JOINTS, 3)
            # Exponential smoothing across windows: consecutive predictions
            # are independent, but a body cannot teleport between them.
            prev = getattr(self, "_prev_joints", None)
            if prev is not None:
                joints = 0.6 * prev + 0.4 * joints
            self._prev_joints = joints
            out["joints"] = [[round(float(v), 4) for v in j] for j in joints]
