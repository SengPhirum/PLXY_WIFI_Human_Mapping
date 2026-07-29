"""End-to-end preprocessing: raw complex CSI stream → model-ready feature
vectors, with fit/transform semantics so normalization and PCA are learned
from training data only (leakage prevention, plan §12).

Pipeline per the thesis plan §11:

    packets → lost-packet repair → amplitude + sanitized phase
            → Hampel outlier removal → low-pass filter
            → per-link/subcarrier normalization → sliding windows
            → per-window statistical features (+ optional PCA)
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .filters import hampel_filter, interpolate_lost_packets, lowpass_filter
from .phase import sanitize_phase


def windows_from_stream(n_samples: int, size: int, step: int) -> list[tuple[int, int]]:
    """Start/end indices of overlapping sliding windows over a stream."""
    if n_samples < size:
        return []
    return [(s, s + size) for s in range(0, n_samples - size + 1, step)]


def extract_window_features(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Feature vector for one window.

    Parameters
    ----------
    amp, phase : arrays (window, n_links, n_subcarriers)

    Statistical features per (link, subcarrier) over time, then flattened:
    amplitude mean/std and phase mean capture the spatial fingerprint;
    amplitude temporal-difference energy captures motion intensity.
    """
    feats = [
        amp.mean(axis=0),                      # spatial fingerprint
        amp.std(axis=0),                       # motion-induced variance
        np.abs(np.diff(amp, axis=0)).mean(axis=0),  # motion energy
        phase.mean(axis=0),                    # sanitized-phase fingerprint
    ]
    return np.concatenate([f.ravel() for f in feats])


class PreprocessPipeline:
    """Stateful preprocessing with train-only fitting of normalization/PCA."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.amp_mean: np.ndarray | None = None
        self.amp_std: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None

    # ------------------------------------------------------------- cleaning

    def clean(self, csi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw complex CSI (time, links, subcarriers) → (amplitude, phase)."""
        p = self.cfg.preprocess
        csi = interpolate_lost_packets(csi)
        amp = np.abs(csi)
        amp = hampel_filter(amp, p.hampel_window, p.hampel_sigmas)
        amp = lowpass_filter(amp, p.lowpass_cutoff_hz, self.cfg.signal.sample_rate_hz)
        phase = sanitize_phase(csi)
        return amp, phase

    # ---------------------------------------------------------- fit / apply

    def fit_normalizer(self, amp: np.ndarray) -> None:
        """Learn per-(link, subcarrier) amplitude statistics from training data."""
        self.amp_mean = amp.mean(axis=0)
        self.amp_std = amp.std(axis=0) + 1e-9

    def normalize(self, amp: np.ndarray) -> np.ndarray:
        if self.amp_mean is None:
            raise RuntimeError("fit_normalizer() must be called first")
        return (amp - self.amp_mean) / self.amp_std

    def fit_pca(self, features: np.ndarray) -> None:
        """Learn a PCA projection of window features from training data."""
        n = self.cfg.preprocess.n_pca_components
        if n <= 0 or features.shape[0] < n:
            return
        self._pca_mean = features.mean(axis=0)
        centered = features - self._pca_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self._pca_components = vt[:n]

    def project(self, features: np.ndarray) -> np.ndarray:
        if self._pca_components is None:
            return features
        return (features - self._pca_mean) @ self._pca_components.T

    # ------------------------------------------------------------ windowing

    def stream_to_features(self, csi: np.ndarray, fit: bool = False) -> np.ndarray:
        """Full pipeline for one recording; returns (n_windows, n_features).

        With ``fit=True`` the normalizer is (re)fitted on this recording —
        only ever do that on training data.
        """
        p = self.cfg.preprocess
        amp, phase = self.clean(csi)
        if fit:
            self.fit_normalizer(amp)
        amp = self.normalize(amp)
        spans = windows_from_stream(csi.shape[0], p.window_size, p.window_step)
        if not spans:
            return np.empty((0, 0))
        return np.stack([
            extract_window_features(amp[s:e], phase[s:e]) for s, e in spans
        ])

    # -------------------------------------------------------- serialization

    def state_dict(self) -> dict:
        return {
            "amp_mean": self.amp_mean,
            "amp_std": self.amp_std,
            "pca_mean": self._pca_mean,
            "pca_components": self._pca_components,
        }

    def load_state_dict(self, state: dict) -> None:
        self.amp_mean = state["amp_mean"]
        self.amp_std = state["amp_std"]
        self._pca_mean = state["pca_mean"]
        self._pca_components = state["pca_components"]
