"""RSSI feature extraction — the mandatory non-CSI baseline (plan RQ6).

RSSI collapses each packet to one power number per link, discarding all
subcarrier structure. Comparing models trained on these features against
CSI features quantifies exactly what the extra spectral detail buys.
"""

from __future__ import annotations

import numpy as np

from ..preprocess.pipeline import windows_from_stream


def rssi_from_csi(csi: np.ndarray) -> np.ndarray:
    """Per-packet, per-link RSSI (dB) from complex CSI (time, links, sc)."""
    import warnings

    with warnings.catch_warnings():
        # Lost packets are all-NaN → nanmean warns; NaNs are handled below.
        warnings.simplefilter("ignore", RuntimeWarning)
        power = np.nanmean(np.abs(csi) ** 2, axis=2)
    # Lost packets: carry the last valid value forward (receiver behaviour).
    for link in range(power.shape[1]):
        col = power[:, link]
        nan = np.isnan(col)
        if nan.any() and not nan.all():
            t = np.arange(len(col))
            col[nan] = np.interp(t[nan], t[~nan], col[~nan])
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(power, 1e-12))


def rssi_window_features(csi: np.ndarray, window: int, step: int) -> np.ndarray:
    """Windowed RSSI features: per-link mean, std, min, max → (n_windows, 4*links)."""
    rssi = rssi_from_csi(csi)
    spans = windows_from_stream(csi.shape[0], window, step)
    feats = []
    for s, e in spans:
        w = rssi[s:e]
        feats.append(np.concatenate([
            w.mean(axis=0), w.std(axis=0), w.min(axis=0), w.max(axis=0),
        ]))
    return np.stack(feats) if feats else np.empty((0, 0))
