"""CSI phase sanitization.

Raw CSI phase from commodity hardware is corrupted per packet by carrier
frequency offset (a random constant across subcarriers) and sampling
frequency offset / packet detection delay (a random *linear slope* across
subcarriers). The standard remedy is to remove the best-fit linear trend
across the subcarrier axis from the unwrapped phase, leaving only the
curvature caused by multipath — which carries geometry information.
"""

from __future__ import annotations

import numpy as np


def sanitize_phase(csi: np.ndarray) -> np.ndarray:
    """Remove per-packet linear phase trend across subcarriers.

    Parameters
    ----------
    csi : complex array (..., n_subcarriers)

    Returns
    -------
    Real array of the same shape: unwrapped, detrended phase.
    """
    phase = np.unwrap(np.angle(csi), axis=-1)
    n = csi.shape[-1]
    k = np.arange(n)
    # Least-squares linear fit per packet, vectorized over leading axes.
    k_mean = k.mean()
    k_center = k - k_mean
    denom = float((k_center ** 2).sum())
    p_mean = phase.mean(axis=-1, keepdims=True)
    slope = ((phase - p_mean) * k_center).sum(axis=-1, keepdims=True) / denom
    return phase - p_mean - slope * k_center
