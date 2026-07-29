"""Time-domain cleaning of CSI streams: lost-packet repair, outlier removal,
and low-pass filtering.

All functions operate on arrays shaped (time, n_links, n_subcarriers) for the
complex CSI, or (time, ...) for real amplitude arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def interpolate_lost_packets(csi: np.ndarray) -> np.ndarray:
    """Linearly interpolate packets whose CSI is NaN (lost/invalid).

    A packet is treated as lost for a link when all its subcarriers are NaN.
    Leading/trailing losses take the nearest valid packet.
    """
    out = csi.copy()
    t = np.arange(csi.shape[0])
    for link in range(csi.shape[1]):
        lost = np.all(np.isnan(out[:, link, :]), axis=1)
        if not lost.any() or lost.all():
            continue
        valid = ~lost
        for sc in range(csi.shape[2]):
            col = out[:, link, sc]
            if np.iscomplexobj(col):
                re = np.interp(t, t[valid], col[valid].real)
                im = np.interp(t, t[valid], col[valid].imag)
                out[:, link, sc] = re + 1j * im
            else:
                out[:, link, sc] = np.interp(t, t[valid], col[valid])
    return out


def hampel_filter(x: np.ndarray, window: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
    """Hampel outlier filter along the time axis (axis 0).

    Replaces samples deviating more than ``n_sigmas`` robust standard
    deviations (1.4826 * MAD) from the rolling median with the median.
    Vectorized over all trailing axes using a strided rolling window.
    """
    if x.shape[0] < window:
        return x.copy()
    half = window // 2
    pad = np.concatenate([x[half - 1::-1], x, x[:-half - 1:-1]], axis=0)
    win = np.lib.stride_tricks.sliding_window_view(pad, window, axis=0)
    med = np.median(win, axis=-1)
    mad = np.median(np.abs(win - med[..., None]), axis=-1)
    thresh = n_sigmas * 1.4826 * mad
    out = x.copy()
    mask = np.abs(x - med[: x.shape[0]]) > thresh[: x.shape[0]]
    out[mask] = med[: x.shape[0]][mask]
    return out


def lowpass_filter(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along the time axis.

    Human-motion energy in CSI amplitude sits below ~10 Hz; everything above
    is noise/interference at typical packet rates.
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return x.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return x.copy()
    return filtfilt(b, a, x, axis=0)
