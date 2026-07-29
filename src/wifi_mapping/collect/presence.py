"""Research-grounded human presence detection from RSSI.

A single RSSI stream carries more than "is the number jumping around".
This module implements four complementary detectors and fuses them into a
three-state decision, following the device-free sensing literature:

1. **Motion** — short-window variance of the *high-passed* RSSI. The
   high-pass (above :data:`MOTION_HP_HZ`) is essential: breathing is itself
   a slow oscillation, so plain variance confuses a still, breathing person
   with a walking one. Filtering above the respiration band makes the
   motion and breathing features orthogonal.
2. **Breathing** — spectral peak in the respiration band (0.16–0.6 Hz,
   ≈10–36 breaths/min). A *stationary* person still modulates the channel
   by breathing; this is the only cue that separates "empty room" from
   "someone sitting still", which pure variance detectors miss entirely.
   Reported feasible on commodity RSSI (see reference/SENSING_RESEARCH.md).
3. **Shadowing** — sustained shift of the mean RSSI away from the
   calibrated empty-room level, caused by a body attenuating the path.
4. **Entropy** — Shannon entropy of the recent RSSI distribution; a quiet
   link sits on one or two quantization levels, an occupied one spreads.

Thresholds are **autonomous**: during a short calibration window the
detector measures each feature's own quiet-state distribution and sets
thresholds at ``median + k·MAD`` (robust to outliers), instead of requiring
hand-tuned per-room constants. This follows the RSSI-distribution
autonomous-threshold approach in the recent literature.

Decision is a hysteretic state machine — ``empty`` → ``stationary`` →
``motion`` — so a walking pause does not flap the output.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import butter, filtfilt

# Respiration band: 10–36 breaths/min. Below 0.16 Hz is drift, above 0.6 Hz
# is small-motion energy rather than breathing.
BREATH_LO_HZ = 0.16
BREATH_HI_HZ = 0.60
# Motion is measured in a band *above* respiration but below the rate at
# which nothing human happens. The upper edge matters: RSSI is quantized to
# 1 dB, and a quantized slow signal produces sharp level-crossing edges
# whose harmonics would otherwise masquerade as movement.
MOTION_LO_HZ = 0.80
MOTION_HI_HZ = 3.00
# Maximum wander (Hz) of the respiration peak for it to count as a real
# breathing rate. 0.05 Hz ≈ 3 breaths/min of drift.
BREATH_STABILITY_HZ = 0.05


@dataclass
class PresenceState:
    """One detector output."""
    t: float = 0.0
    state: str = "calibrating"       # calibrating | empty | stationary | motion
    confidence: float = 0.0
    motion: float = 0.0              # excess variance over quiet floor (dB)
    breathing: float = 0.0           # spectral SNR in the respiration band
    breathing_bpm: float | None = None
    shadow: float = 0.0              # |mean shift| from calibrated level (dB)
    entropy: float = 0.0             # bits
    rssi: float = 0.0
    calibrated: bool = False
    calibration_progress: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _robust_threshold(values: np.ndarray, k: float, floor: float) -> float:
    """median + k·MAD, never below ``floor`` (guards a degenerate calibration)."""
    if len(values) == 0:
        return floor
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return max(floor, med + k * 1.4826 * mad)


def band_snr(x: np.ndarray, fs: float, lo: float, hi: float) -> tuple[float, float]:
    """Spectral SNR inside [lo, hi] Hz and the peak frequency.

    Detrends, applies a Hann window, and compares the strongest in-band bin
    against the median out-of-band power. Returns (snr, peak_hz); snr is 0
    when the window is too short to resolve the band.
    """
    n = len(x)
    if n < 16:
        return 0.0, 0.0
    # Linear detrend: breathing rides on top of slow drift.
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    x = x * np.hanning(n)
    ps = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = (freqs >= lo) & (freqs <= hi)
    out = (freqs > hi) & (freqs < fs / 2)
    if not band.any() or not out.any():
        return 0.0, 0.0
    noise = float(np.median(ps[out])) + 1e-12
    peak_i = int(np.argmax(ps[band]))
    peak_power = float(ps[band][peak_i])
    return peak_power / noise, float(freqs[band][peak_i])


def motion_band(x: np.ndarray, fs: float, lo: float = MOTION_LO_HZ,
                hi: float = MOTION_HI_HZ, order: int = 2) -> np.ndarray:
    """Zero-phase band-pass isolating human-motion rates.

    Falls back to mean-removal when the input is too short to filter.
    """
    nyq = fs / 2.0
    if len(x) < 4 or lo >= nyq:
        return x - x.mean()
    hi = min(hi, nyq * 0.95)
    btype, wn = ("band", [lo / nyq, hi / nyq]) if hi > lo else ("high", lo / nyq)
    b, a = butter(order, wn, btype=btype)
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x - x.mean()
    return filtfilt(b, a, x)


def rssi_entropy(x: np.ndarray) -> float:
    """Shannon entropy (bits) of the RSSI value distribution.

    RSSI is reported in integer dBm, so the natural histogram is over
    1 dB bins: a still link occupies one or two levels (low entropy), an
    occupied one spreads across several.
    """
    if len(x) == 0:
        return 0.0
    counts = np.bincount(np.round(x - x.min()).astype(int))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


class PresenceDetector:
    """Multi-feature presence detection for one RSSI link.

    Parameters
    ----------
    rate_hz : sampling rate of the RSSI stream.
    calibration_s : quiet-room learning period. **The room should be empty
        during this window** — the detector says so via ``state ==
        "calibrating"`` and ``calibration_progress``.
    motion_window_s : short window for the variance feature.
    breath_window_s : long window for respiration. Must span several breath
        cycles: 20 s resolves 0.05 Hz, enough for a 0.16 Hz minimum.
    sensitivity : scales every autonomous threshold; >1 is more
        conservative. The default 1.25 was chosen by sweeping detection
        against false-alarm rate on synthetic links (see
        reference/SENSING_RESEARCH.md): it holds 100% occupancy detection
        down to a 0.4 dB breathing modulation while producing no false
        alarms on empty links, where 1.0 leaks ~5% false alarms and 1.5
        starts losing weak breathers.
    """

    def __init__(self, rate_hz: float = 10.0, calibration_s: float = 20.0,
                 motion_window_s: float = 2.0, breath_window_s: float = 20.0,
                 sensitivity: float = 1.25, hold_s: float = 4.0,
                 motion_persist_s: float = 0.5):
        self.rate_hz = rate_hz
        self.sensitivity = sensitivity
        self.hold_s = hold_s
        # A real body moving produces *sustained* band energy. An isolated
        # 1 dB quantization step produces a single impulse that the
        # band-pass rings on — requiring persistence rejects those.
        self.motion_persist_n = max(2, int(motion_persist_s * rate_hz))
        self._motion_run = 0
        self.motion_n = max(4, int(motion_window_s * rate_hz))
        self.breath_n = max(32, int(breath_window_s * rate_hz))
        self.calibration_n = max(self.breath_n, int(calibration_s * rate_hz))

        self.samples: deque[float] = deque(maxlen=self.breath_n)
        self._calib: list[float] = []
        self._calib_feats: dict[str, list[float]] = {
            "motion": [], "breathing": [], "entropy": []}

        self.calibrated = False
        self.quiet_rssi = 0.0
        self.thresholds = {"motion": 0.6, "breathing": 4.0,
                           "shadow": 1.5, "entropy": 1.5}

        self._last_motion_t = -1e9
        self._last_presence_t = -1e9
        self._state = "calibrating"
        # Recent in-band spectral peaks. A real respiration rate is stable
        # over tens of seconds; an incidental noise peak wanders across the
        # band, so peak stability is a strong false-positive filter.
        self._peaks: deque[float] = deque(maxlen=max(4, int(6 * rate_hz)))

    # ------------------------------------------------------------ features

    def _features(self) -> tuple[float, float, float, float, float]:
        x = np.asarray(self.samples, dtype=float)
        # Motion: energy in the human-movement band only, so that neither a
        # still person's breathing nor RSSI quantization edges register as
        # movement.
        motion = float(np.std(motion_band(x, self.rate_hz)[-self.motion_n:]))
        entropy = rssi_entropy(x)
        shadow = (abs(float(np.mean(x[-self.motion_n:])) - self.quiet_rssi)
                  if self.calibrated else 0.0)
        breathing, peak_hz = (0.0, 0.0)
        if len(x) >= self.breath_n:
            breathing, peak_hz = band_snr(x, self.rate_hz, BREATH_LO_HZ, BREATH_HI_HZ)
        return motion, breathing, peak_hz, shadow, entropy

    # --------------------------------------------------------- calibration

    def _calibrate_step(self) -> None:
        """Accumulate quiet-room statistics, then derive thresholds."""
        motion, breathing, _, _, entropy = self._features()
        self._calib_feats["motion"].append(motion)
        self._calib_feats["breathing"].append(breathing)
        self._calib_feats["entropy"].append(entropy)

        if len(self._calib) < self.calibration_n:
            return

        s = self.sensitivity
        self.quiet_rssi = float(np.median(self._calib))
        m = np.asarray(self._calib_feats["motion"])
        # Only breathing samples measured on a full window are meaningful;
        # the zeros from the warm-up would drag the threshold down.
        b = np.asarray([v for v in self._calib_feats["breathing"] if v > 0])
        e = np.asarray(self._calib_feats["entropy"])
        self.thresholds = {
            # Motion must clearly exceed the quiet high-band variance floor.
            # Floor of 0.35 dB keeps single-LSB steps on a very quiet link
            # from clearing the bar (RSSI is quantized to 1 dB).
            "motion": _robust_threshold(m, 5.0, 0.35) * s,
            # Breathing SNR: an empty link still produces incidental in-band
            # peaks around SNR 3–6, while a real breather measures in the
            # tens-to-hundreds — so the floor sits well clear of the noise.
            "breathing": max(_robust_threshold(b, 6.0, 12.0), 12.0) * s,
            # Body shadowing: RSSI is quantized to 1 dB, so require > 1 dB.
            "shadow": 1.5 * s,
            "entropy": _robust_threshold(e, 5.0, 0.8) * s,
        }
        self.calibrated = True

    def recalibrate(self) -> None:
        """Restart calibration (call after moving the laptop or router)."""
        self._calib.clear()
        for v in self._calib_feats.values():
            v.clear()
        self.calibrated = False
        self._state = "calibrating"

    # -------------------------------------------------------------- update

    def update(self, rssi: float, now: float | None = None) -> PresenceState:
        now = time.time() if now is None else now
        self.samples.append(float(rssi))

        if not self.calibrated:
            self._calib.append(float(rssi))
            self._calibrate_step()
            return PresenceState(
                t=now, state="calibrating", rssi=rssi, calibrated=self.calibrated,
                calibration_progress=min(1.0, len(self._calib) / self.calibration_n),
            )

        motion, breathing, peak_hz, shadow, entropy = self._features()
        th = self.thresholds

        self._motion_run = self._motion_run + 1 if motion > th["motion"] else 0
        motion_hit = self._motion_run >= self.motion_persist_n

        # Breathing requires BOTH a strong in-band peak and a *stable* peak
        # frequency — noise produces strong peaks, but not steady ones.
        if breathing > th["breathing"]:
            self._peaks.append(peak_hz)
        elif self._peaks:
            self._peaks.popleft()
        peak_stable = (len(self._peaks) >= self._peaks.maxlen
                       and float(np.std(self._peaks)) < BREATH_STABILITY_HZ)
        breath_hit = breathing > th["breathing"] and peak_stable
        # Shadowing and entropy corroborate; they never trigger on their own
        # because both drift with AP power control and interference. A
        # sustained shadow *plus* raised entropy is accepted as presence.
        support = (shadow > th["shadow"]) and (entropy > th["entropy"])

        if motion_hit:
            self._last_motion_t = now
            self._last_presence_t = now
        elif breath_hit or support:
            self._last_presence_t = now

        if now - self._last_motion_t < self.hold_s:
            state = "motion"
        elif now - self._last_presence_t < self.hold_s:
            state = "stationary"
        else:
            state = "empty"
        self._state = state

        # Confidence: how far the deciding feature sits above its threshold.
        if state == "motion":
            conf = min(1.0, motion / max(th["motion"], 1e-6) / 3.0)
        elif state == "stationary":
            conf = min(1.0, breathing / max(th["breathing"], 1e-6) / 3.0)
        else:
            conf = min(1.0, 1.0 - motion / max(th["motion"], 1e-6))

        # Report the median of the stable peak run, not the noisy instant.
        bpm = (float(np.median(self._peaks)) * 60.0
               if (breath_hit and state != "motion") else None)
        return PresenceState(
            t=now, state=state, confidence=float(max(0.0, min(1.0, conf))),
            motion=motion, breathing=breathing, breathing_bpm=bpm,
            shadow=shadow, entropy=entropy, rssi=rssi,
            calibrated=True, calibration_progress=1.0,
        )


class MultiLinkPresence:
    """Fuse per-link detectors into one room-level decision.

    Each link (the connected AP, plus any neighbouring APs being scanned)
    gets its own :class:`PresenceDetector` — their quiet levels and noise
    floors differ, so thresholds must be per-link. The room state is the
    strongest link state, which is the correct fusion for detection: a
    person only needs to disturb *one* path to be present.
    """

    def __init__(self, rate_hz: float = 10.0, **kwargs):
        self.rate_hz = rate_hz
        self.kwargs = kwargs
        self.detectors: dict[str, PresenceDetector] = {}
        self.states: dict[str, PresenceState] = {}

    _RANK = {"calibrating": -1, "empty": 0, "stationary": 1, "motion": 2}

    def update(self, link_id: str, rssi: float, now: float | None = None,
               rate_hz: float | None = None) -> PresenceState:
        det = self.detectors.get(link_id)
        if det is None:
            det = self.detectors[link_id] = PresenceDetector(
                rate_hz=rate_hz or self.rate_hz, **self.kwargs)
        st = det.update(rssi, now)
        self.states[link_id] = st
        return st

    def room_state(self) -> dict:
        """Fused summary across all links."""
        live = [s for s in self.states.values() if s.calibrated]
        if not live:
            progress = [s.calibration_progress for s in self.states.values()]
            return {"state": "calibrating", "confidence": 0.0,
                    "calibration_progress": min(progress) if progress else 0.0,
                    "links": {k: v.to_dict() for k, v in self.states.items()}}
        best = max(live, key=lambda s: (self._RANK[s.state], s.confidence))
        # Breathing rate: take it from whichever link sees it most clearly.
        breathing = [s for s in live if s.breathing_bpm]
        bpm = max(breathing, key=lambda s: s.breathing).breathing_bpm if breathing else None
        return {
            "state": best.state,
            "confidence": best.confidence,
            "breathing_bpm": bpm,
            "active_links": sum(1 for s in live if s.state != "empty"),
            "n_links": len(live),
            "calibration_progress": 1.0,
            "links": {k: v.to_dict() for k, v in self.states.items()},
        }

    def recalibrate(self) -> None:
        for det in self.detectors.values():
            det.recalibrate()
