"""Link-health metrics: is the RSSI stream actually usable?

Most "Wi-Fi sensing doesn't work" reports are not algorithm failures — the
input is dead. Three things go wrong, and each has a distinct signature
that these metrics expose:

- **Stale samples.** The driver returns the same cached value repeatedly
  because no frames are arriving. Signature: high ``stale_fraction``.
- **Pinned RSSI.** The link is so stable that every sample rounds to one
  or two dBm levels, so sub-dB breathing modulation is quantized away.
  Signature: ``distinct_levels`` ≤ 2 with a low ``std``.
- **Slow sampling.** The OS call is slower than the requested rate, so the
  effective rate is too low to resolve motion. Signature:
  ``effective_rate_hz`` far below the target.

:func:`assess` turns the numbers into a verdict plus a specific fix, which
is what the dashboard and ``scripts/wifi_diagnose.py`` display.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class LinkHealth:
    samples: int = 0
    effective_rate_hz: float = 0.0
    target_rate_hz: float = 0.0
    stale_fraction: float = 0.0
    distinct_levels: int = 0
    rssi_std: float = 0.0
    rssi_min: float | None = None
    rssi_max: float | None = None
    failed_reads: int = 0
    verdict: str = "unknown"          # good | degraded | dead | unknown
    issues: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class HealthMonitor:
    """Rolling health statistics over the last ``window_s`` of sampling."""

    def __init__(self, target_rate_hz: float = 10.0, window_s: float = 20.0):
        self.target_rate_hz = target_rate_hz
        n = max(16, int(target_rate_hz * window_s))
        self.values: deque[float] = deque(maxlen=n)
        self.times: deque[float] = deque(maxlen=n)
        self.repeats: deque[int] = deque(maxlen=n)
        self.failed_reads = 0
        self._last: float | None = None

    def record(self, rssi: float | None, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if rssi is None:
            self.failed_reads += 1
            return
        self.repeats.append(1 if (self._last is not None and rssi == self._last) else 0)
        self._last = rssi
        self.values.append(float(rssi))
        self.times.append(now)

    def snapshot(self, probe_active: bool = False) -> LinkHealth:
        h = LinkHealth(samples=len(self.values),
                       target_rate_hz=self.target_rate_hz,
                       failed_reads=self.failed_reads)
        if len(self.values) < 4:
            h.verdict = "unknown"
            h.issues.append("not enough samples yet")
            return h

        span = self.times[-1] - self.times[0]
        h.effective_rate_hz = round((len(self.times) - 1) / span, 2) if span > 0 else 0.0
        h.stale_fraction = round(float(np.mean(self.repeats)), 3) if self.repeats else 0.0
        x = np.asarray(self.values)
        h.distinct_levels = int(len(np.unique(np.round(x))))
        h.rssi_std = round(float(np.std(x)), 3)
        h.rssi_min, h.rssi_max = float(x.min()), float(x.max())
        return assess(h, probe_active)


def assess(h: LinkHealth, probe_active: bool = False) -> LinkHealth:
    """Attach a verdict, the issues found, and what to do about each."""
    issues: list[str] = []
    advice: list[str] = []

    if h.effective_rate_hz < h.target_rate_hz * 0.5:
        issues.append(
            f"sampling at {h.effective_rate_hz:.1f} Hz, well below the "
            f"{h.target_rate_hz:.0f} Hz target")
        advice.append("the RSSI query is slow on this system — try a lower "
                      "--rate, or run natively rather than in a VM")

    # A perfectly repeated value stream means the driver is not refreshing.
    if h.stale_fraction > 0.85:
        issues.append(f"{h.stale_fraction:.0%} of samples repeat the previous "
                      "value — the driver is returning a cached RSSI")
        advice.append("keep traffic flowing: the active probe should be on "
                      "(check the device panel), or run `ping <router-ip>`")
        if not probe_active:
            advice.append("active probe is OFF — no gateway was detected; "
                          "start a ping to your router manually")

    if h.distinct_levels <= 2 and h.rssi_std < 0.4:
        issues.append(f"RSSI is pinned to {h.distinct_levels} level(s) "
                      f"(std {h.rssi_std:.2f} dB) — sub-dB modulation is "
                      "quantized away")
        advice.append("move the laptop further from the router, or use the "
                      "2.4 GHz band, so the link has natural variation")

    if h.failed_reads > max(5, h.samples * 0.2):
        issues.append(f"{h.failed_reads} failed RSSI reads")
        advice.append("check the interface is associated (`iw dev` / `netsh "
                      "wlan show interfaces`)")

    if not issues:
        h.verdict = "good"
    elif h.stale_fraction > 0.95 or h.distinct_levels <= 1 or h.effective_rate_hz < 1.0:
        h.verdict = "dead"
    else:
        h.verdict = "degraded"
    h.issues, h.advice = issues, advice
    return h
