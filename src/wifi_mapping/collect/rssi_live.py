"""Live RSSI sensing from the machine's own connected Wi-Fi interface.

Commodity laptops cannot expose CSI without special firmware, but every OS
reports the RSSI of the currently associated access point. A person moving
between (or near) the laptop and the AP perturbs that RSSI measurably —
enough for real-signal presence/motion demos while the ESP32 CSI hardware
is not yet available (thesis plan scope table row 1: presence detection).

Backends, tried in order:

- Linux:   ``/proc/net/wireless`` (no privileges needed), else ``iw dev
  <iface> link``
- macOS:   the ``airport -I`` utility
- Windows: ``netsh wlan show interfaces`` (Signal % → dBm approximation)

Motion metric: rolling standard deviation of RSSI over ~2 s, compared
against a slowly-adapting quiet-baseline (EWMA). This is deliberately
simple and explainable — it is a demo instrument, not a thesis result.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

AIRPORT = ("/System/Library/PrivateFrameworks/Apple80211.framework/"
           "Versions/Current/Resources/airport")


# ----------------------------------------------------------------- parsers

def parse_proc_wireless(text: str) -> dict[str, float]:
    """Parse /proc/net/wireless → {interface: rssi_dbm}."""
    out: dict[str, float] = {}
    for line in text.splitlines()[2:]:
        m = re.match(r"\s*(\S+):\s+\S+\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)", line)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


def parse_iw_link(text: str) -> float | None:
    """Parse `iw dev <iface> link` output → RSSI dBm."""
    m = re.search(r"signal:\s*(-?\d+(?:\.\d+)?)\s*dBm", text)
    return float(m.group(1)) if m else None


def parse_airport(text: str) -> float | None:
    """Parse macOS `airport -I` output → RSSI dBm."""
    m = re.search(r"agrCtlRSSI:\s*(-?\d+)", text)
    return float(m.group(1)) if m else None


def parse_netsh(text: str) -> float | None:
    """Parse Windows `netsh wlan show interfaces` → approximate dBm.

    Windows reports quality percent; the usual approximation is
    dBm ≈ % / 2 − 100 (100% ≈ −50 dBm, 0% ≈ −100 dBm).
    """
    m = re.search(r"Signal\s*:\s*(\d+)\s*%", text)
    return float(m.group(1)) / 2.0 - 100.0 if m else None


# ----------------------------------------------------------------- sampler

class RssiSampler:
    """Poll the connected Wi-Fi interface's RSSI. ``sample()`` → dBm | None."""

    def __init__(self, interface: str | None = None):
        self.interface = interface
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str | None:
        if Path("/proc/net/wireless").exists():
            if self._proc_sample() is not None:
                return "proc"
        if shutil.which("iw"):
            if self._iw_interface() is not None:
                return "iw"
        if Path(AIRPORT).exists():
            return "airport"
        if shutil.which("netsh"):
            return "netsh"
        return None

    def _run(self, cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=3).stdout
        except Exception:
            return ""

    def _proc_sample(self) -> float | None:
        try:
            table = parse_proc_wireless(Path("/proc/net/wireless").read_text())
        except OSError:
            return None
        if self.interface:
            return table.get(self.interface)
        return next(iter(table.values()), None)

    def _iw_interface(self) -> str | None:
        if self.interface:
            return self.interface
        out = self._run(["iw", "dev"])
        m = re.search(r"Interface\s+(\S+)", out)
        return m.group(1) if m else None

    def sample(self) -> float | None:
        if self.backend == "proc":
            return self._proc_sample()
        if self.backend == "iw":
            iface = self._iw_interface()
            return parse_iw_link(self._run(["iw", "dev", iface, "link"])) if iface else None
        if self.backend == "airport":
            return parse_airport(self._run([AIRPORT, "-I"]))
        if self.backend == "netsh":
            return parse_netsh(self._run(["netsh", "wlan", "show", "interfaces"]))
        return None

    @property
    def available(self) -> bool:
        return self.backend is not None


class MotionDetector:
    """Presence/motion from RSSI fluctuation.

    ``update(rssi)`` returns (motion_level, presence). motion_level is the
    rolling std (dB) over ``window_s`` minus the adaptive quiet baseline,
    clipped at 0; presence is True while motion_level exceeds ``threshold``
    within the last ``hold_s`` seconds (hysteresis so walking pauses don't
    flicker the state).
    """

    def __init__(self, window_s: float = 2.0, rate_hz: float = 10.0,
                 threshold: float = 0.6, hold_s: float = 3.0):
        self.buf: deque[float] = deque(maxlen=max(4, int(window_s * rate_hz)))
        self.threshold = threshold
        self.hold_s = hold_s
        self.baseline = 0.0
        self._last_motion_t = 0.0

    def update(self, rssi: float, now: float | None = None) -> tuple[float, bool]:
        import numpy as np

        now = time.time() if now is None else now
        self.buf.append(rssi)
        if len(self.buf) < 4:
            return 0.0, False
        std = float(np.std(self.buf))
        # Track the quiet floor slowly; only adapt downward-ish so a long
        # motion burst doesn't become the new "quiet".
        alpha = 0.02 if std < self.baseline + self.threshold else 0.002
        self.baseline += alpha * (std - self.baseline)
        level = max(0.0, std - self.baseline)
        if level > self.threshold:
            self._last_motion_t = now
        presence = (now - self._last_motion_t) < self.hold_s
        return level, presence


class RssiMonitor:
    """Background thread: sample RSSI at ``rate_hz``, keep latest reading.

    ``latest()`` → dict(rssi, motion, presence, t) or None before first
    successful sample.
    """

    def __init__(self, rate_hz: float = 10.0, interface: str | None = None):
        self.sampler = RssiSampler(interface)
        self.detector = MotionDetector(rate_hz=rate_hz)
        self.rate_hz = rate_hz
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            t0 = time.time()
            rssi = self.sampler.sample()
            if rssi is not None:
                level, presence = self.detector.update(rssi, t0)
                self._latest = {"t": t0, "rssi": rssi,
                                "motion": level, "presence": presence}
            time.sleep(max(0.0, period - (time.time() - t0)))

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="rssi-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest(self) -> dict | None:
        return self._latest
