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

Two practical problems dominate real deployments, and both are handled
here:

- **Stale RSSI.** Drivers only refresh the reported RSSI when frames
  actually arrive. On an idle link the trace flatlines and no detector can
  work. :class:`ActiveProbe` sends steady background traffic to the gateway
  so every sample is fresh — this is usually the difference between "the
  demo does nothing" and "the demo works".
- **One link is not much.** :class:`ApScanner` harvests the RSSI of
  *neighbouring* access points from beacon scans. Each AP↔laptop path
  crosses a different part of the building, giving genuine multi-link
  sensing from a single laptop with no extra hardware.

Detection itself lives in :mod:`presence` (breathing-band spectral
analysis, adaptive thresholds, multi-feature fusion). :class:`MotionDetector`
below is the original simple variance detector, kept as a baseline.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path  # noqa: F401  (used by ActiveProbe/ApScanner)

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


# ------------------------------------------------------- neighbouring APs

def parse_nmcli_scan(text: str) -> dict[str, float]:
    """Parse ``nmcli -t -f BSSID,SIGNAL device wifi list`` → {bssid: dBm}.

    nmcli escapes the colons in a BSSID as ``\\:``; signal is a 0–100
    quality percent, converted with the same approximation as netsh.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # BSSID contains escaped colons, so split on the last unescaped one.
        m = re.match(r"^(.*[0-9A-Fa-f]{2}):(\d{1,3})$", line)
        if not m:
            continue
        bssid = m.group(1).replace("\\:", ":").lower()
        if len(bssid.split(":")) != 6:
            continue
        out[bssid] = float(m.group(2)) / 2.0 - 100.0
    return out


def parse_iw_scan(text: str) -> dict[str, float]:
    """Parse ``iw dev <if> scan dump`` → {bssid: dBm}."""
    out: dict[str, float] = {}
    bssid = None
    for line in text.splitlines():
        m = re.match(r"BSS ([0-9a-fA-F:]{17})", line.strip())
        if m:
            bssid = m.group(1).lower()
            continue
        m = re.search(r"signal:\s*(-?\d+(?:\.\d+)?)\s*dBm", line)
        if m and bssid:
            out[bssid] = float(m.group(1))
            bssid = None
    return out


def parse_netsh_networks(text: str) -> dict[str, float]:
    """Parse ``netsh wlan show networks mode=bssid`` → {bssid: dBm}."""
    out: dict[str, float] = {}
    bssid = None
    for line in text.splitlines():
        m = re.search(r"BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})", line)
        if m:
            bssid = m.group(1).lower()
            continue
        m = re.search(r"Signal\s*:\s*(\d+)\s*%", line)
        if m and bssid:
            out[bssid] = float(m.group(1)) / 2.0 - 100.0
            bssid = None
    return out


def parse_airport_scan(text: str) -> dict[str, float]:
    """Parse macOS ``airport -s`` → {bssid: dBm}."""
    out: dict[str, float] = {}
    for line in text.splitlines()[1:]:
        m = re.search(r"([0-9a-fA-F]{1,2}(?::[0-9a-fA-F]{1,2}){5})\s+(-\d+)", line)
        if m:
            parts = [p.zfill(2) for p in m.group(1).split(":")]
            out[":".join(parts).lower()] = float(m.group(2))
    return out


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


class ActiveProbe:
    """Keep the link busy so the driver refreshes RSSI every sample.

    Without traffic, most drivers report the RSSI of the last received
    frame — which on an idle link can be seconds old, flatlining the trace.
    A low-rate ping to the default gateway (a few hundred bytes/s) forces a
    fresh measurement per interval. Harmless to the network; stops cleanly.
    """

    def __init__(self, target: str | None = None, rate_hz: float = 10.0):
        self.target = target or self._default_gateway()
        self.rate_hz = rate_hz
        self._proc: subprocess.Popen | None = None

    @staticmethod
    def _default_gateway() -> str | None:
        """Best-effort default-gateway lookup across platforms."""
        try:
            out = subprocess.run(["ip", "route"], capture_output=True,
                                 text=True, timeout=3).stdout
            m = re.search(r"default via (\S+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
        for cmd, pat in (
            (["route", "-n", "get", "default"], r"gateway:\s*(\S+)"),   # macOS
            (["ipconfig"], r"Default Gateway.*?:\s*([0-9.]+)"),          # Windows
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=3).stdout
                m = re.search(pat, out)
                if m and m.group(1).strip():
                    return m.group(1).strip()
            except Exception:
                continue
        return None

    @property
    def available(self) -> bool:
        return bool(self.target) and shutil.which("ping") is not None

    def start(self) -> bool:
        if not self.available or self._proc is not None:
            return False
        interval = max(0.05, 1.0 / self.rate_hz)
        is_windows = not Path("/proc").exists() and shutil.which("cmd")
        cmd = (["ping", "-t", self.target] if is_windows
               else ["ping", "-i", f"{interval:.2f}", self.target])
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            # Sub-second ping intervals need root on some systems; retry at 1 Hz.
            try:
                self._proc = subprocess.Popen(
                    ["ping", self.target], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                return True
            except Exception:
                return False

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None


class ApScanner:
    """RSSI of neighbouring access points, for multi-link sensing.

    Every AP the laptop can hear is a separate propagation path through the
    building. Scanning is slow (~1–3 s) and briefly interrupts the
    connection, so this runs at a low rate and is best used for motion, not
    breathing — the connected-link sampler stays fast for that.

    Prefers ``nmcli`` (uses NetworkManager's cache, no root) then ``iw scan
    dump`` (also cached), falling back to platform equivalents.
    """

    def __init__(self, interface: str | None = None, min_rssi: float = -85.0,
                 max_aps: int = 6):
        self.interface = interface
        self.min_rssi = min_rssi
        self.max_aps = max_aps
        self.backend = self._detect_backend()

    def _detect_backend(self) -> str | None:
        if shutil.which("nmcli"):
            return "nmcli"
        if shutil.which("iw"):
            return "iw"
        if Path(AIRPORT).exists():
            return "airport"
        if shutil.which("netsh"):
            return "netsh"
        return None

    @property
    def available(self) -> bool:
        return self.backend is not None

    def _run(self, cmd: list[str], timeout: float = 8.0) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout).stdout
        except Exception:
            return ""

    def scan(self) -> dict[str, float]:
        """{bssid: rssi_dbm} for the strongest neighbouring APs."""
        if self.backend == "nmcli":
            table = parse_nmcli_scan(
                self._run(["nmcli", "-t", "-f", "BSSID,SIGNAL", "device", "wifi", "list"]))
        elif self.backend == "iw":
            iface = self.interface or "wlan0"
            table = parse_iw_scan(self._run(["iw", "dev", iface, "scan", "dump"]))
        elif self.backend == "airport":
            table = parse_airport_scan(self._run([AIRPORT, "-s"]))
        elif self.backend == "netsh":
            table = parse_netsh_networks(
                self._run(["netsh", "wlan", "show", "networks", "mode=bssid"]))
        else:
            return {}
        strong = {b: r for b, r in table.items() if r >= self.min_rssi}
        return dict(sorted(strong.items(), key=lambda kv: -kv[1])[: self.max_aps])


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
    """Background sensing on the machine's own Wi-Fi.

    Runs three things: a fast sampler on the connected link, an optional
    :class:`ActiveProbe` keeping that link's RSSI fresh, and an optional
    slow :class:`ApScanner` adding neighbouring APs as extra links. All
    streams feed :class:`~.presence.MultiLinkPresence`.

    ``latest()`` → dict with the fused room state, per-link detail, and the
    connected link's raw RSSI, or None before the first sample.
    """

    SCAN_RATE_HZ = 0.5   # neighbour scans are slow and disturb the link

    def __init__(self, rate_hz: float = 10.0, interface: str | None = None,
                 active_probe: bool = True, scan_neighbours: bool = True,
                 calibration_s: float = 20.0, sensitivity: float = 1.25):
        from .health import HealthMonitor
        from .presence import MultiLinkPresence

        self.sampler = RssiSampler(interface)
        self.rate_hz = rate_hz
        self.interface = interface
        self.health = HealthMonitor(target_rate_hz=rate_hz)
        self.device: dict = {}
        self._device_refreshed = 0.0
        self.presence = MultiLinkPresence(rate_hz=rate_hz,
                                          calibration_s=calibration_s,
                                          sensitivity=sensitivity)
        # Legacy simple detector, kept so the baseline stays comparable.
        self.detector = MotionDetector(rate_hz=rate_hz)

        self.probe = ActiveProbe(rate_hz=rate_hz) if active_probe else None
        self.scanner = ApScanner(interface) if scan_neighbours else None
        self.probe_active = False

        self._latest: dict | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------- threads

    def _refresh_device(self, now: float) -> None:
        """Adapter/link details are slow to query — refresh every 5 s."""
        if now - self._device_refreshed < 5.0:
            return
        from .wifi_info import get_wifi_info

        self._device_refreshed = now
        try:
            self.device = get_wifi_info(self.interface).to_dict()
        except Exception:
            self.device = {}

    def _sample_loop(self) -> None:
        period = 1.0 / self.rate_hz
        n = 0
        while not self._stop.is_set():
            t0 = time.time()
            rssi = self.sampler.sample()
            self.health.record(rssi, t0)
            self._refresh_device(t0)
            if rssi is not None:
                self.presence.update("connected", rssi, t0)
                level, moving = self.detector.update(rssi, t0)
                room = self.presence.room_state()
                event = {
                    "t": t0,
                    "rssi": rssi,
                    "motion": level,               # legacy baseline fields
                    "presence": moving,
                    "probe_active": self.probe_active,
                    "device": self.device,
                    **room,
                }
                # Spectrum/histogram/health are heavier and change slowly —
                # attach them a few times a second, not on every sample.
                n += 1
                if n % max(1, int(self.rate_hz // 2)) == 0:
                    det = self.presence.detectors.get("connected")
                    event["health"] = self.health.snapshot(self.probe_active).to_dict()
                    if det is not None:
                        event["spectrum"] = det.spectrum()
                        event["histogram"] = det.histogram()
                        event["thresholds"] = {k: round(v, 3)
                                               for k, v in det.thresholds.items()}
                self._latest = event
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _scan_loop(self) -> None:
        period = 1.0 / self.SCAN_RATE_HZ
        while not self._stop.is_set():
            t0 = time.time()
            try:
                for bssid, rssi in self.scanner.scan().items():
                    self.presence.update(f"ap:{bssid}", rssi, t0,
                                         rate_hz=self.SCAN_RATE_HZ)
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.time() - t0)))

    # --------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._stop.clear()
        if self.probe is not None:
            self.probe_active = self.probe.start()
        self._threads = [threading.Thread(target=self._sample_loop, daemon=True,
                                          name="rssi-monitor")]
        if self.scanner is not None and self.scanner.available:
            self._threads.append(threading.Thread(target=self._scan_loop,
                                                  daemon=True, name="ap-scanner"))
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()
        if self.probe is not None:
            self.probe.stop()
            self.probe_active = False

    def recalibrate(self) -> None:
        self.presence.recalibrate()

    def latest(self) -> dict | None:
        return self._latest
