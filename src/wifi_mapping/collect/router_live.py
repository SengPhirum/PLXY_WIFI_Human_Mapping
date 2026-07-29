"""Multi-device sensing through the Wi-Fi router.

The router's AP interface knows the RSSI of *every* associated station
(phone, TV, laptop, plugs…). Each station↔router link crosses a different
part of the home, so per-link RSSI disturbance localizes activity coarsely:
motion on the TV link but not the desk link means someone is moving in the
TV area. This turns one router plus the devices you already own into a
multi-link sensing mesh — the same idea as the ESP32 array, at zero cost
and RSSI (not CSI) fidelity.

Supported sources for the per-station signal table:

- ``ssh``    — OpenWrt/DD-WRT-class routers: runs ``iw dev <if> station
  dump`` over ssh (key-based auth; dropbear and openssh both fine).
- ``local``  — the machine running this code *is* the AP (hostapd):
  runs ``iw`` locally.
- ``demo``   — synthetic stations for UI development and demos.

Consumer routers without shell access do not expose per-station RSSI in any
standard way — see docs/ROUTER_GUIDE.md for options (OpenWrt flashing, or
using a secondary OpenWrt AP as the sensing head).
"""

from __future__ import annotations

import math
import random
import re
import shutil
import subprocess
import threading
import time
from typing import Callable

from .rssi_live import MotionDetector

MAC_RE = re.compile(r"^Station\s+([0-9a-fA-F:]{17})", re.M)


def parse_iw_station_dump(text: str) -> dict[str, dict]:
    """Parse ``iw dev <iface> station dump`` → {mac: {signal, inactive_ms}}.

    Uses "signal avg" when present (smoother), else instantaneous "signal"
    (first value before any per-chain "[...]" list).
    """
    stations: dict[str, dict] = {}
    blocks = re.split(r"(?=^Station\s)", text, flags=re.M)
    for block in blocks:
        m = MAC_RE.match(block)
        if not m:
            continue
        mac = m.group(1).lower()
        entry: dict = {}
        sig_avg = re.search(r"signal avg:\s*(-?\d+)", block)
        sig = re.search(r"signal:\s*(-?\d+)", block)
        if sig_avg:
            entry["signal"] = float(sig_avg.group(1))
        elif sig:
            entry["signal"] = float(sig.group(1))
        else:
            continue
        inactive = re.search(r"inactive time:\s*(\d+)\s*ms", block)
        entry["inactive_ms"] = int(inactive.group(1)) if inactive else 0
        stations[mac] = entry
    return stations


# -------------------------------------------------------------- transports

def make_ssh_transport(host: str, user: str = "root",
                       interfaces: tuple[str, ...] = ("wlan0", "wlan1"),
                       timeout: float = 4.0) -> Callable[[], str]:
    """Return a callable that fetches station dumps from a router via ssh."""
    cmd_remote = "; ".join(f"iw dev {i} station dump 2>/dev/null" for i in interfaces)

    def fetch() -> str:
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                 f"{user}@{host}", cmd_remote],
                capture_output=True, text=True, timeout=timeout)
            return r.stdout
        except Exception:
            return ""
    return fetch


def make_local_transport(interfaces: tuple[str, ...] = ("wlan0",)) -> Callable[[], str]:
    """Station dumps from a local hostapd AP interface."""
    def fetch() -> str:
        if not shutil.which("iw"):
            return ""
        out = []
        for iface in interfaces:
            try:
                r = subprocess.run(["iw", "dev", iface, "station", "dump"],
                                   capture_output=True, text=True, timeout=3)
                out.append(r.stdout)
            except Exception:
                pass
        return "\n".join(out)
    return fetch


def make_demo_transport(n_stations: int = 4, seed: int = 0) -> Callable[[], str]:
    """Synthetic station dump: quiet baselines with a wandering 'person'
    disturbing one link at a time, for UI demos and tests."""
    rng = random.Random(seed)
    macs = [f"aa:bb:cc:00:00:{i:02x}" for i in range(n_stations)]
    base = {m: -45 - 8 * i for i, m in enumerate(macs)}
    t0 = time.time()

    def fetch() -> str:
        t = time.time() - t0
        active = int(t / 8) % n_stations          # person moves link to link
        out = []
        for i, m in enumerate(macs):
            sig = base[m] + rng.gauss(0, 0.3)
            if i == active:
                sig += 4.0 * math.sin(t * 6.0) + rng.gauss(0, 1.2)
            out.append(f"Station {m} (on wlan0)\n"
                       f"\tinactive time:\t{rng.randint(0, 500)} ms\n"
                       f"\tsignal:  \t{sig:.0f} [-60] dBm\n"
                       f"\tsignal avg:\t{sig:.0f} dBm\n")
        return "\n".join(out)
    return fetch


# ----------------------------------------------------------------- monitor

class RouterMonitor:
    """Poll a station-dump transport; per-station motion/presence state.

    ``latest()`` → {"t": ..., "stations": [{mac, rssi, motion, presence,
    inactive_ms}, ...]} sorted by mac. Stations idle for > ``stale_s`` are
    dropped (device left / asleep).
    """

    def __init__(self, transport: Callable[[], str], poll_hz: float = 2.0,
                 threshold: float = 0.8, hold_s: float = 4.0,
                 stale_s: float = 120.0):
        self.transport = transport
        self.poll_hz = poll_hz
        self.threshold = threshold
        self.hold_s = hold_s
        self.stale_s = stale_s
        self.detectors: dict[str, MotionDetector] = {}
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        stations = parse_iw_station_dump(self.transport())
        rows = []
        for mac in sorted(stations):
            info = stations[mac]
            if info["inactive_ms"] > self.stale_s * 1000:
                continue
            det = self.detectors.get(mac)
            if det is None:
                det = self.detectors[mac] = MotionDetector(
                    window_s=3.0, rate_hz=self.poll_hz,
                    threshold=self.threshold, hold_s=self.hold_s)
            level, presence = det.update(info["signal"], now)
            rows.append({"mac": mac, "rssi": info["signal"],
                         "motion": round(level, 3), "presence": presence,
                         "inactive_ms": info["inactive_ms"]})
        self._latest = {"t": now, "stations": rows}
        return self._latest

    def _loop(self) -> None:
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.poll_once(t0)
            except Exception:
                pass
            time.sleep(max(0.0, period - (time.time() - t0)))

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="router-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def latest(self) -> dict | None:
        return self._latest
