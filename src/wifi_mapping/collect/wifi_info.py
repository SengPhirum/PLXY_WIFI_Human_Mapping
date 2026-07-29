"""Cross-platform Wi-Fi adapter and link details.

Answers "is the Wi-Fi device actually working, and what is it doing?" —
adapter name and driver, associated network, BSSID, channel, frequency
band, negotiated bitrate, TX power, and noise floor where the OS exposes
them. Used by the live dashboard's device panel and by
``scripts/wifi_diagnose.py``.

Every value is best-effort: OSes and drivers expose different subsets, so
each field is ``None`` when unavailable rather than faked.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

AIRPORT = ("/System/Library/PrivateFrameworks/Apple80211.framework/"
           "Versions/Current/Resources/airport")


@dataclass
class WifiInfo:
    interface: str | None = None
    driver: str | None = None
    mac: str | None = None
    ssid: str | None = None
    bssid: str | None = None
    channel: int | None = None
    freq_mhz: float | None = None
    band: str | None = None          # "2.4 GHz" | "5 GHz" | "6 GHz"
    width_mhz: int | None = None
    bitrate_mbps: float | None = None
    tx_power_dbm: float | None = None
    rssi_dbm: float | None = None
    noise_dbm: float | None = None
    platform: str | None = None
    source: str | None = None        # which tool produced this

    @property
    def snr_db(self) -> float | None:
        if self.rssi_dbm is not None and self.noise_dbm is not None:
            return round(self.rssi_dbm - self.noise_dbm, 1)
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["snr_db"] = self.snr_db
        return d


def band_of(freq_mhz: float | None) -> str | None:
    if freq_mhz is None:
        return None
    if freq_mhz < 2500:
        return "2.4 GHz"
    if freq_mhz < 5925:
        return "5 GHz"
    return "6 GHz"


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return ""


# ------------------------------------------------------------- parsers

def parse_iw_link_full(text: str) -> dict:
    """Parse ``iw dev <if> link`` (association details)."""
    out: dict = {}
    m = re.search(r"Connected to ([0-9a-fA-F:]{17})", text)
    if m:
        out["bssid"] = m.group(1).lower()
    m = re.search(r"SSID:\s*(.+)", text)
    if m:
        out["ssid"] = m.group(1).strip()
    m = re.search(r"freq:\s*(\d+(?:\.\d+)?)", text)
    if m:
        out["freq_mhz"] = float(m.group(1))
    m = re.search(r"signal:\s*(-?\d+(?:\.\d+)?)\s*dBm", text)
    if m:
        out["rssi_dbm"] = float(m.group(1))
    m = re.search(r"rx bitrate:\s*(\d+(?:\.\d+)?)\s*MBit/s", text)
    if m:
        out["bitrate_mbps"] = float(m.group(1))
    m = re.search(r"(\d+)MHz", text)
    if m:
        out["width_mhz"] = int(m.group(1))
    return out


def parse_iw_info(text: str) -> dict:
    """Parse ``iw dev <if> info`` (adapter details)."""
    out: dict = {}
    m = re.search(r"Interface\s+(\S+)", text)
    if m:
        out["interface"] = m.group(1)
    m = re.search(r"addr\s+([0-9a-fA-F:]{17})", text)
    if m:
        out["mac"] = m.group(1).lower()
    m = re.search(r"channel\s+(\d+)\s*\((\d+)\s*MHz\)(?:.*?width:\s*(\d+))?", text)
    if m:
        out["channel"] = int(m.group(1))
        out["freq_mhz"] = float(m.group(2))
        if m.group(3):
            out["width_mhz"] = int(m.group(3))
    m = re.search(r"txpower\s+(-?\d+(?:\.\d+)?)\s*dBm", text)
    if m:
        out["tx_power_dbm"] = float(m.group(1))
    return out


def parse_airport_info(text: str) -> dict:
    """Parse macOS ``airport -I``."""
    out: dict = {}
    pairs = {
        "agrCtlRSSI": ("rssi_dbm", float),
        "agrCtlNoise": ("noise_dbm", float),
        "SSID": ("ssid", str),
        "BSSID": ("bssid", str),
        "channel": ("channel_raw", str),
        "lastTxRate": ("bitrate_mbps", float),
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key in pairs:
            name, cast = pairs[key]
            try:
                out[name] = cast(val)
            except ValueError:
                pass
    raw = out.pop("channel_raw", None)
    if raw:
        m = re.match(r"(\d+)", raw)
        if m:
            ch = int(m.group(1))
            out["channel"] = ch
            out["freq_mhz"] = float(2407 + 5 * ch if ch <= 14 else 5000 + 5 * ch)
    if isinstance(out.get("bssid"), str):
        out["bssid"] = out["bssid"].lower()
    return out


def parse_netsh_interface(text: str) -> dict:
    """Parse Windows ``netsh wlan show interfaces``."""
    out: dict = {}
    fields = {
        "Name": ("interface", str),
        "Description": ("driver", str),
        "Physical address": ("mac", str),
        "SSID": ("ssid", str),
        "BSSID": ("bssid", str),
        "Channel": ("channel", int),
        "Receive rate (Mbps)": ("bitrate_mbps", float),
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key in fields and val:
            name, cast = fields[key]
            try:
                out[name] = cast(val)
            except ValueError:
                pass
    m = re.search(r"Signal\s*:\s*(\d+)\s*%", text)
    if m:
        out["rssi_dbm"] = float(m.group(1)) / 2.0 - 100.0
    ch = out.get("channel")
    if ch:
        out["freq_mhz"] = float(2407 + 5 * ch if ch <= 14 else 5000 + 5 * ch)
    for k in ("mac", "bssid"):
        if isinstance(out.get(k), str):
            out[k] = out[k].lower()
    return out


def parse_proc_wireless_noise(text: str, interface: str | None = None) -> dict:
    """Noise level column of /proc/net/wireless (often 0 / unsupported)."""
    for line in text.splitlines()[2:]:
        m = re.match(r"\s*(\S+):\s+\S+\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)"
                     r"\s+(-?\d+(?:\.\d*)?)", line)
        if m and (interface is None or m.group(1) == interface):
            noise = float(m.group(4))
            return {"noise_dbm": noise} if noise not in (0.0, 256.0) else {}
    return {}


# --------------------------------------------------------------- gather

def _linux_driver(interface: str) -> str | None:
    link = Path(f"/sys/class/net/{interface}/device/driver")
    try:
        return link.resolve().name
    except OSError:
        return None


def get_wifi_info(interface: str | None = None) -> WifiInfo:
    """Collect adapter + link details from whichever tools exist."""
    info = WifiInfo(platform=platform.system())

    if Path("/proc/net/wireless").exists() or shutil.which("iw"):
        iface = interface
        if iface is None and shutil.which("iw"):
            m = re.search(r"Interface\s+(\S+)", _run(["iw", "dev"]))
            iface = m.group(1) if m else None
        if iface is None:
            try:
                text = Path("/proc/net/wireless").read_text()
                m = re.search(r"^\s*(\S+):", text.splitlines()[2]) if \
                    len(text.splitlines()) > 2 else None
                iface = m.group(1) if m else None
            except Exception:
                iface = None
        if iface:
            info.interface = iface
            info.driver = _linux_driver(iface)
            info.source = "iw"
            if shutil.which("iw"):
                for parsed in (parse_iw_info(_run(["iw", "dev", iface, "info"])),
                               parse_iw_link_full(_run(["iw", "dev", iface, "link"]))):
                    for k, v in parsed.items():
                        if v is not None and hasattr(info, k):
                            setattr(info, k, v)
            try:
                for k, v in parse_proc_wireless_noise(
                        Path("/proc/net/wireless").read_text(), iface).items():
                    setattr(info, k, v)
            except Exception:
                pass

    elif Path(AIRPORT).exists():
        info.source = "airport"
        for k, v in parse_airport_info(_run([AIRPORT, "-I"])).items():
            if hasattr(info, k):
                setattr(info, k, v)

    elif shutil.which("netsh"):
        info.source = "netsh"
        for k, v in parse_netsh_interface(
                _run(["netsh", "wlan", "show", "interfaces"])).items():
            if hasattr(info, k):
                setattr(info, k, v)

    info.band = band_of(info.freq_mhz)
    return info
