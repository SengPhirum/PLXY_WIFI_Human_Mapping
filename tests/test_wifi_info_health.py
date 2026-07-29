"""Tests for Wi-Fi device introspection and link-health diagnosis."""

import numpy as np
import pytest

from wifi_mapping.collect.health import HealthMonitor, LinkHealth, assess
from wifi_mapping.collect.wifi_info import (band_of, get_wifi_info,
                                            parse_airport_info, parse_iw_info,
                                            parse_iw_link_full,
                                            parse_netsh_interface,
                                            parse_proc_wireless_noise)

IW_LINK = """Connected to e8:9f:80:aa:bb:cc (on wlan0)
	SSID: HomeNet-5G
	freq: 5180
	RX: 1234 bytes (10 packets)
	signal: -48 dBm
	rx bitrate: 866.7 MBit/s 80MHz short GI VHT-MCS 9
"""

IW_INFO = """Interface wlan0
	ifindex 3
	addr a4:c3:f0:11:22:33
	type managed
	channel 36 (5180 MHz), width: 80 MHz, center1: 5210 MHz
	txpower 22.00 dBm
"""

AIRPORT_I = """     agrCtlRSSI: -61
     agrCtlNoise: -92
            SSID: HomeNet
           BSSID: E8:9F:80:AA:BB:CC
         channel: 11
      lastTxRate: 144
"""

NETSH_IF = """    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6 AX201 160MHz
    Physical address       : A4:C3:F0:11:22:33
    State                  : connected
    SSID                   : HomeNet
    BSSID                  : E8:9F:80:AA:BB:CC
    Channel                : 6
    Receive rate (Mbps)    : 144.4
    Signal                 : 86%
"""


# ------------------------------------------------------------- device info

def test_band_of():
    assert band_of(2437) == "2.4 GHz"
    assert band_of(5180) == "5 GHz"
    assert band_of(6115) == "6 GHz"
    assert band_of(None) is None


def test_parse_iw_link_full():
    out = parse_iw_link_full(IW_LINK)
    assert out["bssid"] == "e8:9f:80:aa:bb:cc"
    assert out["ssid"] == "HomeNet-5G"
    assert out["freq_mhz"] == 5180.0
    assert out["rssi_dbm"] == -48.0
    assert out["bitrate_mbps"] == pytest.approx(866.7)


def test_parse_iw_info():
    out = parse_iw_info(IW_INFO)
    assert out["interface"] == "wlan0"
    assert out["mac"] == "a4:c3:f0:11:22:33"
    assert out["channel"] == 36
    assert out["freq_mhz"] == 5180.0
    assert out["width_mhz"] == 80
    assert out["tx_power_dbm"] == 22.0


def test_parse_airport_info():
    out = parse_airport_info(AIRPORT_I)
    assert out["rssi_dbm"] == -61.0
    assert out["noise_dbm"] == -92.0
    assert out["ssid"] == "HomeNet"
    assert out["bssid"] == "e8:9f:80:aa:bb:cc"
    assert out["channel"] == 11
    assert out["freq_mhz"] == 2462.0     # 2407 + 5*11


def test_parse_netsh_interface():
    out = parse_netsh_interface(NETSH_IF)
    assert out["interface"] == "Wi-Fi"
    assert "AX201" in out["driver"]
    assert out["bssid"] == "e8:9f:80:aa:bb:cc"
    assert out["channel"] == 6
    assert out["rssi_dbm"] == -57.0      # 86% -> 86/2 - 100
    assert out["freq_mhz"] == 2437.0


def test_parse_proc_wireless_noise_skips_unsupported():
    text = ("Inter-| sta-|   Quality        |   Discarded packets\n"
            " face | tus | link level noise |  nwid crypt frag retry\n"
            " wlan0: 0000   54.  -56.  -92        0     0    0     0\n")
    assert parse_proc_wireless_noise(text, "wlan0") == {"noise_dbm": -92.0}
    zero = text.replace("-92", "0")
    assert parse_proc_wireless_noise(zero, "wlan0") == {}   # 0 = unsupported


def test_snr_and_dict_roundtrip():
    from wifi_mapping.collect.wifi_info import WifiInfo
    info = WifiInfo(rssi_dbm=-56.0, noise_dbm=-92.0)
    assert info.snr_db == 36.0
    assert info.to_dict()["snr_db"] == 36.0
    assert WifiInfo().snr_db is None


def test_get_wifi_info_never_raises():
    """Must degrade to empty fields rather than crash on machines with no
    wireless (CI containers, wired desktops)."""
    info = get_wifi_info()
    assert info.platform is not None
    assert isinstance(info.to_dict(), dict)


# ------------------------------------------------------------ link health

def test_healthy_link():
    mon = HealthMonitor(target_rate_hz=10.0)
    rng = np.random.default_rng(0)
    for i in range(200):
        mon.record(round(-56 + rng.normal(0, 0.8)), now=i * 0.1)
    h = mon.snapshot(probe_active=True)
    assert h.verdict == "good"
    assert h.effective_rate_hz == pytest.approx(10.0, abs=0.5)
    assert h.distinct_levels >= 3


def test_stale_link_is_flagged_with_advice():
    """The flat-trace failure: driver returns a cached value forever."""
    mon = HealthMonitor(target_rate_hz=10.0)
    for i in range(200):
        mon.record(-56.0, now=i * 0.1)
    h = mon.snapshot(probe_active=False)
    assert h.verdict == "dead"
    assert h.stale_fraction > 0.95
    assert any("cached" in i for i in h.issues)
    assert any("probe" in a.lower() for a in h.advice)


def test_pinned_rssi_is_flagged():
    """Too-stable link: the value sits on one level with rare excursions,
    so sub-dB breathing modulation is quantized away entirely.

    Note a link that *alternates* levels every sample has std ≈ 0.5 dB and
    is healthy — that is dither, and it preserves sub-dB information.
    """
    mon = HealthMonitor(target_rate_hz=10.0)
    for i in range(200):
        mon.record(-57.0 if i % 40 == 0 else -56.0, now=i * 0.1)
    h = mon.snapshot(probe_active=True)
    assert h.distinct_levels == 2
    assert h.rssi_std < 0.4
    assert any("pinned" in i for i in h.issues)


def test_dithered_two_level_link_is_not_pinned():
    mon = HealthMonitor(target_rate_hz=10.0)
    for i in range(200):
        mon.record(-56.0 if i % 2 else -57.0, now=i * 0.1)
    h = mon.snapshot(probe_active=True)
    assert not any("pinned" in i for i in h.issues)


def test_slow_sampling_is_flagged():
    mon = HealthMonitor(target_rate_hz=10.0)
    rng = np.random.default_rng(0)
    for i in range(50):
        mon.record(round(-56 + rng.normal(0, 0.8)), now=i * 1.0)   # 1 Hz
    h = mon.snapshot(probe_active=True)
    assert h.verdict in ("degraded", "dead")
    assert any("below" in i for i in h.issues)


def test_failed_reads_flagged():
    h = assess(LinkHealth(samples=10, failed_reads=50,
                          effective_rate_hz=10.0, target_rate_hz=10.0,
                          distinct_levels=5, rssi_std=1.0))
    assert any("failed" in i for i in h.issues)


def test_insufficient_samples_is_unknown():
    assert HealthMonitor().snapshot().verdict == "unknown"
