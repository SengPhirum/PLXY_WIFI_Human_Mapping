import numpy as np

from wifi_mapping.collect.rssi_live import (MotionDetector, parse_airport,
                                            parse_iw_link, parse_netsh,
                                            parse_proc_wireless)

PROC = """Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE
 face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22
 wlan0: 0000   54.  -56.  -256        0      0      0      0      0        0
"""

IW = """Connected to aa:bb:cc:dd:ee:ff (on wlan0)
	SSID: HomeNet
	freq: 2412
	signal: -58 dBm
	tx bitrate: 144.4 MBit/s
"""

AIRPORT = """     agrCtlRSSI: -61
     agrExtRSSI: 0
            SSID: HomeNet
"""

NETSH = """    Name                   : Wi-Fi
    State                  : connected
    Signal                 : 86%
"""


def test_parse_proc_wireless():
    assert parse_proc_wireless(PROC) == {"wlan0": -56.0}
    assert parse_proc_wireless("junk\nheader\n") == {}


def test_parse_iw_link():
    assert parse_iw_link(IW) == -58.0
    assert parse_iw_link("Not connected.") is None


def test_parse_airport():
    assert parse_airport(AIRPORT) == -61.0
    assert parse_airport("AirPort: Off") is None


def test_parse_netsh():
    assert parse_netsh(NETSH) == 86 / 2 - 100  # -57 dBm
    assert parse_netsh("There is no wireless interface") is None


def test_motion_detector_flags_fluctuation():
    det = MotionDetector(window_s=2.0, rate_hz=10.0, threshold=0.6, hold_s=1.0)
    rng = np.random.default_rng(0)
    t = 0.0
    # Quiet period: tiny noise → no presence after warm-up.
    for _ in range(60):
        t += 0.1
        level, presence = det.update(-56 + rng.normal(0, 0.1), now=t)
    assert not presence
    # Person walking: several-dB swings → presence flips on.
    for _ in range(30):
        t += 0.1
        level, presence = det.update(-56 + rng.normal(0, 3.0), now=t)
    assert presence
    # Quiet again for longer than hold_s → presence releases.
    for _ in range(80):
        t += 0.1
        level, presence = det.update(-56 + rng.normal(0, 0.1), now=t)
    assert not presence
