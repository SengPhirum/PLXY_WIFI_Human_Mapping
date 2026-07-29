"""Presence-detection tests.

The scenario tests are the regression form of the evaluation used to tune
the detector (see reference/SENSING_RESEARCH.md): synthetic RSSI streams
that model an empty link, a breathing stationary person, and a walking
person — including the 1 dB quantization real drivers report.
"""

import numpy as np
import pytest

from wifi_mapping.collect.presence import (BREATH_HI_HZ, BREATH_LO_HZ,
                                           MultiLinkPresence, PresenceDetector,
                                           band_snr, motion_band, rssi_entropy)
from wifi_mapping.collect.rssi_live import (parse_airport_scan, parse_iw_scan,
                                            parse_netsh_networks,
                                            parse_nmcli_scan)

FS = 10.0


def _quiet(n, rng, base=-56.0, noise=0.35):
    return np.round(base + rng.normal(0, noise, n))


def _breathing(n, rng, bpm=15.0, depth=0.9, base=-56.0, noise=0.35):
    t = np.arange(n) / FS
    return np.round(base - 1.8 + depth * np.sin(2 * np.pi * (bpm / 60) * t)
                    + rng.normal(0, noise, n))


def _walking(n, rng, base=-56.0, noise=0.35):
    t = np.arange(n) / FS
    swing = 4.0 * np.sin(2 * np.pi * 0.7 * t) + 2.5 * np.sin(2 * np.pi * 1.9 * t + 1)
    return np.round(base + swing + rng.normal(0, noise * 3, n))


def _run(calib, test, **kw):
    det = PresenceDetector(rate_hz=FS, calibration_s=25.0, **kw)
    t = 0.0
    for v in calib:
        t += 1 / FS
        det.update(v, now=t)
    assert det.calibrated, "detector failed to calibrate"
    states, last = [], None
    for v in test:
        t += 1 / FS
        last = det.update(v, now=t)
        states.append(last.state)
    tail = states[len(states) // 2:]
    return tail, last


# ------------------------------------------------------------- primitives

def test_band_snr_finds_breathing_peak():
    t = np.arange(400) / FS
    x = np.sin(2 * np.pi * 0.25 * t) + np.random.default_rng(0).normal(0, 0.05, 400)
    snr, peak = band_snr(x, FS, BREATH_LO_HZ, BREATH_HI_HZ)
    assert snr > 50
    assert peak == pytest.approx(0.25, abs=0.03)   # 15 breaths/min


def test_band_snr_low_on_noise():
    x = np.random.default_rng(1).normal(0, 1.0, 400)
    snr, _ = band_snr(x, FS, BREATH_LO_HZ, BREATH_HI_HZ)
    assert snr < 30


def test_motion_band_rejects_breathing_but_keeps_walking():
    """The band-pass must separate the two — otherwise a still, breathing
    person reads as movement (the bug this filter exists to fix)."""
    t = np.arange(400) / FS
    breath = np.sin(2 * np.pi * 0.25 * t)
    walk = np.sin(2 * np.pi * 1.5 * t)
    assert np.std(motion_band(breath, FS)) < 0.1
    assert np.std(motion_band(walk, FS)) > 0.5


def test_entropy_rises_with_spread():
    flat = np.full(100, -56.0)
    spread = np.array([-56, -55, -57, -54, -58] * 20, dtype=float)
    assert rssi_entropy(flat) == pytest.approx(0.0)
    assert rssi_entropy(spread) > 1.5


# --------------------------------------------------------------- scenarios

def test_empty_room_reports_empty():
    rng = np.random.default_rng(0)
    tail, _ = _run(_quiet(300, rng), _quiet(600, rng))
    assert tail.count("empty") / len(tail) > 0.9


def test_stationary_breathing_person_detected():
    """The capability plain variance detectors lack: a person sitting still."""
    rng = np.random.default_rng(0)
    tail, last = _run(_quiet(300, rng), _breathing(600, rng, bpm=15))
    assert tail.count("stationary") / len(tail) > 0.9
    assert last.breathing_bpm == pytest.approx(15, abs=4)


def test_walking_person_detected_as_motion():
    rng = np.random.default_rng(0)
    tail, _ = _run(_quiet(300, rng), _walking(600, rng))
    assert tail.count("motion") / len(tail) > 0.9


def test_person_leaving_returns_to_empty():
    """Release latency is ~5–6 s (hold time + filter settling), so the
    state clears well within 15 s of the person leaving."""
    rng = np.random.default_rng(0)
    seq = np.concatenate([_walking(200, rng), _quiet(400, rng)])
    det = PresenceDetector(rate_hz=FS, calibration_s=25.0)
    t = 0.0
    for v in _quiet(300, rng):
        t += 1 / FS
        det.update(v, now=t)
    states = []
    for v in seq:
        t += 1 / FS
        states.append(det.update(v, now=t).state)
    assert states[150] == "motion"                    # still occupied
    assert all(s == "empty" for s in states[-150:])   # cleared within 15 s


def test_calibration_is_reported_and_recalibration_resets():
    rng = np.random.default_rng(0)
    det = PresenceDetector(rate_hz=FS, calibration_s=25.0)
    st = det.update(-56.0, now=0.1)
    assert st.state == "calibrating" and not st.calibrated
    assert 0.0 < st.calibration_progress < 1.0
    t = 0.1
    for v in _quiet(300, rng):
        t += 1 / FS
        det.update(v, now=t)
    assert det.calibrated
    det.recalibrate()
    assert not det.calibrated


# ------------------------------------------------------------- multi-link

def test_multilink_takes_strongest_state():
    rng = np.random.default_rng(0)
    fusion = MultiLinkPresence(rate_hz=FS, calibration_s=25.0)
    quiet, walk = _quiet(300, rng), _walking(300, rng)
    t = 0.0
    for i in range(300):                      # calibrate both links
        t += 1 / FS
        fusion.update("connected", quiet[i], now=t)
        fusion.update("ap:aa", quiet[i], now=t)
    assert fusion.room_state()["state"] != "calibrating"
    for i in range(300):                      # only one link is disturbed
        t += 1 / FS
        fusion.update("connected", walk[i], now=t)
        fusion.update("ap:aa", _quiet(1, rng)[0], now=t)
    room = fusion.room_state()
    assert room["state"] == "motion"
    assert room["n_links"] == 2 and room["active_links"] >= 1


# ----------------------------------------------------------- scan parsers

def test_parse_nmcli_scan():
    text = "AA\\:BB\\:CC\\:DD\\:EE\\:01:86\nAA\\:BB\\:CC\\:DD\\:EE\\:02:40\ngarbage\n"
    out = parse_nmcli_scan(text)
    assert out == {"aa:bb:cc:dd:ee:01": -57.0, "aa:bb:cc:dd:ee:02": -80.0}


def test_parse_iw_scan():
    text = ("BSS aa:bb:cc:dd:ee:01(on wlan0)\n\tsignal: -42.00 dBm\n"
            "BSS aa:bb:cc:dd:ee:02(on wlan0)\n\tsignal: -71.00 dBm\n")
    assert parse_iw_scan(text) == {"aa:bb:cc:dd:ee:01": -42.0,
                                   "aa:bb:cc:dd:ee:02": -71.0}


def test_parse_netsh_networks():
    text = ("    BSSID 1                 : aa:bb:cc:dd:ee:01\n"
            "         Signal             : 90%\n"
            "    BSSID 2                 : aa:bb:cc:dd:ee:02\n"
            "         Signal             : 50%\n")
    assert parse_netsh_networks(text) == {"aa:bb:cc:dd:ee:01": -55.0,
                                          "aa:bb:cc:dd:ee:02": -75.0}


def test_parse_airport_scan():
    text = ("                SSID BSSID             RSSI CHANNEL\n"
            "             HomeNet aa:bb:cc:dd:ee:1  -62  6\n")
    assert parse_airport_scan(text) == {"aa:bb:cc:dd:ee:01": -62.0}
