#!/usr/bin/env python3
"""Diagnose whether this machine's Wi-Fi can be used for sensing.

Run this FIRST if wifi_live.py isn't behaving. It reports what adapter was
found, what the OS exposes, how the RSSI stream actually behaves over a
short capture, and — if something is wrong — exactly what to change.

    python scripts/wifi_diagnose.py              # 20 s check
    python scripts/wifi_diagnose.py --seconds 60 # longer, more reliable
    python scripts/wifi_diagnose.py --no-probe   # test without the ping
"""

import argparse
import time

import _bootstrap  # noqa: F401

from wifi_mapping.collect.health import HealthMonitor
from wifi_mapping.collect.rssi_live import ActiveProbe, ApScanner, RssiSampler
from wifi_mapping.collect.wifi_info import get_wifi_info

TICK = "✓"
CROSS = "✗"


def row(label: str, value) -> None:
    print(f"  {label:<18} {value if value not in (None, '') else '—'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interface", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print("Wi-Fi sensing diagnostic")
    print("=" * 62)

    # ---------------------------------------------------------- 1. adapter
    print("\n[1] Wi-Fi adapter")
    info = get_wifi_info(args.interface)
    row("platform", info.platform)
    row("interface", info.interface)
    row("driver", info.driver)
    row("MAC", info.mac)
    row("source tool", info.source)
    if info.interface is None:
        print(f"\n  {CROSS} No wireless interface found.")
        print("     Wi-Fi sensing needs a real wireless adapter. This fails "
              "inside VMs,\n     WSL, containers, or on a wired-only machine. "
              "Run on the host OS of\n     a Wi-Fi-connected laptop.")
        return

    # ------------------------------------------------------------- 2. link
    print("\n[2] Associated link")
    row("SSID", info.ssid)
    row("BSSID", info.bssid)
    row("channel", info.channel)
    row("frequency", f"{info.freq_mhz:.0f} MHz" if info.freq_mhz else None)
    row("band", info.band)
    row("bitrate", f"{info.bitrate_mbps:.0f} Mbps" if info.bitrate_mbps else None)
    row("TX power", f"{info.tx_power_dbm:.0f} dBm" if info.tx_power_dbm else None)
    row("RSSI", f"{info.rssi_dbm:.0f} dBm" if info.rssi_dbm is not None else None)
    row("noise", f"{info.noise_dbm:.0f} dBm" if info.noise_dbm is not None else None)
    row("SNR", f"{info.snr_db} dB" if info.snr_db is not None else None)
    if info.band == "5 GHz":
        print("\n  note: 5 GHz links are often very stable, which is bad for "
              "sensing —\n        2.4 GHz usually carries more body-induced "
              "variation.")

    # --------------------------------------------------------- 3. sampling
    sampler = RssiSampler(args.interface)
    if not sampler.available:
        print(f"\n  {CROSS} No RSSI backend available "
              "(/proc/net/wireless, iw, airport, netsh).")
        return
    print(f"\n[3] RSSI sampling  (backend: {sampler.backend})")

    probe = None
    if not args.no_probe:
        probe = ActiveProbe(rate_hz=args.rate)
        started = probe.start()
        row("active probe", f"{TICK} pinging {probe.target}" if started
            else f"{CROSS} unavailable (no gateway found)")
    else:
        row("active probe", "disabled by --no-probe")

    health = HealthMonitor(target_rate_hz=args.rate)
    print(f"  sampling for {args.seconds:.0f}s ", end="", flush=True)
    period, t_end = 1.0 / args.rate, time.time() + args.seconds
    next_dot = time.time() + args.seconds / 20
    while time.time() < t_end:
        t0 = time.time()
        health.record(sampler.sample(), t0)
        if time.time() > next_dot:
            print(".", end="", flush=True)
            next_dot += args.seconds / 20
        time.sleep(max(0.0, period - (time.time() - t0)))
    print()
    if probe is not None:
        probe.stop()

    h = health.snapshot(probe_active=bool(probe and probe._proc))
    row("samples", h.samples)
    row("effective rate", f"{h.effective_rate_hz:.1f} Hz "
                          f"(target {h.target_rate_hz:.0f})")
    row("repeated values", f"{h.stale_fraction:.0%}")
    row("distinct levels", h.distinct_levels)
    row("RSSI std", f"{h.rssi_std:.2f} dB")
    row("RSSI range", f"{h.rssi_min:.0f} … {h.rssi_max:.0f} dBm"
        if h.rssi_min is not None else None)
    row("failed reads", h.failed_reads)

    # ------------------------------------------------------- 4. neighbours
    print("\n[4] Neighbouring access points (extra sensing links)")
    scanner = ApScanner(args.interface)
    if not scanner.available:
        row("scanner", f"{CROSS} none available")
    else:
        aps = scanner.scan()
        row("scanner", scanner.backend)
        row("APs found", len(aps))
        for bssid, rssi in list(aps.items())[:6]:
            print(f"      {bssid}   {rssi:6.0f} dBm")

    # ---------------------------------------------------------- 5. verdict
    print("\n" + "=" * 62)
    mark = {"good": TICK, "degraded": "!", "dead": CROSS}.get(h.verdict, "?")
    print(f"VERDICT: {mark} {h.verdict.upper()}")
    print("=" * 62)
    if h.issues:
        print("\nIssues:")
        for i in h.issues:
            print(f"  - {i}")
    if h.advice:
        print("\nWhat to do:")
        for a in h.advice:
            print(f"  * {a}")
    if h.verdict == "good":
        print("\nThe link looks usable. Start the detector with:")
        print("    python scripts/wifi_live.py")
        print("Keep the room empty for the first ~20 s while it calibrates.")
    elif h.verdict == "dead":
        print("\nSensing will not work until the issues above are fixed —")
        print("the detector cannot recover information the driver never provides.")


if __name__ == "__main__":
    main()
