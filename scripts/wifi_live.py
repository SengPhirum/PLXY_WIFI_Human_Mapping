#!/usr/bin/env python3
"""Live human detection from THIS machine's connected Wi-Fi (no ESP32).

Samples the real RSSI of the associated access point at 10 Hz and runs a
multi-feature detector (collect/presence.py) that reports three states:

    empty  |  person present but still  |  person moving

Detecting a *still* person is the hard part, and is done by finding the
respiration peak (0.16–0.6 Hz) in the RSSI spectrum — a stationary body's
breathing modulates the channel measurably. The detector also reports the
estimated breathing rate. Thresholds are learned automatically during a
short calibration window, so **keep the room empty for the first ~20 s**.

Two things it does to make real signal usable:
  * an active probe (ping to the gateway) so the driver refreshes RSSI
    every sample — without traffic the trace flatlines;
  * optional scanning of neighbouring APs, each an extra sensing link.

Scope: presence sensing only. One RSSI link carries no position
information, and laptops cannot expose CSI without special firmware, so
localization needs the ESP32 links (docs/HARDWARE_GUIDE.md) and pose needs
more links still (docs/POSE_GUIDE.md). Measured performance and the
research basis: reference/SENSING_RESEARCH.md.

    python scripts/wifi_live.py                     # http://127.0.0.1:8000
    python scripts/wifi_live.py --interface wlan0   # pick a specific NIC
    python scripts/wifi_live.py --sensitivity 1.0   # detect weaker signals
    python scripts/wifi_live.py --no-probe          # don't generate traffic
"""

import argparse

import _bootstrap  # noqa: F401

from wifi_mapping.config import load_config
from wifi_mapping.collect.rssi_live import RssiSampler
from wifi_mapping.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--interface", default=None, help="wireless interface name")
    ap.add_argument("--sensitivity", type=float, default=1.25,
                    help="threshold scale; <1 detects weaker signals with more "
                         "false alarms, >1 is more conservative (default 1.25)")
    ap.add_argument("--calibration", type=float, default=20.0,
                    help="quiet-room calibration seconds (keep the room empty)")
    ap.add_argument("--no-probe", action="store_true",
                    help="do not ping the gateway to keep RSSI fresh")
    ap.add_argument("--no-scan", action="store_true",
                    help="do not scan neighbouring APs for extra links")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    probe = RssiSampler(args.interface)
    if not probe.available:
        raise SystemExit(
            "No wireless interface found. This machine has no accessible "
            "Wi-Fi (checked /proc/net/wireless, iw, airport, netsh).\n"
            "Run this script on a laptop connected to Wi-Fi, or use the "
            "simulator demo: python scripts/run_demo.py"
        )
    rssi = probe.sample()
    print(f"wifi backend: {probe.backend}, current RSSI: "
          f"{rssi if rssi is not None else 'n/a'} dBm")

    app = create_app(cfg, predictor=None, mode="rssi",
                     wifi_interface=args.interface,
                     rssi_options={
                         "sensitivity": args.sensitivity,
                         "calibration_s": args.calibration,
                         "active_probe": not args.no_probe,
                         "scan_neighbours": not args.no_scan,
                     })
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"\n=== dashboard: http://{host}:{port}  |  body view: "
          f"http://{host}:{port}/body  (Ctrl-C to stop) ===\n"
          f"1. Keep the room EMPTY for the first {args.calibration:.0f}s "
          "(calibration — the dashboard shows a progress bar).\n"
          "2. Then walk in, and also try sitting still: the detector should "
          "report 'person moving' and 'person present (still)' respectively, "
          "with an estimated breathing rate.\n")
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
