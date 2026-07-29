#!/usr/bin/env python3
"""Live sensing from THIS machine's connected Wi-Fi interface (no ESP32).

Streams the real RSSI of the currently associated access point at 10 Hz,
detects motion/presence from signal fluctuation, and serves the dashboard
(signal panel) plus the /body avatar view.

This is presence/motion sensing only: one laptop↔AP link carries no
position information, and laptops cannot expose CSI without special
firmware. Localization needs the ESP32 links (docs/HARDWARE_GUIDE.md) or
the simulator demo (scripts/run_demo.py).

    python scripts/wifi_live.py                     # http://127.0.0.1:8000
    python scripts/wifi_live.py --interface wlan0   # pick a specific NIC
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
                     wifi_interface=args.interface)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"\n=== dashboard: http://{host}:{port}  |  body view: "
          f"http://{host}:{port}/body  (Ctrl-C to stop) ===\n"
          "Walk between this machine and your Wi-Fi router — the motion "
          "meter and presence state react to the real signal.\n")
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
