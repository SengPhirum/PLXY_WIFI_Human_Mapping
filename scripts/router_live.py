#!/usr/bin/env python3
"""Whole-home activity sensing through the Wi-Fi router.

Polls the router for the RSSI of EVERY connected device (phone, TV,
laptop, …) and runs per-link motion detection: movement between the router
and a device disturbs that device's link, so with a handful of stationary
devices spread around the home you get a coarse activity map — which link
region is occupied — from hardware you already own.

Requires a router with shell access exposing `iw` (OpenWrt, DD-WRT, GL.iNet
and similar; see docs/ROUTER_GUIDE.md, including how to set up ssh keys).
Configure the router host and, optionally, MAC → name/position mapping in
config/default.yaml under `router:`.

Examples:
    python scripts/router_live.py --demo               # no router needed: fake stations
    python scripts/router_live.py                      # uses config router.host/user
    python scripts/router_live.py --host 192.168.1.1 --user root
    python scripts/router_live.py --local --interfaces wlan0   # this machine IS the AP

Then open http://127.0.0.1:8000 — floor plan shows router→device links,
line width/colour = live motion on that link; the device table lists RSSI,
motion level, and state for every station.
"""

import argparse

import _bootstrap  # noqa: F401

from wifi_mapping.collect.router_live import (make_demo_transport,
                                              make_local_transport,
                                              make_ssh_transport,
                                              parse_iw_station_dump)
from wifi_mapping.config import load_config
from wifi_mapping.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default=None, help="router address (default: config router.host)")
    ap.add_argument("--user", default=None, help="ssh user (default: config router.user)")
    ap.add_argument("--interfaces", nargs="+", default=None,
                    help="AP interfaces to poll (default: config router.interfaces)")
    ap.add_argument("--local", action="store_true",
                    help="this machine is the AP: run iw locally instead of ssh")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic stations — try the UI without any router")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--bind", default=None, help="server bind host")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg = cfg.router or {}
    interfaces = tuple(args.interfaces or rcfg.get("interfaces") or ["wlan0", "wlan1"])

    if args.demo:
        transport = make_demo_transport()
        # Give the demo stations positions so the map lights up.
        cfg.router = {**rcfg, "position": rcfg.get("position", [2.5, 0.15]),
                      "devices": {
                          "aa:bb:cc:00:00:00": {"name": "TV", "pos": [0.6, 5.4]},
                          "aa:bb:cc:00:00:01": {"name": "Desk PC", "pos": [4.4, 5.2]},
                          "aa:bb:cc:00:00:02": {"name": "Printer", "pos": [4.6, 0.9]},
                          "aa:bb:cc:00:00:03": {"name": "Tablet", "pos": [0.8, 1.2]},
                      }}
        print("demo mode: synthetic stations, no router contacted")
    elif args.local:
        transport = make_local_transport(interfaces)
        print(f"local AP mode: polling iw on {', '.join(interfaces)}")
    else:
        host = args.host or rcfg.get("host")
        user = args.user or rcfg.get("user", "root")
        if not host:
            raise SystemExit("no router host: pass --host or set router.host "
                             "in config/default.yaml (or try --demo)")
        transport = make_ssh_transport(host, user, interfaces)
        print(f"router mode: ssh {user}@{host}, interfaces {', '.join(interfaces)}")
        stations = parse_iw_station_dump(transport())
        if stations:
            print(f"reachable: {len(stations)} station(s): "
                  + ", ".join(sorted(stations)))
        else:
            print("WARNING: no stations returned. Check: ssh key auth "
                  f"({user}@{host} without password), `iw` present on the "
                  "router, interface names (try: ssh router 'iw dev'). "
                  "Continuing — will keep retrying.")

    app = create_app(cfg, predictor=None, mode="router",
                     router_transport=transport)
    host_ = args.bind or cfg.server.host
    port = args.port or cfg.server.port
    print(f"\n=== dashboard: http://{host_}:{port} (Ctrl-C to stop) ===\n")
    import uvicorn
    uvicorn.run(app, host=host_, port=port, log_level="warning")


if __name__ == "__main__":
    main()
