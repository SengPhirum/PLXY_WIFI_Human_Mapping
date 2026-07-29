#!/usr/bin/env python3
"""Generate a simulated CSI dataset following the collection protocol.

Sessions are spread over days and participants so that all the plan's
independence splits (session / day / person) are exercisable.

Example (the demo defaults):
    python scripts/generate_dataset.py --name demo --sessions 8 --days 4 --persons 4
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from wifi_mapping.config import load_config
from wifi_mapping.dataset import generate_session, save_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="config YAML (default: config/default.yaml)")
    ap.add_argument("--name", default="demo", help="dataset name (directory under data/datasets)")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--persons", type=int, default=4)
    ap.add_argument("--packets-per-zone", type=int, default=600)
    ap.add_argument("--spots-per-zone", type=int, default=6)
    ap.add_argument("--walk-seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.resolve(cfg.paths.datasets_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"config": args.config or "config/default.yaml",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "args": vars(args), "sessions": []}
    t0 = time.time()
    for i in range(args.sessions):
        day = i % args.days
        person = (i // args.days) % args.persons if args.persons > 1 else 0
        s = generate_session(
            cfg, session_id=i, day=day, person=person,
            packets_per_zone=args.packets_per_zone,
            spots_per_zone=args.spots_per_zone,
            walk_seconds=args.walk_seconds,
            seed=args.seed + i,
        )
        path = out_dir / f"session_{i:02d}.npz"
        save_session(s, path)
        manifest["sessions"].append({**s["meta"], "file": path.name,
                                     "packets": int(s["csi"].shape[0])})
        print(f"  session {i:02d}: day={day} person={person} "
              f"packets={s['csi'].shape[0]} -> {path.name}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"dataset '{args.name}' written to {out_dir} "
          f"({args.sessions} sessions, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
