#!/usr/bin/env python3
"""Generate a simulated pose dataset (articulated body, 5-posture vocabulary).

Each capture is one window-length recording of a person at a random position
and heading performing one posture; labels are the posture id and the
heading-invariant local joint positions.

    python scripts/generate_pose_dataset.py --config config/pose.yaml \\
        --name pose --sessions 8 --captures 800
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from wifi_mapping.config import load_config
from wifi_mapping.dataset import generate_pose_session, save_pose_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/pose.yaml")
    ap.add_argument("--name", default="pose")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--persons", type=int, default=4)
    ap.add_argument("--captures", type=int, default=800,
                    help="postures recorded per session")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.resolve(cfg.paths.datasets_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"config": args.config, "kind": "pose",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "args": vars(args), "sessions": []}
    t0 = time.time()
    for i in range(args.sessions):
        day = i % args.days
        person = (i // args.days) % args.persons if args.persons > 1 else 0
        s = generate_pose_session(cfg, session_id=i, day=day, person=person,
                                  n_captures=args.captures, seed=args.seed + i)
        path = out_dir / f"pose_session_{i:02d}.npz"
        save_pose_session(s, path)
        manifest["sessions"].append({**s["meta"], "file": path.name,
                                     "captures": int(len(s["posture"]))})
        print(f"  session {i:02d}: day={day} person={person} "
              f"captures={len(s['posture'])} -> {path.name} "
              f"({time.time() - t0:.0f}s)", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"pose dataset '{args.name}' -> {out_dir} "
          f"({args.sessions} sessions, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
