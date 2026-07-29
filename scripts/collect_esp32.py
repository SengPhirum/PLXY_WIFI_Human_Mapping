#!/usr/bin/env python3
"""Guided CSI collection from ESP32 receivers (real hardware).

Standing capture: the operator walks to each announced spot, presses Enter,
stands still for the capture duration, repeats. Walking capture: the
operator walks a pre-measured path at steady pace between Enter presses
(positions are interpolated along the declared path).

Sessions are written in the same .npz format the simulator produces, so
train.py / evaluate.py work unchanged on real data.

Examples:
    # check links are alive
    python scripts/collect_esp32.py --ports RX0=/dev/ttyUSB0 RX1=/dev/ttyUSB1 RX2=/dev/ttyUSB2 --check

    # record session 3, day 1, participant 2, 4 spots per zone, 10 s per spot
    python scripts/collect_esp32.py --ports RX0=/dev/ttyUSB0 RX1=/dev/ttyUSB1 RX2=/dev/ttyUSB2 \\
        --session 3 --day 1 --person 2 --spots-per-zone 4 --seconds-per-spot 10
"""

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np

from wifi_mapping.collect import CsiSerialReader, SessionRecorder
from wifi_mapping.config import load_config


def parse_ports(items: list[str]) -> dict[str, str]:
    ports = {}
    for item in items:
        link, _, port = item.partition("=")
        if not port:
            raise SystemExit(f"--ports entries must be LINK=PORT, got: {item}")
        ports[link] = port
    return ports


def check_links(reader: CsiSerialReader, seconds: float = 5.0) -> None:
    print(f"listening {seconds:.0f}s ...")
    t_end = time.time() + seconds
    for _ in reader.packets(timeout=0.5):
        if time.time() > t_end:
            break
    for link, st in reader.stats.items():
        rate = st["packets"] / seconds
        status = "OK" if rate > 50 else ("LOW" if rate > 5 else "DEAD")
        print(f"  {link}: {rate:6.1f} pkt/s  errors={st['errors']}  [{status}]")


def capture_spot(cfg, reader, recorder, x: float, y: float, seconds: float) -> None:
    """Drain packets and tick frames at the configured rate for one spot."""
    dt = 1.0 / cfg.signal.sample_rate_hz
    t_end = time.time() + seconds
    next_tick = time.time()
    while time.time() < t_end:
        drained = 0
        while drained < 1000:
            try:
                recorder.offer(reader.queue.get_nowait())
                drained += 1
            except Exception:
                break
        now = time.time()
        if now >= next_tick:
            recorder.tick(x, y)
            next_tick += dt
        time.sleep(min(0.002, max(0.0, next_tick - time.time())))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ports", nargs="+", required=True, metavar="LINK=PORT")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--check", action="store_true", help="verify link packet rates and exit")
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--person", type=int, default=0)
    ap.add_argument("--dataset", default="hardware")
    ap.add_argument("--spots-per-zone", type=int, default=4)
    ap.add_argument("--seconds-per-spot", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0, help="seed for spot placement")
    args = ap.parse_args()

    cfg = load_config(args.config)
    reader = CsiSerialReader(parse_ports(args.ports), baudrate=args.baud)
    reader.start()
    try:
        if args.check:
            check_links(reader)
            return

        rng = np.random.default_rng(args.seed + args.session)
        recorder = SessionRecorder(cfg, args.session, args.day, args.person)
        cw = cfg.room.width / cfg.grid.cols
        ch = cfg.room.depth / cfg.grid.rows

        print(f"\nSession {args.session}: {cfg.grid.n_zones} zones × "
              f"{args.spots_per_zone} spots × {args.seconds_per_spot:.0f}s")
        print("Mark each announced spot on the floor (tape + tape measure), "
              "stand on it, press Enter, hold still.\n")
        for zone in range(cfg.grid.n_zones):
            zx, zy = cfg.zone_center(zone)
            for spot in range(args.spots_per_zone):
                px = zx + rng.uniform(-0.4, 0.4) * cw
                py = zy + rng.uniform(-0.4, 0.4) * ch
                input(f"zone {zone:2d} spot {spot + 1}/{args.spots_per_zone}: "
                      f"stand at x={px:.2f} y={py:.2f} then press Enter ")
                capture_spot(cfg, reader, recorder, px, py, args.seconds_per_spot)
                recorder.new_segment()
                print(f"   captured. (loss events so far: {recorder.packet_loss_events})")

        out = (cfg.resolve(cfg.paths.datasets_dir) / args.dataset /
               f"session_{args.session:02d}.npz")
        recorder.save(out)
        print(f"\nsaved -> {out}")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
