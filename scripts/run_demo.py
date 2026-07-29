#!/usr/bin/env python3
"""One-command demo: dataset → model → live dashboard with a simulated
walking person. Skips any stage whose output already exists.

    python scripts/run_demo.py                # http://127.0.0.1:8000
    python scripts/run_demo.py --fresh        # regenerate everything
    python scripts/run_demo.py --model mlp    # different model type
"""

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from wifi_mapping.config import load_config
from wifi_mapping.persistence import load_bundle
from wifi_mapping.server import LivePredictor, create_app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="demo")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--fresh", action="store_true",
                    help="regenerate the dataset and retrain even if cached")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--sim-seed", type=int, default=7,
                    help="seed for the live walking person")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scripts = Path(__file__).parent
    dataset_dir = cfg.resolve(cfg.paths.datasets_dir) / args.dataset
    bundle_path = cfg.resolve(cfg.paths.models_dir) / f"{args.model}_{args.dataset}.pkl"

    # Stage 1: dataset.
    if args.fresh or not (dataset_dir / "manifest.json").exists():
        print("=== generating dataset (first run takes a few minutes) ===")
        subprocess.run([sys.executable, str(scripts / "generate_dataset.py"),
                        "--name", args.dataset]
                       + (["--config", args.config] if args.config else []),
                       check=True)
    else:
        print(f"=== dataset '{args.dataset}' found, skipping generation ===")

    # Stage 2: model.
    if args.fresh or not bundle_path.exists():
        print("=== training model ===")
        subprocess.run([sys.executable, str(scripts / "train.py"),
                        "--dataset", args.dataset, "--model", args.model]
                       + (["--config", args.config] if args.config else []),
                       check=True)
    else:
        print(f"=== model bundle {bundle_path.name} found, skipping training ===")

    # Stage 3: live server.
    pipeline, models, extra = load_bundle(bundle_path, cfg)
    if extra.get("rssi_features"):
        sys.exit("rssi_knn bundles are for baseline comparison only — "
                 "the live demo needs a CSI model (rf/knn/mlp/svm).")
    predictor = LivePredictor(cfg, pipeline, models)
    app = create_app(cfg, predictor, mode="sim", sim_seed=args.sim_seed)

    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    print(f"\n=== dashboard: http://{host}:{port} (Ctrl-C to stop) ===\n")
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
