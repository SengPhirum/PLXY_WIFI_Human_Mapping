#!/usr/bin/env python3
"""Train zone + xy models on a dataset with a leakage-aware split, report
held-out metrics, and save a deployable model bundle.

Example:
    python scripts/train.py --dataset demo --model rf --split-by session
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from wifi_mapping.config import load_config
from wifi_mapping.dataset import load_dataset, sessions_to_arrays
from wifi_mapping.eval import (classification_report_dict, localization_report,
                               split_sessions)
from wifi_mapping.models import MODEL_TYPES, create_model
from wifi_mapping.persistence import save_bundle
from wifi_mapping.preprocess import PreprocessPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="demo")
    ap.add_argument("--model", default="rf", choices=MODEL_TYPES)
    ap.add_argument("--split-by", default="session",
                    choices=["session", "day", "person"],
                    help="independence requirement for the held-out test set")
    ap.add_argument("--out", default=None, help="bundle filename (default: <model>_<dataset>.pkl)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sessions = load_dataset(cfg.resolve(cfg.paths.datasets_dir) / args.dataset)
    train_s, test_s = split_sessions(sessions, by=args.split_by)
    print(f"split by {args.split_by}: {len(train_s)} train / {len(test_s)} test sessions")

    rssi = args.model == "rssi_knn"
    pipeline = PreprocessPipeline(cfg)
    t0 = time.time()
    tr = sessions_to_arrays(cfg, train_s, pipeline, fit=True, rssi=rssi)
    te = sessions_to_arrays(cfg, test_s, pipeline, rssi=rssi)
    print(f"features: {tr['X'].shape} train / {te['X'].shape} test "
          f"({time.time() - t0:.1f}s)")

    results = {"model": args.model, "dataset": args.dataset,
               "split_by": args.split_by, "n_train": len(tr["X"]),
               "n_test": len(te["X"])}
    models = {}

    clf = create_model(args.model, "zone", seed=args.seed)
    clf.fit(tr["X"], tr["zone"])
    rep = classification_report_dict(te["zone"], clf.predict(te["X"]))
    results["zone"] = rep
    models["zone"] = clf
    print(f"zone: accuracy={rep['accuracy']:.3f} macro_f1={rep['macro_f1']:.3f}")

    reg = create_model(args.model, "xy", seed=args.seed)
    reg.fit(tr["X"], tr["xy"])
    rep = localization_report(te["xy"], reg.predict(te["X"]))
    results["xy"] = rep
    models["xy"] = reg
    print(f"xy: mean={rep['mean_error_m']:.3f}m median={rep['median_error_m']:.3f}m "
          f"p90={rep['p90_error_m']:.3f}m within1m={rep['within_1m']:.2f}")

    out_name = args.out or f"{args.model}_{args.dataset}.pkl"
    bundle_path = cfg.resolve(cfg.paths.models_dir) / out_name
    save_bundle(bundle_path, cfg, pipeline, models,
                extra={"results": results, "rssi_features": rssi})
    results_path = bundle_path.with_suffix(".results.json")
    results_path.write_text(json.dumps(results, indent=2))
    print(f"bundle -> {bundle_path}\nresults -> {results_path}")


if __name__ == "__main__":
    main()
