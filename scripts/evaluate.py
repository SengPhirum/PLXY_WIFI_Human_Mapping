#!/usr/bin/env python3
"""Run the baseline comparison matrix (plan §14): every model × every
independence split, on one dataset. Prints a summary table and writes
JSON + Markdown results for the thesis.

Example:
    python scripts/evaluate.py --dataset demo --models rssi_knn knn rf mlp
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
from wifi_mapping.preprocess import PreprocessPipeline


def evaluate_one(cfg, sessions, model_type: str, split_by: str, seed: int) -> dict:
    train_s, test_s = split_sessions(sessions, by=split_by)
    rssi = model_type == "rssi_knn"
    pipeline = PreprocessPipeline(cfg)
    tr = sessions_to_arrays(cfg, train_s, pipeline, fit=True, rssi=rssi)
    te = sessions_to_arrays(cfg, test_s, pipeline, rssi=rssi)

    t0 = time.time()
    clf = create_model(model_type, "zone", seed=seed)
    clf.fit(tr["X"], tr["zone"])
    zone = classification_report_dict(te["zone"], clf.predict(te["X"]))
    reg = create_model(model_type, "xy", seed=seed)
    reg.fit(tr["X"], tr["xy"])
    xy = localization_report(te["xy"], reg.predict(te["X"]))
    return {"model": model_type, "split_by": split_by,
            "zone_accuracy": zone["accuracy"], "zone_macro_f1": zone["macro_f1"],
            "xy_mean_m": xy["mean_error_m"], "xy_median_m": xy["median_error_m"],
            "xy_p90_m": xy["p90_error_m"], "within_1m": xy["within_1m"],
            "train_s": round(time.time() - t0, 1),
            "n_train": len(tr["X"]), "n_test": len(te["X"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="demo")
    ap.add_argument("--models", nargs="+", default=["rssi_knn", "knn", "rf", "mlp"],
                    choices=MODEL_TYPES)
    ap.add_argument("--splits", nargs="+", default=["session", "day", "person"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sessions = load_dataset(cfg.resolve(cfg.paths.datasets_dir) / args.dataset)

    rows = []
    for split_by in args.splits:
        for model_type in args.models:
            print(f"→ {model_type} / split-by-{split_by} ...", flush=True)
            try:
                rows.append(evaluate_one(cfg, sessions, model_type, split_by, args.seed))
            except (ValueError, ImportError) as e:
                print(f"  skipped: {e}")

    header = (f"{'model':10} {'split':8} {'zone acc':>8} {'macroF1':>8} "
              f"{'mean(m)':>8} {'med(m)':>7} {'p90(m)':>7} {'<1m':>5}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['model']:10} {r['split_by']:8} {r['zone_accuracy']:8.3f} "
                     f"{r['zone_macro_f1']:8.3f} {r['xy_mean_m']:8.3f} "
                     f"{r['xy_median_m']:7.3f} {r['xy_p90_m']:7.3f} {r['within_1m']:5.2f}")
    print("\n".join(lines))

    out_dir = cfg.resolve(cfg.paths.data_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"eval_{args.dataset}_{stamp}.json").write_text(json.dumps(rows, indent=2))

    md = ["| model | split | zone acc | macro F1 | mean err (m) | median (m) | p90 (m) | <1 m |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['split_by']} | {r['zone_accuracy']:.3f} | "
                  f"{r['zone_macro_f1']:.3f} | {r['xy_mean_m']:.3f} | "
                  f"{r['xy_median_m']:.3f} | {r['xy_p90_m']:.3f} | {r['within_1m']:.2f} |")
    (out_dir / f"eval_{args.dataset}_{stamp}.md").write_text("\n".join(md) + "\n")
    print(f"\nresults saved to {out_dir}/eval_{args.dataset}_{stamp}.{{json,md}}")


if __name__ == "__main__":
    main()
