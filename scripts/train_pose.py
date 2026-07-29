#!/usr/bin/env python3
"""Train the pose models (posture classifier + joint regressor) with a
leakage-aware split, report keypoint metrics, and save a deployable bundle.

    python scripts/train_pose.py --config config/pose.yaml --dataset pose
    python scripts/train_pose.py --config config/pose.yaml --split-by person
    python scripts/train_pose.py --config config/pose.yaml --scaling   # data curve
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np

from wifi_mapping.config import load_config
from wifi_mapping.dataset import load_pose_session, pose_sessions_to_arrays
from wifi_mapping.eval import split_sessions
from wifi_mapping.models.pose import (create_joint_regressor,
                                      create_posture_classifier, fuse_pose,
                                      mpjpe, per_joint_errors)
from wifi_mapping.persistence import save_bundle
from wifi_mapping.preprocess import PreprocessPipeline
from wifi_mapping.simulate.body_model import JOINTS, POSTURES
from sklearn.metrics import confusion_matrix, f1_score


def load_pose_dataset(directory):
    files = sorted(directory.glob("pose_session_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no pose_session_*.npz in {directory} — run "
            "scripts/generate_pose_dataset.py first")
    return [load_pose_session(f) for f in files]


def fit_and_score(tr, te, seed=0, subset=None):
    Xtr, ptr, jtr = tr["X"], tr["posture"], tr["joints"]
    if subset is not None and subset < len(Xtr):
        idx = np.random.default_rng(0).choice(len(Xtr), subset, replace=False)
        Xtr, ptr, jtr = Xtr[idx], ptr[idx], jtr[idx]

    clf = create_posture_classifier(seed).fit(Xtr, ptr)
    proba = clf.predict_proba(te["X"])
    pred_posture = clf.classes_[proba.argmax(axis=1)]

    reg = create_joint_regressor(seed).fit(Xtr, jtr)
    raw_joints = reg.predict(te["X"])
    # Align class probabilities to the full posture vocabulary before fusing.
    full_proba = np.zeros((len(proba), len(POSTURES)))
    full_proba[:, clf.classes_] = proba
    fused = fuse_pose(raw_joints, full_proba)

    return {
        "n_train": int(len(Xtr)),
        "posture_accuracy": float((pred_posture == te["posture"]).mean()),
        "posture_macro_f1": float(f1_score(te["posture"], pred_posture,
                                           average="macro", zero_division=0)),
        "joints_raw": mpjpe(raw_joints, te["joints"]),
        "joints_fused": mpjpe(fused, te["joints"]),
        "confusion": confusion_matrix(
            te["posture"], pred_posture,
            labels=list(range(len(POSTURES)))).tolist(),
        "per_joint_cm": per_joint_errors(fused, te["joints"]).round(1).tolist(),
    }, (clf, reg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/pose.yaml")
    ap.add_argument("--dataset", default="pose")
    ap.add_argument("--split-by", default="session",
                    choices=["session", "day", "person"])
    ap.add_argument("--scaling", action="store_true",
                    help="report the accuracy-vs-training-size curve")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sessions = load_pose_dataset(cfg.resolve(cfg.paths.datasets_dir) / args.dataset)
    train_s, test_s = split_sessions(sessions, by=args.split_by)
    print(f"split by {args.split_by}: {len(train_s)} train / {len(test_s)} test sessions")

    pipeline = PreprocessPipeline(cfg)
    t0 = time.time()
    tr = pose_sessions_to_arrays(cfg, train_s, pipeline, fit=True)
    te = pose_sessions_to_arrays(cfg, test_s, pipeline)
    print(f"features: {tr['X'].shape} train / {te['X'].shape} test "
          f"({time.time() - t0:.0f}s)")

    if args.scaling:
        print("\ntraining-size scaling curve (posture accuracy / fused MPJPE):")
        curve = []
        for frac in (0.1, 0.25, 0.5, 1.0):
            n = max(50, int(len(tr["X"]) * frac))
            r, _ = fit_and_score(tr, te, args.seed, subset=n)
            curve.append(r)
            print(f"  n={r['n_train']:5d}  acc={r['posture_accuracy']:.3f}  "
                  f"MPJPE={r['joints_fused']['mpjpe_cm']:.1f} cm  "
                  f"PCK@20={r['joints_fused']['pck_20cm']:.2f}")
        out = cfg.resolve(cfg.paths.data_dir) / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"pose_scaling_{args.dataset}.json").write_text(json.dumps(curve, indent=2))
        return

    res, (clf, reg) = fit_and_score(tr, te, args.seed)
    print(f"\nposture: accuracy={res['posture_accuracy']:.3f} "
          f"macro_f1={res['posture_macro_f1']:.3f}")
    print("confusion (rows=true " + ", ".join(POSTURES) + "):")
    for name, row in zip(POSTURES, res["confusion"]):
        print(f"  {name:6} {row}")
    jr, jf = res["joints_raw"], res["joints_fused"]
    print(f"joints raw  : MPJPE={jr['mpjpe_cm']:.1f} cm  "
          f"PCK@10={jr['pck_10cm']:.2f}  PCK@20={jr['pck_20cm']:.2f}")
    print(f"joints fused: MPJPE={jf['mpjpe_cm']:.1f} cm  "
          f"PCK@10={jf['pck_10cm']:.2f}  PCK@20={jf['pck_20cm']:.2f}")
    print("per-joint error (cm): " + ", ".join(
        f"{j}={e}" for j, e in zip(JOINTS, res["per_joint_cm"])))

    bundle = cfg.resolve(cfg.paths.models_dir) / f"pose_{args.dataset}.pkl"
    save_bundle(bundle, cfg, pipeline, {"posture": clf, "joints": reg},
                extra={"results": res, "kind": "pose",
                       "postures": POSTURES, "joints": JOINTS})
    bundle.with_suffix(".results.json").write_text(json.dumps(res, indent=2))
    print(f"\nbundle -> {bundle}")


if __name__ == "__main__":
    main()
