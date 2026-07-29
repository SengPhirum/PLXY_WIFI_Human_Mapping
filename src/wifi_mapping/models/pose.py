"""Pose estimation from CSI: posture classifier + joint-position regressor.

Architecture follows the two-head design used across the Wi-Fi pose
literature (Person-in-WiFi, DensePose-from-WiFi, MM-Fi baselines): a shared
CSI feature representation feeds (a) a discrete posture/activity head and
(b) a continuous keypoint head. We keep both heads as MLPs over the
engineered window features so the whole pipeline trains in minutes on CPU;
:mod:`wifi_mapping.models.deep` holds the optional torch CNN path for the
same job.

Prediction fusion (:func:`fuse_pose`) is what makes the live view stable:
the classifier's posture drives a kinematic prior (the canonical skeleton
for that posture), and the regressor's joint estimate is blended toward it
by the classifier's confidence. This is standard practice for noisy
keypoint regression and prevents the anatomically impossible skeletons a
raw regressor produces on hard windows.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor

from ..simulate.body_model import N_JOINTS, POSTURES, ArticulatedBody


def create_posture_classifier(seed: int = 0) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=800,
                         early_stopping=True, n_iter_no_change=15,
                         random_state=seed)


def create_joint_regressor(seed: int = 0) -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(512, 256), max_iter=800,
                        early_stopping=True, n_iter_no_change=15,
                        random_state=seed)


# ------------------------------------------------------------------ metrics

def mpjpe(pred: np.ndarray, true: np.ndarray) -> dict:
    """Mean Per-Joint Position Error and related keypoint metrics.

    Both arrays are (N, N_JOINTS*3) or (N, N_JOINTS, 3), in metres.
    PCK@d is the fraction of joints within d metres of ground truth — the
    standard keypoint metric (reported at 10 cm and 20 cm here).
    """
    pred = pred.reshape(-1, N_JOINTS, 3)
    true = true.reshape(-1, N_JOINTS, 3)
    err = np.linalg.norm(pred - true, axis=2)          # (N, K)
    return {
        "mpjpe_m": float(err.mean()),
        "mpjpe_cm": float(err.mean() * 100),
        "median_joint_error_cm": float(np.median(err) * 100),
        "pck_10cm": float((err <= 0.10).mean()),
        "pck_20cm": float((err <= 0.20).mean()),
        "worst_joint_cm": float(err.mean(axis=0).max() * 100),
        "n": int(len(pred)),
    }


def per_joint_errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean error per joint (N_JOINTS,) in cm — for the error breakdown."""
    pred = pred.reshape(-1, N_JOINTS, 3)
    true = true.reshape(-1, N_JOINTS, 3)
    return np.linalg.norm(pred - true, axis=2).mean(axis=0) * 100


# ------------------------------------------------------------------- priors

def canonical_skeletons(height: float = 1.72, phase: float = 0.0) -> np.ndarray:
    """Canonical local joints per posture: (n_postures, N_JOINTS, 3)."""
    body = ArticulatedBody(height=height)
    return np.stack([body.joints_local(p, phase) for p in POSTURES])


def fuse_pose(joint_pred: np.ndarray, posture_proba: np.ndarray,
              prior_weight: float = 0.5,
              height: float = 1.72, phase: float = 0.0) -> np.ndarray:
    """Blend a regressed skeleton toward the classifier's posture prior.

    Parameters
    ----------
    joint_pred : (N_JOINTS*3,) or (N, N_JOINTS*3) regressor output
    posture_proba : (n_postures,) or (N, n_postures) class probabilities
    prior_weight : how far to pull toward the prior when the classifier is
        fully confident; the actual pull is ``prior_weight * max_proba``,
        so an unsure classifier barely moves the regression.
    """
    single = joint_pred.ndim == 1
    jp = np.atleast_2d(joint_pred).reshape(-1, N_JOINTS, 3)
    pp = np.atleast_2d(posture_proba)
    priors = canonical_skeletons(height, phase)          # (P, K, 3)
    expected = pp @ priors.reshape(len(priors), -1)      # (N, K*3)
    expected = expected.reshape(-1, N_JOINTS, 3)
    w = (prior_weight * pp.max(axis=1))[:, None, None]
    out = (1.0 - w) * jp + w * expected
    return out[0] if single else out
