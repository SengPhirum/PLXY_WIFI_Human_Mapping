"""Evaluation metrics from plan §13.

Classification (zone task): accuracy, macro/weighted F1, confusion matrix.
Localization (xy task): mean/median error, RMSE, 90th percentile, and the
fraction of predictions within 0.5 m / 1 m / 2 m.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def localization_report(xy_true: np.ndarray, xy_pred: np.ndarray) -> dict:
    """Euclidean-error statistics for coordinate predictions (n, 2)."""
    err = np.linalg.norm(xy_pred - xy_true, axis=1)
    return {
        "mean_error_m": float(err.mean()),
        "median_error_m": float(np.median(err)),
        "rmse_m": float(np.sqrt((err ** 2).mean())),
        "p90_error_m": float(np.percentile(err, 90)),
        "within_0.5m": float((err <= 0.5).mean()),
        "within_1m": float((err <= 1.0).mean()),
        "within_2m": float((err <= 2.0).mean()),
        "n": int(len(err)),
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels.tolist(),
        "n": int(len(y_true)),
    }
