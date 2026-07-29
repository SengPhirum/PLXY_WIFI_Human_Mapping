"""Model factory: one place that knows how to build every baseline and
candidate model from the thesis plan §11.2.

``task`` is "zone" (grid-cell classification) or "xy" (coordinate
regression). All returned objects follow the sklearn fit/predict API,
including the optional torch models, so training and evaluation code is
model-agnostic.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputRegressor

MODEL_TYPES = ["rssi_knn", "knn", "svm", "rf", "mlp", "cnn", "cnn_gru"]


def create_model(model_type: str, task: str, seed: int = 0) -> Any:
    """Build a model. "rssi_knn" is the same KNN as "knn" — it differs only
    in that the caller feeds it RSSI features instead of CSI features."""
    if task not in ("zone", "xy"):
        raise ValueError(f"unknown task: {task}")

    if model_type in ("knn", "rssi_knn"):
        return (KNeighborsClassifier(n_neighbors=5, weights="distance")
                if task == "zone"
                else KNeighborsRegressor(n_neighbors=5, weights="distance"))

    if model_type == "svm":
        return (SVC(kernel="rbf", C=10.0, probability=True, random_state=seed)
                if task == "zone"
                else MultiOutputRegressor(SVR(kernel="rbf", C=10.0)))

    if model_type == "rf":
        return (RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                if task == "zone"
                else RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1))

    if model_type == "mlp":
        kw = dict(hidden_layer_sizes=(256, 128), max_iter=400,
                  early_stopping=True, random_state=seed)
        return MLPClassifier(**kw) if task == "zone" else MLPRegressor(**kw)

    if model_type in ("cnn", "cnn_gru"):
        from .deep import TorchWindowModel  # requires optional torch install
        return TorchWindowModel(arch=model_type, task=task, seed=seed)

    raise ValueError(f"unknown model type: {model_type} (choose from {MODEL_TYPES})")
