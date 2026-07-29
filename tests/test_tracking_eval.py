import numpy as np
import pytest

from wifi_mapping.eval import (classification_report_dict, localization_report,
                               split_sessions)
from wifi_mapping.tracking import KalmanTracker2D


def test_kalman_reduces_noise():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 100)
    truth = np.stack([1 + 0.3 * t, 2 + 0.2 * t], axis=1)
    noisy = truth + rng.normal(0, 0.5, truth.shape)
    smoothed = KalmanTracker2D(dt=0.1).smooth_path(noisy, dt=0.1)
    raw_err = np.linalg.norm(noisy - truth, axis=1).mean()
    smooth_err = np.linalg.norm(smoothed - truth, axis=1).mean()
    assert smooth_err < raw_err * 0.8


def test_localization_report():
    true = np.zeros((4, 2))
    pred = np.array([[0.3, 0.4], [0, 0], [0, 2.0], [1.0, 0]])  # errors .5, 0, 2, 1
    rep = localization_report(true, pred)
    assert rep["mean_error_m"] == pytest.approx(0.875)
    assert rep["within_1m"] == pytest.approx(0.75)
    assert rep["within_2m"] == pytest.approx(1.0)


def test_classification_report():
    rep = classification_report_dict(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert rep["accuracy"] == pytest.approx(2 / 3)


def test_split_sessions_holds_out_whole_groups():
    sessions = [{"meta": {"session": i, "day": i // 2}} for i in range(6)]
    train, test = split_sessions(sessions, by="day")
    train_days = {s["meta"]["day"] for s in train}
    test_days = {s["meta"]["day"] for s in test}
    assert train_days.isdisjoint(test_days)
    assert len(train) + len(test) == 6


def test_split_sessions_rejects_single_group():
    with pytest.raises(ValueError):
        split_sessions([{"meta": {"day": 0}}] * 3, by="day")
