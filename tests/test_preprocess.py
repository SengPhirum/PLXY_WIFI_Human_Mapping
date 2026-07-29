import numpy as np

from wifi_mapping.preprocess import (PreprocessPipeline, hampel_filter,
                                     interpolate_lost_packets, sanitize_phase)
from wifi_mapping.preprocess.pipeline import windows_from_stream


def test_hampel_removes_spikes():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, (200, 2, 4))
    x[50, 0, 0] = 25.0
    out = hampel_filter(x, window=11, n_sigmas=3.0)
    assert abs(out[50, 0, 0]) < 1.0
    # Non-outlier samples pass through (almost) unchanged.
    assert np.abs(out[100] - x[100]).max() < 1e-9


def test_interpolate_lost_packets():
    x = np.ones((10, 1, 3), dtype=complex)
    x[4] = np.nan
    out = interpolate_lost_packets(x)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out[4].real, 1.0)


def test_sanitize_phase_removes_linear_trend():
    n = 52
    k = np.arange(n)
    # Pure CFO + SFO corruption on a flat channel → sanitized phase ≈ 0.
    csi = np.exp(1j * (1.3 + 0.07 * k))[None, :]
    out = sanitize_phase(csi)
    assert np.abs(out).max() < 1e-6


def test_windows_from_stream():
    assert windows_from_stream(100, 50, 25) == [(0, 50), (25, 75), (50, 100)]
    assert windows_from_stream(30, 50, 25) == []


def test_pipeline_fit_then_transform(cfg):
    from wifi_mapping.simulate import CsiSimulator

    sim = CsiSimulator(cfg, seed=0)
    csi = sim.record_static(2.0, 3.0, 120)
    pipe = PreprocessPipeline(cfg)
    feats = pipe.stream_to_features(csi, fit=True)
    assert feats.shape[0] == len(windows_from_stream(120, 50, 25))
    assert np.isfinite(feats).all()
