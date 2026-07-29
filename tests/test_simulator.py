import numpy as np

from wifi_mapping.simulate import CsiSimulator, WalkGenerator


def test_csi_shape_and_position_sensitivity(cfg):
    sim = CsiSimulator(cfg, seed=0)
    csi = sim.csi_at(1.0, 1.0)
    assert csi.shape == (cfg.links.n_links, cfg.signal.n_subcarriers)
    # Different positions must produce different amplitude fingerprints.
    a = np.abs(sim.csi_at(1.0, 1.0))
    b = np.abs(sim.csi_at(4.0, 5.0))
    valid = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    assert valid.any()
    assert np.abs(a[valid] - b[valid]).mean() > 0.01


def test_packet_loss_marked_as_nan(cfg):
    cfg.signal.packet_loss = 1.0
    sim = CsiSimulator(cfg, seed=0)
    assert np.isnan(sim.csi_at(2.0, 3.0)).all()


def test_record_static_has_time_variation(cfg):
    cfg.signal.packet_loss = 0.0
    sim = CsiSimulator(cfg, seed=0)
    rec = sim.record_static(2.0, 3.0, 20)
    assert rec.shape[0] == 20
    assert np.abs(np.diff(np.abs(rec), axis=0)).mean() > 0  # jitter + noise


def test_walker_stays_in_room(cfg):
    walker = WalkGenerator(cfg, seed=1)
    path = walker.trajectory(500, 0.1)
    assert path[:, 0].min() >= 0 and path[:, 0].max() <= cfg.room.width
    assert path[:, 1].min() >= 0 and path[:, 1].max() <= cfg.room.depth
    # Walker actually moves.
    assert np.linalg.norm(path[-1] - path[0]) + np.abs(np.diff(path, axis=0)).sum() > 1.0
