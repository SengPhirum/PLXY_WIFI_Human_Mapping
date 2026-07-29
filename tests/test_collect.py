import numpy as np

from wifi_mapping.collect import parse_csi_line


ESP_LINE = ('CSI_DATA,101,aa:bb:cc:dd:ee:ff,-42,11,1,7,0,0,0,0,0,0,-92,0,1,1,'
            '128,0,"[12,-3,14,-1,10,2,9,-4]"')


def test_parse_valid_line():
    pkt = parse_csi_line(ESP_LINE, "RX1", timestamp=123.0)
    assert pkt is not None
    assert pkt.link_id == "RX1"
    assert pkt.seq == 101
    assert pkt.rssi == -42.0
    assert pkt.timestamp == 123.0
    # [imag, real] interleaving: first subcarrier = -3 + 12j
    np.testing.assert_allclose(pkt.csi[0], complex(-3, 12))
    assert len(pkt.csi) == 4


def test_parse_rejects_noise_lines():
    assert parse_csi_line("I (1234) wifi: connected", "RX0") is None
    assert parse_csi_line("CSI_DATA,malformed", "RX0") is None
    assert parse_csi_line('CSI_DATA,1,mac,-40,"[1,2,3]"', "RX0") is None  # odd length


def test_end_to_end_sim_dataset_roundtrip(cfg, tmp_path):
    """Simulated session → save → load → features → model → sane accuracy."""
    from wifi_mapping.dataset import (generate_session, load_session,
                                      save_session, sessions_to_arrays)
    from wifi_mapping.models import create_model
    from wifi_mapping.preprocess import PreprocessPipeline

    s = generate_session(cfg, 0, day=0, person=0, packets_per_zone=100,
                         spots_per_zone=2, walk_seconds=5, seed=0)
    path = tmp_path / "session_00.npz"
    save_session(s, path)
    loaded = load_session(path)
    assert loaded["meta"]["session"] == 0
    np.testing.assert_array_equal(loaded["seg"], s["seg"])

    pipe = PreprocessPipeline(cfg)
    d = sessions_to_arrays(cfg, [loaded], pipe, fit=True)
    assert d["X"].shape[0] == len(d["zone"]) == len(d["xy"])
    assert np.isfinite(d["X"]).all()

    m = create_model("knn", "zone")
    m.fit(d["X"], d["zone"])
    # Trivially fitting its own training data — just proves the plumbing.
    assert m.score(d["X"], d["zone"]) > 0.5
