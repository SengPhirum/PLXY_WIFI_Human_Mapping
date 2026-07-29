import numpy as np
import pytest

from wifi_mapping.models.pose import (canonical_skeletons, fuse_pose, mpjpe,
                                      per_joint_errors)
from wifi_mapping.preprocess.pipeline import spectral_band_features
from wifi_mapping.simulate import CsiSimulator
from wifi_mapping.simulate.body_model import (JOINT_GAINS, JOINTS, N_JOINTS,
                                              POSTURES, ArticulatedBody)


def test_body_joint_count_and_anatomy():
    body = ArticulatedBody()
    j = body.joints_local("idle", 0.0)
    assert j.shape == (N_JOINTS, 3)
    # Head above neck above hips above ankles.
    assert j[JOINTS.index("head")][2] > j[JOINTS.index("neck")][2]
    assert j[JOINTS.index("neck")][2] > j[JOINTS.index("r_hip")][2]
    assert j[JOINTS.index("r_hip")][2] > j[JOINTS.index("r_ankle")][2]
    # Left/right symmetry about x for a symmetric posture.
    assert j[JOINTS.index("r_shoulder")][0] == pytest.approx(
        -j[JOINTS.index("l_shoulder")][0])


def test_postures_are_geometrically_distinct():
    body = ArticulatedBody()
    poses = {p: body.joints_local(p, 0.0) for p in POSTURES}
    # Sitting lowers the hips substantially.
    assert poses["sit"][JOINTS.index("r_hip")][2] < poses["idle"][JOINTS.index("r_hip")][2] - 0.3
    # T-pose extends the wrists laterally far beyond idle.
    assert abs(poses["tpose"][JOINTS.index("r_wrist")][0]) > \
           abs(poses["idle"][JOINTS.index("r_wrist")][0]) + 0.3
    # Waving raises the right wrist above the shoulder.
    assert poses["wave"][JOINTS.index("r_wrist")][2] > \
           poses["wave"][JOINTS.index("r_shoulder")][2]


def test_joints_world_applies_heading_and_translation():
    body = ArticulatedBody()
    local = body.joints_local("idle", 0.0)
    world = body.joints_world("idle", 0.0, np.array([2.0, 3.0]), np.pi / 2)
    # Heights unchanged, root translated, and x/y rotated.
    np.testing.assert_allclose(world[:, 2], local[:, 2])
    np.testing.assert_allclose(world[JOINTS.index("neck"), :2], [2.0, 3.0], atol=1e-9)


def test_csi_sequence_matches_shape_and_reacts_to_posture(cfg):
    cfg.signal.packet_loss = 0.0
    sim = CsiSimulator(cfg, seed=0)
    body = ArticulatedBody()
    root = np.array([2.5, 3.0])
    seq_idle = np.stack([body.joints_world("idle", 0.0, root, 0.0)] * 8)
    seq_tpose = np.stack([body.joints_world("tpose", 0.0, root, 0.0)] * 8)
    a = sim.csi_sequence_joints(seq_idle, JOINT_GAINS)
    b = sim.csi_sequence_joints(seq_tpose, JOINT_GAINS)
    assert a.shape == (8, cfg.links.n_links, cfg.signal.n_subcarriers)
    # Different limb configurations must change the amplitude fingerprint.
    assert np.abs(np.abs(a).mean(axis=0) - np.abs(b).mean(axis=0)).mean() > 1e-4


def test_spectral_features_separate_motion_rates():
    """A 1 s window has 1 Hz bin spacing, so gait (~1 Hz) and waving
    (~3 Hz) land in distinct bins — that separation is what lets the
    posture classifier tell movement types apart."""
    fs, n, links = 100.0, 100, 2
    t = np.arange(n) / fs
    slow = np.ones((n, links, 4)) * np.sin(2 * np.pi * 1.0 * t)[:, None, None]
    fast = np.ones((n, links, 4)) * np.sin(2 * np.pi * 3.0 * t)[:, None, None]
    prof_slow = spectral_band_features(slow, fs).reshape(links, -1)
    prof_fast = spectral_band_features(fast, fs).reshape(links, -1)
    assert prof_slow[0].argmax() == 1      # 1 Hz bin
    assert prof_fast[0].argmax() == 3      # 3 Hz bin
    # Profiles are normalized per link.
    np.testing.assert_allclose(prof_slow.sum(axis=1), 1.0, rtol=1e-6)


def test_spectral_features_flag_static_bodies():
    """A motionless body has no AC energy: the profile collapses to DC."""
    fs, n, links = 100.0, 100, 2
    static = np.ones((n, links, 4)) * 0.5
    prof = spectral_band_features(static, fs).reshape(links, -1)
    assert prof[0].argmax() == 0


def test_mpjpe_and_pck():
    true = np.zeros((2, N_JOINTS, 3))
    pred = true.copy()
    pred[:, 0, 0] = 0.30      # one joint off by 30 cm
    m = mpjpe(pred, true)
    assert m["mpjpe_cm"] == pytest.approx(100 * 0.30 / N_JOINTS)
    assert m["pck_10cm"] == pytest.approx((N_JOINTS - 1) / N_JOINTS)
    assert m["worst_joint_cm"] == pytest.approx(30.0)
    assert per_joint_errors(pred, true)[0] == pytest.approx(30.0)


def test_fuse_pose_pulls_toward_confident_prior():
    priors = canonical_skeletons()
    assert priors.shape == (len(POSTURES), N_JOINTS, 3)
    tpose_idx = POSTURES.index("tpose")
    proba = np.zeros(len(POSTURES))
    proba[tpose_idx] = 1.0
    garbage = np.zeros(N_JOINTS * 3)
    fused = fuse_pose(garbage, proba, prior_weight=0.5)
    # With full confidence the fused pose sits halfway to the canonical one.
    np.testing.assert_allclose(fused, priors[tpose_idx] * 0.5, atol=1e-9)


def test_fuse_pose_ignores_prior_when_unsure():
    proba = np.full(len(POSTURES), 1.0 / len(POSTURES))
    raw = np.tile([0.1, 0.2, 1.0], N_JOINTS)
    fused = fuse_pose(raw, proba, prior_weight=0.5).ravel()
    # Low confidence → the regression dominates.
    assert np.abs(fused - raw).mean() < 0.35
