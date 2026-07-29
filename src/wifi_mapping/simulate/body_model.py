"""Articulated body model for pose-from-Wi-Fi simulation.

The single-point person model in :mod:`csi_simulator` is enough for
localization, but pose estimation needs the RF signature of *body
configuration*: each body part scatters the signal from a different place,
so raising an arm or sitting down changes the multipath sum measurably.
This module models a person as 14 joint scatterers (COCO-style subset)
animated through a small posture vocabulary:

    idle | walk | sit | wave | tpose

This mirrors — at simulator fidelity — what DensePose-from-WiFi (CMU,
arXiv:2301.00250) and Person-in-WiFi learn from real 3×3-antenna CSI: the
mapping from multipath perturbation patterns to body configuration. See
reference/POSE_FROM_WIFI.md for the research context and the honest scope
statement (simulator-validated pipeline, not a claim about real-CSI
accuracy).
"""

from __future__ import annotations

import numpy as np

# Joint order is the contract between simulator, models, and the /body
# renderer — do not reorder without updating body.html's EDGES.
JOINTS = [
    "head", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
]
N_JOINTS = len(JOINTS)

POSTURES = ["idle", "walk", "sit", "wave", "tpose"]

# Radar-like scattering weight per joint (torso-adjacent parts reflect more).
JOINT_GAINS = np.array([
    0.45, 0.50,
    0.35, 0.25, 0.20,
    0.35, 0.25, 0.20,
    0.45, 0.30, 0.22,
    0.45, 0.30, 0.22,
])


class ArticulatedBody:
    """Compute world-frame joint positions for a posture + walk phase.

    Local frame before heading rotation: x lateral (right+), y forward,
    z up; root is the mid-hip point projected to the floor at (0, 0).
    """

    def __init__(self, height: float = 1.72):
        s = height / 1.72  # scale all segment lengths to subject height
        self.s = s
        self.hip_h = 0.92 * s
        self.neck_h = 1.46 * s
        self.head_h = 1.64 * s
        self.shoulder_w = 0.20 * s
        self.hip_w = 0.10 * s
        self.upper_arm = 0.29 * s
        self.forearm = 0.26 * s
        self.thigh = 0.44 * s
        self.shin = 0.42 * s

    # ---------------------------------------------------------------- poses

    def _arms_down(self, phase: float, swing: float) -> dict:
        a = np.sin(phase) * 0.7 * swing
        return {"r": (a, 0.35), "l": (-a, 0.35)}  # (swing angle, elbow bend)

    def joints_local(self, posture: str, phase: float) -> np.ndarray:
        """(N_JOINTS, 3) local-frame joint positions."""
        j = np.zeros((N_JOINTS, 3))
        sit = posture == "sit"
        walk = posture == "walk"
        # Sitting drops the pelvis and folds the legs forward.
        hip_h = 0.46 * self.s if sit else self.hip_h
        bob = 0.02 * abs(np.cos(phase)) if walk else 0.0
        neck_h = hip_h + (self.neck_h - self.hip_h) + bob
        head_h = hip_h + (self.head_h - self.hip_h) + bob

        j[JOINTS.index("head")] = [0, 0.02, head_h]
        j[JOINTS.index("neck")] = [0, 0, neck_h]
        j[JOINTS.index("r_hip")] = [+self.hip_w, 0, hip_h]
        j[JOINTS.index("l_hip")] = [-self.hip_w, 0, hip_h]
        j[JOINTS.index("r_shoulder")] = [+self.shoulder_w, 0, neck_h - 0.02]
        j[JOINTS.index("l_shoulder")] = [-self.shoulder_w, 0, neck_h - 0.02]

        # --- arms
        if posture == "tpose":
            for side, sgn in (("r", +1), ("l", -1)):
                sh = j[JOINTS.index(f"{side}_shoulder")]
                j[JOINTS.index(f"{side}_elbow")] = sh + [sgn * self.upper_arm, 0, 0]
                j[JOINTS.index(f"{side}_wrist")] = sh + [sgn * (self.upper_arm + self.forearm), 0, 0]
        elif posture == "wave":
            # Left arm down, right arm raised with the forearm oscillating
            # (~2 Hz for a natural wave — a distinct spectral signature).
            sh = j[JOINTS.index("r_shoulder")]
            el = sh + [0.10, 0, self.upper_arm * 0.9]
            osc = 0.35 * np.sin(phase * 6.0)
            j[JOINTS.index("r_elbow")] = el
            j[JOINTS.index("r_wrist")] = el + [
                self.forearm * np.sin(osc), 0, self.forearm * np.cos(osc)]
            a, bend = self._arms_down(phase, 0.0)["l"]
            self._arm_down_into(j, "l", a, bend)
        else:
            swing = 1.0 if walk else 0.08
            arms = self._arms_down(phase, swing)
            for side in ("r", "l"):
                a, bend = arms[side]
                self._arm_down_into(j, side, a, bend)

        # --- legs
        if sit:
            for side, sgn in (("r", +1), ("l", -1)):
                hip = j[JOINTS.index(f"{side}_hip")]
                knee = hip + [0, self.thigh * 0.95, -0.05]
                j[JOINTS.index(f"{side}_knee")] = knee
                j[JOINTS.index(f"{side}_ankle")] = knee + [0, 0.05, -self.shin * 0.95]
        else:
            swing = 0.55 if walk else 0.0
            for side, sgn in (("r", +1), ("l", -1)):
                a = sgn * np.sin(phase) * swing
                hip = j[JOINTS.index(f"{side}_hip")]
                knee = hip + [0, self.thigh * np.sin(a), -self.thigh * np.cos(a)]
                kb = max(0.0, -a) * 1.1
                j[JOINTS.index(f"{side}_knee")] = knee
                j[JOINTS.index(f"{side}_ankle")] = knee + [
                    0, self.shin * np.sin(a - kb), -self.shin * np.cos(a - kb)]
        return j

    def _arm_down_into(self, j: np.ndarray, side: str, a: float, bend: float) -> None:
        sgn = +1 if side == "r" else -1
        sh = j[JOINTS.index(f"{side}_shoulder")]
        el = sh + [sgn * 0.02, self.upper_arm * np.sin(a), -self.upper_arm * np.cos(a)]
        j[JOINTS.index(f"{side}_elbow")] = el
        j[JOINTS.index(f"{side}_wrist")] = el + [
            sgn * 0.01, self.forearm * np.sin(a + bend), -self.forearm * np.cos(a + bend)]

    # ---------------------------------------------------------------- world

    def joints_world(self, posture: str, phase: float, root_xy: np.ndarray,
                     heading: float) -> np.ndarray:
        """Rotate local joints by heading (about z) and translate to root."""
        local = self.joints_local(posture, phase)
        c, s = np.cos(heading), np.sin(heading)
        world = local.copy()
        world[:, 0] = local[:, 0] * c - local[:, 1] * s + root_xy[0]
        world[:, 1] = local[:, 0] * s + local[:, 1] * c + root_xy[1]
        return world
