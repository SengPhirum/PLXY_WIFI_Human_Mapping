"""Constant-velocity Kalman filter for smoothing sequential (x, y) estimates.

Raw per-window position predictions jitter because each window is estimated
independently; a person, however, moves continuously. The filter fuses the
constant-velocity motion model with the noisy measurements, producing the
smooth trajectory drawn on the dashboard (plan §11 pipeline tail).
"""

from __future__ import annotations

import numpy as np


class KalmanTracker2D:
    """State: [x, y, vx, vy]. Call :meth:`update` once per position estimate."""

    def __init__(self, process_noise: float = 0.35,
                 measurement_noise: float = 0.45, dt: float = 0.5):
        self.q = process_noise
        self.r = measurement_noise
        self.dt = dt
        self.x: np.ndarray | None = None  # state
        self.P: np.ndarray | None = None  # covariance
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    def _Q(self, dt: float) -> np.ndarray:
        # White-acceleration model.
        q = self.q ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        return q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                             [0, dt4 / 4, 0, dt3 / 2],
                             [dt3 / 2, 0, dt2, 0],
                             [0, dt3 / 2, 0, dt2]])

    def reset(self) -> None:
        self.x = None
        self.P = None

    def update(self, zx: float, zy: float, dt: float | None = None) -> tuple[float, float]:
        """Feed one measurement; returns the smoothed (x, y)."""
        dt = self.dt if dt is None else dt
        z = np.array([zx, zy], dtype=float)

        if self.x is None:
            self.x = np.array([zx, zy, 0.0, 0.0])
            self.P = np.diag([self.r ** 2, self.r ** 2, 1.0, 1.0])
            return zx, zy

        F, Q = self._F(dt), self._Q(dt)
        # Predict.
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # Update.
        R = np.eye(2) * self.r ** 2
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])

    def smooth_path(self, path: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Offline smoothing of an (n, 2) trajectory."""
        self.reset()
        return np.array([self.update(px, py, dt) for px, py in path])
