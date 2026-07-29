"""Geometry-based multipath CSI simulator.

Models each TX→RX link as a sum of complex ray contributions per subcarrier:

- a static component: line-of-sight ray + first-order wall reflections
  (image-source method on the four walls),
- a dynamic component: a ray scattered off the human body, whose path length
  is TX→person→RX, attenuated by total travelled distance and a radar-like
  cross-section factor,
- realistic hardware impairments: per-packet random phase offset (CFO/PLL),
  a linear phase slope across subcarriers (SFO/packet-detection delay),
  additive Gaussian noise, and random packet loss.

The per-packet random phase offsets deliberately destroy raw phase — exactly
as on real ESP32 hardware — so the preprocessing stage's phase sanitization
and amplitude-based features are exercised honestly.

This is *not* a full ray tracer; it is a controllable stand-in that makes
position information learnable through the same physical mechanism real CSI
sensing relies on (path-length-dependent phase across subcarriers and
body-shadowing of the LoS), which is sufficient for developing and demoing
the entire pipeline before hardware arrives.
"""

from __future__ import annotations

import numpy as np

from ..config import Config

C = 299_792_458.0  # speed of light (m/s)


class CsiSimulator:
    """Generate CSI packets for a person at position (x, y) in the room."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        s = cfg.signal
        # Subcarrier frequencies centred on the carrier.
        k = np.arange(s.n_subcarriers) - s.n_subcarriers // 2
        self.freqs = s.carrier_freq_hz + k * (s.bandwidth_hz / s.n_subcarriers)
        self.tx = np.array(cfg.links.tx.pos)
        self.rx = [np.array(r.pos) for r in cfg.links.rx]
        # Wall x/y coordinates for image-source reflections.
        self._walls = [
            ("x", 0.0), ("x", cfg.room.width),
            ("y", 0.0), ("y", cfg.room.depth),
        ]
        # Static channel is fixed per simulator instance (per "environment").
        self._static = [self._static_channel(rx) for rx in self.rx]

    # ------------------------------------------------------------------ rays

    def _ray(self, length: float, gain: float) -> np.ndarray:
        """Complex response of one ray across subcarriers."""
        phase = -2.0 * np.pi * self.freqs * (length / C)
        amp = gain / max(length, 0.5)  # free-space-like 1/d amplitude decay
        return amp * np.exp(1j * phase)

    def _static_channel(self, rx: np.ndarray) -> np.ndarray:
        """LoS + first-order wall reflections for one link."""
        h = self._ray(float(np.linalg.norm(rx - self.tx)), gain=1.0)
        for axis, coord in self._walls:
            img = self.tx.copy()
            i = 0 if axis == "x" else 1
            img[i] = 2.0 * coord - img[i]
            length = float(np.linalg.norm(rx - img))
            h = h + self._ray(length, gain=0.35)  # reflection loss
        return h

    def _person_ray(self, rx: np.ndarray, person: np.ndarray) -> np.ndarray:
        """Scattered ray TX→person→RX plus LoS shadowing by the body."""
        d1 = float(np.linalg.norm(person - self.tx))
        d2 = float(np.linalg.norm(rx - person))
        scattered = self._ray(d1 + d2, gain=1.2)
        # Body shadowing: attenuate LoS when the person stands near the
        # straight TX→RX segment (within ~0.5 m).
        seg = rx - self.tx
        t = float(np.clip(np.dot(person - self.tx, seg) / np.dot(seg, seg), 0, 1))
        closest = self.tx + t * seg
        gap = float(np.linalg.norm(person - closest))
        shadow = 1.0 - 0.6 * np.exp(-(gap / 0.5) ** 2)
        return scattered, shadow

    # --------------------------------------------------------------- packets

    def csi_at(self, x: float, y: float, z: float = 1.0) -> np.ndarray:
        """One CSI packet per link for a person at (x, y, z).

        Returns
        -------
        complex array of shape (n_links, n_subcarriers). Entries are NaN for
        links whose packet was lost.
        """
        s = self.cfg.signal
        person = np.array([x, y, z])
        out = np.empty((len(self.rx), s.n_subcarriers), dtype=complex)
        for i, rx in enumerate(self.rx):
            if self.rng.random() < s.packet_loss:
                out[i] = np.nan
                continue
            scattered, shadow = self._person_ray(rx, person)
            h = self._static[i] * shadow + scattered
            # Hardware impairments: random common phase + linear slope (SFO).
            cfo = self.rng.uniform(0, 2 * np.pi)
            slope = self.rng.uniform(-0.1, 0.1)
            k = np.arange(s.n_subcarriers)
            h = h * np.exp(1j * (cfo + slope * k))
            h = h + (self.rng.normal(0, s.noise_std, h.shape)
                     + 1j * self.rng.normal(0, s.noise_std, h.shape))
            out[i] = h
        return out

    def csi_empty(self) -> np.ndarray:
        """One CSI packet per link with no person in the room."""
        s = self.cfg.signal
        out = np.empty((len(self.rx), s.n_subcarriers), dtype=complex)
        for i in range(len(self.rx)):
            if self.rng.random() < s.packet_loss:
                out[i] = np.nan
                continue
            h = self._static[i].copy()
            cfo = self.rng.uniform(0, 2 * np.pi)
            slope = self.rng.uniform(-0.1, 0.1)
            k = np.arange(s.n_subcarriers)
            h = h * np.exp(1j * (cfo + slope * k))
            h = h + (self.rng.normal(0, s.noise_std, h.shape)
                     + 1j * self.rng.normal(0, s.noise_std, h.shape))
            out[i] = h
        return out

    def rssi_at(self, csi: np.ndarray) -> np.ndarray:
        """Per-link RSSI (dB) derived from a CSI packet (NaN-safe)."""
        power = np.nanmean(np.abs(csi) ** 2, axis=1)
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(power)

    # ------------------------------------------------------------ recordings

    def record_static(self, x: float, y: float, n_packets: int,
                      jitter: float = 0.012) -> np.ndarray:
        """CSI time series for a person standing at (x, y).

        Small positional jitter models breathing/body sway so consecutive
        packets are not identical. Shape: (n_packets, n_links, n_subcarriers).
        """
        frames = []
        for _ in range(n_packets):
            jx = x + self.rng.normal(0, jitter)
            jy = y + self.rng.normal(0, jitter)
            frames.append(self.csi_at(jx, jy))
        return np.stack(frames)

    def record_path(self, path_xy: np.ndarray) -> np.ndarray:
        """CSI time series along a trajectory of shape (n_packets, 2)."""
        return np.stack([self.csi_at(px, py) for px, py in path_xy])


class WalkGenerator:
    """Generate smooth random walking trajectories inside the room.

    Used both for building the moving-person training data and for driving
    the live demo. Waypoint-based: pick a random target, walk towards it at
    ~walking speed, repeat.
    """

    def __init__(self, cfg: Config, speed: float = 0.8,
                 seed: int | None = None, margin: float = 0.4):
        self.cfg = cfg
        self.speed = speed  # m/s
        self.margin = margin
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([cfg.room.width / 2, cfg.room.depth / 2])
        self.target = self._new_target()

    def _new_target(self) -> np.ndarray:
        return np.array([
            self.rng.uniform(self.margin, self.cfg.room.width - self.margin),
            self.rng.uniform(self.margin, self.cfg.room.depth - self.margin),
        ])

    def step(self, dt: float) -> np.ndarray:
        """Advance the walker by dt seconds; returns current (x, y)."""
        to_target = self.target - self.pos
        dist = float(np.linalg.norm(to_target))
        if dist < 0.15:
            self.target = self._new_target()
            to_target = self.target - self.pos
            dist = float(np.linalg.norm(to_target))
        step_len = min(self.speed * dt, dist)
        self.pos = self.pos + to_target / dist * step_len
        # Gait wobble.
        self.pos = self.pos + self.rng.normal(0, 0.01, 2)
        self.pos[0] = float(np.clip(self.pos[0], 0.05, self.cfg.room.width - 0.05))
        self.pos[1] = float(np.clip(self.pos[1], 0.05, self.cfg.room.depth - 0.05))
        return self.pos.copy()

    def trajectory(self, n_steps: int, dt: float) -> np.ndarray:
        return np.stack([self.step(dt) for _ in range(n_steps)])
