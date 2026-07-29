"""Session recording with ground-truth labelling for real hardware captures.

Aligns packets from multiple receivers into fixed-rate CSI frames and stores
them in the same ``.npz`` session format the simulator produces, so the rest
of the pipeline is identical for real and simulated data (plan §9, §10).

Ground truth entry is manual for standing captures (operator keys in the
grid position being occupied) and marker-based for walks; camera/UWB
integration is left as a documented extension point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..dataset import save_session
from .serial_reader import CsiPacket


class SessionRecorder:
    """Accumulates aligned CSI frames plus ground truth into a session file.

    Packets arrive asynchronously per link; a frame is emitted at the
    configured sample rate containing the most recent packet per link
    (NaN-filled for links that produced nothing in the frame interval —
    the same convention the simulator uses for lost packets).
    """

    def __init__(self, cfg: Config, session_id: int, day: int, person: int,
                 room: str | None = None):
        self.cfg = cfg
        self.link_ids = [r.id for r in cfg.links.rx]
        self.n_sc = cfg.signal.n_subcarriers
        self.meta = {
            "session": session_id,
            "day": day,
            "person": person,
            "room": room or cfg.room.name,
            "kind": "hardware",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._frames: list[np.ndarray] = []
        self._xy: list[tuple[float, float]] = []
        self._seg: list[int] = []
        self._latest: dict[str, CsiPacket | None] = {l: None for l in self.link_ids}
        self._seg_id = 0
        self._last_seq: dict[str, int] = {}
        self.packet_loss_events = 0

    # ---------------------------------------------------------------- frames

    def offer(self, pkt: CsiPacket) -> None:
        """Feed one packet from the serial reader (any link, any order)."""
        if pkt.link_id not in self._latest:
            return
        prev = self._last_seq.get(pkt.link_id)
        if prev is not None and pkt.seq > prev + 1:
            self.packet_loss_events += pkt.seq - prev - 1
        self._last_seq[pkt.link_id] = pkt.seq
        self._latest[pkt.link_id] = pkt

    def tick(self, x: float, y: float) -> None:
        """Emit one aligned frame with the current ground-truth position.

        Call at the configured sample rate while recording.
        """
        frame = np.full((len(self.link_ids), self.n_sc), np.nan, dtype=complex)
        now = time.time()
        max_age = 2.0 / self.cfg.signal.sample_rate_hz
        for i, link in enumerate(self.link_ids):
            pkt = self._latest[link]
            if pkt is None or now - pkt.timestamp > max_age:
                continue  # stale → treat as lost
            csi = pkt.csi
            if len(csi) >= self.n_sc:
                frame[i] = csi[: self.n_sc]
            else:
                frame[i, : len(csi)] = csi
        self._frames.append(frame)
        self._xy.append((x, y))
        self._seg.append(self._seg_id)

    def new_segment(self) -> None:
        """Start a new contiguous capture (next standing spot / next walk)."""
        self._seg_id += 1

    # ----------------------------------------------------------------- save

    def save(self, path: Path) -> Path:
        if not self._frames:
            raise RuntimeError("no frames recorded")
        self.meta["packet_loss_events"] = self.packet_loss_events
        self.meta["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        session = {
            "csi": np.stack(self._frames).astype(np.complex64),
            "xy": np.array(self._xy, dtype=np.float32),
            "seg": np.array(self._seg, dtype=np.int32),
            "meta": self.meta,
        }
        save_session(session, path)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(self.meta, indent=2))
        return path
