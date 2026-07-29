"""FastAPI application: mapping dashboard + live position WebSocket.

Three live sources:

- ``mode="sim"``  — a simulated person walks the room; the simulator's CSI
  is fed through the trained model in real time. The ground-truth position
  is broadcast alongside the prediction so the dashboard can show live
  localization error. This is the no-hardware demo.
- ``mode="hw"``   — frames come from ESP32 receivers via
  ``scripts/collect_esp32.py --live`` posting to /api/frame (documented in
  docs/SETUP_GUIDE.md). No ground truth in this mode.
- ``mode="rssi"`` — REAL Wi-Fi signal from this machine's connected
  wireless interface (collect/rssi_live.py): live RSSI waveform plus
  motion/presence detection. No localization — a single laptop link
  carries no position information; the dashboard switches to a signal
  panel and the body view shows a presence-reactive avatar.
- ``mode="router"`` — REAL Wi-Fi via the router: per-station RSSI of every
  connected device (collect/router_live.py), per-link motion detection,
  activity drawn on the floor plan for devices with configured positions.

Endpoints:
    GET  /               dashboard (single-file HTML/JS)
    GET  /body           wireframe body view (avatar rendering of the track)
    GET  /api/config     room geometry, grid, node layout for rendering
    GET  /api/status     packet/prediction counters, model info
    POST /api/frame      push one CSI frame (hardware mode)
    WS   /ws             stream of prediction/signal events (JSON)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..config import Config
from ..simulate import CsiSimulator, WalkGenerator
from .live import LivePredictor

STATIC_DIR = Path(__file__).parent / "static"


class Broadcaster:
    """Fan out prediction events to all connected WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def send(self, event: dict) -> None:
        msg = json.dumps(event)
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def create_app(cfg: Config, predictor: LivePredictor | None, mode: str = "sim",
               sim_seed: int | None = None,
               wifi_interface: str | None = None,
               router_transport=None) -> FastAPI:
    app = FastAPI(title="Wi-Fi Human Mapping", version="0.1.0")
    hub = Broadcaster()
    state: dict[str, Any] = {
        "mode": mode, "frames": 0, "predictions": 0,
        "started": time.time(), "last_latency_ms": None,
    }

    # ------------------------------------------------------------ live loop

    async def sim_loop() -> None:
        """Real-time simulated person: generate CSI at the configured packet
        rate, predict, and broadcast prediction + ground truth."""
        sim = CsiSimulator(cfg, seed=sim_seed)
        walker = WalkGenerator(cfg, speed=0.7, seed=sim_seed)
        dt = 1.0 / cfg.signal.sample_rate_hz
        # Generate in small batches so the event loop stays responsive.
        batch = max(1, int(cfg.signal.sample_rate_hz // 20))
        next_tick = time.perf_counter()
        while True:
            for _ in range(batch):
                gt = walker.step(dt)
                event = predictor.push_frame(sim.csi_at(gt[0], gt[1]))
                state["frames"] += 1
                if event is not None:
                    event.update(gt_x=float(gt[0]), gt_y=float(gt[1]),
                                 mode="sim")
                    if "x" in event:
                        event["error_m"] = float(np.hypot(
                            event["x"] - gt[0], event["y"] - gt[1]))
                    state["predictions"] += 1
                    state["last_latency_ms"] = event["latency_ms"]
                    await hub.send(event)
            next_tick += batch * dt
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))

    async def rssi_loop() -> None:
        """Real Wi-Fi from the connected interface → signal/motion events."""
        from ..collect.rssi_live import RssiMonitor

        monitor = RssiMonitor(rate_hz=10.0, interface=wifi_interface)
        if not monitor.sampler.available:
            state["error"] = ("no wireless interface backend found "
                              "(/proc/net/wireless, iw, airport, netsh)")
            return
        monitor.start()
        state["backend"] = monitor.sampler.backend
        last_sent = 0.0
        while True:
            await asyncio.sleep(0.1)
            reading = monitor.latest()
            if reading is None or reading["t"] <= last_sent:
                continue
            last_sent = reading["t"]
            state["frames"] += 1
            state["predictions"] += 1
            await hub.send({**reading, "mode": "rssi"})

    async def router_loop() -> None:
        """Per-device RSSI from the router → per-link activity events."""
        from ..collect.router_live import RouterMonitor

        rcfg = cfg.router or {}
        monitor = RouterMonitor(
            router_transport,
            poll_hz=float(rcfg.get("poll_hz", 2.0)),
            threshold=float(rcfg.get("threshold", 0.8)),
        )
        monitor.start()
        state["backend"] = "router"
        devices = rcfg.get("devices") or {}
        last_sent = 0.0
        while True:
            await asyncio.sleep(1.0 / (2 * monitor.poll_hz))
            reading = monitor.latest()
            if reading is None or reading["t"] <= last_sent:
                continue
            last_sent = reading["t"]
            for s in reading["stations"]:
                meta = devices.get(s["mac"]) or {}
                s["name"] = meta.get("name", s["mac"][-8:])
                if "pos" in meta:
                    s["pos"] = meta["pos"]
            state["frames"] += 1
            state["predictions"] += 1
            await hub.send({"t": reading["t"], "mode": "router",
                            "stations": reading["stations"]})

    @app.on_event("startup")
    async def _startup() -> None:
        if mode == "sim":
            asyncio.create_task(sim_loop())
        elif mode == "rssi":
            asyncio.create_task(rssi_loop())
        elif mode == "router":
            asyncio.create_task(router_loop())

    # ------------------------------------------------------------ endpoints

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/body")
    async def body_view() -> FileResponse:
        return FileResponse(STATIC_DIR / "body.html")

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        return JSONResponse({
            "room": {"name": cfg.room.name, "width": cfg.room.width,
                     "depth": cfg.room.depth},
            "grid": {"cols": cfg.grid.cols, "rows": cfg.grid.rows},
            "tx": {"id": cfg.links.tx.id, "pos": list(cfg.links.tx.pos)},
            "rx": [{"id": r.id, "pos": list(r.pos)} for r in cfg.links.rx],
            "mode": mode,
            "update_hz": cfg.signal.sample_rate_hz / cfg.preprocess.window_step,
            "router": {
                "position": (cfg.router or {}).get("position"),
                "devices": (cfg.router or {}).get("devices") or {},
            },
        })

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse({**state, "uptime_s": time.time() - state["started"],
                             "clients": len(hub.clients)})

    @app.post("/api/frame")
    async def api_frame(payload: dict) -> JSONResponse:
        """Hardware mode: one aligned frame as {"re": [[..]], "im": [[..]]}."""
        if predictor is None:
            return JSONResponse({"ok": False, "error": "no model loaded in this mode"},
                                status_code=409)
        frame = np.array(payload["re"], dtype=float) \
            + 1j * np.array(payload["im"], dtype=float)
        event = predictor.push_frame(frame)
        state["frames"] += 1
        if event is not None:
            event["mode"] = "hw"
            state["predictions"] += 1
            state["last_latency_ms"] = event["latency_ms"]
            await hub.send(event)
        return JSONResponse({"ok": True, "predicted": event is not None})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive pings from the client
        except WebSocketDisconnect:
            hub.disconnect(ws)

    return app
