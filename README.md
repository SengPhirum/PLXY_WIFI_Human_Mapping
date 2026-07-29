# Wi-Fi Human Mapping

Privacy-preserving indoor human localization and movement mapping using
commodity Wi-Fi Channel State Information (CSI) and lightweight machine
learning — the implementation companion to the master's thesis plan in
[`reference/THESIS_PLAN.md`](reference/THESIS_PLAN.md).

A person changes how Wi-Fi signals propagate through a room. This project
detects that change on multiple low-cost TX→RX links, predicts the person's
grid zone and (x, y) coordinates from cleaned CSI features, Kalman-smooths
the track, and renders a live floor-plan dashboard with position, trajectory,
and occupancy heatmap. **No camera, no wearable.**

![pipeline](docs/ARCHITECTURE.md): CSI packets → cleaning → sliding windows →
features → zone/xy models → Kalman → dashboard.

## Quick start (no hardware needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py          # → open http://127.0.0.1:8000
```

The first run simulates a full data-collection campaign (8 sessions across
4 days and 4 participants, ~2 min), trains a Random-Forest localizer with an
honest session-independent split, then starts the dashboard with a simulated
person walking the room — predictions and live localization error on screen.
The **/body** page adds a wireframe-avatar view of the live track (dual-pane,
mesh-on-black style). Full walkthrough: **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)**.

### Pose estimation demo

```bash
python scripts/run_demo.py --pose     # → http://127.0.0.1:8000/body
```

Predicts a **14-joint body skeleton and posture class from CSI**, and
renders a wireframe mesh built around the predicted joints with the
ground-truth skeleton overlaid — dual-pane, in the style of
DensePose-from-WiFi. Measured on simulated CSI with a session-independent
split: posture accuracy **0.583** (chance 0.20; `walk` 98%), **MPJPE
17.7 cm**, PCK@20 cm 0.64. Research basis, feasibility analysis, and the
scope statement: **[reference/POSE_FROM_WIFI.md](reference/POSE_FROM_WIFI.md)**
and **[docs/POSE_GUIDE.md](docs/POSE_GUIDE.md)**.

**Real signal, today, no ESP32:**

- `python scripts/wifi_live.py` — presence/motion sensing on the laptop's
  own connected-link RSSI (single link).
- `python scripts/router_live.py` — **whole-home mode**: polls the router
  for the RSSI of *every* connected device and maps per-link motion onto
  the floor plan (which link region is occupied). Works with OpenWrt-class
  routers over ssh; `--demo` tries the UI without any router. See
  [docs/ROUTER_GUIDE.md](docs/ROUTER_GUIDE.md).

Precise (x, y) localization still requires the multi-link CSI setup
(simulator today, ESP32 when hardware arrives).

With real ESP32 hardware: **[docs/HARDWARE_GUIDE.md](docs/HARDWARE_GUIDE.md)**
and [firmware/esp32-csi/README.md](firmware/esp32-csi/README.md).

## What's here

| Path | Contents |
|---|---|
| `src/wifi_mapping/` | The package: simulator, body model, preprocessing, models, tracking, eval, server |
| `scripts/` | `generate_dataset` → `train` → `evaluate` → `run_demo`, plus `generate_pose_dataset`/`train_pose`, `collect_esp32`, `wifi_live`, `router_live` |
| `config/default.yaml` | Room geometry, grid, link layout, signal + pipeline parameters |
| `config/pose.yaml` | Pose configuration: 6-link receiver ring, no PCA |
| `firmware/esp32-csi/` | ESP32 flashing and verification guide |
| `tests/` | pytest suite — 35 tests, fast, no hardware |
| `docs/` | Setup guide, hardware guide, architecture notes |
| `reference/` | **Research notes, thesis plan, TODO, and project status for continuing work** |

## Results (simulated dataset, session-independent test)

Run `python scripts/evaluate.py` to reproduce; numbers land in
`data/results/`. See `reference/PROJECT_STATUS.md` for the latest table and
interpretation.

## Thesis mapping

| Plan section | Where implemented |
|---|---|
| §7 hardware platform | `firmware/esp32-csi/`, `collect/serial_reader.py` |
| §9 collection protocol | `scripts/collect_esp32.py`, `dataset.generate_session` |
| §11 processing pipeline | `preprocess/` (Hampel, low-pass, phase sanitization, windows, PCA) |
| §11.2 models | `models/` (RSSI-KNN, KNN/SVM/RF/MLP, optional CNN & CNN-GRU) |
| §12 leakage prevention | `eval/splits.py` (session/day/person-independent) |
| §13 metrics | `eval/metrics.py`, `models/pose.py` (MPJPE/PCK) |
| §15 dashboard | `server/` (FastAPI + WebSocket + canvas floor plan + body view) |
| pose feasibility study | `simulate/body_model.py`, `models/pose.py`, `reference/POSE_FROM_WIFI.md` |

## License / ethics

Research prototype for consenting, controlled experiments. Follow the data
governance rules in the thesis plan (§9.2): informed consent, anonymous
participant codes, no unnecessary video retention.
