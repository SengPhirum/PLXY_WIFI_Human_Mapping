# Architecture

## Data flow

```
                     SIMULATION PATH                      HARDWARE PATH
              ┌────────────────────────┐        ┌─────────────────────────────┐
              │ simulate/csi_simulator │        │ ESP32 × 3 → USB serial       │
              │  multipath + person    │        │ collect/serial_reader        │
              │  + CFO/SFO + loss      │        │ collect/session (alignment)  │
              └───────────┬────────────┘        └──────────────┬──────────────┘
                          └────────────┬───────────────────────┘
                                       ▼
                  session .npz  (csi, xy, seg, meta)   ← dataset.py
                                       ▼
                    preprocess/  clean → normalize → window → features
                    (Hampel, low-pass, phase sanitization, PCA;
                     normalizer/PCA fitted on TRAIN sessions only)
                                       ▼
              ┌── offline ────────────────────────────┐   ┌── online ─────────┐
              │ eval/splits  session/day/person-      │   │ server/live       │
              │              independent split        │   │  ring buffer +    │
              │ models/      rssi_knn|knn|svm|rf|mlp  │   │  same features    │
              │              (+cnn, cnn_gru w/ torch) │   │  → predict        │
              │ eval/metrics zone + localization      │   │  → tracking/kalman│
              │ persistence  model bundle .pkl        │   │  → WebSocket      │
              └───────────────────────────────────────┘   └─────────┬─────────┘
                                                                    ▼
                                                     server/static/index.html
                                                     (canvas floor plan, trail,
                                                      heatmap, stat tiles)
```

## Key design decisions

**One session format for sim and hardware.** `generate_session` (simulator)
and `SessionRecorder` (ESP32) emit identical `.npz` files, so every
downstream stage — training, evaluation, live inference — is source-agnostic.
Swapping simulated for real data is a directory name.

**Segments prevent silent label mixing.** A session stores a `seg` id per
packet; windows and filters never cross segment boundaries, so a window
can't straddle the position jump between two standing spots.

**Leakage discipline is structural, not procedural.** Normalization and PCA
are stateful on `PreprocessPipeline` and fitted only inside
`sessions_to_arrays(..., fit=True)`, which train/evaluate call only on the
training partition. `split_sessions` refuses to split anything finer than
whole sessions.

**The model bundle carries its preprocessing.** `persistence.save_bundle`
stores pipeline state + models + config snapshot together; the live server
can't accidentally run a model against differently-normalized features.

**Deep models are optional.** `models/registry.py` builds sklearn baselines
unconditionally; `cnn`/`cnn_gru` import torch lazily and fail with an
actionable message if it's absent. The demo never needs a GPU.

**Simulator realism budget.** The simulator reproduces the mechanisms that
matter for the pipeline (subcarrier-dependent multipath fingerprints, body
shadowing, per-packet random CFO/SFO phase corruption, packet loss,
cross-day environment drift) and skips what doesn't (full EM ray tracing,
antenna patterns). Honest limitation: simulated results validate the
*pipeline*, not the thesis's empirical claims — those need the hardware
dataset. See `reference/RESEARCH_NOTES.md` § Simulator.

## Module map

| Module | Responsibility | Key entry points |
|---|---|---|
| `config.py` | Typed YAML config, room/zone geometry | `load_config`, `Config.zone_of` |
| `simulate/csi_simulator.py` | Synthetic CSI + walking trajectories | `CsiSimulator.csi_at`, `WalkGenerator.step` |
| `collect/serial_reader.py` | ESP-CSI serial parsing, multi-port reader | `parse_csi_line`, `CsiSerialReader` |
| `collect/session.py` | Frame alignment + ground truth → .npz | `SessionRecorder` |
| `dataset.py` | Session generation/IO, feature assembly | `generate_session`, `sessions_to_arrays` |
| `preprocess/` | Filters, phase, normalization, windows | `PreprocessPipeline.stream_to_features` |
| `models/` | Model factory, RSSI features, torch models | `create_model` |
| `tracking/kalman.py` | Constant-velocity smoothing | `KalmanTracker2D.update` |
| `eval/` | Group splits, metrics | `split_sessions`, `localization_report` |
| `server/` | Live inference + dashboard | `create_app`, `LivePredictor` |
| `persistence.py` | Model bundles | `save_bundle`, `load_bundle` |

## Performance envelope (laptop CPU, defaults)

- Simulation: ~8 k packets/s (real-time factor ×80 at 100 Hz)
- Feature extraction per 1 s window: ~5 ms
- RF inference per window: <10 ms → end-to-end latency ≪ update period (0.5 s)
- Training on the demo dataset (8 sessions): ~1 min
