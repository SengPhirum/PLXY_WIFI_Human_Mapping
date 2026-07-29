# Project Status & Handoff

> **Purpose of this file:** let anyone — including a different Claude
> account/session with zero context — continue this project without
> re-deriving anything. Read this first, then `TODO.md` for next actions.
> Keep this file updated at the end of every working session.

**Last updated:** 2026-07-29
**State:** Software prototype complete and verified end-to-end on simulated
data. No hardware yet (per plan timeline, purchase is month 3).

## What this project is

Implementation of the master's thesis plan in `THESIS_PLAN.md` (converted
from the uploaded .docx — that file is the requirements source of truth):
privacy-preserving indoor human localization using Wi-Fi CSI from low-cost
ESP32 links, with zone classification, (x, y) regression, Kalman-smoothed
tracking, and a live floor-plan dashboard. Prepared for Seng Phirum
(sengphirum143@gmail.com), thesis timeline July 2026 → ~July 2027.

## How to get running from a fresh clone

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                          # 17 tests should pass
python scripts/run_demo.py      # → http://127.0.0.1:8000
```

Full instructions: `docs/SETUP_GUIDE.md`. Codebase map:
`docs/ARCHITECTURE.md`. Hardware path: `docs/HARDWARE_GUIDE.md` +
`firmware/esp32-csi/README.md`.

## What is DONE (verified working)

- **Core package** `src/wifi_mapping/`: YAML config; geometry-based
  multipath CSI simulator (LoS + wall reflections + body scattering/
  shadowing + CFO/SFO phase corruption + packet loss + cross-day drift);
  preprocessing (lost-packet interpolation, Hampel, low-pass, phase
  sanitization, per-link normalization, sliding windows, PCA) with strict
  train-only fitting; models via one factory (rssi_knn / knn / svm / rf /
  mlp sklearn + cnn / cnn_gru torch-optional); constant-velocity Kalman;
  leakage-aware session/day/person splits; full metric set from plan §13.
- **Collection tooling**: ESP-CSI serial parser (unit-tested against the
  documented line format), multi-port threaded reader, guided collection
  script writing hardware sessions in the *same* .npz format as simulation.
- **Server + dashboard**: FastAPI + WebSocket; canvas floor plan with live
  predicted position, ground-truth ✕ (sim mode), trail, occupancy heatmap,
  zone highlight, stat tiles (zone/position/error/latency/rate), recent-
  predictions table. Hardware frames can be POSTed to `/api/frame`.
- **Scripts**: `generate_dataset` → `train` → `evaluate` → `run_demo`
  (one-command demo with caching), `collect_esp32` (hardware).
- **Tests**: 17 pytest cases — simulator physics, filters, phase
  sanitization, windowing, Kalman, metrics, split integrity, serial parsing,
  save/load round-trip + end-to-end mini-pipeline. All passing.

## Current results (simulated `demo` dataset, 8 sessions / 4 days / 4 people)

`python scripts/evaluate.py --dataset demo` reproduces (~5 min):

| model | split | zone acc | macro F1 | mean err (m) | median (m) | p90 (m) | <1 m |
|---|---|---|---|---|---|---|---|
| rssi_knn | session | 0.470 | 0.445 | 0.894 | 0.639 | 2.124 | 0.70 |
| knn | session | 0.319 | 0.277 | 1.263 | 0.984 | 2.632 | 0.51 |
| **rf** | session | **0.547** | 0.511 | 0.928 | 0.735 | 1.901 | 0.67 |
| mlp | session | 0.530 | 0.481 | 0.894 | 0.715 | 1.878 | 0.67 |
| rssi_knn | day | 0.402 | 0.363 | 1.049 | 0.830 | 2.086 | 0.60 |
| knn | day | 0.307 | 0.263 | 1.328 | 1.200 | 2.507 | 0.45 |
| **rf** | day | **0.519** | 0.465 | 0.974 | 0.758 | 1.856 | 0.61 |
| mlp | day | 0.391 | 0.311 | 1.057 | 0.900 | 1.802 | 0.55 |
| rssi_knn | person | 0.474 | 0.444 | 0.845 | 0.613 | 1.819 | 0.72 |
| knn | person | 0.306 | 0.272 | 1.247 | 0.989 | 2.652 | 0.51 |
| **rf** | person | **0.490** | 0.441 | 0.988 | 0.800 | 2.003 | 0.60 |
| mlp | person | 0.463 | 0.417 | 0.927 | 0.747 | 1.848 | 0.64 |

Interpretation (details in `RESEARCH_NOTES.md` §4): RF best overall; zone
chance level is 6.25%; ~0.9 m mean error is in the realistic range for
commodity fingerprinting. **Known simulator artifact:** RSSI-KNN is
unrealistically strong here — don't cite it for RQ6; that comparison needs
hardware data.

## Key context a newcomer needs (read before changing code)

1. **Leakage discipline is the hill to die on** (plan §12): splits are by
   whole sessions; normalizer/PCA fit only via `sessions_to_arrays(...,
   fit=True)` on the training partition. Any change that breaks this
   invalidates every number produced.
2. **Sessions carry `seg` ids** for contiguous captures; windows/filters
   must never cross segment boundaries (position jumps).
3. **Sim and hardware share one session format** — that equivalence is the
   whole hardware-transition strategy; don't fork the formats.
4. **Simulator knobs were tuned deliberately** (sway ≈1 cm, re-mount σ=4 mm,
   scatter gain 1.2) — rationale in `DECISIONS.md` D8; don't "fix" them
   casually, results shift.
5. The `.gitignore` excludes `data/` — datasets/models/results are
   reproducible artifacts, regenerate with the scripts.

## Repository state

- Branch: `claude/project-setup-guide-uw9d9p` (designated working branch).
- Remote: `sengphirum/plxy_wifi_human_mapping` (GitHub).
- Environment used for verification: Python 3.11.15, numpy 2.4.6,
  sklearn 1.9.0; torch NOT installed (deep models unexercised — code
  review-complete but not runtime-verified; first torch user should run
  `pytest` + a quick `--model cnn` train).

## Immediate next steps (full list in TODO.md)

1. Doppler/STFT features + temporal CNN-GRU on walking data (software-only).
2. Link-count/placement ablation script — feeds thesis contribution B.
3. Cross-environment calibration experiment design — thesis contribution A
   (the recommended novelty), can be prototyped entirely in simulation.
4. When hardware arrives: pilot per `docs/HARDWARE_GUIDE.md`, then re-run
   the evaluation matrix on real sessions.

## Session log

- **2026-07-29** (Claude Code, initial build): converted thesis plan docx →
  `THESIS_PLAN.md`; implemented entire package, scripts, server, dashboard,
  tests, docs, reference folder; tuned simulator for honest-but-learnable
  fingerprints (see RESEARCH_NOTES §4); generated demo dataset; ran full
  baseline matrix; verified live demo server end-to-end.
