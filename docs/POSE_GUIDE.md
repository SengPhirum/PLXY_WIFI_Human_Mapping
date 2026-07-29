# Pose Estimation Guide

How to run, understand, and extend the pose track — the part of this
project that predicts a **body skeleton** from Wi-Fi channel state, and
renders it as the dual-pane wireframe view.

Research basis, feasibility analysis, and the honest scope statement:
**[../reference/POSE_FROM_WIFI.md](../reference/POSE_FROM_WIFI.md)**. Read
that before quoting any number from here in the thesis.

## Run it

```bash
python scripts/run_demo.py --pose      # → http://127.0.0.1:8000/body
```

Three stages, cached between runs: generate the pose dataset (~6 min the
first time), train the posture classifier + joint regressor (~5 min), then
serve the live view. A simulated person cycles through postures every few
seconds; the model sees only CSI.

**What you see:** the solid viridis mesh is built around the *predicted*
14-joint skeleton. The dashed figure is the simulated body's ground-truth
skeleton — the gap between them is the live joint error, also reported
numerically (MPJPE in cm) along with the predicted posture, the
classifier's confidence, and whether the posture matches truth.

## The pipeline

```
articulated body (14 joints, 5 postures)
   → per-joint scatterers + LoS shadowing  [simulate/csi_simulator.py]
   → CSI stream
   → static-background subtraction (empty-room baseline)
   → Hampel / low-pass / phase sanitization
   → window features + motion spectrum     [preprocess/pipeline.py]
   → posture MLP  ─┐
   → joints  MLP  ─┴→ fuse_pose (prior blending by confidence)
   → temporal smoothing → skeleton → mesh renderer  [server/static/body.html]
```

Postures: `idle`, `walk`, `sit`, `wave`, `tpose`.
Joints (order is a contract with the renderer): head, neck, r/l shoulder,
elbow, wrist, r/l hip, knee, ankle.

## Train and evaluate yourself

```bash
# bigger/smaller dataset
python scripts/generate_pose_dataset.py --config config/pose.yaml \
    --name pose --sessions 8 --captures 1200

# leakage-aware evaluation, three independence protocols
python scripts/train_pose.py --config config/pose.yaml --dataset pose --split-by session
python scripts/train_pose.py --config config/pose.yaml --dataset pose --split-by day
python scripts/train_pose.py --config config/pose.yaml --dataset pose --split-by person

# how much does more data buy? (thesis-relevant curve)
python scripts/train_pose.py --config config/pose.yaml --dataset pose --scaling
```

Reported metrics: posture accuracy, macro-F1, confusion matrix; MPJPE (cm),
median joint error, PCK@10 cm, PCK@20 cm, worst joint, and the per-joint
error breakdown — before and after prior fusion.

## Why `config/pose.yaml` uses six receivers

Pose needs far more spatial diversity than localization. DensePose-from-WiFi
used two 3-antenna routers = **9 links**; MM-Fi uses a 1×3 MIMO array over
114 subcarriers. The localization config's 3 links are not enough: measured
here, moving from 3 → 6 links raised posture accuracy from ≈0.39 to ≈0.47
with an identical model. If you add receivers, add them to `links.rx` and
regenerate — everything downstream adapts automatically.

## Measured results (simulated, session-independent)

8 sessions × 1200 captures, 6 links, 7200 train / 2400 test windows:

| Metric | Value |
|---|---|
| Posture accuracy | **0.583** (chance 0.20), macro-F1 0.582 |
| ├ `walk` | **98%** correct |
| ├ `wave` | 68% correct |
| └ `idle` / `sit` / `tpose` | 30–49% — mutually confused |
| MPJPE (raw → fused) | 18.8 cm → **17.7 cm** |
| PCK@20 cm | **0.64** |
| Best joints | hips 11.7 cm, shoulders/neck/head ~12 cm |
| Worst joints | wrists 29.7–36.9 cm, elbows 20.6–24.0 cm |

How to read this:

- **Dynamic postures are easy; static postures are hard.** Gait produces an
  unmistakable ~1 Hz spectral peak; waving sits at ~2–3 Hz. But `idle`,
  `sit` and `tpose` are *static* — the only cue is a change in the
  standing-wave fingerprint, comparable in size to the noise floor. This is
  the physically correct outcome and exactly why published systems need
  many antennas, deep networks, and large in-environment datasets.
- **Error grows from the torso outward** (hips ~12 cm → wrists ~35 cm).
  That proximal-accurate / distal-hard gradient is the same pattern
  reported in camera-based and Wi-Fi pose estimation, and it is good
  evidence the model learns body structure rather than memorizing windows.
- **The two levers that moved the numbers** were link count (3 → 6 links:
  0.39 → 0.47 accuracy) and data volume (4× data: 0.47 → 0.583, MPJPE
  ≈24 → 17.7 cm). Neither curve had flattened.

The honest headline: **pose from low-cost Wi-Fi is observability-limited,
not model-limited.** Adding links and data helps; swapping the MLP for a
CNN on this data does not rescue static-posture discrimination.

## Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| Receiver count/placement | `config/pose.yaml` `links.rx` | Biggest single lever on accuracy |
| `window_size` | `config/pose.yaml` `preprocess` | FFT resolution = `sample_rate / window_size`; 1 s → 1 Hz bins. Longer windows resolve slower motions but blur transitions |
| `n_pca_components` | same | Keep at 0 for pose — PCA discards the detail the joint regressor needs |
| `prior_weight` | `models/pose.py: fuse_pose` | How hard to snap to the canonical posture. Higher = prettier, less faithful |
| Smoothing 0.6/0.4 | `server/live.py: _add_pose` | Temporal stability of the live skeleton |

## Moving to real CSI

Ranked by value per effort (full reasoning in the research note):

1. **Train on MM-Fi** — a public dataset with 40 subjects, 27 actions, 4
   environments and 17-joint labels from real 1×3 MIMO CSI. No hardware
   purchase, no labelling campaign, and the model code here transfers.
2. Buy CSI-capable hardware giving 6–9 links (2 × Nexmon-compatible
   3-antenna routers, or 6 ESP32-S3 nodes).
3. Camera-supervised collection: webcam + MediaPipe/OpenPose writing
   17-joint labels synchronized to CSI — the teacher–student setup every
   published system uses. `collect/session.py` already records
   synchronized sessions; add a keypoint label stream.
4. Only then swap the MLP heads for the torch CNN in `models/deep.py`.

**Do not** attempt pose from the RSSI modes (`wifi_live.py`,
`router_live.py`). RSSI is one number per link; pose needs the full
subcarrier vector. That is a firmware limit, not a modelling one.
