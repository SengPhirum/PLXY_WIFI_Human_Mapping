# Pose Estimation from Wi-Fi — Research Review & Implementation Basis

Research notes behind the pose track in this repository. The target image
(dense body mesh overlaid on people in a room) is the headline figure of
**DensePose From WiFi**, CMU 2022. This document establishes what that work
actually does, what it requires, what is reproducible here, and what this
repository therefore implements and claims.

---

## 1. The reference works

### DensePose From WiFi (Geng, Huang, De la Torre — CMU, arXiv:2301.00250)

- **Hardware:** two commodity 3-antenna routers (TP-Link AC1750-class,
  ≈ $30 each) facing each other → **3 × 3 = 9 antenna links**, 56
  subcarriers of CSI per link.
- **Pipeline (three stages):**
  1. amplitude & phase sanitization of raw CSI;
  2. a two-branch encoder–decoder that maps sanitized CSI into 2D
     "image-like" feature maps in the camera's spatial domain;
  3. a modified **DensePose-RCNN** head that predicts UV surface
     coordinates plus 17 COCO keypoints.
- **Supervision:** *teacher–student*. A camera runs image-based DensePose;
  its output is the training label for the Wi-Fi student network. The
  camera is required for **training**, not for inference — this is the
  core trick, and it's the same trick Person-in-WiFi and Wi2Vi use.
- **Results:** same-layout AP ≈ 43.5 (with keypoint supervision + transfer
  learning, from a 42.9 baseline).
- **Stated limitation — the important one:** on an **unseen layout**
  (trained on 15 layouts, tested on 1 held out) AP drops **43.5 → 27.3**
  and dpAP·GPS **45.3 → 25.4**. Wi-Fi propagation differs so much between
  environments that cross-environment deployment is unsolved. The authors
  note image-based models suffer the same domain-generalization problem.

### Related work that frames the feasibility question

| Work | Hardware | What it shows |
|---|---|---|
| Person-in-WiFi (ICCV'19) | 3×3 links, Intel 5300 | Joint heatmaps + part affinity fields from CSI; the origin of the "pose from Wi-Fi" line |
| Person-in-WiFi 3D (CVPR'24) | Many antennas | Multi-person 3D pose; confirms more antennas → more capability |
| WiPose / MetaFi / MM-Fi baselines | 1×3 MIMO, 114 subcarriers | Standard CSI→keypoint regression baselines |
| **MM-Fi dataset** (NeurIPS'23) | 1×3 MIMO, 114 subcarriers | **40 subjects, 27 actions, 4 environments, 320k synchronized frames**, 17-joint ground truth from stereo IR cameras — the public dataset to use when hardware exists |
| RuView / ESP32 mesh projects (2026) | 4–6 ESP32 nodes | Community attempts at ESP32-based pose; consistently report that a model **must be trained per environment**, and that camera-supervised pipelines are still the bottleneck |

### What is definitively NOT possible

- **Pose from RSSI.** Consumer laptops and routers expose RSSI — *one
  number per link*. Pose needs CSI: 52+ complex subcarrier values per link
  per packet. This is a hardware/firmware limit, not a modelling one. The
  router mode in this repo (`scripts/router_live.py`) is therefore
  presence/motion sensing, and can never be upgraded to pose.
- **Zero-shot deployment in a new room.** Every published system degrades
  sharply across layouts (see the 43.5 → 27.3 drop above).
- **Pose without training labels.** Every system is camera-supervised
  during data collection.

---

## 2. What this repository can honestly build today

| Requirement of the CMU result | Status here |
|---|---|
| 9 CSI links | ❌ no CSI hardware yet (ESP32 purchase is plan month 3). Simulator configured with a **6-link array** (`config/pose.yaml`) as the pose-appropriate geometry |
| Camera-supervised DensePose labels | ❌ no camera rig, no labelled capture campaign |
| Large in-environment dataset | ⚠️ simulated: 8 sessions × 1200 captures |
| Dense UV surface (DensePose) | ❌ out of scope — dense surface needs the RCNN + UV supervision. We predict a **14-joint skeleton**, then render a mesh around it |
| End-to-end pipeline, metrics, live view | ✅ implemented and evaluated |

**Decision (see DECISIONS.md D16):** implement the *full pose pipeline* —
articulated body physics → CSI → features → posture classifier + joint
regressor → fused skeleton → live mesh rendering — against the **simulator**,
with real keypoint metrics (MPJPE, PCK) and leakage-aware splits. This
validates the pipeline and reproduces the literature's qualitative findings;
it is not, and is never presented as, evidence about real-CSI accuracy.

---

## 3. What the implementation does

**Body model** (`simulate/body_model.py`): 14 joints (COCO subset), five
postures — `idle, walk, sit, wave, tpose` — with kinematic animation
(gait swing, ~2 Hz wave oscillation, seated leg fold).

**Physics** (`simulate/csi_simulator.py: csi_sequence_joints`): each joint
is a scatterer contributing a TX→joint→RX ray with a per-part radar
cross-section weight; the trunk additionally shadows the LoS. Vectorized
over time and joints (≈30× faster than the naive loop), which is what makes
dataset generation practical.

**Features** (`preprocess/pipeline.py`): the localization features
(amplitude mean/std/motion-energy, sanitized-phase mean) **plus** a
per-link **motion spectrum** (`spectral_band_features`) — FFT band powers
over 0–0.5, 0.5–1, 1–2, 2–3.5, 3.5–6, 6+ Hz, log-scaled and per-link
normalized. This is the Doppler-style feature the plan lists in §11.1, and
it is what separates movement *types*. Static-background subtraction using
the session's empty-room baseline is applied first.

**Models** (`models/pose.py`): posture MLP classifier + joint MLP
regressor, then **`fuse_pose`** — blend the regressed skeleton toward the
canonical skeleton of the predicted posture, weighted by classifier
confidence. This is the standard prior-fusion trick for noisy keypoint
regression; it is what makes the live figure anatomically plausible instead
of a twitching point cloud.

**Metrics** (`models/pose.py`): MPJPE (cm), median joint error, PCK@10cm,
PCK@20cm, worst-joint error, per-joint breakdown, posture accuracy/macro-F1
and confusion — all under session/day/person-independent splits.

---

## 4. Findings from the simulated pose experiments

Final run: 8 sessions × 1200 captures (9600 windows), 6 links,
session-independent split (6 train / 2 test sessions, 7200 / 2400 windows).

```
posture accuracy 0.583   macro-F1 0.582        (chance = 0.20)
confusion (rows = true):
  idle  [146,   1, 238,  37,  68]
  walk  [  5, 479,   0,   4,   0]     ← 98% correct
  sit   [134,   0, 232,  30,  81]
  wave  [ 27,   0,  49, 320,  75]     ← 68% correct
  tpose [ 73,   0, 112,  68, 221]
joints raw   MPJPE 18.8 cm   PCK@10 0.47   PCK@20 0.63
joints fused MPJPE 17.7 cm   PCK@10 0.46   PCK@20 0.64
per-joint (cm): head 12.1 · neck 12.0 · shoulders 12.0 · hips 11.7
                elbows 20.6–24.0 · knees 13.5 · ankles 19.0
                wrists 29.7–36.9   ← worst
```

1. **Dynamic postures are easy; static postures are hard.** `walk` is
   classified at 98% (its ~1 Hz gait spectrum is unmistakable) and `wave`
   at 68%. `idle` ↔ `sit` ↔ `tpose` confuse heavily — they differ only in
   a *static* multipath fingerprint whose posture-induced change is
   comparable to the noise floor. **This is the physically correct
   result**, and it mirrors why the literature needs many links and deep
   networks.
2. **Error grows from the torso outward.** Hips/shoulders/neck land at
   ~12 cm; wrists at 30–37 cm. This proximal-accurate / distal-hard
   gradient is exactly the pattern reported in camera-based *and* Wi-Fi
   pose estimation — extremities are small, fast, and weakly reflective.
   It is a strong sanity signal that the pipeline is learning body
   structure rather than memorizing.
3. **Link count matters more than model choice.** Going from the 3-link
   localization array to the 6-link pose ring improved posture accuracy
   from ≈0.39 to ≈0.47 with an identical model — consistent with CMU using
   9 links and Person-in-WiFi 3D using more.
4. **Data volume is the second lever.** 4× the training data moved
   accuracy 0.47 → 0.583 and MPJPE ≈24 cm → 17.7 cm, with no architecture
   change. The curve had not flattened — `train_pose.py --scaling`
   measures this properly and is the experiment to run before concluding
   anything about model capacity.
5. **Position conditioning does not help.** Feeding the (noisy) predicted
   (x, y) into the pose model as a cascade changed nothing measurable —
   the bottleneck is signal observability, not position confounding.
6. **Motion-spectrum features are the single biggest feature win** for
   posture; window-mean amplitude alone cannot distinguish wave from idle.
   Note the resolution constraint discovered while testing: a 1 s window
   gives 1 Hz FFT bins, so sub-1 Hz structure is unresolvable — an
   argument for longer windows if slow motions must be separated.
7. **Prior fusion helps modestly on MPJPE (18.8 → 17.7 cm) and greatly on
   plausibility** — it is what keeps the live skeleton anatomical instead
   of a twitching point cloud.

These points are, in themselves, a defensible experimental section for the
thesis chapter on pose feasibility — with the honest headline: *pose from
low-cost Wi-Fi is observability-limited, not model-limited.*

---

## 5. The path to the real thing (ranked, for the thesis)

1. **Use MM-Fi** (public: 40 subjects, 27 actions, 4 environments, 17-joint
   labels, 1×3 MIMO CSI). This removes the hardware and labelling blocker
   entirely and lets the same model code train on *real* CSI. **Highest
   value per effort — do this before buying anything.**
2. Buy 2 × 3-antenna CSI-capable routers (Nexmon-compatible) or 6 ESP32-S3
   nodes → 6–9 links, matching the papers' geometry.
3. Camera-supervised collection: webcam + an off-the-shelf pose estimator
   (MediaPipe/OpenPose) writing 17-joint labels synchronized to CSI —
   exactly the teacher–student setup. `collect/session.py` already stores
   synchronized sessions; add a keypoint label stream.
4. Swap the MLP heads for the torch CNN in `models/deep.py` (or a CNN-GRU
   over window sequences) once real data volume justifies it.
5. Report cross-environment degradation honestly — it is the field's open
   problem and matches thesis contribution option A.

---

## Sources

- [DensePose From WiFi (arXiv:2301.00250)](https://arxiv.org/abs/2301.00250)
- [DensePose From WiFi — PDF](https://arxiv.org/pdf/2301.00250)
- [CMU's DensePose From WiFi — Synced review](https://syncedreview.com/2023/01/17/cmus-densepose-from-wifi-an-affordable-accessible-and-secure-approach-to-human-sensing/)
- [MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset (arXiv:2305.10345)](https://arxiv.org/abs/2305.10345)
- [Towards Robust and Realistic Human Pose Estimation via WiFi Signals (arXiv:2501.09411)](https://arxiv.org/html/2501.09411v1)
- [Awesome-WiFi-CSI-Sensing (paper/dataset index)](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing)
- [RuView — ESP32-node Wi-Fi sensing project](https://github.com/ruvnet/RuView)
- [RuView coverage: ESP32 nodes for presence/pose/vitals (CNX Software)](https://www.cnx-software.com/2026/03/26/ruview-project-leverages-esp32-nodes-for-presence-detection-pose-estimation-and-breathing-heart-rate-monitoring/)
