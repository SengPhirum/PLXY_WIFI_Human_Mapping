# TODO — Wi-Fi Human Mapping

Ordered roughly by thesis value ÷ effort. Check off in place, keep history.
See `PROJECT_STATUS.md` for what is already done, and the 12-month plan in
`THESIS_PLAN.md` §17 for scheduling context.

## Pose track — next steps (highest research value)

- [ ] **Train the pose models on MM-Fi** (public dataset: 40 subjects, 27
      actions, 4 environments, 17-joint labels, real 1×3 MIMO CSI). This
      is the single highest-value action in the whole project: it converts
      the pose pipeline from simulator-validated to **real-CSI validated
      with zero hardware purchase**. Write an MM-Fi loader that emits the
      existing pose-session format, then `train_pose.py` works unchanged.
- [ ] Run `train_pose.py --scaling` to completion and put the
      accuracy-vs-data curve in the thesis — the curve had not flattened
      at 7200 windows, so it bounds how much more data is worth collecting.
- [ ] Link-count ablation for pose specifically (3 / 4 / 6 / 9 receivers) —
      quantifies the observability claim and directly informs what hardware
      to buy. Nearly free in simulation.
- [ ] Torch CNN/CNN-GRU pose heads (`models/deep.py`) once MM-Fi data is
      in — only worth it on real data volume, not on the simulated set.
- [ ] Longer windows (2–4 s) to resolve sub-1 Hz motion; currently 1 s
      windows give 1 Hz FFT bins, which is the resolution floor.
- [ ] Camera-supervised labelling rig (webcam + MediaPipe → 17-joint
      labels synchronized to CSI) — required for own-hardware pose data.

## Now (software, no hardware required)

- [ ] **Run `scripts/wifi_live.py` on a real laptop** (container has no
      wireless): confirm the RSSI backend detection on the user's actual
      OS, tune MotionDetector threshold/hold to that environment. User's
      first test failed — walk through SETUP_GUIDE §3b troubleshooting
      table (VM/WSL? flat RSSI from driver caching? power-save?).
- [ ] **Run `scripts/router_live.py` against a real router.** Needs
      OpenWrt/GL.iNet/DD-WRT-class ssh access (ROUTER_GUIDE.md). If the
      user's router is a stock ISP box, the cheapest path is a $20–30
      OpenWrt AP as dedicated sensing head — decide and order early.
- [ ] Router mode upgrades once real data flows: per-link adaptive
      thresholds, station sparklines in the dashboard, activity history
      timeline, optional MQTT/webhook output for home automation.
- [ ] Body view polish (optional): camera orbit control, second ghost
      avatar for ground truth in sim mode, trail ribbon on the floor.

- [ ] **Doppler / STFT features** (plan §11.1): add short-time Fourier
      spectrum of per-link principal amplitude component to
      `preprocess/pipeline.py:extract_window_features`; ablate vs current
      features on the walking subset.
- [ ] **Temporal sequence models**: feed sequences of consecutive windows to
      `models/deep.py:_CnnGru` (the module supports it; `sessions_to_arrays`
      currently emits independent windows — add a `--seq-len` path).
- [ ] **Link-count / placement ablation** (contribution B, nearly free in
      sim): script that sweeps 1–4 receivers and grid placements, regenerates
      datasets, and plots error vs cost. New file `scripts/ablate_links.py`.
- [ ] **Presence detection baseline** (plan scope table row 1): empty-room
      vs occupied classifier; simulator already has `csi_empty()`.
- [ ] **Confidence intervals + statistical tests** (plan §13.3): bootstrap
      the per-window errors in `evaluate.py`; paired test between models,
      report effect sizes.
- [ ] **Trajectory metrics** (plan §13.3): trajectory RMSE / lost-track rate
      on the walking segments; currently only per-window error is reported.
- [ ] Hardware-mode live bridge: small loop that reads
      `CsiSerialReader`, aligns frames (reuse `SessionRecorder` logic), and
      POSTs to `/api/frame` (see docs/HARDWARE_GUIDE.md §4).
- [ ] Wavelet-denoising preprocessing variant (plan §11.1) + ablation flag.
- [ ] Dockerfile + `docker compose up demo` (plan §15 suggested stack).

## Blocked on hardware purchase (plan §17 month 3–4)

- [ ] Buy 4× ESP32 DevKitC + mounts (see docs/HARDWARE_GUIDE.md shopping list).
- [ ] Flash ESP-CSI, run `collect_esp32.py --check`, verify >90 pkt/s/link.
- [ ] Validate `parse_csi_line` against the real firmware's exact CSV layout
      (column index for RSSI may need adjusting — noted in the parser).
- [ ] Pilot: 2 participants × 2 sessions, `--spots-per-zone 2`; compare
      hardware feature distributions vs simulator (sanity histograms).
- [ ] Main collection campaign per THESIS_PLAN §9 (≥2 rooms, ≥10
      participants, multiple days) — only after pilot review with supervisor.
- [ ] Re-run full `evaluate.py` matrix on `--dataset hardware`; update
      RESEARCH_NOTES §4 with real numbers.

## Thesis-specific (writing / research)

- [ ] Choose final thesis title (candidates in THESIS_PLAN §2) and primary
      contribution (recommended: A — cross-environment adaptation).
- [ ] Expand literature matrix in RESEARCH_NOTES §2 to 25–40 papers.
- [ ] Cross-environment adaptation experiment: train in room A (sim),
      calibrate with k windows in room B, plot error vs k. This is the
      novelty experiment — design it before hardware arrives.
- [ ] Ethics/consent paperwork (institution-specific) before any participant
      data.
- [ ] Chapter 4 (System Design) can be drafted now from docs/ARCHITECTURE.md.

## Done

- [x] Pose track: articulated body simulator, per-joint CSI scattering,
      motion-spectrum features, posture + joint models, prior fusion,
      live skeleton mesh view, research review (2026-07-29)
- [x] Real-Wi-Fi RSSI live mode + motion/presence detection + signal panel
      (2026-07-29)
- [x] /body wireframe avatar view, sim + rssi modes (2026-07-29)
- [x] Core package: config, simulator, preprocessing, models, tracking,
      eval, persistence (2026-07-29)
- [x] Leakage-aware splits + full metrics (2026-07-29)
- [x] ESP32 serial parser + session recorder + firmware guide (2026-07-29)
- [x] FastAPI live server + dashboard + demo scripts (2026-07-29)
- [x] Test suite (17 tests) + end-to-end verification (2026-07-29)
- [x] Baseline comparison matrix on simulated dataset (2026-07-29)
