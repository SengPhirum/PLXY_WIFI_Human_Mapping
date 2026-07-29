# TODO — Wi-Fi Human Mapping

Ordered roughly by thesis value ÷ effort. Check off in place, keep history.
See `PROJECT_STATUS.md` for what is already done, and the 12-month plan in
`THESIS_PLAN.md` §17 for scheduling context.

## Now (software, no hardware required)

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

- [x] Core package: config, simulator, preprocessing, models, tracking,
      eval, persistence (2026-07-29)
- [x] Leakage-aware splits + full metrics (2026-07-29)
- [x] ESP32 serial parser + session recorder + firmware guide (2026-07-29)
- [x] FastAPI live server + dashboard + demo scripts (2026-07-29)
- [x] Test suite (17 tests) + end-to-end verification (2026-07-29)
- [x] Baseline comparison matrix on simulated dataset (2026-07-29)
