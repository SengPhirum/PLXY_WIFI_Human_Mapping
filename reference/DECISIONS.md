# Decision Log

Why the code is the way it is. Add entries at the top; never delete —
supersede.

---

## 2026-07-29 e — Presence detection rebuild

**D21. Breathing detection is the capability, not an extra.** Rolling
variance only ever detects *movement*; a person sitting still reads as an
empty room, which is the single biggest failure of naive Wi-Fi presence
sensing. The literature establishes that respiration is observable in
commodity RSSI, so respiration-band spectral analysis (0.16–0.6 Hz) is now
the primary stationary-presence cue. Cost: it needs a ~20 s window, hence
the ~20 s warm-up and ~5 s release latency — accepted and documented.

**D22. Motion must be band-limited, not just high-passed.** Three tuning
bugs, all worth remembering because each looked like a threshold problem
and was actually a physics problem:
  1. plain variance conflates breathing with walking → band-pass *above*
     the respiration band;
  2. RSSI is quantized to 1 dB, and an isolated level step rings a
     high-pass filter into what looks like motion → cap the band at 3 Hz
     (nothing human is faster) and require ~0.5 s of *sustained* energy;
  3. noise produces strong in-band spectral peaks, but they wander →
     require the respiration peak to be stable (< 0.05 Hz) before
     declaring a breathing person.

**D23. Autonomous thresholds, not constants.** Thresholds come from
`median + k·MAD` of each feature's own quiet-room distribution during
calibration, with physically-motivated floors (e.g. the motion floor sits
above a single 1 dB LSB step). Removes per-room tuning; the price is that
the room must be empty during calibration, which the UI states and offers a
`recalibrate` button for.

**D24. Default sensitivity 1.25, chosen by sweep not by feel.** Measured
across breathing depths and noise levels: 1.0 leaks ~5% false alarms, 1.5
starts losing weak breathers, 1.25 gives full detection to 0.4 dB with zero
false alarms. Recorded in SENSING_RESEARCH.md §4 so it can be re-derived.

**D25. Active probing ships on by default.** RSSI only refreshes when
frames arrive; an idle link flatlines and no algorithm can help. A low-rate
ping to the gateway is a few hundred bytes/s and is the difference between
a working and a dead demo. Opt out with `--no-probe`.

---

## 2026-07-29 d — Pose estimation track

**D16. Build the pose *pipeline*, not a fake DensePose.** The requested
reference image is CMU's DensePose-from-WiFi: dense UV body surface, from
9 antenna links, supervised by a camera running image-based DensePose. We
have neither CSI hardware nor a camera rig, so dense surface regression is
not reproducible here. Decision: implement the complete pose pipeline
(articulated body physics → CSI → features → posture + 14-joint models →
fused skeleton → mesh rendering) evaluated in simulation with *real*
keypoint metrics (MPJPE, PCK) under leakage-aware splits. Rejected
alternatives: (a) render a canned animation and call it pose estimation —
dishonest; (b) refuse the request — the pipeline genuinely is buildable and
the findings are thesis-relevant. Every surface (page text, docs, research
note) states the simulator scope explicitly.

**D17. 14 joints, 5 postures — not 17 COCO keypoints.** The COCO-17 set
includes facial keypoints (eyes, ears) that carry no Wi-Fi observability
whatsoever; regressing them would manufacture fake precision. The 14-joint
subset is the RF-observable skeleton. Postures are a small vocabulary
because posture *class* is what a 6-link array can plausibly support.

**D18. Separate `config/pose.yaml` with a 6-receiver ring.** Pose needs
spatial diversity the 3-link localization array cannot give (measured:
3→6 links moved accuracy 0.39→0.47). Keeping it a separate config avoids
degrading the localization track and makes the link-count dependence an
explicit, testable variable rather than a hidden assumption.

**D19. Prior fusion instead of raw regression output.** A raw joint
regressor produces anatomically impossible skeletons on hard windows.
`fuse_pose` blends toward the predicted posture's canonical skeleton,
weighted by classifier confidence — improves MPJPE (18.8→17.7 cm) and
greatly improves plausibility. Kept as a *post-process* so the raw
regression metrics remain reportable and honest.

**D20. Motion-spectrum features use raw FFT bins, not named bands.** The
first implementation used semantic bands (0–0.5, 0.5–1 Hz …); a unit test
exposed that a 1 s window has 1 Hz bin spacing, so the sub-1 Hz bands could
never be populated. Replaced with the first 8 FFT bins, normalized per
link. Lesson recorded because it is a real resolution constraint on any
window-based Doppler feature here.

---

## 2026-07-29 c — Router integration

**D14. Router polling over ssh + `iw`, not vendor APIs.** Per-station RSSI
has no cross-vendor API; the one near-universal interface is `iw dev <if>
station dump` on OpenWrt-class firmware. We poll it over key-auth ssh
(subprocess, BatchMode) rather than shipping vendor-specific scrapers for
stock firmwares — those are brittle, undocumented, and often absent.
Stock-router users get three documented paths (ROUTER_GUIDE.md): flash
OpenWrt, add a cheap OpenWrt sensing AP, or hostapd on the laptop
(`--local`). Monitor-mode packet sniffing was rejected: needs root +
monitor-capable NIC and breaks normal Wi-Fi use while running.

**D15. Router mode claims "which link", never coordinates.** Per-link RSSI
disturbance localizes to link regions only. The dashboard therefore draws
link lines and activity halos — not a person marker — and the docs say so.
Fusing multiple disturbed links into rough (x,y) is possible future work
but needs real data to justify.

---

## 2026-07-29 b — Real-signal mode and body view

**D11. Real Wi-Fi = RSSI presence sensing, stated plainly.** The user asked
to "enhance the demo with real Wi-Fi signal from the current device".
Laptops cannot expose CSI without special firmware (Intel 5300 / Nexmon
class hardware), and one laptop↔AP link carries no position information.
Rather than fake localization on real signal, `mode="rssi"` implements what
the signal genuinely supports — live RSSI + motion/presence detection
(thesis scope row 1) — and the UI says so. Alternative rejected: silently
driving the localizer with simulated data while displaying "real Wi-Fi".

**D12. Body view is avatar visualization, not pose estimation.** The user's
reference image is DensePose-from-WiFi-style mesh recovery — multi-antenna
CSI, deep networks, large labelled datasets; the thesis plan itself rules
pose out of scope. Implemented instead: a wireframe humanoid *rendering* of
the real localization output (position, speed, heading drive a synthesized
walk cycle), dual-pane in the reference image's visual style, with an
explicit in-page disclaimer. This keeps the demo compelling without
overclaiming — do not remove the disclaimer.

**D13. Canvas 3D projection by hand, no three.js.** ~150 lines of pinhole
projection + polyline primitives keep the zero-build, zero-CDN property of
the dashboard (also required offline). Revisit only if the view needs
textures/lighting.

---

## 2026-07-29 — Initial implementation decisions

**D1. Simulator-first development.** No hardware exists yet (plan month 3),
but the entire pipeline, experiment design, and dashboard can be built and
honestly evaluated against a physics-based simulator that reproduces the
*mechanisms* real CSI sensing uses (multipath fingerprints, body shadowing,
phase corruption, packet loss). Alternative (wait for hardware) would idle
months 1–3. Risk accepted: simulated accuracy ≠ real accuracy; mitigated by
identical session format so hardware data drops in with zero code change.

**D2. sklearn for defaults, torch optional.** RF/KNN/SVM/MLP cover the
plan's traditional baselines and train in seconds on CPU; torch (~1.5 GB
installed) would make the quick-start heavy for marginal demo benefit. Deep
models (`cnn`, `cnn_gru`) are implemented but import torch lazily.

**D3. Vanilla-JS single-file dashboard, not Vue/React.** The plan suggests
Vue/Nuxt or React (§15); for a thesis demo a zero-build single HTML file
served by FastAPI removes node/npm from the setup entirely. Revisit only if
the dashboard needs multi-page operations views.

**D4. NPZ session files, not PostgreSQL/Parquet.** One compressed .npz per
session (~10 MB) with a JSON manifest is versionable, copyable, and
trivially loadable; a database adds operational burden with no query need
yet. Parquet export of *window features* remains a cheap add if wanted.

**D5. Segment ids inside sessions.** Windows/filters must not cross the
position jump between two standing captures; encoding contiguous captures as
`seg` ids in the session file makes the constraint structural (enforced in
`sessions_to_arrays`) instead of procedural.

**D6. Amplitude-centric features; sanitized phase as secondary.** ESP32
phase is unusable raw (random CFO/SFO per packet — reproduced in the
simulator). Amplitude mean/std/motion-energy + phase-curvature mean is the
standard commodity-hardware compromise in the literature.

**D7. Fingerprint (ML) approach, not geometry (AoA/ToF).** ESP32 has one
antenna and no timing resolution — SpotFi-style geometry is impossible on
this hardware tier; fingerprinting with dense coverage is the viable path
(and matches the plan's model list).

**D8. Simulation realism parameters.** Standing sway σ≈1 cm (not 5 cm —
tested: large sway destroys window-mean fingerprints and contradicts human
biomechanics); scattered-ray gain 1.2 with 60% max LoS shadowing (visible
but not dominant person signal); cross-day device re-mount σ=4 mm (≈ 0.2
rad phase shift — noticeable cross-day degradation without erasing the
fingerprint, matching literature behaviour qualitatively). These were tuned
by experiment — see RESEARCH_NOTES §4; treat as simulator calibration knobs,
not physical truth.

**D9. Kalman constant-velocity for tracking** (not particle filter):
adequate for single-person walking speeds, closed-form, no tuning burden
beyond two noise scalars exposed in config. Particle filter listed in plan
§11.2 stays a future comparison.

**D10. 20-component PCA default.** Full features are 624-dim for 3 links;
PCA(20) keeps RF/KNN fast and slightly improves cross-session error on the
demo dataset. Set `n_pca_components: 0` to ablate.
