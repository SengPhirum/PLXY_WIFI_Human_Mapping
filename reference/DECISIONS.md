# Decision Log

Why the code is the way it is. Add entries at the top; never delete —
supersede.

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
