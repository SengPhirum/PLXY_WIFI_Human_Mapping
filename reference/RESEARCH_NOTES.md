# Research Notes — Wi-Fi CSI Human Localization

Working notes behind the implementation decisions. Companion to
`THESIS_PLAN.md` (requirements) and `DECISIONS.md` (what we chose and why).

## 1. CSI sensing theory in five paragraphs

**What CSI is.** For each received Wi-Fi packet, the receiver estimates the
complex channel response H(f) at every OFDM subcarrier (52 usable on ESP32's
20 MHz LLTF). H(f) is the sum of all propagation paths: line of sight, wall/
furniture reflections, and — crucially — reflections off the human body.
Amplitude |H| and phase ∠H per subcarrier form the "fingerprint" per packet.

**Why a person is visible.** A body (mostly water) both *blocks* paths
(shadowing of the LoS, several dB) and *adds* a scattered path whose length
is TX→body→RX. The scattered path interferes with static paths differently
at each subcarrier (path-length difference → frequency-dependent phase →
ripple across the subcarrier axis). Both effects depend on where the body
is, which is what makes localization possible.

**Why phase is hard on commodity hardware.** Each packet suffers a random
common phase offset (CFO/PLL) and a random linear phase slope across
subcarriers (SFO + packet-detection delay). Raw phase is therefore useless
across packets. Standard fix (implemented in `preprocess/phase.py`): unwrap
across subcarriers, subtract the best-fit line → what survives is multipath
*curvature*. Amplitude is unaffected and is the workhorse feature.

**Spatial decorrelation is the fundamental difficulty.** Multipath
fingerprints decorrelate over ~λ/2 ≈ 6 cm at 2.4 GHz. Fingerprint methods
therefore interpolate poorly between training points; dense spatial coverage
(many spots per zone + walking data) is not optional. Shadowing-based cues
vary much more smoothly (~0.5 m) and carry the generalizable signal —
this is why our features mix window means (fingerprint) with variance/motion
energy (shadowing dynamics).

**What limits generalization.** Moving a device by centimetres, opening a
door, or a different body re-shapes the static multipath → cross-day and
cross-room degradation is the norm in the literature and the main open
problem (→ thesis contribution option A: cheap recalibration / domain
adaptation).

## 2. Literature matrix (seed — expand to 25–40 papers)

| Study | Setup | Method | Result | Relevance / limits |
|---|---|---|---|---|
| WiTrack (NSDI'14) | Custom FMCW radio, not Wi-Fi | ToF geometry | 10–13 cm median 3D | Geometry-based upper bound; not commodity hardware |
| SpotFi (SIGCOMM'15) | Intel 5300, 3 antennas | AoA+ToF super-resolution | ~40 cm median | Needs multi-antenna phase — ESP32 can't; motivates fingerprint route |
| Widar3.0 (TPAMI) | Intel 5300 | BVP domain-independent gesture feature | cross-domain gesture recognition | Blueprint for domain-independent features (RQ4) |
| EI (MobiCom'18) | CSI, adversarial nets | Environment-independent activity | — | Domain-adversarial training idea for contribution A |
| Person-in-WiFi 3D (CVPR'24) | Many antennas + labels | End-to-end pose | 3D pose, multi-person | Confirms plan's "not primary scope" for pose |
| BFMSense (NSDI'24) | Beamforming feedback (no CSI access) | BFM as sensing signal | sensing w/o CSI firmware | Fallback if ESP32 CSI proves too unstable |
| ESP-CSI demos (Espressif) | ESP32 | amplitude-based detection | presence/motion demos | Exactly our hardware tier; validates feasibility |
| IEEE 802.11bf-2025 | standard | sensing measurement service | — | Future-proofing section for Chapter 2 |

Fill remaining rows using the comparison fields from THESIS_PLAN §5.

## 3. Design of the simulator (and its honest limits)

Implemented in `simulate/csi_simulator.py`:

- Static channel per link: LoS ray + 4 first-order image-source wall
  reflections (gain 0.35, 1/d decay).
- Person: scattered ray TX→person→RX (gain 1.2) + Gaussian LoS shadowing
  (up to 60% amplitude, σ = 0.5 m from the LoS segment).
- Impairments: per-packet uniform CFO phase, uniform SFO slope ±0.1 rad/sc,
  additive complex Gaussian noise (σ=0.05), 2% packet loss (NaN frames),
  ~1 cm standing sway, per-day device re-mount jitter (σ=4 mm).

Validated behaviours: raw phase is useless until sanitized (as on real
hardware); accuracy collapses if windows leak across sessions (mirrors
literature); accuracy degrades cross-day (re-mount jitter) and cross-person
(different sway/speed).

**Limits — do not oversell in the thesis:** no antenna patterns, no
frequency-selective materials, no second-order reflections, no other moving
objects/people, simplified body scattering (point scatterer). Simulated
results validate the *pipeline and experiment design*, never the empirical
claims. The moment hardware sessions exist, re-run everything on
`--dataset hardware`.

## 4. Empirical findings so far (simulated, honest splits)

From the tuning experiments during initial development (see PROJECT_STATUS
for the current full table):

- **Dense spatial coverage is decisive.** 1 spot/zone/session →
  cross-session accuracy ≈ chance. 6 spots/zone + 60 s walking → 0.85–0.93 m
  mean error, ~55% zone accuracy (16 zones, chance 6%).
- **Standing sway magnitude matters enormously** in simulation: cm-scale
  sway (realistic) preserves window-mean fingerprints; 5+ cm (unrealistic)
  destroys them. On hardware, expect the analogous effect from posture
  changes → collect both still and fidgety captures.
- **RF is the best zone classifier; CSI-KNN is the weakest** (raw
  fingerprint distances decorrelate across sessions; tree ensembles cope).
- **Caveat: RSSI-KNN is unrealistically strong in simulation** (≈0.9 m,
  close to CSI-RF). The simulator's per-link power is a clean geometric
  function of position with no interference, AGC quantization, or
  temperature drift — the things that ruin RSSI in reality. Expect the
  RSSI-vs-CSI gap (RQ6) to open up decisively on hardware data; do not
  cite the simulated RSSI numbers as evidence either way.
- Session-independent ≥ day-independent ≥ person-independent accuracy —
  degradation ordering matches the plan's risk register.

## 5. Feature/pipeline choices worth defending in the thesis

- Window features = per-(link, subcarrier) amplitude mean, std,
  |Δamp| mean (motion energy), sanitized-phase mean → 4·links·52 dims →
  optional PCA(20). Simple, fast (<10 ms), interpretable; leaves obvious
  headroom for the deep models (Doppler/STFT features are scaffolded in the
  plan §11.1 but not yet implemented — see TODO).
- Hampel (11, 3σ) before low-pass: order matters, spikes would smear.
- Normalization is per-(link, subcarrier) z-score fitted on train sessions
  only — absorbs per-board amplitude scale differences (hardware reality).
- Kalman (constant velocity) roughly halves visual jitter at walking speed;
  measurement noise 0.45 m ≈ observed regressor error, process noise 0.35
  tuned for ~0.7 m/s walking. Re-tune both on hardware data.

## 6. Open research questions (ranked for thesis value)

1. **Cheap cross-environment calibration** (contribution A): how few
   labelled windows in a new room recover 90% of in-room accuracy? Fine-tune
   the last layer / re-fit normalizer only / prototype-nearest-neighbour.
2. **Link count & placement ablation** (contribution B): simulator sweep is
   nearly free (edit config, regenerate) — do it before buying more boards.
3. Temporal deep models (CNN-GRU over window sequences) vs window-independent
   models on the *walking* subset specifically.
4. Robustness: packet-loss injection curves, congested-channel noise.

## 7. Privacy & ethics notes (plan §9.2, §16)

- CSI cannot image faces, but gait/motion patterns are weakly identifying;
  treat participant codes + CSI as personal data (consent, retention limit).
- The live system stores only positions and aggregated heatmaps — by design
  no raw-CSI retention outside labelled research sessions.
- For the defence: contrast with camera systems (identity capture by
  default) — this is the "privacy-preserving" claim, state it precisely.
