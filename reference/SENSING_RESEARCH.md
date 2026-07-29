# Human Detection from RSSI — Research Basis and Measured Performance

Research notes and evaluation behind `collect/presence.py`, the detector
used by the live Wi-Fi modes (`scripts/wifi_live.py`, `scripts/router_live.py`).

The question this answers: **given only RSSI — one signal-strength number
per link — how well can a human be detected, and what exactly limits it?**

---

## 1. What the literature establishes

| Finding | Source | What we took from it |
|---|---|---|
| Device-free detection works on RSSI variance; entropy of the RSSI distribution is a stronger feature than raw variance | Information-entropy RSSI methods; variance-based radio tomography | Both variance and entropy are computed as features |
| **Thresholds can be derived autonomously** from the observed RSSI distribution instead of hand-tuned per room, updating as the environment changes | [MDPI Electronics 2025, RSSI distribution-based method with autonomous threshold](https://www.mdpi.com/2079-9292/15/2/491) | Calibration phase learns each feature's quiet distribution; thresholds = median + k·MAD |
| **A stationary person's breathing is observable in RSSI** on standard Wi-Fi/Zigbee transceivers — inhaling/exhaling modulates the channel measurably | [Catch a Breath (arXiv:1307.0084)](https://arxiv.org/pdf/1307.0084); [Wi-Fi Beyond Communications (arXiv:2407.05155)](https://arxiv.org/pdf/2407.05155) | Respiration-band spectral detection is the core of the "still person" capability |
| Breathing is recoverable in roughly 0.16–1.2 Hz (10–72 breaths/min); the breathing signal is **not** resolvable during motion interference | Same | Band set to 0.16–0.6 Hz; breathing is only consulted when motion is absent |
| Presence can be declared from spectral components in the breathing band | Presence/pulse detection patents (US 11448726, US 11828872) | Spectral peak + stability is the stationary-presence trigger |
| Every AP a device hears is a separate propagation path; beacons carry per-AP RSSI | AP beacon scanning / RSSI fingerprinting literature | `ApScanner` turns neighbouring APs into extra sensing links |

**The gap this closes:** the original detector here was rolling variance
only. Variance detects *movement*. A person sitting still produces almost
no variance and was reported as an empty room — the single biggest failure
mode of naive Wi-Fi presence sensing, and what the breathing feature fixes.

---

## 2. What the detector does

Four features per link, all computed on a rolling buffer:

1. **Motion** — standard deviation of the RSSI **band-passed to 0.8–3 Hz**.
   The band matters in both directions:
   - the *low* edge excludes breathing, so a still person is never
     mislabelled as moving;
   - the *high* edge excludes the harmonics produced when a slowly-varying
     signal crosses RSSI's 1 dB quantization levels — without it, an
     isolated 1 dB step rings the filter and reads as movement.
2. **Breathing** — spectral SNR in 0.16–0.6 Hz (detrended, Hann-windowed,
   in-band peak vs out-of-band median), **plus a peak-stability test**: a
   real respiration rate holds steady (< 0.05 Hz wander) over seconds,
   whereas an incidental noise peak wanders across the band. Stability is
   what makes this usable rather than a false-alarm generator.
3. **Shadowing** — sustained mean-RSSI shift from the calibrated empty
   level (a body attenuates the path).
4. **Entropy** — Shannon entropy of the RSSI value histogram over 1 dB bins.

**Autonomous thresholds.** A calibration window (default 20 s, room empty)
measures each feature's own quiet distribution; thresholds are set at
`median + k·MAD`, with physically-motivated floors (e.g. the motion floor
sits above a single-LSB step because RSSI is quantized to 1 dB). No
per-room hand tuning.

**Decision.** Hysteretic three-state machine — `empty` → `stationary` →
`motion`. Motion additionally requires **persistence** (~0.5 s of sustained
band energy) so impulse artefacts cannot trigger it. Shadowing and entropy
never trigger on their own — both drift with AP power control — but
together they count as corroborating presence.

**Multi-link fusion.** Each link (connected AP + scanned neighbours) gets
its own detector, because quiet levels and noise floors differ per link.
Room state is the strongest link's state: a person only has to disturb one
path to be present.

---

## 3. Two engineering fixes that matter more than the algorithm

- **Active probing.** Drivers refresh RSSI only when frames arrive. On an
  idle link the trace flatlines and *no* detector can work — this is the
  single most common reason a Wi-Fi sensing demo "does nothing".
  `ActiveProbe` pings the default gateway at the sample rate to guarantee a
  fresh measurement per interval.
- **Neighbouring APs as extra links.** `ApScanner` harvests per-BSSID RSSI
  from beacon scans (nmcli / `iw scan dump` / netsh / airport). Each AP sits
  in a different place, so each link crosses a different part of the
  building — genuine multi-link sensing from one laptop, no hardware. Scans
  are slow (~1–3 s) and briefly disturb the connection, so scanned links run
  at ~0.5 Hz and serve motion detection; breathing stays on the fast
  connected link.

---

## 4. Measured performance

Synthetic RSSI streams modelling each room state, **including the 1 dB
quantization real drivers report**. Evaluation harness reproduced in
`tests/test_presence.py`; tuning runs used 6 seeds per cell.

### State classification (30 s calibration, 60 s test)

| Scenario | Result |
|---|---|
| Empty room | 100% `empty` |
| Stationary person, 15 bpm | 100% `stationary`, rate estimated **15 bpm** |
| Stationary person, 22 bpm | 100% `stationary`, rate estimated **21 bpm** |
| Walking person | 100% `motion` |
| Walk then leave | 100% returns to `empty`; **release latency ≈ 5–6 s** |

### Occupancy detection (human detected at all) vs signal conditions

Rows = peak RSSI modulation from breathing (dB); columns = link jitter (dB):

| depth ↓ / noise → | 0.2 | 0.35 | 0.5 | 0.8 |
|---|---|---|---|---|
| 1.5 dB | 100% | 100% | 100% | 100% |
| 0.9 dB | 100% | 100% | 100% | 100% |
| 0.6 dB | 100% | 100% | 100% | 100% |
| 0.4 dB | 100% | 100% | 81% | 41% |
| 0.3 dB | 100% | 96% | 40% | 26% |
| 0.2 dB | 34% | 25% | 13% | 17% |

False-alarm rate on empty links: **0%** at the default sensitivity.

### Sensitivity knob (noise 0.35 dB)

| sensitivity | d=1.2 | d=0.6 | d=0.4 | d=0.3 | false alarms |
|---|---|---|---|---|---|
| 1.0 | 100% | 100% | 100% | 96% | 5% |
| **1.25 (default)** | 100% | 100% | 100% | 76% | **0%** |
| 1.5 | 100% | 100% | 99% | 50% | 0% |
| 2.0 | 100% | 100% | 88% | 15% | 0% |

**1.25 is the chosen default**: full detection down to a 0.4 dB breathing
modulation with no false alarms. Lower it if the room is quiet and
detections are being missed; raise it in an interference-heavy environment.

### Operating envelope, stated plainly

- Reliable down to **≈0.4 dB** of breathing-induced RSSI modulation.
- Below ≈0.3 dB it degrades, and on very noisy links (≥0.8 dB jitter) a
  still person needs ≈0.6 dB to be seen.
- **A small amount of link noise helps.** RSSI is quantized to 1 dB, so a
  very quiet link (0.2 dB jitter) can lock onto a single level and lose sub-dB
  modulation entirely — classic dither behaviour. Real links usually supply
  enough natural jitter.

---

## 5. Honest limits

- These numbers come from **synthetic RSSI**, calibrated to real
  quantization and plausible noise. They validate the algorithm and set
  expectations; they are not field measurements. Real-world confounds not
  modelled: interference bursts, AP transmit-power control, multipath
  fading from non-human movement (fans, doors, pets), and body orientation.
- **Presence only.** One RSSI link carries no position information. This
  cannot be extended to localization or pose — that needs CSI (see
  `POSE_FROM_WIFI.md`).
- Breathing detection assumes **one** stationary person; two people
  breathing at different rates put two peaks in the band, and the
  stability test may reject both.
- Calibration assumes the room is empty. If it is not, the quiet baseline
  absorbs the person and they become invisible. The dashboard states this
  and offers a `recalibrate` control.

## 6. Next steps

- Validate against real captures on a laptop — record RSSI with the room
  empty / occupied-still / occupied-walking and re-run the same evaluation.
- Per-link adaptive re-calibration (slow drift tracking) so long sessions
  do not need manual recalibration.
- Fuse the router mode's per-device links into the same detector so a whole
  home reports one occupancy state per area.

## Sources

- [A Device-Free Human Detection System Using 2.4 GHz Wireless Networks and an RSSI Distribution-Based Method with Autonomous Threshold (MDPI Electronics, 2025)](https://www.mdpi.com/2079-9292/15/2/491)
- [Catch a Breath: Non-invasive Respiration Rate Monitoring via Wireless Communication (arXiv:1307.0084)](https://arxiv.org/pdf/1307.0084)
- [Wi-Fi Beyond Communications: Respiration Monitoring and Motion Detection Using COTS Devices (arXiv:2407.05155)](https://arxiv.org/pdf/2407.05155)
- [Device-free indoor human presence detection based on information entropy of RSSI variations](https://www.researchgate.net/publication/258022986_Device-free_indoor_human_presence_detection_method_based_on_the_information_entropy_of_RSSI_variations)
- [Robust WiFi Respiration Sensing in the Presence of Interfering Individual (IEEE TMC 2023)](http://staff.ustc.edu.cn/~dongheng/dhfiles/2023TMCxuecheng.pdf)
- [BreatheBand: Fine-grained Respiration Monitoring Using WiFi (ACM TOSN)](https://dl.acm.org/doi/10.1145/3582079)
