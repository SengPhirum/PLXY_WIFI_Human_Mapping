# Master's Thesis Planning Guide — Wi-Fi Human Mapping

> Faithful Markdown conversion of the uploaded planning document
> *WiFi_Human_Mapping_Thesis_Research_Plan.docx* (July 2026, prepared for
> Seng Phirum). This is the project's source of truth for scope and
> requirements; the repository implements it.

## Executive Summary

Wi-Fi human mapping uses changes in radio propagation caused by the human
body to infer presence, position, movement, activity, or pose. For a
master's degree, the most achievable and academically defensible scope is a
camera-free system that uses commodity Wi-Fi Channel State Information (CSI)
to detect one person, estimate the person's indoor position, track movement
over time, and generate a floor-plan heatmap.

**Recommended thesis direction:** Privacy-Preserving Indoor Human
Localization and Movement Mapping Using Commodity Wi-Fi CSI and Lightweight
Machine Learning.

The recommended minimum contribution is single-person zone classification
and continuous movement tracking using multiple low-cost Wi-Fi sensing
links. Multi-person occupancy and activity recognition should be treated as
optional extensions. Full-body pose reconstruction is substantially more
difficult and should not be the primary scope unless advanced hardware, a
large labelled dataset, and considerable GPU resources are available.

## 1. Define the Research Scope

| Scope | Output | Difficulty | Recommendation |
|---|---|---|---|
| Presence detection | Detect occupied versus empty space | Low | Baseline only |
| Room-zone classification | Predict a labelled room region | Medium | Minimum thesis target |
| Single-person x,y localization | Estimate physical coordinates | Medium-High | Recommended core |
| Movement-path tracking | Estimate a continuous trajectory | High | Recommended extension |
| Occupancy counting | Estimate number of people | High | Optional |
| Multi-person localization | Locate several people separately | Very high | Stretch goal |
| Skeleton/pose estimation | Estimate body joints or pose | Very high | Not primary scope |

A practical room can initially be divided into a 4 × 4 grid, producing 16
labelled positions. The model predicts the occupied grid cell; a later
regression model can estimate continuous x,y coordinates. Sequential
predictions can then be smoothed and displayed as a live movement path and
historical occupancy heatmap.

## 2. Recommended Thesis Titles

- Privacy-Preserving Human Localization and Movement Mapping Using Wi-Fi Channel State Information
- Low-Cost Indoor Human Tracking Using Commodity Wi-Fi CSI and Edge Machine Learning
- Cross-Environment Human Localization Using Wi-Fi CSI and Domain-Adaptive Deep Learning
- Camera-Free Indoor Occupancy Mapping Using Multi-Link Wi-Fi Channel State Information
- Real-Time Human Movement Heatmap Generation Using ESP32-Based Wi-Fi Sensing

## 3. Research Aim and Objectives

### 3.1 Main objective

Design, implement, and evaluate a low-cost, privacy-preserving system that
uses Wi-Fi CSI to locate and track a person in an indoor environment without
requiring a wearable device or camera during normal operation.

### 3.2 Specific objectives

1. Build a repeatable multi-link Wi-Fi CSI data-capture platform.
2. Design a data collection and ground-truth labelling protocol.
3. Remove noise, packet anomalies, and hardware-related signal variation.
4. Extract spatial and temporal features associated with human position and movement.
5. Compare RSSI, traditional machine learning, and deep-learning baselines.
6. Generate real-time location markers, trajectories, and floor-plan heatmaps.
7. Evaluate accuracy across different people, sessions, days, layouts, and rooms.
8. Measure inference latency, computing requirements, and deployment feasibility.
9. Assess privacy, consent, data governance, and practical deployment limitations.

## 4. Research Questions and Hypotheses

| ID | Research question |
|---|---|
| RQ1 | How accurately can commodity Wi-Fi CSI localize a person indoors? |
| RQ2 | How do transmitter and receiver placement affect localization performance? |
| RQ3 | Which CSI preprocessing and machine-learning approach produces the best results? |
| RQ4 | How well does the model generalize to unseen people, days, and room layouts? |
| RQ5 | Can the system provide real-time mapping with acceptable edge-computing latency? |
| RQ6 | How does CSI-based localization compare with RSSI-based localization? |

Example hypothesis: Multi-link CSI combined with temporal deep learning will
produce significantly lower localization error than RSSI and single-link CSI
approaches.

## 5. Literature Review Plan

Establish the sensing principles, available platforms, leading approaches,
unresolved limitations, and the gap addressed by the thesis:

- Wi-Fi RSSI-based positioning and fingerprinting.
- CSI amplitude, phase, subcarriers, multipath, and Doppler effects.
- Device-free passive human detection and localization.
- Activity, gesture, respiration, gait, and pose sensing.
- Fingerprint-based versus geometry/model-based localization.
- Traditional machine learning, deep learning, and temporal modelling.
- Cross-person, cross-day, cross-device, and cross-room generalization.
- Multi-person interference and non-line-of-sight conditions.
- Privacy, ethics, security, and informed consent.
- IEEE 802.11bf-2025 enhancements for WLAN sensing.

For every reviewed study, record: publication (authors, title, venue, year);
sensing setup (hardware, frequency, bandwidth, antennas, links); study
design (rooms, participants, positions, activities, sessions); method
(preprocessing, features, model, validation); results (accuracy,
localization error, latency); limitations (generality, cost,
reproducibility, privacy); availability (dataset, source code, hardware
instructions).

## 6. Proposed Research Contribution

| Option | Contribution | Assessment |
|---|---|---|
| A. Cross-environment adaptation | Reduce calibration data needed when moving the system to a new room | Strong recommendation |
| B. Low-cost multi-link sensing | Quantify how ESP32 node count and placement affect accuracy and cost | Highly feasible |
| C. Edge-optimized inference | Compress and deploy the localization model on a mini-PC or embedded device | Practical contribution |
| D. Robustness under change | Evaluate doors, furniture, congestion, NLOS, and time variation | Strong evaluation |
| E. Privacy-aware analytics | Produce anonymous occupancy and movement maps without identity storage | Applied contribution |

Recommended novelty: combine low-cost multi-link CSI with cross-day and
cross-room evaluation, then introduce a lightweight calibration or
domain-adaptation method that reduces the amount of labelled data required
in a new environment.

## 7. Hardware and Platform Selection

### 7.1 Affordable prototype

- Three or four CSI-capable ESP32 receiver nodes.
- One dedicated Wi-Fi access point or ESP32 transmitter.
- One laptop, workstation, or mini-PC for collection and inference.
- Fixed mounts or tripods to maintain repeatable geometry.
- Measuring tape or laser distance meter.
- Optional overhead camera, AprilTags, or UWB tags for ground truth only.

### 7.2 Research-grade alternatives

| Platform | Strength | Main consideration |
|---|---|---|
| ESP32 / ESP-CSI | Low cost and reproducible | Limited RF control and device-specific CSI behaviour |
| Intel 5300 CSI Tool | Extensively used in earlier literature | Old hardware and difficult sourcing |
| Nexmon CSI | Supports selected Broadcom chips | Device/firmware compatibility must be managed |
| PicoScenes | Advanced Wi-Fi sensing and experimental control | More setup complexity and hardware dependence |
| 802.11bf-capable platform | Standards-aligned future direction | Hardware/tool availability may be limited |

Before purchasing equipment, run a small compatibility proof-of-concept:
confirm that the selected devices expose stable CSI, packet timestamps,
subcarrier values, and a controllable packet rate.

## 8. Experimental Environment Design

Use at least two indoor environments (laboratory/classroom and
office/meeting room). Document each room so experiments can be reproduced:
measure dimensions and obstacles; create a digital floor plan and coordinate
system; mark ground-truth coordinates and labelled zones; fix and record
device position, height, angle, orientation; record materials and furniture;
fix channel, bandwidth, packet rate, transmit power; keep an experiment log.

| Design item | Suggested starting point |
|---|---|
| Room size | ≈ 5 × 6 m for the initial prototype |
| Grid | 4 × 4 grid → 16 labelled positions |
| Links | One transmitter and three receivers initially |
| Device height | Test ≈ 1.0–1.5 m and document the selected value |
| Frequency | Compare 2.4 GHz and 5 GHz if supported |
| Participants | Target at least 10–20 consenting participants |
| Sessions | Repeat across multiple days and environmental states |

## 9. Data Collection Protocol

### 9.1 Scenarios

Empty-room baseline; standing in each grid position; sitting in selected
positions; different orientations at the same position; walking along
straight/diagonal/turning paths; different participants, clothing, carried
objects; door open vs closed and furniture unchanged vs rearranged; LOS vs
NLOS; different days, times, and Wi-Fi traffic conditions.

### 9.2 Data fields

| Category | Required fields |
|---|---|
| Signal | CSI amplitude, sanitized phase where reliable, RSSI, subcarrier index |
| Packet | Timestamp, sequence number, packet rate, loss indicator |
| Link | Transmitter, receiver, antenna, channel, bandwidth |
| Ground truth | x,y coordinate, zone, path, posture/activity, orientation |
| Context | Anonymous participant code, room, session, date, layout state |
| Quality | Clock offset, capture errors, unusual interference, operator notes |

Data governance: anonymous participant identifiers, informed consent,
defined retention periods, ground-truth video stored separately with
restricted access. The final operational system should not require video.

## 10. Ground Truth and Synchronization

Model accuracy cannot exceed label quality. Candidate approaches: floor
markers, overhead camera, AprilTags, UWB reference tag, controlled moving
platform. Synchronize CSI receivers, ground-truth source, and video clocks;
record a sync event at session start; measure and correct fixed clock
offsets; check packet sequence continuity; validate a sample of labels
manually before large-scale collection.

## 11. Signal Processing and Modelling Pipeline

End-to-end: Wi-Fi packets → CSI extraction → packet validation →
amplitude/phase cleaning → filtering → sliding-window segmentation →
feature/model input → position prediction → trajectory smoothing →
floor-plan heatmap.

### 11.1 Preprocessing methods to compare

Missing-packet and invalid-subcarrier removal; Hampel/median/low-pass
filtering; wavelet denoising; phase sanitization and linear trend removal;
per-link and per-subcarrier normalization; static-background subtraction;
PCA; STFT/Doppler-spectrum extraction; fixed-length and overlapping sliding
windows.

### 11.2 Baseline and candidate models

| Model | Purpose |
|---|---|
| RSSI KNN fingerprinting | Minimum non-CSI baseline |
| CSI KNN / SVM / Random Forest | Traditional machine-learning baselines |
| Multilayer Perceptron | Simple neural baseline |
| 1D CNN | Learn spatial/subcarrier patterns |
| CNN-LSTM or CNN-GRU | Learn spatial and temporal patterns |
| Temporal Convolutional Network | Efficient temporal modelling |
| Lightweight Transformer | Long-range temporal relationships |
| Kalman/particle filter | Smooth sequential position estimates |

## 12. Dataset Splitting and Leakage Prevention

Do not randomly distribute neighbouring windows from the same recording into
both training and testing sets — that leaks session-specific signal
characteristics and produces unrealistically high performance.

| Validation design | Purpose |
|---|---|
| Session-independent | Test on recordings not used for training |
| Day-independent | Test on a different collection day |
| Person-independent | Leave one or more participants out |
| Room-independent | Train in one room and test in another |
| Layout-independent | Test after moving furniture or changing doors |
| Device-independent | Test replacement or repositioned devices, if feasible |

All normalization parameters, feature transformations, and hyperparameter
decisions must be fitted using training and validation data only; the final
test set remains untouched until the method is finalized.

## 13. Evaluation Framework

### 13.1 Classification metrics

Accuracy; precision and recall; macro and weighted F1; confusion matrix;
false-positive and false-negative rates.

### 13.2 Coordinate-localization metrics

eᵢ = √[(x̂ᵢ − xᵢ)² + (ŷᵢ − yᵢ)²]; mean and median localization error; RMSE;
90th-percentile error; percentage within 0.5 m / 1 m / 2 m.

### 13.3 Tracking and system metrics

| Area | Measures |
|---|---|
| Tracking | Trajectory RMSE, path-distance error, lost-track rate, continuity |
| Real-time performance | End-to-end latency and update frequency |
| Efficiency | CPU, memory, model size, power |
| Reliability | Packet-loss tolerance and recovery time |
| Generalization | Degradation across people, days, rooms, layouts |

Report confidence intervals; use appropriate statistical tests with effect
sizes, not only p-values.

## 14. Required Experiments

RSSI vs CSI; raw CSI vs alternative preprocessing; traditional ML vs deep
learning; one vs multiple receivers; alternative placements; coarse vs fine
grid; known vs unseen person; same vs different day; same vs different room;
static vs moving; LOS vs NLOS; stable vs rearranged layout; clean vs
congested traffic; real-time latency and resources; ablation of links,
preprocessing, features, temporal modelling, smoothing.

## 15. Prototype Software Components

| Component | Functions |
|---|---|
| CSI collection | Receiver firmware/client, packet sequence checks, link configuration |
| Dataset management | Session metadata, labels, raw files, versioning, validation |
| Preprocessing | Filtering, normalization, segmentation, feature extraction |
| Model pipeline | Training, validation, testing, experiment tracking |
| Inference service | Real-time prediction API and trajectory smoothing |
| Mapping dashboard | Floor plan, live marker, path, heatmap, history |
| Operations | Logging, model version, sensor status, performance monitoring |

Suggested technologies: Python, NumPy, SciPy, Pandas, PyTorch or TensorFlow,
FastAPI, PostgreSQL or Parquet, Vue/Nuxt or React, Docker, MLflow or
TensorBoard.

## 16. Risk Register and Mitigation

| Risk | Mitigation |
|---|---|
| CSI instability | Fix channel, device geometry, packet rate, calibration |
| Model memorizes sessions | Session/day/person-independent testing |
| Insufficient dataset | Pilot the protocol; calculate coverage before full collection |
| Clock misalignment | Synchronized clocks, sequence numbers, sync events |
| Poor cross-room accuracy | Transfer learning, calibration, domain adaptation |
| Multi-person failure | Keep single-person mapping the guaranteed core |
| Inaccurate labels | Camera, AprilTags, or UWB during controlled collection |
| Scope too large | Activity recognition and multi-person tracking optional |
| Ethics/privacy | Consent, anonymization, limited video retention |
| Weak reproducibility | Version data, code, config, seeds, trained models |

## 17. Twelve-Month Work Plan

| Month | Main activities |
|---|---|
| 1 | Define problem, scope, objectives, research questions, ethics needs |
| 2–3 | Literature review and thesis proposal |
| 3 | Select, purchase, validate sensing hardware |
| 4 | Build CSI collection, synchronization, labelling tools |
| 5 | Pilot study; refine experimental protocol |
| 6–7 | Collect and quality-check the main dataset |
| 7–8 | Implement preprocessing and baseline models |
| 8–9 | Develop and optimize the proposed method |
| 9–10 | Generalization, robustness, ablation experiments |
| 10 | Real-time mapping dashboard |
| 11 | Analyse results; write thesis chapters |
| 12 | Finalize thesis, paper, presentation, defence |

## 18. Deliverables and Acceptance Criteria

Working multi-link CSI sensing prototype; documented quality-checked
dataset; reproducible preprocessing/model code; RSSI and CSI baselines;
proposed localization/tracking method; real-time mapping dashboard;
cross-person/day/environment evaluation; privacy/ethics/deployment
analysis; completed thesis and defence; optional paper.

**Minimum successful thesis:** single-person presence detection, zone
localization, movement tracking, and heatmap generation using at least three
low-cost CSI links, evaluated on unseen participants and different days.

## 19. Immediate Action Checklist

- ☐ Confirm thesis target: zone classification or continuous x,y localization
- ☐ Select primary novelty: cross-environment adaptation, low-cost deployment, or robustness
- ☐ Review 25–40 core papers; populate the literature matrix
- ☐ Confirm access to a laboratory/classroom and a second test room
- ☐ Select CSI hardware; validate one TX–RX link
- ☐ Draft experimental protocol and consent documentation
- ☐ Create floor plan, coordinate system, grid, ground-truth method
- ☐ Run a small pilot (two participants, two sessions)
- ☐ Build RSSI and CSI KNN baselines before any deep model
- ☐ Review pilot results with supervisor before scaling collection

## 20. Suggested Thesis Chapter Structure

| Chapter | Title | Main content |
|---|---|---|
| 1 | Introduction | Background, problem, questions, objectives, scope, contributions |
| 2 | Literature Review | Wi-Fi sensing theory, platforms, methods, gaps |
| 3 | Research Methodology | Design, hardware, participants, data, ethics, metrics |
| 4 | System Design | Architecture, collection, preprocessing, models, dashboard |
| 5 | Results | Baselines, proposed method, generalization, robustness, efficiency |
| 6 | Discussion | Interpretation, comparison, limitations, privacy, validity |
| 7 | Conclusion | Contributions, recommendations, future work |

## References and Starting Resources

1. Adib et al., *WiTrack: 3D Tracking via Body Radio Reflections*, USENIX NSDI 2014.
2. Zheng et al., *Widar3.0: Zero-Effort Cross-Domain Gesture Recognition With Wi-Fi*, IEEE TPAMI.
3. Yan et al., *Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi*, CVPR 2024.
4. Yi et al., *BFMSense: WiFi Sensing Using Beamforming Feedback Matrix*, USENIX NSDI 2024.
5. Espressif Systems, *ESP-CSI Programming Guide* — github.com/espressif/esp-csi
6. *Linux 802.11n CSI Tool* for Intel 5300 — dhalperi.github.io/linux-80211n-csitool
7. *Nexmon CSI: CSI Extraction on Broadcom Wi-Fi Chips* — github.com/seemoo-lab/nexmon_csi
8. Jiang et al., *PicoScenes Wi-Fi Sensing Platform* — ps.zpj.io
9. IEEE Standards Association, *IEEE 802.11bf-2025: WLAN Sensing*.
