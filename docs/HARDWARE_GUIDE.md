# Hardware Guide — Real ESP32 CSI Collection

How to move from the simulated demo to real measurements. Firmware flashing
details live in [../firmware/esp32-csi/README.md](../firmware/esp32-csi/README.md);
this guide covers the room setup and the collection workflow.

## Shopping list (≈ $30–60 total)

- 4 × ESP32 DevKitC boards (3 receivers + 1 transmitter — or use a router as TX)
- 3+ tripods / camera clamps, USB cables, powered USB hub
- Masking tape + tape measure (floor grid and ground-truth marking)

## 1. Room preparation (plan §8)

1. Measure the room; update `room.width` / `room.depth` in `config/default.yaml`.
2. Choose the coordinate origin (a corner), mark the 4×4 zone grid on the
   floor with tape at zone boundaries.
3. Mount TX and RX nodes at the positions/heights in the config — or move the
   nodes and **write the measured positions back into the config** (±2 cm).
   Corner placement with the TX on the opposite wall maximizes link-crossing
   coverage.
4. Log everything (photos, sketch, wall materials, furniture) in a session
   notes file — future-you must be able to rebuild the setup exactly.

## 2. Flash and verify

Follow `firmware/esp32-csi/README.md`, then:

```bash
python scripts/collect_esp32.py \
    --ports RX0=/dev/ttyUSB0 RX1=/dev/ttyUSB1 RX2=/dev/ttyUSB2 --check
```

All links should show >90 pkt/s at the default 100 Hz TX rate. Fix DEAD/LOW
links before collecting anything.

## 3. Collect sessions (plan §9)

One session = one participant, one day. Start every session with an
empty-room capture, then the guided standing + walking protocol:

```bash
python scripts/collect_esp32.py \
    --ports RX0=/dev/ttyUSB0 RX1=/dev/ttyUSB1 RX2=/dev/ttyUSB2 \
    --session 0 --day 0 --person 0 \
    --spots-per-zone 4 --seconds-per-spot 10
```

The script announces each spot (`zone 3 spot 2/4: stand at x=1.87 y=2.10`);
the operator tapes the point, the participant stands on it, Enter starts the
10 s capture. ~11 min of standing data per session at the defaults.
Repeat across ≥2 days and ≥2 participants so the independence splits in
`scripts/train.py --split-by day|person` are possible.

Sessions land in `data/datasets/hardware/session_XX.npz` — the **same format
as simulated data**, so the whole downstream pipeline is unchanged:

```bash
python scripts/train.py --dataset hardware --model rf --split-by session
python scripts/evaluate.py --dataset hardware --models rssi_knn knn rf mlp
```

## 4. Live inference with hardware

The live server accepts frames over HTTP in hardware mode: run the collector
in live mode piping aligned frames to `/api/frame` (see
`src/wifi_mapping/server/app.py`); a ready-made bridge is
`scripts/collect_esp32.py`'s `capture_spot` loop with the recorder swapped
for HTTP posts — left as a small integration task, tracked in
`reference/TODO.md`.

## 5. Data quality checklist (plan §10, §16)

- [ ] Empty-room baseline recorded at session start
- [ ] Packet loss events < 5% (printed by the collector after each spot)
- [ ] Node positions re-measured after any bump; config updated
- [ ] Wi-Fi channel fixed (no auto-selection) and noted
- [ ] Participant consent recorded; only anonymous person codes in metadata
- [ ] Session notes: date, operator, layout changes, anomalies

## Known hardware realities to expect

- ESP32 CSI amplitude is stable, phase is not — the pipeline already relies
  on amplitude + sanitized phase only.
- Different individual boards have different amplitude scales; per-link
  normalization (already in the pipeline) absorbs most of it, but keep the
  same physical board in the same role across a dataset.
- 2.4 GHz interference (neighbours' APs, Bluetooth) shows up as extra noise
  and packet loss; prefer nights/weekends or a quiet channel.
