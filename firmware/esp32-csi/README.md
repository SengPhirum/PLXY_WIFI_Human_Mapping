# ESP32 CSI Firmware Setup

The receivers run Espressif's official **ESP-CSI** example firmware — there is
no custom firmware to maintain in this repo. This document covers flashing,
configuration, and verifying that the serial output matches what
`wifi_mapping.collect.serial_reader` expects.

## Hardware (per the thesis plan §7.1)

| Qty | Item | Role | Notes |
|----:|------|------|-------|
| 3 | ESP32 DevKitC (ESP32-WROOM-32) | CSI receivers RX0–RX2 | Any ESP32 with ESP-IDF ≥5.0 support works; ESP32-S3 also fine |
| 1 | ESP32 DevKitC *or* a dedicated Wi-Fi router | Transmitter TX0 | A fixed AP sending periodic traffic |
| 1 | Laptop / mini-PC with 3 free USB ports (or a powered hub) | Collection host | Runs `scripts/collect_esp32.py` |
| — | Tripods / wall mounts, measuring tape | Repeatable geometry | Positions go into `config/default.yaml` |

## 1. Install ESP-IDF

```bash
# Linux/macOS — ESP-IDF v5.1+
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32 && . ./export.sh
```

## 2. Get ESP-CSI and build the receiver example

```bash
git clone https://github.com/espressif/esp-csi.git
cd esp-csi/examples/get-started/csi_recv
idf.py set-target esp32
idf.py menuconfig   # optional: set Wi-Fi channel to match config/default.yaml
idf.py build
```

## 3. Flash each receiver

Plug in one board at a time and note which USB port maps to which physical
node (label the boards RX0/RX1/RX2 with tape):

```bash
idf.py -p /dev/ttyUSB0 flash monitor   # Windows: COM3 etc.
```

## 4. Set up the transmitter

Option A (recommended for control): flash `esp-csi/examples/get-started/
csi_send` onto the TX board — it broadcasts packets at a configurable rate.

Option B: use a dedicated router/AP on a fixed channel and have the
receivers listen to its beacons/traffic (set the same channel in menuconfig).
Fix the channel — do **not** leave auto channel selection on.

## 5. Verify serial output

With `idf.py monitor` (or `screen /dev/ttyUSB0 921600`) you should see one
CSV line per packet:

```
CSI_DATA,1234,aa:bb:cc:dd:ee:ff,-42,11,...,128,"[12,-3,14,-1,...]"
```

The trailing bracketed array is interleaved `[imag, real]` per subcarrier.
Quick host-side check that the parser accepts your firmware's exact format:

```bash
python -c "
from wifi_mapping.collect import parse_csi_line
line = open('/dev/ttyUSB0').readline()   # or paste a captured line
print(parse_csi_line(line, 'RX0'))
"
```

If your ESP-IDF version emits a different column layout and the parser
returns wrong RSSI, adjust the column index in
`src/wifi_mapping/collect/serial_reader.py` (`head + 3`) — the CSI array
itself is located by the brackets and is layout-independent.

## 6. Collection-time checklist (plan §8–§10)

- Fix channel, bandwidth (20 MHz), TX packet rate (100 Hz), and TX power.
- Mount all nodes at the height recorded in `config/default.yaml`; measure
  positions to ±2 cm and update the config if anything moves.
- Start `scripts/collect_esp32.py` and confirm all links report packets
  (>90 packets/s each) before recording.
- Record the empty-room baseline first, every session.
- Log anomalies (people entering, doors, interference) in the session notes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No `CSI_DATA` lines | Wrong channel; TX not sending; CSI not enabled in menuconfig |
| Very low packet rate | Crowded channel — pick another; check TX rate config |
| Garbled serial text | Baud rate mismatch — firmware default is 921600 |
| `Permission denied /dev/ttyUSB0` | `sudo usermod -aG dialout $USER` then re-login |
| Ports swap after reboot | Use `/dev/serial/by-id/...` paths in `--ports` |
