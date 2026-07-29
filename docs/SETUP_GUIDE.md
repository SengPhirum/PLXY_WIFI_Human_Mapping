# Demo Setup Guide

Step-by-step instructions to run the full Wi-Fi Human Mapping demo from a
fresh clone — **no hardware required**. For real ESP32 collection, do this
guide first, then continue with [HARDWARE_GUIDE.md](HARDWARE_GUIDE.md).

## 0. Requirements

- Python **3.10+** (3.11 recommended)
- ~1 GB disk for the virtualenv, ~200 MB for the generated demo dataset
- Any OS (Linux/macOS/Windows); a modern browser for the dashboard

## 1. Install

```bash
git clone <this-repo>
cd PLXY_WIFI_Human_Mapping
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (only for the `cnn` / `cnn_gru` models):

```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```

## 2. Sanity check

```bash
pytest            # 17 tests, a few seconds, no hardware touched
```

## 3. One-command demo

```bash
python scripts/run_demo.py
```

This runs three stages, skipping any that are already cached under `data/`:

1. **Dataset generation** (~2 min): simulates 8 collection sessions
   (4 days × 4 participants) in the 5×6 m room described in
   `config/default.yaml` — person standing at 6 spots in each of the 16 grid
   zones plus 60 s of walking per session, through a geometry-based
   multipath CSI simulator with realistic phase corruption and packet loss.
2. **Training** (~1 min): Random Forest zone classifier + (x, y) regressor
   on cleaned CSI features, evaluated with a **session-independent** split
   (test sessions never seen in training — see `reference/RESEARCH_NOTES.md`
   on why random window splits would be cheating).
3. **Live dashboard**: opens a FastAPI server at **http://127.0.0.1:8000**.
   A simulated person walks the room; their CSI is pushed through the trained
   model in real time.

### What you should see

- A floor plan with the 4×4 zone grid, TX (▲) and three RX (■) nodes.
- A blue dot: the smoothed predicted position, updating ~2× per second,
  with a fading trail.
- A gray ✕: the simulated ground-truth position — the gap between ✕ and the
  blue dot **is** the live localization error, also shown numerically.
- An accumulating occupancy heatmap (toggleable), the predicted zone
  highlighted on the grid, and stat tiles for zone, position, error,
  inference latency, and update rate.
- Typical performance out of the box: mean error ≈ 0.6–0.9 m, ~70% of
  updates within 1 m, latency well under 50 ms on a laptop CPU.

### Useful variants

```bash
python scripts/run_demo.py --model mlp        # neural baseline instead of RF
python scripts/run_demo.py --fresh            # regenerate data + retrain
python scripts/run_demo.py --port 8080        # different port
python scripts/run_demo.py --sim-seed 21      # different walking pattern
```

### Body view (wireframe avatar)

While any mode is running, open **http://127.0.0.1:8000/body** (or click
*body view →* in the dashboard header): a dual-pane rendering — perspective
room view on a light background, mesh-only view on black — of a viridis
wireframe human standing at the model's live predicted position, walking
animation driven by the estimated speed and heading.

Honest framing (repeated on the page itself): this is an **avatar
visualization of the localization output**, not pose estimation. The
measured quantities are position and movement; body shape and limb motion
are synthesized for display. Wi-Fi pose recovery à la DensePose-from-WiFi
needs multi-antenna CSI and large labelled datasets, and is explicitly out
of thesis scope (plan §1).

## 3b. Real Wi-Fi signal from the current device (no ESP32 needed)

On a **laptop connected to Wi-Fi** (not inside a container/VM), you can run
the demo against the real signal of the currently associated access point:

```bash
python scripts/wifi_live.py        # → dashboard + /body in real-signal mode
```

This samples the connected link's RSSI at 10 Hz (Linux
`/proc/net/wireless`/`iw`, macOS `airport`, Windows `netsh`) and runs live
motion/presence detection on the fluctuation. Walk between the laptop and
the router: the RSSI trace visibly swings, the presence pill flips to
*motion detected*, and the /body avatar wakes up and reacts.

What this mode honestly is: **presence/motion sensing on real signal**
(thesis plan scope row 1). A single laptop↔AP link carries no position
information, and laptops cannot expose CSI without special firmware — so
there is no localization here. Localization = simulator demo today, ESP32
multi-link CSI when the hardware arrives ([HARDWARE_GUIDE.md](HARDWARE_GUIDE.md)).

**If `wifi_live.py` "doesn't work" for you, check in this order:**

| Symptom | Cause / fix |
|---|---|
| `No wireless interface found` | You're in a VM/WSL/container or on Ethernet — these have no Wi-Fi device. Run on the host OS of a Wi-Fi-connected laptop. WSL2 specifically cannot see the Wi-Fi adapter; use native Windows Python there. |
| Trace is a flat line, never any motion | Some drivers cache RSSI aggressively. Move *between* laptop and router (not beside), within ~3 m of the straight line. Generate link traffic (`ping 192.168.1.1` in another terminal) — RSSI updates with traffic. |
| Motion never triggers presence | Fluctuation below threshold — the detector needs >0.6 dB excess std. Try a closer router, 2.4 GHz instead of 5 GHz, or the router mode below (multiple links = much better sensitivity). |
| Trace updates very slowly | Power-save NIC. Disable Wi-Fi power management (`sudo iw dev wlan0 set power_save off`). |

## 3c. Whole-home sensing through the router (all connected devices)

The single laptop link above is the weakest possible setup. The **router**
sees the RSSI of *every* connected device — each device↔router link is a
sensing tripwire crossing a different part of the home:

```bash
python scripts/router_live.py --demo    # try the UI right now, no router needed
python scripts/router_live.py           # real: OpenWrt-class router via ssh
```

The dashboard switches to an activity map: device markers on the floor
plan, router→device link lines that thicken and light up with live motion
on that link, plus a per-device table (RSSI, motion, state). Requirements,
router compatibility (OpenWrt / GL.iNet / DD-WRT; what to do with stock ISP
boxes), MAC→position mapping, and tuning: **[ROUTER_GUIDE.md](ROUTER_GUIDE.md)**.

## 4. Reproduce the baseline comparison (thesis §14)

```bash
python scripts/evaluate.py --dataset demo \
    --models rssi_knn knn rf mlp --splits session day person
```

Prints a model × split matrix (zone accuracy, macro-F1, mean/median/p90
localization error, fraction within 1 m) and writes JSON + Markdown to
`data/results/`. `rssi_knn` is the RSSI-only baseline — expect it to be
clearly worse than the CSI models; that gap is thesis research question RQ6.

Individual stages can also be run directly:

```bash
python scripts/generate_dataset.py --name big --sessions 16 --days 8 --persons 8
python scripts/train.py --dataset big --model rf --split-by person
```

## 5. Experiment with the configuration

Everything is in `config/default.yaml`: room size, grid resolution, node
positions, packet rate, window/step, filters, PCA, Kalman noise. Pass an
alternative file with `--config`:

```bash
cp config/default.yaml config/myroom.yaml   # edit dimensions/links
python scripts/run_demo.py --config config/myroom.yaml --dataset myroom
```

Ideas that map directly to plan §14 experiments: drop to 1 receiver
(delete two `rx:` entries), coarsen the grid to 2×2, raise `noise_std`,
raise `packet_loss`, shrink `spots_per_zone`.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no session_*.npz files` | Dataset stage skipped/failed — run `scripts/generate_dataset.py` |
| Dashboard loads, no movement | Model bundle missing → check `data/models/`; browser console for WS errors |
| Port already in use | `--port 8001` |
| Slow generation | Reduce `--sessions` or `--packets-per-zone`; it's pure NumPy, no GPU needed |
| `ImportError: torch` | You picked `--model cnn` without installing `requirements-torch.txt` |
| Blank page over SSH | Forward the port: `ssh -L 8000:127.0.0.1:8000 host` |
