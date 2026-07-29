# Router Guide — Sensing Every Connected Device

`scripts/router_live.py` turns the Wi-Fi router into the sensing hub: the
router knows the RSSI of **every associated device** (phones, TVs, laptops,
smart plugs…), and each device↔router link crosses a different part of the
home. Motion between the router and a device disturbs that link, so a
handful of *stationary* devices spread around the home gives a coarse
live activity map — which link region is occupied — with no extra hardware.

```
   TV ────────┐                      link "TV" quiet
   Desk PC ───┤ router               link "Desk PC" quiet
   Printer ───┤        ← person →    link "Printer" DISTURBED  ⇒ activity near printer
   Tablet ────┘                      link "Tablet" quiet
```

Try the UI right now, with no router at all:

```bash
python scripts/router_live.py --demo     # → http://127.0.0.1:8000
```

## What kind of router works

Per-station RSSI requires shell access to the router — consumer stock
firmware does not expose it in any standard way.

| Router | Works? | How |
|---|---|---|
| OpenWrt (any device) | ✅ | ssh + `iw` — supported out of the box |
| GL.iNet travel routers | ✅ | OpenWrt-based; enable ssh in admin UI |
| DD-WRT / FreshTomato | ✅ | enable sshd; `iw` present on most builds |
| Stock TP-Link/Netgear/ISP boxes | ❌ directly | no per-station RSSI API — see options below |
| This PC as hotspot (hostapd) | ✅ | `--local` flag, no ssh needed |

**Options when your main router is a stock ISP box:**

1. **Cheapest real fix:** add one OpenWrt access point (a $20–30 used
   TP-Link/Xiaomi, or a GL.iNet) as the *sensing AP*: give it its own SSID,
   connect your stationary devices to it, and point `router_live.py` at it.
   The main router keeps serving the internet untouched.
2. **Hotspot mode:** run a hostapd hotspot on the collection laptop and
   connect devices to it, then `--local`.
3. Check if the stock firmware has a telnet/ssh debug mode exposing
   `iw`/`wl` — some vendors do (search "<model> ssh access").

## Setup (OpenWrt-class router)

1. **Enable key-based ssh** so polling needs no password:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/router_sensing -N ""
   # OpenWrt (dropbear): paste the .pub line into /etc/dropbear/authorized_keys
   #   or via LuCI: System → Administration → SSH-Keys
   ssh root@192.168.1.1 "iw dev"        # note the AP interface names
   ssh root@192.168.1.1 "iw dev wlan0 station dump"   # must print stations
   ```

2. **Configure** `config/default.yaml` → `router:` — host, user, the
   interface names from step 1, and (recommended) the MAC → name/position
   map for your stationary devices. Positions are metres in the same floor
   plan as everything else; find MACs in the router's DHCP lease table.

3. **Run:**

   ```bash
   python scripts/router_live.py
   ```

   The startup log lists reachable stations. Open the dashboard: device
   markers on the floor plan, link lines that thicken/colour with live
   motion, and a per-device table (RSSI, motion level, state). Unmapped
   devices appear in the table only.

## Interpreting it honestly (for the thesis)

- This is **link-disturbance sensing on RSSI**: presence and coarse
  *which-link* localization, not (x, y) coordinates. It complements — not
  replaces — the CSI pipeline: same room geometry, same dashboard, and it
  answers plan RQ6's "what can RSSI alone do" from a deployment angle.
- Detection quality depends on geometry (links that cross walking paths
  detect best), device chatter (idle devices update RSSI less often —
  phones asleep may go minutes between beacons; wired-powered devices like
  TVs and printers are the most reliable anchors), and the router's RSSI
  averaging.
- Tune `router.threshold` (dB of excess fluctuation that counts as motion)
  per home: lower = more sensitive, more false positives from interference.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no stations returned` at startup | ssh key auth not set (`ssh router` must work without password); wrong interface names (`ssh router 'iw dev'`); router lacks `iw` (install `iw` package on OpenWrt) |
| Stations listed but never any motion | Devices too idle — use powered, chatty devices; lower `router.threshold`; walk *between* router and device, not beside |
| Motion flickers constantly | Raise `router.threshold`; microwave ovens and neighbours' traffic add noise on 2.4 GHz |
| Devices missing from the map | Add their MAC under `router.devices` with a `pos`; check they're associated to the polled interface (5 GHz devices may be on `wlan1`) |
| Works on 2.4 GHz only | Poll both radios: `interfaces: ["wlan0", "wlan1"]` (names vary: `phy0-ap0`, `ath0`, …) |
