import time

from wifi_mapping.collect.router_live import (RouterMonitor,
                                              make_demo_transport,
                                              parse_iw_station_dump)

DUMP = """Station aa:bb:cc:dd:ee:01 (on wlan0)
	inactive time:	120 ms
	rx bytes:	123456
	signal:  	-52 [-55, -58] dBm
	signal avg:	-53 dBm
	tx bitrate:	144.4 MBit/s
Station aa:bb:cc:dd:ee:02 (on wlan0)
	inactive time:	40 ms
	signal:  	-64 [-66] dBm
Station aa:bb:cc:dd:ee:03 (on wlan1)
	inactive time:	400000 ms
	signal:  	-70 dBm
	signal avg:	-71 dBm
"""


def test_parse_station_dump_prefers_signal_avg():
    st = parse_iw_station_dump(DUMP)
    assert st["aa:bb:cc:dd:ee:01"]["signal"] == -53.0   # avg preferred
    assert st["aa:bb:cc:dd:ee:02"]["signal"] == -64.0   # falls back to instant
    assert st["aa:bb:cc:dd:ee:01"]["inactive_ms"] == 120
    assert len(st) == 3


def test_parse_station_dump_ignores_garbage():
    assert parse_iw_station_dump("") == {}
    assert parse_iw_station_dump("command failed: No such device (-19)") == {}


def test_monitor_drops_stale_stations():
    mon = RouterMonitor(lambda: DUMP, poll_hz=2.0, stale_s=120.0)
    reading = mon.poll_once(now=1000.0)
    macs = [s["mac"] for s in reading["stations"]]
    assert "aa:bb:cc:dd:ee:03" not in macs   # inactive 400 s > stale_s
    assert len(macs) == 2


def test_monitor_detects_motion_on_disturbed_link():
    transport = make_demo_transport(n_stations=2, seed=1)
    mon = RouterMonitor(transport, poll_hz=5.0, threshold=0.5)
    t = time.time()
    last = None
    for i in range(120):  # 24 simulated seconds: quiet + disturbance phases
        last = mon.poll_once(now=t + i * 0.2)
        time.sleep(0.001)
    assert last is not None and len(last["stations"]) == 2
    # The demo transport disturbs one link at a time — at least one station
    # must have accumulated a clearly non-zero motion level.
    assert max(s["motion"] for s in last["stations"]) > 0.3
