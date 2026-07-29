"""Read CSI packets from ESP32 receivers over USB serial.

Works with Espressif's ESP-CSI example firmware (esp-csi/examples/get-started/
csi_recv), which prints one CSV line per received packet:

    CSI_DATA,<seq>,<mac>,<rssi>,<rate>,...,<len>,"[i0,q0,i1,q1,...]"

The exact column count differs between ESP-IDF versions, so the parser
anchors on the leading "CSI_DATA" tag, takes RSSI from its documented
position, and takes the bracketed int array at the end as interleaved
imaginary/real pairs per subcarrier (ESP32 order: [imag, real] per
subcarrier for the LLTF).

See firmware/esp32-csi/README.md for flashing instructions.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import numpy as np

_ARRAY_RE = re.compile(r"\[([-0-9, ]+)\]")


@dataclass
class CsiPacket:
    link_id: str            # receiver node id (e.g. "RX0")
    timestamp: float        # host receive time (s)
    seq: int                # firmware sequence counter
    rssi: float             # dB
    csi: np.ndarray         # complex (n_subcarriers,)
    raw: str = field(repr=False, default="")


def parse_csi_line(line: str, link_id: str = "RX?",
                   timestamp: float | None = None) -> CsiPacket | None:
    """Parse one serial line; returns None for non-CSI lines (logs, boot)."""
    if "CSI_DATA" not in line:
        return None
    m = _ARRAY_RE.search(line)
    if not m:
        return None
    try:
        vals = np.array([int(v) for v in m.group(1).replace(" ", "").split(",") if v])
    except ValueError:
        return None
    if len(vals) < 4 or len(vals) % 2:
        return None
    # Interleaved [imag, real] per subcarrier.
    csi = vals[1::2].astype(float) + 1j * vals[0::2].astype(float)

    fields = line.split(",")
    try:
        head = fields.index("CSI_DATA")
        seq = int(fields[head + 1])
        rssi = float(fields[head + 3])
    except (ValueError, IndexError):
        seq, rssi = -1, float("nan")
    return CsiPacket(
        link_id=link_id,
        timestamp=time.time() if timestamp is None else timestamp,
        seq=seq,
        rssi=rssi,
        csi=csi,
        raw=line,
    )


class CsiSerialReader:
    """Background reader for one or more ESP32 receivers.

    Each receiver is one USB serial port. Packets from all ports are pushed
    into a single queue as :class:`CsiPacket` with the link id attached.

    Usage::

        reader = CsiSerialReader({"RX0": "/dev/ttyUSB0", "RX1": "/dev/ttyUSB1"})
        reader.start()
        for packet in reader.packets(timeout=1.0):
            ...
        reader.stop()
    """

    def __init__(self, ports: dict[str, str], baudrate: int = 921600):
        self.ports = ports
        self.baudrate = baudrate
        self.queue: Queue[CsiPacket] = Queue(maxsize=100_000)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.stats = {link: {"packets": 0, "errors": 0} for link in ports}

    def _read_port(self, link_id: str, port: str) -> None:
        import serial  # pyserial; imported here so the sim-only demo needs no hardware deps

        with serial.Serial(port, self.baudrate, timeout=1.0) as ser:
            while not self._stop.is_set():
                try:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                except Exception:
                    self.stats[link_id]["errors"] += 1
                    continue
                pkt = parse_csi_line(line, link_id)
                if pkt is None:
                    continue
                self.stats[link_id]["packets"] += 1
                if not self.queue.full():
                    self.queue.put(pkt)

    def start(self) -> None:
        self._stop.clear()
        for link_id, port in self.ports.items():
            t = threading.Thread(target=self._read_port, args=(link_id, port),
                                 daemon=True, name=f"csi-{link_id}")
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def packets(self, timeout: float = 1.0):
        """Yield packets until stopped; silently idles through timeouts."""
        while not self._stop.is_set():
            try:
                yield self.queue.get(timeout=timeout)
            except Empty:
                continue
