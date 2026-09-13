"""USB-serial link from the laptop AI to the ESP32 buzzer (Stage 11).

Pipeline slice implemented here::

    TemporalState + AlertStatus  ->  word_for()  ->  BuzzerLink  --USB serial-->  ESP32  ->  buzzer
    (Stages 9 and 10)                ALERT/MILD/    background thread:            hardware/drowsiness_alarm/
                                     DROWSY/CLEAR   connect, send on change,      drowsiness_alarm.ino
                                                    1 s heartbeat, read replies,
                                                    reconnect when unplugged

The ESP32 runs no AI. It receives one word and drives one buzzer. The laptop's
own Stage 10 alerts keep working with or without the board.

Protocol (ASCII lines, '\\n' terminated, 115200 8N1; full table in hardware/README.md)
--------------------------------------------------------------------------------------
    laptop -> ESP32                        ESP32 -> laptop
    HELLO    laptop present, arm watchdog  OK HELLO
    ALERT    buzzer silent                 OK ALERT
    MILD     one chirp every 2 s           OK MILD
    DROWSY   fast beeping                  OK DROWSY
    CLEAR    silence now, until next MILD/DROWSY   OK CLEAR
    PING                                   PONG <uptime_ms>
    STATUS                                 STATUS <state> buzzer=<0|1> age_ms=<n>
    TEST     local demo (no laptop needed) OK TEST ... TEST DONE
    (boot)                                 READY drowsiness_alarm v1 pin=23 passive=0 watchdog_ms=3000
    (no command for 3 s after HELLO)       LINK LOST   -> buzzer forced silent

Mapping from the AI to a word (``word_for``)
--------------------------------------------
    alert audio muted by the driver ('d')  -> CLEAR   (the buzzer is dismissed with the laptop audio)
    otherwise                              -> the Stage 9 state: ALERT / MILD / DROWSY
    program exit                           -> CLEAR, then the port is closed

Design rules
------------
* **Never block the vision loop.** ``set_state()`` stores a word; a daemon
  thread owns the port: open, HELLO, send on change, heartbeat every second,
  read replies, reopen every 3 s while the board is unplugged.
* **Auto-detect only real USB-serial bridges** (CP210x, CH340, FTDI, Espressif
  native USB) by VID/PID. Bluetooth "Standard Serial over Bluetooth" ports are
  never opened - this laptop exposes six of them.
* **Testable without hardware.** ``FakeESP32`` implements the firmware's
  protocol on a fake port so ``--self-test`` checks the Python side end to end.

Run directly::

    python -m src.hardware --list                    # serial ports, ESP32 candidates marked
    python -m src.hardware --test [--port COM9]      # READY/HELLO/PING, MILD 3 s, DROWSY 3 s, CLEAR
    python -m src.hardware --monitor [--port COM9]   # type words, see replies (Ctrl+C to quit)
    python -m src.hardware --send DROWSY             # one word, print the reply
    python -m src.hardware --self-test               # protocol logic against FakeESP32, no hardware
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple
from collections import deque

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover
    serial = None  # type: ignore
    list_ports = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

BAUD = 115200
WORDS = ("ALERT", "MILD", "DROWSY", "CLEAR")
HEARTBEAT_S = 1.0
RECONNECT_S = 3.0
REPLY_TIMEOUT_S = 2.0

# USB-serial bridges found on ESP32 dev boards (vendor id, product id)
ESP32_USB_IDS = {
    (0x10C4, 0xEA60): "Silicon Labs CP210x",
    (0x1A86, 0x7523): "WCH CH340",
    (0x1A86, 0x55D4): "WCH CH9102",
    (0x0403, 0x6001): "FTDI FT232",
    (0x0403, 0x6015): "FTDI FT231X",
    (0x303A, 0x1001): "Espressif USB JTAG/serial",
    (0x303A, 0x0002): "Espressif USB CDC",
}
ESP32_DESCRIPTION_HINTS = ("CP210", "CH340", "CH910", "FTDI", "USB-SERIAL", "USB SERIAL", "SILICON LABS", "ESPRESSIF")


# --- port discovery ------------------------------------------------------------

@dataclass
class PortInfo:
    device: str
    description: str
    vid: Optional[int]
    pid: Optional[int]
    esp32_hint: Optional[str]              # why we think this is an ESP32 bridge, or None

    @property
    def is_bluetooth(self) -> bool:
        return "BLUETOOTH" in self.description.upper()


def list_serial_ports() -> List[PortInfo]:
    if list_ports is None:
        raise ImportError("pyserial is not installed: pip install pyserial")
    out = []
    for p in list_ports.comports():
        hint = ESP32_USB_IDS.get((p.vid, p.pid)) if (p.vid is not None and p.pid is not None) else None
        if hint is None and any(h in (p.description or "").upper() for h in ESP32_DESCRIPTION_HINTS):
            hint = "description matches"
        out.append(PortInfo(p.device, p.description or "", p.vid, p.pid, hint))
    return out


def find_esp32_port() -> Optional[str]:
    """The single most plausible ESP32 port, or None. Never a Bluetooth port."""
    candidates = [p for p in list_serial_ports() if p.esp32_hint and not p.is_bluetooth]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.esp32_hint == "description matches", p.device))
    return candidates[0].device


# --- the mapping from the AI to a word ---------------------------------------------

def word_for(temporal_state, alert_status=None) -> str:
    """Stage 9 + Stage 10 -> protocol word. Kept separate from BuzzerLink so it
    can be unit-tested and reused by Stage 12."""
    if temporal_state is None:
        return "CLEAR"
    if alert_status is not None and alert_status.dismissed:
        return "CLEAR"
    state = temporal_state.state
    return state if state in WORDS else "CLEAR"


# --- the link ---------------------------------------------------------------------

@dataclass
class LinkStats:
    connects: int = 0
    disconnects: int = 0
    sent: int = 0
    acks: int = 0
    errors: int = 0
    last_sent: str = ""
    last_reply: str = ""
    last_reply_wall: float = 0.0
    ready_line: str = ""
    link_lost_reports: int = 0
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=50))


class BuzzerLink:
    """Owns the serial port on a daemon thread. Main-thread API:
    ``set_state(word)``, ``connected``, ``status_text()``, ``close()``."""

    def __init__(self, port: str = "auto", baud: int = BAUD, heartbeat_s: float = HEARTBEAT_S,
                 reconnect_s: float = RECONNECT_S, open_port: Optional[Callable[[str, int], object]] = None,
                 echo: bool = True, autostart: bool = True) -> None:
        if serial is None and open_port is None:
            raise ImportError("pyserial is not installed: pip install pyserial ({})".format(_IMPORT_ERROR))
        self.port_request = port
        self.baud = baud
        self.heartbeat_s = heartbeat_s
        self.reconnect_s = reconnect_s
        self.echo = echo
        self._open_port = open_port or self._open_real_port
        self._desired = "ALERT"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.port: Optional[str] = None
        self.connected = False
        self.stats = LinkStats()
        self._ser = None
        self._thread = threading.Thread(target=self._run, name="esp32-link", daemon=True)
        if autostart:
            self._thread.start()

    # -- main-thread API -------------------------------------------------------------
    def set_state(self, word: str) -> None:
        word = word.upper()
        if word not in WORDS:
            raise ValueError("unknown word {!r}; use one of {}".format(word, WORDS))
        with self._lock:
            changed = word != self._desired
            self._desired = word
        if changed:
            self._wake.set()                 # send immediately, do not wait for the heartbeat

    @property
    def desired(self) -> str:
        with self._lock:
            return self._desired

    def status_text(self) -> str:
        if self.connected:
            age = time.time() - self.stats.last_reply_wall if self.stats.last_reply_wall else float("inf")
            return "ESP32 {} | sent {} | last reply {} ({:.0f} s ago)".format(
                self.port, self.stats.last_sent, self.stats.last_reply or "-", age)
        return "ESP32 not connected ({})".format(
            "searching..." if self.port_request == "auto" else self.port_request)

    def close(self, timeout: float = 3.0) -> None:
        """Send CLEAR so the buzzer is silent, then close the port."""
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)

    # -- worker ----------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.stats.recent.append(msg)
        if self.echo:
            print("[esp32] {}".format(msg))

    def _open_real_port(self, port: str, baud: int):
        return serial.Serial(port, baud, timeout=0.05, write_timeout=1.0)

    def _connect(self) -> bool:
        port = self.port_request
        if port == "auto":
            port = find_esp32_port()
            if port is None:
                return False
        try:
            self._ser = self._open_port(port, self.baud)
        except Exception as exc:
            self._log("cannot open {}: {}".format(port, exc))
            self.stats.errors += 1
            return False
        self.port = port
        self.connected = True
        self.stats.connects += 1
        self._log("connected to {} @ {}".format(port, self.baud))
        # The board resets when the port opens (DTR). Give it up to 2 s to say READY,
        # then arm its watchdog with HELLO. A board that was already running just answers.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = self._read_line()
            if line is None:
                time.sleep(0.02)
                continue
            if line.startswith("READY"):
                break
        self._write("HELLO")
        self._await_reply(REPLY_TIMEOUT_S)
        return True

    def _disconnect(self, why: str) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        if self.connected:
            self.stats.disconnects += 1
            self._log("disconnected: {}".format(why))
        self.connected = False

    def _write(self, word: str) -> bool:
        try:
            self._ser.write((word + "\n").encode("ascii"))
            self.stats.sent += 1
            self.stats.last_sent = word
            return True
        except Exception as exc:
            self.stats.errors += 1
            self._disconnect("write failed: {}".format(exc))
            return False

    def _read_line(self) -> Optional[str]:
        try:
            raw = self._ser.readline()
        except Exception as exc:
            self.stats.errors += 1
            self._disconnect("read failed: {}".format(exc))
            return None
        if not raw:
            return None
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            return None
        self.stats.last_reply = line
        self.stats.last_reply_wall = time.time()
        if line.startswith("READY"):
            self.stats.ready_line = line
            self._log("board: {}".format(line))
        elif line.startswith("OK"):
            self.stats.acks += 1
        elif line.startswith("LINK LOST"):
            self.stats.link_lost_reports += 1
            self._log("board reports LINK LOST (it was not fed for 3 s)")
        elif line.startswith("ERR"):
            self.stats.errors += 1
            self._log("board: {}".format(line))
        return line

    def _await_reply(self, timeout: float) -> Optional[str]:
        deadline = time.time() + timeout
        while time.time() < deadline and self._ser is not None:
            line = self._read_line()
            if line is not None:
                return line
        return None

    def _run(self) -> None:
        last_sent_word, last_sent_at = None, 0.0
        while not self._stop.is_set():
            if not self.connected:
                if not self._connect():
                    self._wake.wait(self.reconnect_s)
                    self._wake.clear()
                    continue
                last_sent_word = None
            word = self.desired
            now = time.time()
            if word != last_sent_word or now - last_sent_at >= self.heartbeat_s:
                changed = word != last_sent_word
                if self._write(word):
                    last_sent_word, last_sent_at = word, now
                    if changed:                       # heartbeats are not logged, only changes
                        self._log("-> {}".format(word))
            # drain replies for up to one heartbeat, waking early on a state change
            deadline = time.time() + self.heartbeat_s
            while self.connected and time.time() < deadline and not self._stop.is_set():
                self._read_line()
                if self._wake.is_set():
                    self._wake.clear()
                    break
                time.sleep(0.01)
        # shutdown: leave the buzzer silent
        if self.connected:
            self._write("CLEAR")
            self._await_reply(0.5)
            self._disconnect("closed")


# --- a fake board for tests --------------------------------------------------------

class FakeESP32:
    """Behaves like drowsiness_alarm.ino on a fake serial port: same replies,
    same watchdog (scaled by ``time_scale`` so tests run fast)."""

    def __init__(self, watchdog_s: float = 3.0, reset_on_open: bool = True) -> None:
        self.rx = bytearray()
        self.tx: Deque[bytes] = deque()
        self.state = "ALERT"
        self.armed = False
        self.watchdog_s = watchdog_s
        self.last_cmd = time.time()
        self.received: List[str] = []
        self.buzzer_pattern = "silent"
        self.closed = False
        self.boot_ms = time.time()
        if reset_on_open:
            self._say("READY drowsiness_alarm v1 pin=23 passive=0 watchdog_ms={}".format(int(watchdog_s * 1000)))

    # serial-like interface used by BuzzerLink
    def write(self, data: bytes) -> int:
        self.rx += data
        while b"\n" in self.rx:
            line, _, self.rx = self.rx.partition(b"\n")
            self._handle(line.decode("ascii", errors="replace").strip())
        return len(data)

    def readline(self) -> bytes:
        self._tick()
        return self.tx.popleft() if self.tx else b""

    def close(self) -> None:
        self.closed = True

    # firmware behaviour
    def _say(self, text: str) -> None:
        self.tx.append((text + "\n").encode("ascii"))

    def _handle(self, line: str) -> None:
        if not line:
            return
        self.received.append(line)
        self.last_cmd = time.time()
        cmd = line.upper()
        if self.state == "LOST":
            self.state = "ALERT"
        if cmd == "HELLO":
            self.armed = True
            self._say("OK HELLO")
        elif cmd in WORDS:
            self.state = cmd
            self.buzzer_pattern = {"ALERT": "silent", "CLEAR": "silent", "MILD": "chirp every 2 s",
                                   "DROWSY": "fast beeping"}[cmd]
            self._say("OK " + cmd)
        elif cmd == "PING":
            self._say("PONG {}".format(int((time.time() - self.boot_ms) * 1000)))
        elif cmd == "STATUS":
            self._say("STATUS {} buzzer={} age_ms={}".format(
                self.state, 0 if self.buzzer_pattern == "silent" else 1, int((time.time() - self.last_cmd) * 1000)))
        else:
            self._say("ERR unknown " + line)

    def _tick(self) -> None:
        if self.armed and self.state != "LOST" and time.time() - self.last_cmd > self.watchdog_s:
            self.state = "LOST"
            self.buzzer_pattern = "silent"
            self._say("LINK LOST")


# --- command-line tools ------------------------------------------------------------

def cmd_list() -> int:
    ports = list_serial_ports()
    if not ports:
        print("no serial ports found")
        return 1
    for p in ports:
        ids = "{:04X}:{:04X}".format(p.vid, p.pid) if p.vid is not None else "----:----"
        tag = "  <- ESP32? ({})".format(p.esp32_hint) if (p.esp32_hint and not p.is_bluetooth) else ""
        print("{:<6} {:<45} {}{}".format(p.device, p.description[:45], ids, tag))
    chosen = find_esp32_port()
    print("auto-detect would use: {}".format(chosen or "nothing (plug the ESP32 in, then re-run)"))
    return 0


def _open_or_die(port: str):
    if port == "auto":
        found = find_esp32_port()
        if found is None:
            print("ERROR: no ESP32 USB-serial port found. Run --list; pass --port COMx if the bridge is unusual.",
                  file=sys.stderr)
            return None, None
        port = found
    try:
        ser = serial.Serial(port, BAUD, timeout=0.1, write_timeout=1.0)
    except Exception as exc:
        print("ERROR: cannot open {}: {}".format(port, exc), file=sys.stderr)
        print("       Close the Arduino Serial Monitor - only one program can hold the port.", file=sys.stderr)
        return None, None
    return ser, port


def _ask(ser, word: str, wait_s: float = 1.0) -> List[str]:
    ser.write((word + "\n").encode("ascii"))
    replies, deadline = [], time.time() + wait_s
    while time.time() < deadline:
        raw = ser.readline()
        if raw:
            replies.append(raw.decode("ascii", errors="replace").strip())
            if replies[-1].startswith(("OK", "PONG", "STATUS", "ERR")):
                break
    print("   -> {:<8} <- {}".format(word, " | ".join(replies) if replies else "(no reply)"))
    return replies


def cmd_test(port: str) -> int:
    """Step 2 of the Stage 11 test plan: Python -> ESP32 without the AI."""
    ser, port = _open_or_die(port)
    if ser is None:
        return 1
    print("[esp32] opened {} @ {}. Waiting for READY (the board resets when the port opens)...".format(port, BAUD))
    ready, deadline = None, time.time() + 3.0
    while time.time() < deadline:
        raw = ser.readline()
        if raw and raw.startswith(b"READY"):
            ready = raw.decode("ascii", errors="replace").strip()
            break
    print("   {}".format(ready or "no READY seen (board may have been running already; continuing)"))
    ok = True
    ok &= any(r == "OK HELLO" for r in _ask(ser, "HELLO"))
    ok &= any(r.startswith("PONG") for r in _ask(ser, "PING"))
    print("[esp32] MILD for 3 s - expect one short chirp every 2 s (and the blue LED with it)")
    ok &= any(r == "OK MILD" for r in _ask(ser, "MILD"))
    for _ in range(3):
        time.sleep(1.0); _ask(ser, "MILD", 0.3)               # heartbeat, as the live loop does
    print("[esp32] DROWSY for 3 s - expect fast beeping")
    ok &= any(r == "OK DROWSY" for r in _ask(ser, "DROWSY"))
    for _ in range(3):
        time.sleep(1.0); _ask(ser, "DROWSY", 0.3)
    print("[esp32] CLEAR - expect silence")
    ok &= any(r == "OK CLEAR" for r in _ask(ser, "CLEAR"))
    _ask(ser, "STATUS")
    print("[esp32] watchdog: sending nothing for 4 s - expect the board to report LINK LOST and blink slowly")
    lost, deadline = False, time.time() + 4.5
    while time.time() < deadline:
        raw = ser.readline()
        if raw and b"LINK LOST" in raw:
            lost = True
            print("   <- LINK LOST")
            break
    ok &= lost
    _ask(ser, "ALERT")
    ser.close()
    print("[esp32] {}".format("TEST PASSED" if ok else "TEST FAILED - see the replies above"))
    return 0 if ok else 1


def cmd_monitor(port: str) -> int:
    ser, port = _open_or_die(port)
    if ser is None:
        return 1
    print("[esp32] monitor on {}. Type ALERT / MILD / DROWSY / CLEAR / PING / STATUS / TEST, Ctrl+C to quit.".format(port))
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            raw = ser.readline()
            if raw:
                print("   <- " + raw.decode("ascii", errors="replace").strip())

    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            word = input().strip()
            if word:
                ser.write((word + "\n").encode("ascii"))
    except (KeyboardInterrupt, EOFError):
        pass
    stop.set()
    ser.write(b"CLEAR\n")
    time.sleep(0.2)
    ser.close()
    return 0


def cmd_send(port: str, word: str) -> int:
    ser, port = _open_or_die(port)
    if ser is None:
        return 1
    time.sleep(1.5)                                          # board reset on open
    ser.reset_input_buffer()
    replies = _ask(ser, word)
    ser.close()
    return 0 if replies else 1


# --- self-test (no hardware) -----------------------------------------------------------

def self_test() -> int:
    from src.temporal import TemporalState, ALERT, MILD, DROWSY

    def ts(state):
        return TemporalState(t=0.0, state=state, state_since=0.0, reasons=[], sufficient=True, window_fill_s=60,
                             frames=1, valid_frames=1, invalid_rate=0.0, perclos=0.0, closed_now=False,
                             closure_now_s=0.0, longest_closure_s=0.0, blink_count=0, blink_rate_per_min=0.0,
                             mean_blink_s=float("nan"), yawn_count=0, yawn_rate_per_min=0.0, nod_count=0,
                             pitch_baseline_deg=float("nan"), fusion="fused")

    class Muted:
        dismissed = True

    class Live:
        dismissed = False

    # 1. mapping
    assert word_for(None) == "CLEAR"
    assert word_for(ts(ALERT)) == "ALERT" and word_for(ts(MILD)) == "MILD" and word_for(ts(DROWSY)) == "DROWSY"
    assert word_for(ts(DROWSY), Muted()) == "CLEAR" and word_for(ts(DROWSY), Live()) == "DROWSY"
    print("[self-test] word_for: states map 1:1, driver mute -> CLEAR: ok")

    # 2. the fake board speaks the protocol
    board = FakeESP32(watchdog_s=0.4)
    assert board.readline().startswith(b"READY")
    board.write(b"hello\n"); assert board.readline() == b"OK HELLO\n"
    board.write(b"MILD\r\n"); assert board.readline() == b"OK MILD\n" and board.buzzer_pattern == "chirp every 2 s"
    board.write(b"BOGUS\n"); assert board.readline().startswith(b"ERR unknown")
    time.sleep(0.5)
    assert board.readline() == b"LINK LOST\n" and board.state == "LOST" and board.buzzer_pattern == "silent"
    print("[self-test] FakeESP32: READY, case-insensitive commands, ERR, watchdog -> LINK LOST: ok")

    # 3. the link: connects, HELLOs, sends on change immediately, heartbeats, CLEARs on close
    boards: List[FakeESP32] = []

    def factory(port, baud):
        b = FakeESP32(watchdog_s=0.6)
        boards.append(b)
        return b

    link = BuzzerLink(port="FAKE", heartbeat_s=0.2, reconnect_s=0.1, open_port=factory, echo=False)
    deadline = time.time() + 2.0
    while not (link.connected and link.stats.acks >= 2) and time.time() < deadline:
        time.sleep(0.01)
    b = boards[0]
    assert link.connected and b.received[:2] == ["HELLO", "ALERT"], b.received
    tick = time.perf_counter()
    link.set_state("DROWSY")
    cost_ms = (time.perf_counter() - tick) * 1000
    assert cost_ms < 5.0, cost_ms
    deadline = time.time() + 1.0
    while "DROWSY" not in b.received and time.time() < deadline:
        time.sleep(0.005)
    latency_ms = (time.time() - (deadline - 1.0)) * 1000
    assert "DROWSY" in b.received and b.buzzer_pattern == "fast beeping", b.received
    time.sleep(0.7)                                           # > 3 heartbeats, well inside the 0.6 s watchdog
    assert b.received.count("DROWSY") >= 3 and b.state == "DROWSY", b.received
    assert link.stats.link_lost_reports == 0, "heartbeat must keep the board's watchdog fed"
    link.set_state("CLEAR"); time.sleep(0.1)
    assert b.state == "CLEAR" and b.buzzer_pattern == "silent"
    link.set_state("MILD"); time.sleep(0.1)
    link.close()
    assert b.received[-1] == "CLEAR" and b.closed and not link.connected, b.received[-3:]
    print("[self-test] BuzzerLink: HELLO+state on connect, set_state() {:.2f} ms, DROWSY on the wire in {:.0f} ms, "
          "heartbeat fed the watchdog, CLEAR on close: ok".format(cost_ms, latency_ms))

    # 4. reconnect: the board vanishes (read raises), the link reopens on a new one
    class Flaky(FakeESP32):
        def __init__(self):
            super().__init__(watchdog_s=5.0)
            self.die_after = 3

        def readline(self):
            if len(self.received) >= self.die_after:
                raise OSError("device unplugged")
            return super().readline()

    made: List[FakeESP32] = []

    def factory2(port, baud):
        b = Flaky() if not made else FakeESP32(watchdog_s=5.0)
        made.append(b)
        return b

    link = BuzzerLink(port="FAKE", heartbeat_s=0.1, reconnect_s=0.1, open_port=factory2, echo=False)
    deadline = time.time() + 3.0
    while not (len(made) >= 2 and made[1].received[:1] == ["HELLO"]) and time.time() < deadline:
        time.sleep(0.01)
    assert len(made) >= 2 and link.stats.disconnects >= 1 and made[1].received[:1] == ["HELLO"], (
        len(made), link.stats.disconnects, made[-1].received[:3])
    link.close()
    print("[self-test] reconnect after unplug: {} disconnect(s), new board greeted with HELLO: ok".format(
        link.stats.disconnects))

    # 5. no board at all: the link stays harmless and reports it
    link = BuzzerLink(port="FAKE", heartbeat_s=0.1, reconnect_s=0.05,
                      open_port=lambda p, b: (_ for _ in ()).throw(OSError("no such port")), echo=False)
    time.sleep(0.3)
    link.set_state("DROWSY")
    assert not link.connected and "not connected" in link.status_text() and link.stats.errors >= 2
    link.close()
    print("[self-test] no board: set_state() harmless, status says not connected, keeps retrying: ok")

    # 6. auto-detect never picks a Bluetooth port (this machine has six of them)
    for p in list_serial_ports():
        if p.is_bluetooth:
            assert find_esp32_port() != p.device
    print("[self-test] auto-detect: {} port(s) listed, Bluetooth excluded, would use {}: ok".format(
        len(list_serial_ports()), find_esp32_port()))
    print("[self-test] ALL PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 11: laptop -> ESP32 buzzer link.")
    parser.add_argument("--port", default="auto", help="COMx, or auto (default) = first USB-serial bridge found")
    parser.add_argument("--list", action="store_true", help="List serial ports and the auto-detect choice")
    parser.add_argument("--test", action="store_true", help="Protocol test against the real board")
    parser.add_argument("--monitor", action="store_true", help="Interactive terminal to the board")
    parser.add_argument("--send", metavar="WORD", help="Send one word and print the reply")
    parser.add_argument("--self-test", action="store_true", help="Test the Python side against a fake board")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if serial is None:
        print("ERROR: pyserial is not installed: pip install pyserial", file=sys.stderr)
        return 1
    if args.list:
        return cmd_list()
    if args.test:
        return cmd_test(args.port)
    if args.monitor:
        return cmd_monitor(args.port)
    if args.send:
        return cmd_send(args.port, args.send.upper())
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
