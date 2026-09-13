"""Timestamped session logging to SQLite (Stage 12).

One database file (default ``logs/sessions.db``) holds every run of the
detector, whether it was started from the OpenCV tool (``src.features``) or
from the Streamlit dashboard (``src.app``). Three tables:

    sessions   one row per run: start/end wall time, camera, the thresholds in
               force (JSON), frame count, duration, and a summary (JSON) written
               at the end.
    events     one row per thing that happened, with both clocks:
                 kind='transition'  ALERT -> MILD etc. with the Stage 9 reason
                 kind='alert'       every Stage 10 event (RAISED, BEEP, VOICE, DISMISSED, RESET ...)
                 kind='driver'      mute / reset pressed
                 kind='link'        ESP32 connected / disconnected
               plus the detection metrics at that instant (state, PERCLOS,
               closure, EAR, MAR, yaw, pitch).
    metrics    a snapshot of every live metric at a fixed interval (default
               1 s): EAR, MAR, PERCLOS, blink rate / duration, closure, yawns,
               nods, pose, FPS, inference time, invalid-frame rate, CNN
               probability, alert level, buzzer word, link state.

Why SQLite rather than CSV: the dashboard reads while the detector writes
(WAL mode, separate connections), past sessions are queryable without parsing
file names, and Stage 15's evaluation can join events to metrics by time.
The per-frame ``--record`` CSV of Stage 3 still exists for full-rate traces;
this store is the coarser, always-on record.

Two clocks appear in every row: ``wall_ts`` (ISO-8601 local wall clock with
milliseconds - what a human and a report need) and ``t_s`` (seconds since the
session's first frame on the pipeline's monotonic clock - what lines up with
the ``--record`` CSV and the alert CSV).

Run directly::

    python -m src.session_log --list                      # sessions in logs/sessions.db
    python -m src.session_log --show 3                    # summary + events of session 3
    python -m src.session_log --export 3 --out logs/s3    # events.csv + metrics.csv
    python -m src.session_log --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_DB = Path("logs/sessions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    source      TEXT,
    config      TEXT,
    frames      INTEGER,
    duration_s  REAL,
    summary     TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    wall_ts       TEXT NOT NULL,
    t_s           REAL,
    kind          TEXT NOT NULL,
    event         TEXT NOT NULL,
    from_state    TEXT,
    to_state      TEXT,
    level         TEXT,
    detail        TEXT,
    state         TEXT,
    perclos       REAL,
    closure_now_s REAL,
    ear           REAL,
    mar           REAL,
    yaw_deg       REAL,
    pitch_deg     REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         INTEGER NOT NULL REFERENCES sessions(id),
    wall_ts            TEXT NOT NULL,
    t_s                REAL,
    state              TEXT,
    sufficient         INTEGER,
    valid              INTEGER,
    face_found         INTEGER,
    ear                REAL,
    ear_left           REAL,
    ear_right          REAL,
    mar                REAL,
    yaw_deg            REAL,
    pitch_deg          REAL,
    roll_deg           REAL,
    cnn_closed_prob    REAL,
    perclos            REAL,
    blink_rate_per_min REAL,
    mean_blink_s       REAL,
    closure_now_s      REAL,
    longest_closure_s  REAL,
    yawn_count         INTEGER,
    yawn_rate_per_min  REAL,
    nod_count          INTEGER,
    invalid_rate       REAL,
    fps                REAL,
    inference_ms       REAL,
    cnn_ms             REAL,
    alert_level        TEXT,
    alert_dismissed    INTEGER,
    buzzer_word        TEXT,
    esp32_connected    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_session  ON events(session_id, id);
CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id, id);
"""

EVENT_COLUMNS = ["session_id", "wall_ts", "t_s", "kind", "event", "from_state", "to_state", "level", "detail",
                 "state", "perclos", "closure_now_s", "ear", "mar", "yaw_deg", "pitch_deg"]
METRIC_COLUMNS = ["session_id", "wall_ts", "t_s", "state", "sufficient", "valid", "face_found", "ear", "ear_left",
                  "ear_right", "mar", "yaw_deg", "pitch_deg", "roll_deg", "cnn_closed_prob", "perclos",
                  "blink_rate_per_min", "mean_blink_s", "closure_now_s", "longest_closure_s", "yawn_count",
                  "yawn_rate_per_min", "nod_count", "invalid_rate", "fps", "inference_ms", "cnn_ms", "alert_level",
                  "alert_dismissed", "buzzer_word", "esp32_connected"]


def iso_now(wall: Optional[float] = None) -> str:
    return datetime.fromtimestamp(wall if wall is not None else time.time()).isoformat(timespec="milliseconds")


def _clean(value: Any) -> Any:
    """SQLite cannot store NaN meaningfully; store NULL instead."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return int(value)
    return value


# --- writer ---------------------------------------------------------------------

class SessionLogger:
    """Append-only writer. Safe to call from any thread (one internal lock);
    the connection is opened lazily on the first call, in WAL mode so readers
    never block the detector."""

    def __init__(self, path: Path = DEFAULT_DB, metrics_interval_s: float = 1.0, echo: bool = False) -> None:
        self.path = Path(path)
        self.metrics_interval_s = metrics_interval_s
        self.echo = echo
        self.session_id: Optional[int] = None
        self.started_wall: Optional[float] = None
        self.t0: Optional[float] = None
        self.events_written = 0
        self.metrics_written = 0
        self._last_metrics_t = -math.inf
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        return self._conn

    def start(self, source: str = "", config: Optional[Dict[str, Any]] = None, wall: Optional[float] = None) -> int:
        wall = wall if wall is not None else time.time()
        with self._lock:
            conn = self._connect()
            cur = conn.execute("INSERT INTO sessions (started_at, source, config) VALUES (?, ?, ?)",
                               (iso_now(wall), source, json.dumps(config or {}, default=str)))
            conn.commit()
            self.session_id = int(cur.lastrowid)
            self.started_wall = wall
        if self.echo:
            print("[session] #{} started -> {}".format(self.session_id, self.path))
        return self.session_id

    def end(self, frames: int = 0, duration_s: float = 0.0, summary: Optional[Dict[str, Any]] = None,
            wall: Optional[float] = None) -> None:
        if self.session_id is None:
            return
        with self._lock:
            conn = self._connect()
            conn.execute("UPDATE sessions SET ended_at=?, frames=?, duration_s=?, summary=? WHERE id=?",
                         (iso_now(wall), frames, round(duration_s, 3), json.dumps(summary or {}, default=str),
                          self.session_id))
            conn.commit()
        if self.echo:
            print("[session] #{} ended: {} events, {} metric rows -> {}".format(
                self.session_id, self.events_written, self.metrics_written, self.path))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- time base --------------------------------------------------------------------
    def rel_t(self, t: Optional[float]) -> Optional[float]:
        """Pipeline seconds since the session's first frame."""
        if t is None:
            return None
        if self.t0 is None:
            self.t0 = t
        return round(t - self.t0, 3)

    # -- writes -------------------------------------------------------------------------
    def log_event(self, kind: str, event: str, t: Optional[float] = None, wall: Optional[float] = None,
                  from_state: Optional[str] = None, to_state: Optional[str] = None, level: Optional[str] = None,
                  detail: str = "", state: Optional[str] = None, perclos: Optional[float] = None,
                  closure_now_s: Optional[float] = None, ear: Optional[float] = None, mar: Optional[float] = None,
                  yaw_deg: Optional[float] = None, pitch_deg: Optional[float] = None) -> None:
        if self.session_id is None:
            return
        row = (self.session_id, iso_now(wall), self.rel_t(t), kind, event, from_state, to_state, level, detail,
               state, perclos, closure_now_s, ear, mar, yaw_deg, pitch_deg)
        with self._lock:
            conn = self._connect()
            conn.execute("INSERT INTO events ({}) VALUES ({})".format(
                ", ".join(EVENT_COLUMNS), ", ".join("?" * len(EVENT_COLUMNS))), tuple(_clean(v) for v in row))
            conn.commit()
            self.events_written += 1
        if self.echo:
            print("[session] {} {} {}".format(kind, event, detail))

    def log_metrics(self, t: float, values: Dict[str, Any], wall: Optional[float] = None, force: bool = False) -> bool:
        """Write one metrics row if ``metrics_interval_s`` has passed since the
        last one (or ``force``). Returns True when a row was written."""
        if self.session_id is None:
            return False
        if not force and t - self._last_metrics_t < self.metrics_interval_s:
            return False
        self._last_metrics_t = t
        row = {"session_id": self.session_id, "wall_ts": iso_now(wall), "t_s": self.rel_t(t)}
        for key in METRIC_COLUMNS[3:]:
            row[key] = _clean(values.get(key))
        with self._lock:
            conn = self._connect()
            conn.execute("INSERT INTO metrics ({}) VALUES ({})".format(
                ", ".join(METRIC_COLUMNS), ", ".join("?" * len(METRIC_COLUMNS))),
                tuple(row[k] for k in METRIC_COLUMNS))
            conn.commit()
            self.metrics_written += 1
        return True


# --- readers (separate connection per call: safe from any thread / process) ------------

def _read(path: Path, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    conn = sqlite3.connect("file:{}?mode=ro".format(path.as_posix()), uri=True, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]
    finally:
        conn.close()


def list_sessions(path: Path = DEFAULT_DB, limit: int = 50) -> List[Dict[str, Any]]:
    rows = _read(path, "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,))
    for r in rows:
        r["config"] = json.loads(r["config"]) if r.get("config") else {}
        r["summary"] = json.loads(r["summary"]) if r.get("summary") else {}
    return rows


def get_session(path: Path, session_id: int) -> Optional[Dict[str, Any]]:
    rows = _read(path, "SELECT * FROM sessions WHERE id=?", (session_id,))
    if not rows:
        return None
    r = rows[0]
    r["config"] = json.loads(r["config"]) if r.get("config") else {}
    r["summary"] = json.loads(r["summary"]) if r.get("summary") else {}
    return r


def read_events(path: Path, session_id: int, kinds: Optional[Iterable[str]] = None, limit: int = 0,
                newest_first: bool = False) -> List[Dict[str, Any]]:
    sql, params = "SELECT * FROM events WHERE session_id=?", [session_id]
    if kinds:
        kinds = list(kinds)
        sql += " AND kind IN ({})".format(", ".join("?" * len(kinds)))
        params += kinds
    sql += " ORDER BY id {}".format("DESC" if newest_first else "ASC")
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return _read(path, sql, params)


def read_metrics(path: Path, session_id: int, limit: int = 0) -> List[Dict[str, Any]]:
    sql, params = "SELECT * FROM metrics WHERE session_id=? ORDER BY id ASC", [session_id]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return _read(path, sql, params)


def export_csv(path: Path, session_id: int, out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in (("events", read_events(path, session_id)), ("metrics", read_metrics(path, session_id))):
        target = out_dir / "session{}_{}.csv".format(session_id, name)
        with open(target, "w", newline="", encoding="utf-8") as fh:
            columns = list(rows[0].keys()) if rows else (["id"] + (EVENT_COLUMNS if name == "events" else METRIC_COLUMNS))
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        written.append(target)
    return written


def describe_session(s: Dict[str, Any]) -> str:
    summary = s.get("summary") or {}
    dur = s.get("duration_s") or 0.0
    return "#{:<3} {}  {:>6.0f} s  {:>6} frames  {}  final {}  transitions {}  alerts {}".format(
        s["id"], (s.get("started_at") or "")[:19], dur, s.get("frames") or 0,
        (s.get("source") or "")[:32].ljust(32), summary.get("final_state", "?"),
        summary.get("transitions", "?"), summary.get("alert_events", "?"))


# --- command line -------------------------------------------------------------------------

def cmd_list(path: Path) -> int:
    sessions = list_sessions(path)
    if not sessions:
        print("no sessions in {}".format(path))
        return 1
    for s in sessions:
        print(describe_session(s))
    return 0


def cmd_show(path: Path, session_id: int) -> int:
    s = get_session(path, session_id)
    if s is None:
        print("no session #{} in {}".format(session_id, path))
        return 1
    print(describe_session(s))
    print("config : {}".format(json.dumps(s["config"])))
    print("summary: {}".format(json.dumps(s["summary"], indent=1)))
    events = read_events(path, session_id)
    print("{} events:".format(len(events)))
    for e in events:
        print("  {}  t={:>7}  {:<10} {:<12} {} -> {}  {}".format(
            e["wall_ts"][11:23], e["t_s"] if e["t_s"] is not None else "-", e["kind"], e["event"],
            e["from_state"] or "", e["to_state"] or e["level"] or "", e["detail"] or ""))
    print("{} metric rows".format(len(read_metrics(path, session_id))))
    return 0


def self_test() -> int:
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="session_log_")) / "test.db"
    log = SessionLogger(tmp, metrics_interval_s=1.0)
    sid = log.start(source="fake camera", config={"perclos_mild": 0.15, "window_s": 60})
    assert sid == 1
    # metrics are throttled to the interval; NaN becomes NULL
    assert log.log_metrics(100.0, {"state": "ALERT", "ear": 0.31, "mean_blink_s": float("nan"), "fps": 20.0})
    assert not log.log_metrics(100.5, {"state": "ALERT"})           # too soon
    assert log.log_metrics(101.0, {"state": "ALERT", "ear": 0.30})
    assert log.log_metrics(101.2, {"state": "MILD"}, force=True)
    log.log_event("transition", "ALERT->MILD", t=101.2, from_state="ALERT", to_state="MILD", detail="PERCLOS 20%",
                  state="MILD", perclos=0.2, ear=0.21)
    log.log_event("alert", "RAISED", t=101.2, level="VISUAL", detail="state MILD", state="MILD", perclos=0.2)
    log.log_event("driver", "mute", t=105.0, state="MILD")
    log.log_event("link", "connected", t=100.1, detail="COM9")
    # writes from another thread are fine
    th = threading.Thread(target=lambda: log.log_event("alert", "BEEP", t=104.2, level="BEEP", state="DROWSY"))
    th.start(); th.join()
    # a concurrent reader sees committed rows while the writer is still open
    ev = read_events(tmp, sid)
    assert [e["event"] for e in ev] == ["ALERT->MILD", "RAISED", "mute", "connected", "BEEP"], [e["event"] for e in ev]
    assert ev[0]["t_s"] == 1.2 and ev[0]["from_state"] == "ALERT" and ev[0]["perclos"] == 0.2, ev[0]
    assert read_events(tmp, sid, kinds=["alert"], newest_first=True)[0]["event"] == "BEEP"
    m = read_metrics(tmp, sid)
    assert len(m) == 3 and m[0]["mean_blink_s"] is None and m[0]["ear"] == 0.31 and m[0]["t_s"] == 0.0, m[0]
    log.end(frames=60, duration_s=3.0, summary={"final_state": "MILD", "transitions": 1, "alert_events": 2})
    s = get_session(tmp, sid)
    assert s["ended_at"] and s["frames"] == 60 and s["summary"]["final_state"] == "MILD" and s["config"]["window_s"] == 60
    assert list_sessions(tmp)[0]["id"] == sid
    # a second session in the same file gets id 2 and its own rows
    log2 = SessionLogger(tmp)
    sid2 = log2.start(source="another")
    log2.log_event("transition", "ALERT->DROWSY", t=5.0, from_state="ALERT", to_state="DROWSY")
    assert sid2 == 2 and len(read_events(tmp, sid2)) == 1 and len(read_events(tmp, sid)) == 5
    log2.end(); log2.close(); log.close()
    files = export_csv(tmp, sid, tmp.parent / "export")
    rows = list(csv.DictReader(open(files[0], encoding="utf-8")))
    assert len(rows) == 5 and rows[0]["event"] == "ALERT->MILD" and files[1].name == "session1_metrics.csv"
    print("[self-test] schema, throttled metrics, NaN->NULL, cross-thread writes, concurrent reads, "
          "two sessions, CSV export: ok")
    print(describe_session(get_session(tmp, sid)))
    print("[self-test] ALL PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 12: session log (SQLite).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--list", action="store_true")
    p.add_argument("--show", type=int, metavar="ID")
    p.add_argument("--export", type=int, metavar="ID")
    p.add_argument("--out", type=Path, default=None, help="With --export: output folder (default logs/session<ID>)")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.list:
        return cmd_list(args.db)
    if args.show is not None:
        return cmd_show(args.db, args.show)
    if args.export is not None:
        out = args.out or Path("logs") / "session{}".format(args.export)
        for f in export_csv(args.db, args.export, out):
            print("wrote {}".format(f))
        return 0
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
