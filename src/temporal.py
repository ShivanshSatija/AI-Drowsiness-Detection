"""Temporal drowsiness analysis over a 60-second sliding window (Stage 9).

Pipeline slice implemented here::

    per-frame observations (Stages 3-8)          60 s sliding window
      EAR, CNN eye state, MAR, pitch, validity -> PERCLOS, blink duration, yawn rate,
                                                  nod count, invalid-frame rate
                                               -> state machine ALERT -> MILD -> DROWSY
                                                  with hysteresis

Principles
----------
* A single closed-eye frame never means drowsiness. It moves PERCLOS by one
  frame and may start a closure *event*; decisions come only from window
  statistics and event durations.
* Invalid frames (Stage 4) are excluded from every eye, mouth and pose
  statistic - numerator and denominator alike - and counted only for the
  invalid-frame rate. A short invalid gap inside a closure pauses the event
  rather than splitting it.
* Hysteresis is a Schmitt trigger plus dwell times: separate enter / exit
  thresholds, a hold before escalating, a longer hold before de-escalating,
  and a minimum time in MILD / DROWSY. Only a microsleep (closure >= 1.5 s,
  i.e. ~30 frames) escalates without the dwell.
* Head nods are detected on pitch *changes* against a rolling-median baseline,
  because absolute pitch carries a camera-placement offset (Stage 4 measured
  -11 deg at rest on the laptop).

Per-frame eye state
-------------------
``fusion`` selects how a frame is judged closed:

    "cnn"    mean P(CLOSED) over the available eyes
    "ear"    sigmoid((ear_closed_thr - EAR) / ear_soft_scale)
    "fused"  mean of the two when both exist, otherwise whichever exists (default)

The frame counts as closed when the resulting probability >= closed_prob_thr.
Stage 15's ablation switches these modes.

Threshold provenance
--------------------
Every value in ``TemporalConfig`` is configurable. Their status:

    MEASURED   values taken from this project's own live measurements
    INITIAL    literature-derived or engineering starting points
    TUNE       decision thresholds that must be set from recorded sessions

    ear_closed_thr 0.20        INITIAL, bracketed by MEASURED data: developer open
                               0.34-0.40, closed 0.05-0.19 (Stages 3/5). Per-person.
    mar_yawn_thr 0.60          INITIAL, bracketed by MEASURED data: closed <= 0.05,
                               wide open 0.87-1.00. Talking is untested -> TUNE.
    min_blink 0.08 s           INITIAL (one frame at 20 FPS is 0.05 s; blinks are
    max_blink 0.50 s           typically 100-400 ms in the literature)
    microsleep 1.5 s           INITIAL, conservative (literature 0.5-1 s and up) -> TUNE
    PERCLOS mild 0.15 / 0.10   INITIAL from the driver-monitoring literature
    PERCLOS drowsy 0.30 / 0.22 (Wierwille et al. / NHTSA PERCLOS levels) -> TUNE
    yawns, nods counts         INITIAL -> TUNE
    nod_drop 15 deg            INITIAL -> TUNE
    dwell / hold times         INITIAL engineering values -> TUNE
    window 60 s                roadmap requirement

Run directly::

    python -m src.temporal --self-test
    python -m src.temporal --replay data/session.csv [--plot out.png]   # a --record CSV from src.features
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

ALERT, MILD, DROWSY = "ALERT", "MILD", "DROWSY"
STATES = (ALERT, MILD, DROWSY)
LEVEL = {ALERT: 0, MILD: 1, DROWSY: 2}


# --- configuration -----------------------------------------------------------

@dataclass
class TemporalConfig:
    """All thresholds of the temporal layer. See the module docstring for the
    provenance of each value (MEASURED / INITIAL / TUNE)."""

    window_s: float = 60.0                 # roadmap: 60-second sliding window
    fusion: str = "fused"                  # "cnn" | "ear" | "fused"

    # per-frame eye closure
    ear_closed_thr: float = 0.20           # INITIAL, bracketed by measured open/closed EAR; per-person -> TUNE
    ear_soft_scale: float = 0.03           # INITIAL: softness of the EAR -> probability sigmoid
    closed_prob_thr: float = 0.5           # frame counts as closed at/above this fused probability

    # closure events
    min_blink_s: float = 0.08              # INITIAL: shorter closures are noise (a lone frame at 20 FPS)
    max_blink_s: float = 0.50              # INITIAL: longer closures are "prolonged", not blinks
    long_closure_s: float = 0.50           # a closure this long counts as "long"
    long_closures_mild: int = 2            # INITIAL -> TUNE: long closures in the window that support MILD
    microsleep_s: float = 1.5              # INITIAL -> TUNE: closure that forces DROWSY without dwell
    closure_gap_s: float = 0.30            # invalid gap tolerated inside one closure event

    # yawns
    mar_yawn_thr: float = 0.60             # INITIAL, bracketed by measured MAR; talking untested -> TUNE
    min_yawn_s: float = 1.0                # INITIAL

    # head nods (pitch drop below a rolling-median baseline)
    nod_drop_deg: float = 15.0             # INITIAL -> TUNE
    nod_recover_frac: float = 0.5          # nod ends when pitch recovers to within this fraction of the drop
    min_nod_s: float = 0.3                 # INITIAL
    max_nod_s: float = 3.0                 # INITIAL: longer = sustained looking down, not a nod

    # state machine (Schmitt thresholds + dwell)
    perclos_mild_enter: float = 0.15       # INITIAL (literature) -> TUNE
    perclos_mild_exit: float = 0.10
    perclos_drowsy_enter: float = 0.30     # INITIAL (literature) -> TUNE
    perclos_drowsy_exit: float = 0.22
    yawns_mild: int = 2                    # yawns in the window that support MILD  (INITIAL -> TUNE)
    nods_mild: int = 2                     # nods in the window that support MILD   (INITIAL -> TUNE)
    support_for_drowsy: int = 3            # yawns + nods that lift PERCLOS-mild to DROWSY (INITIAL -> TUNE)
    up_dwell_s: float = 2.0                # enter condition must hold this long to escalate
    down_dwell_s: float = 10.0             # exit condition must hold this long to de-escalate
    mild_min_hold_s: float = 5.0
    drowsy_min_hold_s: float = 10.0

    # data sufficiency
    min_valid_frames: int = 60             # ~3 s at 20 FPS before any escalation
    max_invalid_rate: float = 0.60         # above this the window is unreliable: hold state, flag it


# --- observations and events ---------------------------------------------------

@dataclass
class Observation:
    """One frame's inputs to the temporal layer. ``None`` = not available."""

    t: float                               # seconds, monotonic
    valid: bool                            # Stage 4 frame validity
    ear: Optional[float] = None            # mean EAR
    mar: Optional[float] = None
    pitch_deg: Optional[float] = None
    cnn_closed_prob: Optional[float] = None  # mean P(CLOSED) over the available eyes


@dataclass
class Event:
    kind: str                              # "blink" | "closure" | "yawn" | "nod"
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _Run:
    """An in-progress run of consecutive positive frames."""
    start: float
    last: float
    frames: int = 1
    extra: float = 0.0                     # e.g. nod baseline at onset


@dataclass
class TemporalState:
    """Everything the display, the logger and the alert layer need."""

    t: float
    state: str
    state_since: float
    reasons: List[str]
    sufficient: bool                       # enough valid data to trust the metrics
    window_fill_s: float                   # seconds of data in the window
    frames: int
    valid_frames: int
    invalid_rate: float
    perclos: float
    closed_now: bool
    closure_now_s: float                   # duration of the ongoing closure, if any
    longest_closure_s: float               # longest completed or ongoing closure in the window
    blink_count: int
    blink_rate_per_min: float
    mean_blink_s: float                    # NaN when no blinks
    yawn_count: int
    yawn_rate_per_min: float
    nod_count: int
    pitch_baseline_deg: float              # NaN without pitch data
    fusion: str

    def as_row(self) -> Dict[str, object]:
        return {"state": self.state, "perclos": round(self.perclos, 4),
                "blink_rate_per_min": round(self.blink_rate_per_min, 2),
                "mean_blink_s": "" if math.isnan(self.mean_blink_s) else round(self.mean_blink_s, 3),
                "closure_now_s": round(self.closure_now_s, 2),
                "yawn_rate_per_min": round(self.yawn_rate_per_min, 2), "nod_count": self.nod_count,
                "invalid_rate": round(self.invalid_rate, 4), "sufficient": int(self.sufficient)}


# --- helpers -------------------------------------------------------------------

def closed_probability(obs: Observation, config: TemporalConfig) -> Optional[float]:
    """Fused per-frame probability that the eyes are closed; None if no source."""
    ear_p = None
    if obs.ear is not None and not math.isnan(obs.ear):
        ear_p = 1.0 / (1.0 + math.exp(-(config.ear_closed_thr - obs.ear) / config.ear_soft_scale))
    cnn_p = obs.cnn_closed_prob if obs.cnn_closed_prob is not None and not math.isnan(obs.cnn_closed_prob) else None
    if config.fusion == "ear":
        return ear_p
    if config.fusion == "cnn":
        return cnn_p
    if ear_p is None:
        return cnn_p
    if cnn_p is None:
        return ear_p
    return 0.5 * (ear_p + cnn_p)


# --- engine --------------------------------------------------------------------

class TemporalEngine:
    """Feed ``update(Observation)`` once per frame; read the returned state."""

    def __init__(self, config: Optional[TemporalConfig] = None) -> None:
        self.config = config or TemporalConfig()
        if self.config.fusion not in ("cnn", "ear", "fused"):
            raise ValueError("fusion must be cnn, ear or fused")
        # window of (t, valid, closed or None, pitch or None)
        self._frames: Deque[Tuple[float, bool, Optional[bool], Optional[float]]] = deque()
        self._events: Deque[Event] = deque()
        self._closure: Optional[_Run] = None
        self._yawn: Optional[_Run] = None
        self._nod: Optional[_Run] = None
        self._dt_estimate = 0.05                # refined from the stream
        self._last_t: Optional[float] = None
        self.state = ALERT
        self.state_since = 0.0
        self._enter_since: Dict[str, Optional[float]] = {MILD: None, DROWSY: None}
        self._exit_since: Optional[float] = None
        self.transitions: List[Tuple[float, str, str, str]] = []   # (t, from, to, reason)

    # -- window maintenance ------------------------------------------------------
    def _trim(self, now: float) -> None:
        horizon = now - self.config.window_s
        while self._frames and self._frames[0][0] < horizon:
            self._frames.popleft()
        while self._events and self._events[0].end < horizon:
            self._events.popleft()

    def _end_run(self, run: _Run, kind_short: str, kind_long: Optional[str], min_s: float,
                 max_s: Optional[float]) -> None:
        duration = run.last - run.start + self._dt_estimate
        if duration < min_s:
            return
        if max_s is not None and duration > max_s and kind_long is not None:
            self._events.append(Event(kind_long, run.start, run.start + duration))
        elif max_s is None or duration <= max_s or kind_long is None:
            self._events.append(Event(kind_short, run.start, run.start + duration))

    # -- per-frame update ----------------------------------------------------------
    def update(self, obs: Observation) -> TemporalState:
        cfg = self.config
        t = obs.t
        if self._last_t is None:
            self.state_since = t          # clocks are monotonic seconds, not zero-based
        elif t > self._last_t:
            dt = t - self._last_t
            if 0.005 < dt < 0.5:
                self._dt_estimate = 0.9 * self._dt_estimate + 0.1 * dt
        self._last_t = t

        closed: Optional[bool] = None
        pitch: Optional[float] = None
        if obs.valid:
            p = closed_probability(obs, cfg)
            closed = None if p is None else (p >= cfg.closed_prob_thr)
            if obs.pitch_deg is not None and not math.isnan(obs.pitch_deg):
                pitch = obs.pitch_deg
        self._frames.append((t, obs.valid, closed, pitch))
        self._trim(t)

        # -- closure events (blinks / prolonged closures) -------------------------
        if closed is True:
            if self._closure is None:
                self._closure = _Run(t, t)
            else:
                self._closure.last, self._closure.frames = t, self._closure.frames + 1
        elif closed is False:
            if self._closure is not None:
                self._end_run(self._closure, "blink", "closure", cfg.min_blink_s, cfg.max_blink_s)
                self._closure = None
        else:  # invalid or no eye source: pause; end the run if the gap grows too long
            if self._closure is not None and t - self._closure.last > cfg.closure_gap_s:
                self._end_run(self._closure, "blink", "closure", cfg.min_blink_s, cfg.max_blink_s)
                self._closure = None

        # -- yawns ---------------------------------------------------------------------
        mouth_open = (obs.valid and obs.mar is not None and not math.isnan(obs.mar)
                      and obs.mar >= cfg.mar_yawn_thr)
        if mouth_open:
            if self._yawn is None:
                self._yawn = _Run(t, t)
            else:
                self._yawn.last, self._yawn.frames = t, self._yawn.frames + 1
        elif self._yawn is not None and (obs.valid or t - self._yawn.last > cfg.closure_gap_s):
            self._end_run(self._yawn, "yawn", None, cfg.min_yawn_s, None)
            self._yawn = None

        # -- nods ---------------------------------------------------------------------
        pitches = [p for (_, v, _, p) in self._frames if v and p is not None]
        baseline = statistics.median(pitches) if len(pitches) >= 10 else math.nan
        if pitch is not None and not math.isnan(baseline):
            if self._nod is None:
                if pitch < baseline - cfg.nod_drop_deg:
                    self._nod = _Run(t, t, extra=baseline)
            else:
                recovered = pitch > self._nod.extra - cfg.nod_drop_deg * cfg.nod_recover_frac
                if recovered:
                    self._end_run(self._nod, "nod", "lookdown", cfg.min_nod_s, cfg.max_nod_s)
                    self._nod = None
                else:
                    self._nod.last, self._nod.frames = t, self._nod.frames + 1
        elif self._nod is not None and t - self._nod.last > cfg.closure_gap_s:
            self._end_run(self._nod, "nod", "lookdown", cfg.min_nod_s, cfg.max_nod_s)
            self._nod = None

        # -- window metrics ---------------------------------------------------------
        frames = len(self._frames)
        valid_frames = sum(1 for (_, v, _, _) in self._frames if v)
        eye_frames = [c for (_, v, c, _) in self._frames if v and c is not None]
        perclos = (sum(1 for c in eye_frames if c) / len(eye_frames)) if eye_frames else 0.0
        invalid_rate = (frames - valid_frames) / frames if frames else 0.0
        window_fill = (t - self._frames[0][0]) if frames > 1 else 0.0
        minutes = max(window_fill, 1e-6) / 60.0

        blinks = [e for e in self._events if e.kind == "blink"]
        closures = [e for e in self._events if e.kind == "closure"]
        yawns = [e for e in self._events if e.kind == "yawn"]
        nods = [e for e in self._events if e.kind == "nod"]
        closure_now = (t - self._closure.start + self._dt_estimate) if self._closure is not None else 0.0
        longest_closure = max([e.duration for e in closures] + [closure_now] + [e.duration for e in blinks] + [0.0])
        mean_blink = float(np.mean([e.duration for e in blinks])) if blinks else math.nan
        sufficient = valid_frames >= cfg.min_valid_frames and invalid_rate <= cfg.max_invalid_rate

        # -- state machine ---------------------------------------------------------
        reasons: List[str] = []
        support = len(yawns) + len(nods)
        long_closures = sum(1 for e in closures if e.duration >= cfg.long_closure_s) + (
            1 if closure_now >= cfg.long_closure_s else 0)
        drowsy_enter = (perclos >= cfg.perclos_drowsy_enter
                        or (perclos >= cfg.perclos_mild_enter and support >= cfg.support_for_drowsy))
        mild_enter = (perclos >= cfg.perclos_mild_enter or len(yawns) >= cfg.yawns_mild
                      or len(nods) >= cfg.nods_mild or long_closures >= cfg.long_closures_mild)
        microsleep = closure_now >= cfg.microsleep_s or any(e.duration >= cfg.microsleep_s for e in closures)
        drowsy_stay = perclos >= cfg.perclos_drowsy_exit or microsleep or (
            perclos >= cfg.perclos_mild_exit and support >= cfg.support_for_drowsy)
        mild_stay = (perclos >= cfg.perclos_mild_exit or len(yawns) >= cfg.yawns_mild
                     or len(nods) >= cfg.nods_mild or long_closures >= cfg.long_closures_mild)

        if perclos >= cfg.perclos_mild_enter:
            reasons.append("PERCLOS {:.0%}".format(perclos))
        if microsleep:
            reasons.append("closure {:.1f} s".format(max(closure_now, max([e.duration for e in closures] + [0.0]))))
        elif long_closures >= cfg.long_closures_mild:
            reasons.append("{} long closures (max {:.1f} s)".format(long_closures, longest_closure))
        if len(yawns) >= cfg.yawns_mild:
            reasons.append("{} yawns".format(len(yawns)))
        if len(nods) >= cfg.nods_mild:
            reasons.append("{} nods".format(len(nods)))

        if sufficient:
            # escalation: microsleep is immediate; everything else needs the dwell
            if self.state != DROWSY and microsleep:
                self._transition(t, DROWSY, "microsleep {:.1f} s".format(closure_now or longest_closure))
            else:
                for level, cond in ((DROWSY, drowsy_enter), (MILD, mild_enter)):
                    if LEVEL[level] <= LEVEL[self.state]:
                        self._enter_since[level] = None
                        continue
                    if cond:
                        if self._enter_since[level] is None:
                            self._enter_since[level] = t
                        elif t - self._enter_since[level] >= cfg.up_dwell_s:
                            self._transition(t, level, "; ".join(reasons) or "threshold held {:.0f} s".format(
                                cfg.up_dwell_s))
                            break
                    else:
                        self._enter_since[level] = None
            # de-escalation: exit condition must hold for the down dwell, after the minimum hold
            stay = drowsy_stay if self.state == DROWSY else mild_stay if self.state == MILD else True
            if self.state != ALERT and not stay:
                if self._exit_since is None:
                    self._exit_since = t
                min_hold = cfg.drowsy_min_hold_s if self.state == DROWSY else cfg.mild_min_hold_s
                if t - self._exit_since >= cfg.down_dwell_s and t - self.state_since >= min_hold:
                    target = MILD if (self.state == DROWSY and mild_stay) else ALERT
                    self._transition(t, target, "recovered")
            else:
                self._exit_since = None
        else:
            reasons = ["insufficient data ({} valid frames, {:.0%} invalid)".format(valid_frames, invalid_rate)]

        return TemporalState(
            t=t, state=self.state, state_since=self.state_since, reasons=reasons, sufficient=sufficient,
            window_fill_s=window_fill, frames=frames, valid_frames=valid_frames, invalid_rate=invalid_rate,
            perclos=perclos, closed_now=closed is True, closure_now_s=closure_now,
            longest_closure_s=longest_closure, blink_count=len(blinks),
            blink_rate_per_min=len(blinks) / minutes if window_fill > 1.0 else 0.0, mean_blink_s=mean_blink,
            yawn_count=len(yawns), yawn_rate_per_min=len(yawns) / minutes if window_fill > 1.0 else 0.0,
            nod_count=len(nods), pitch_baseline_deg=baseline, fusion=cfg.fusion)

    def reset(self, t: float, reason: str = "manual reset") -> None:
        """Driver acknowledged the alert: back to ALERT with an empty window.

        The window is cleared on purpose - the closures that caused the episode
        would otherwise re-trigger MILD/DROWSY after the 2 s dwell. Detection
        restarts from scratch: ~3 s of insufficient data, then normal rules, so a
        driver who is still drowsy is re-detected within seconds (a new microsleep
        escalates immediately). Logged as a transition so replays show it."""
        if self.state != ALERT:
            self._transition(t, ALERT, reason)
        else:
            self.transitions.append((t, ALERT, ALERT, reason))
            self.state_since = t
        self._frames.clear()
        self._events.clear()
        self._closure = self._yawn = self._nod = None
        self._last_t = None

    def _transition(self, t: float, new_state: str, reason: str) -> None:
        self.transitions.append((t, self.state, new_state, reason))
        self.state, self.state_since = new_state, t
        self._exit_since = None
        for level in self._enter_since:
            self._enter_since[level] = None


# --- display -------------------------------------------------------------------

STATE_COLORS = {ALERT: (0, 200, 0), MILD: (0, 200, 255), DROWSY: (0, 0, 255)}


def draw_temporal_panel(frame: np.ndarray, ts: TemporalState, x: int, y: int) -> None:
    """Compact top-right panel: state, PERCLOS, blinks, closure, yawns, nods,
    invalid rate. Call after mirroring."""
    import cv2

    def put(text, dx, dy, color=(255, 255, 255), scale=0.5, thick=1):
        cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    color = STATE_COLORS[ts.state] if ts.sufficient else (160, 160, 160)
    put(ts.state, 0, 0, color, 0.9, 2)
    put("{:.0f} s".format(max(0.0, ts.t - ts.state_since)), 118, 0, (200, 200, 200), 0.5)
    put("PERCLOS {:5.1%}   win {:2.0f}/{:.0f} s".format(ts.perclos, ts.window_fill_s, 60.0), 0, 20)
    blink = "blinks {:4.1f}/min".format(ts.blink_rate_per_min)
    blink += "  {:.0f} ms".format(ts.mean_blink_s * 1000) if not math.isnan(ts.mean_blink_s) else ""
    put(blink, 0, 38)
    put("closure now {:.1f} s  max {:.1f} s".format(ts.closure_now_s, ts.longest_closure_s), 0, 56,
        (0, 140, 255) if ts.closed_now else (255, 255, 255))
    put("yawns {:.1f}/min   nods {}".format(ts.yawn_rate_per_min, ts.nod_count), 0, 74)
    put("invalid {:4.0%}   eyes: {}".format(ts.invalid_rate, ts.fusion), 0, 92, (200, 200, 200))
    if ts.reasons:
        put("; ".join(ts.reasons)[:46], 0, 110, color, 0.45)


# --- replay of a recorded session ---------------------------------------------

def observations_from_csv(path: Path, default_fps: float = 20.0) -> List[Observation]:
    """Turn a --record CSV from src.features into Observations.

    Time comes from the detector's monotonic timestamp_ms where a face was
    found; frames without a face are placed by interpolation at the median
    frame interval (fallback: default_fps)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    stamps = [float(r["timestamp_ms"]) / 1000.0 for r in rows if r.get("timestamp_ms")]
    dt = float(np.median(np.diff(stamps))) if len(stamps) > 2 else 1.0 / default_fps
    if not (0.005 < dt < 0.5):
        dt = 1.0 / default_fps
    obs: List[Observation] = []
    t_prev = None
    for r in rows:
        if r.get("timestamp_ms"):
            t = float(r["timestamp_ms"]) / 1000.0
        else:
            t = (t_prev + dt) if t_prev is not None else 0.0
        if t_prev is not None and t <= t_prev:
            t = t_prev + dt
        t_prev = t
        valid = r.get("valid") == "1"

        def num(key):
            v = r.get(key, "")
            return float(v) if v not in ("", None) else None

        probs = []
        for side in ("left", "right"):
            state, conf = r.get("cnn_{}_state".format(side)), r.get("cnn_{}_conf".format(side))
            if state and conf:
                c = float(conf)
                probs.append(c if state == "CLOSED" else 1.0 - c)
        obs.append(Observation(t=t, valid=valid, ear=num("ear_mean"), mar=num("mar"), pitch_deg=num("pitch_deg"),
                               cnn_closed_prob=float(np.mean(probs)) if probs else None))
    return obs


def replay(path: Path, config: TemporalConfig, plot: Optional[Path] = None) -> int:
    obs = observations_from_csv(path)
    engine = TemporalEngine(config)
    history: List[TemporalState] = [engine.update(o) for o in obs]
    if not history:
        print("no rows in {}".format(path))
        return 1
    last = history[-1]
    print("[replay] {} frames, {:.1f} s, fusion {}".format(len(obs), obs[-1].t - obs[0].t, config.fusion))
    print("[replay] final: state {} | PERCLOS {:.1%} | blinks {} (mean {} ms, {:.1f}/min) | longest closure {:.2f} s"
          " | yawns {} | nods {} | invalid {:.1%}".format(
              last.state, last.perclos, last.blink_count,
              "n/a" if math.isnan(last.mean_blink_s) else "{:.0f}".format(last.mean_blink_s * 1000),
              last.blink_rate_per_min, last.longest_closure_s, last.yawn_count, last.nod_count, last.invalid_rate))
    print("[replay] state transitions:")
    for t, a, b, why in engine.transitions or []:
        print("   {:7.1f} s  {} -> {}  ({})".format(t - obs[0].t, a, b, why))
    if not engine.transitions:
        print("   none")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = np.array([h.t - obs[0].t for h in history])
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(ts, [o.ear if o.ear is not None else np.nan for o in obs], lw=0.8, label="EAR")
        axes[0].plot(ts, [o.cnn_closed_prob if o.cnn_closed_prob is not None else np.nan for o in obs], lw=0.8,
                     label="CNN P(closed)")
        axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)
        axes[1].plot(ts, [h.perclos for h in history], label="PERCLOS", color="tab:orange")
        axes[1].axhline(config.perclos_mild_enter, ls="--", color="gold", lw=0.8)
        axes[1].axhline(config.perclos_drowsy_enter, ls="--", color="red", lw=0.8)
        axes[1].plot(ts, [h.closure_now_s / 3 for h in history], lw=0.8, color="tab:purple", label="closure now (s/3)")
        axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)
        axes[2].step(ts, [LEVEL[h.state] for h in history], where="post", color="tab:red")
        axes[2].set_yticks([0, 1, 2]); axes[2].set_yticklabels(STATES); axes[2].set_xlabel("seconds")
        axes[2].grid(alpha=0.3)
        fig.suptitle("Temporal replay: {}".format(path.name))
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print("[replay] plot -> {}".format(plot))
    return 0


# --- self-test ------------------------------------------------------------------

def _stream(engine: TemporalEngine, seconds: float, fps: float, t0: float, closed_pattern, mar=0.02, pitch=-10.0,
            valid_pattern=None) -> Tuple[float, TemporalState]:
    """Feed synthetic frames. closed_pattern(t) -> bool; valid_pattern(t) -> bool."""
    dt = 1.0 / fps
    n = int(round(seconds * fps))
    state = None
    t = t0
    for i in range(n):
        t = t0 + i * dt
        closed = closed_pattern(t)
        valid = valid_pattern(t) if valid_pattern else True
        ear = 0.10 if closed else 0.35
        pitch_v = pitch(t) if callable(pitch) else pitch
        mar_v = mar(t) if callable(mar) else mar
        state = engine.update(Observation(t, valid, ear=ear, mar=mar_v, pitch_deg=pitch_v,
                                          cnn_closed_prob=0.9 if closed else 0.1))
    return t + dt, state


def self_test() -> int:
    fps = 20.0

    # 1. eyes open: ALERT, PERCLOS 0, no blinks; insufficient data at first.
    eng = TemporalEngine()
    s_early = eng.update(Observation(0.0, True, ear=0.35, mar=0.0, pitch_deg=-10, cnn_closed_prob=0.1))
    assert not s_early.sufficient and s_early.state == ALERT
    t, s = _stream(eng, 30, fps, 0.05, lambda t: False)
    assert s.sufficient and s.state == ALERT and s.perclos == 0.0 and s.blink_count == 0, s
    print("[self-test] eyes open 30 s: ALERT, PERCLOS 0, no blinks; first frames flagged insufficient")

    # 2. a single closed frame is not a blink and does not change state.
    eng = TemporalEngine()
    t, _ = _stream(eng, 10, fps, 0.0, lambda t: False)
    t, _ = _stream(eng, 1 / fps, fps, t, lambda t: True)           # exactly one closed frame
    t, s = _stream(eng, 5, fps, t, lambda t: False)
    assert s.blink_count == 0 and s.state == ALERT and s.perclos < 0.01, (s.blink_count, s.state, s.perclos)
    print("[self-test] one closed frame: no blink event, PERCLOS {:.3f}, still ALERT".format(s.perclos))

    # 3. normal blinking: 150 ms closures every 4 s -> ~15/min, mean 150 ms, ALERT.
    eng = TemporalEngine()
    blink = lambda t: (t % 4.0) < 0.15
    t, s = _stream(eng, 60, fps, 0.0, blink)
    assert 12 <= s.blink_count <= 16, s.blink_count
    assert 0.10 <= s.mean_blink_s <= 0.22, s.mean_blink_s
    assert s.state == ALERT and s.perclos < 0.06, (s.state, s.perclos)
    print("[self-test] blinking: {} blinks/min, mean {:.0f} ms, PERCLOS {:.1%}, ALERT".format(
        s.blink_count, s.mean_blink_s * 1000, s.perclos))

    # 4. prolonged closure: DROWSY once the closure passes 1.5 s, held afterwards.
    eng = TemporalEngine()
    t, _ = _stream(eng, 10, fps, 0.0, lambda t: False)
    t, s_mid = _stream(eng, 1.0, fps, t, lambda t: True)
    assert s_mid.state == ALERT, "1 s closure must not yet be DROWSY"
    t, s_long = _stream(eng, 1.0, fps, t, lambda t: True)          # now 2 s closed
    assert s_long.state == DROWSY and s_long.closure_now_s >= 1.5, (s_long.state, s_long.closure_now_s)
    t, s_after = _stream(eng, 5, fps, t, lambda t: False)
    assert s_after.state == DROWSY, "DROWSY must hold for its minimum time"
    assert s_after.longest_closure_s >= 1.9, s_after.longest_closure_s
    print("[self-test] 2 s closure: ALERT at 1 s, DROWSY at {:.1f} s (microsleep), held; longest {:.2f} s".format(
        s_long.closure_now_s, s_after.longest_closure_s))

    # 5. PERCLOS escalation with dwell, then graded recovery DROWSY -> MILD -> ALERT without flapping.
    eng = TemporalEngine(TemporalConfig(microsleep_s=99.0))       # isolate the PERCLOS path
    heavy = lambda t: (t % 2.0) < 0.8                            # 40 % closed in 0.8 s closures
    t, s = _stream(eng, 60, fps, 0.0, heavy)
    assert s.state == DROWSY and s.perclos > 0.3, (s.state, s.perclos)
    first_drowsy = [tr for tr in eng.transitions if tr[2] == DROWSY][0][0]
    # sufficiency needs 60 valid frames (3 s at 20 FPS) and the up dwell adds 2 s: earliest 5.0 s
    assert first_drowsy >= 4.9, "escalation needs sufficiency plus the up dwell, got {:.1f}s".format(first_drowsy)
    t, s_mid = _stream(eng, 40, fps, t, lambda t: False)
    assert s_mid.state == MILD, "0.8 s closures still in the window must hold MILD, got {}".format(s_mid.state)
    t, s = _stream(eng, 40, fps, t, lambda t: False)             # window now clear of closures for > 10 s
    assert s.state == ALERT and s.perclos == 0.0, (s.state, s.perclos)
    path = [tr[2] for tr in eng.transitions]
    assert path == [DROWSY, MILD, ALERT] or path == [MILD, DROWSY, MILD, ALERT], path
    print("[self-test] PERCLOS 40%: DROWSY after {:.1f} s; recovery {} with {} transitions".format(
        first_drowsy, " -> ".join(path), len(path)))

    # 6. hovering around the MILD threshold does not oscillate (Schmitt + dwell).
    eng = TemporalEngine(TemporalConfig(microsleep_s=99.0))
    hover = lambda t: (t % 5.0) < (0.8 if int(t // 5) % 2 == 0 else 0.6)   # ~16 % then ~12 % closed
    t, s = _stream(eng, 120, fps, 0.0, hover)
    changes = [tr for tr in eng.transitions]
    assert len(changes) <= 2, "state flapped: {}".format(changes)
    print("[self-test] PERCLOS hovering 12-16 %: {} transition(s) in 120 s, final {}".format(len(changes), s.state))

    # 7. yawns: MAR high for 1.5 s at t = 0, 12, 24, 36 -> 4 yawns in 40 s, MILD.
    eng = TemporalEngine()
    yawn_mar = lambda t: 0.85 if (t % 12.0) < 1.5 else 0.02
    t, s = _stream(eng, 40, fps, 0.0, lambda t: False, mar=yawn_mar)
    assert s.yawn_count == 4 and s.state == MILD, (s.yawn_count, s.state)
    print("[self-test] yawns: {} counted, rate {:.1f}/min -> MILD".format(s.yawn_count, s.yawn_rate_per_min))

    # 8. nods: pitch dips 20 deg for 0.8 s, three times -> 3 nods (baseline from the median).
    eng = TemporalEngine()
    nod_pitch = lambda t: -30.0 if 20.0 <= (t % 10.0) < 20.8 or (t % 10.0) < 0.8 and t > 5 else -10.0
    t, s = _stream(eng, 40, fps, 0.0, lambda t: False, pitch=nod_pitch)
    assert s.nod_count >= 3, s.nod_count
    assert abs(s.pitch_baseline_deg + 10.0) < 1.0, s.pitch_baseline_deg
    print("[self-test] nods: {} counted against baseline {:.0f} deg".format(s.nod_count, s.pitch_baseline_deg))

    # 9. invalid frames are excluded: closed eyes on invalid frames must not raise PERCLOS.
    eng = TemporalEngine()
    invalid = lambda t: (t % 1.0) < 0.3                         # 30 % of frames invalid
    t, s = _stream(eng, 40, fps, 0.0, lambda t: invalid(t), valid_pattern=lambda t: not invalid(t))
    assert s.perclos == 0.0 and abs(s.invalid_rate - 0.30) < 0.03 and s.state == ALERT, (s.perclos, s.invalid_rate)
    eng = TemporalEngine()
    t, s = _stream(eng, 20, fps, 0.0, lambda t: True, valid_pattern=lambda t: (t % 1.0) >= 0.7)  # 70 % invalid
    assert not s.sufficient and s.state == ALERT, "unreliable window must hold ALERT and be flagged"

    # 9b. manual reset: DROWSY -> ALERT at once, window emptied, re-detected if closures continue.
    engine = TemporalEngine(TemporalConfig())
    t, s = _stream(engine, 8.0, 20.0, 0.0, lambda t: t >= 5.0)      # 5 s open, then closed -> microsleep
    assert s.state == DROWSY, s.state
    engine.reset(t)
    t, s = _stream(engine, 1.0, 20.0, t, lambda t: False)
    assert s.state == ALERT and not s.sufficient and s.perclos == 0.0 and s.longest_closure_s == 0.0, s
    assert engine.transitions[-1][1:] == (DROWSY, ALERT, "manual reset"), engine.transitions[-1]
    t, s = _stream(engine, 6.0, 20.0, t, lambda t: True)             # still drowsy -> back within seconds
    assert s.state == DROWSY, "a still-drowsy driver must be re-detected after a reset, got {}".format(s.state)
    print("[self-test] manual reset: DROWSY -> ALERT, window cleared, re-detected {:.1f} s later: ok".format(
        engine.transitions[-1][0] - engine.transitions[-2][0]))
    print("[self-test] invalid frames: excluded from PERCLOS (0.0 at 30 % invalid); 70 % invalid -> insufficient")

    # 10. fusion fallbacks.
    cfg = TemporalConfig()
    assert closed_probability(Observation(0, True, ear=0.10, cnn_closed_prob=None), cfg) > 0.9
    assert closed_probability(Observation(0, True, ear=None, cnn_closed_prob=0.8), cfg) == 0.8
    assert closed_probability(Observation(0, True, ear=None, cnn_closed_prob=None), cfg) is None
    assert closed_probability(Observation(0, True, ear=0.35, cnn_closed_prob=0.9), TemporalConfig(fusion="ear")) < 0.1
    print("[self-test] fusion: EAR-only, CNN-only, none and forced modes behave")

    # 11. drawing and replay round trip via a synthetic CSV.
    import cv2  # noqa: F401
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_temporal_panel(canvas, s, 400, 30)
    assert canvas.any()
    tmp = Path(__file__).resolve().parents[1] / "data" / "_selftest_temporal.csv"
    tmp.parent.mkdir(exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "face_found", "inference_ms", "ear_left", "ear_right", "ear_mean", "mar",
                    "eye_width_left_px", "eye_width_right_px", "mouth_width_px", "timestamp_ms", "yaw_deg",
                    "pitch_deg", "roll_deg", "pose_method", "valid", "invalid_reasons", "cnn_left_state",
                    "cnn_left_conf", "cnn_right_state", "cnn_right_conf", "cnn_ms"])
        for i in range(400):
            closed = 100 <= i < 160                                # a 3 s closure
            w.writerow([i, 1, 10, 0.1 if closed else 0.35, 0.1 if closed else 0.35, 0.1 if closed else 0.35, 0.02,
                        40, 40, 60, int(i * 50), 5, -10, 0, "matrix", 1, "",
                        "CLOSED" if closed else "OPEN", 0.9, "CLOSED" if closed else "OPEN", 0.9, 4])
    obs = observations_from_csv(tmp)
    assert len(obs) == 400 and abs(obs[1].t - obs[0].t - 0.05) < 1e-6
    eng = TemporalEngine()
    final = [eng.update(o) for o in obs][-1]
    assert final.state == DROWSY and final.longest_closure_s > 2.5, (final.state, final.longest_closure_s)
    tmp.unlink()
    print("[self-test] panel drawn; CSV replay reproduces a 3 s closure -> DROWSY")

    print("[self-test] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 9: temporal drowsiness analysis.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--replay", type=Path, help="Replay a --record CSV from src.features")
    parser.add_argument("--plot", type=Path, help="With --replay: write a PNG of EAR/PERCLOS/state over time")
    parser.add_argument("--fusion", choices=("cnn", "ear", "fused"), default="fused")
    parser.add_argument("--window", type=float, default=60.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.replay:
        return replay(args.replay, TemporalConfig(window_s=args.window, fusion=args.fusion), args.plot)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
