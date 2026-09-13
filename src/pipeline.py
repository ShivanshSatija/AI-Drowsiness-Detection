"""One shared per-frame pipeline for every front end (Stage 12).

Until Stage 11 the glue between the stages lived inside ``src.features.run_demo``
together with the OpenCV window. Stage 12 needs the same chain without a
window, driven from a background thread, so the glue moved here unchanged::

    frame -> FaceLandmarkDetector (2) -> compute_features (3) -> estimate_pose (4)
          -> assess_frame (4) -> extract_eye_crops (5) -> EyeStateClassifier (7/8)
          -> Observation -> TemporalEngine (9) -> AlertManager (10) -> BuzzerLink (11)
          -> SessionLogger (12)

No detection algorithm is defined in this file; every stage keeps its own
module and its own self-test. This module only *calls* them in the same order
with the same arguments as before and returns everything in a ``FrameResult``.

Two front ends use it:

    src/features.py   OpenCV window, keys, --record CSV       (development / debugging tool)
    src/app.py        Streamlit dashboard via PipelineWorker  (demonstration / monitoring)

``PipelineWorker`` runs a ``FrameSource`` through the pipeline on a daemon
thread and keeps the latest annotated frame (JPEG) plus a ``Snapshot`` of
every metric for a UI to poll. The camera keeps running at its own rate no
matter how often the UI redraws.

Run directly (headless, no window, no Streamlit - the dashboard's backend on its own)::

    python -m src.pipeline --device 0 --max-frames 300
    python -m src.pipeline --device data/two_faces.avi --no-cnn --no-serial --min-face-px 60 --min-eye-px 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.alert import LEVEL_NAMES, NONE, AlertConfig, AlertManager, AlertStatus, draw_alert_overlay
from src.capture import CameraConfig, CameraError, FPSCounter, FrameSource, create_source
from src.eye_cnn import MODEL_PATH, EyePreprocessConfig, EyeStateClassifier, draw_eye_boxes, extract_eye_crops
from src.features import (FrameAssessment, GeometricFeatures, InvalidFrameTracker, ValidityConfig, assess_frame,
                          compute_features, draw_feature_geometry)
from src.hardware import BuzzerLink, word_for
from src.headpose import HeadPose, PoseConfig, draw_pose, estimate_pose
from src.landmarks import (FaceLandmarkDetector, FaceLandmarks, LandmarkConfig, LandmarkModelError,
                           draw_ignored_faces, draw_landmarks)
from src.session_log import SessionLogger
from src.temporal import STATE_COLORS, Observation, TemporalConfig, TemporalEngine, TemporalState


# --- per-frame result ------------------------------------------------------------

@dataclass
class FrameResult:
    """Everything one frame produced, in the order the stages produced it."""

    frame_index: int
    t: float                                  # pipeline monotonic seconds (assessment.time_s)
    wall: float                               # time.time() when the frame was processed
    face: Optional[FaceLandmarks]
    feats: Optional[GeometricFeatures]
    pose: Optional[HeadPose]
    assessment: FrameAssessment
    crops: Tuple                              # (left, right) EyeCrop or None
    eye_states: Optional[List]                # [(state, conf) | None, ...] from the CNN
    cnn_ms: float
    observation: Observation
    temporal_state: TemporalState
    alert_status: Optional[AlertStatus]
    buzzer_word: Optional[str]
    transitions: List[Tuple[float, str, str, str]]   # new Stage 9 transitions on this frame
    alert_events: List[Dict[str, Any]]                # new Stage 10 events on this frame
    fps: float
    inference_ms: float


# --- the pipeline ---------------------------------------------------------------------

class DrowsinessPipeline:
    """Owns the per-frame chain and every session-wide statistic. Call
    ``process(frame)`` once per frame from one thread; ``dismiss()`` /
    ``reset()`` from any thread; ``close()`` once at the end."""

    def __init__(self, detector: FaceLandmarkDetector, classifier: Optional[EyeStateClassifier] = None,
                 pose_config: Optional[PoseConfig] = None, validity_config: Optional[ValidityConfig] = None,
                 eye_config: Optional[EyePreprocessConfig] = None, temporal_config: Optional[TemporalConfig] = None,
                 alert_config: Optional[AlertConfig] = None, alerts: bool = True,
                 serial_port: Optional[str] = "auto", session_log: Optional[SessionLogger] = None,
                 cnn_status: str = "", echo: bool = True) -> None:
        self.detector = detector
        self.classifier = classifier
        self.cnn_status = cnn_status
        self.pose_config = pose_config or PoseConfig()
        self.validity_config = validity_config or ValidityConfig()
        self.eye_config = eye_config or EyePreprocessConfig()
        self.echo = echo
        self.serial_port = serial_port

        # Stage 9
        self.temporal = TemporalEngine(temporal_config or TemporalConfig())
        # Stage 10: the alert layer only ever sees TemporalState - never frames or features.
        self.alerter: Optional[AlertManager] = (
            AlertManager(alert_config or AlertConfig(), on_event=self._on_alert_event) if alerts else None)
        # Stage 11: the ESP32 buzzer link. Its thread owns the port; one set_state() per frame here.
        self.buzzer: Optional[BuzzerLink] = None
        if serial_port:
            try:
                self.buzzer = BuzzerLink(port=serial_port, echo=echo)
            except ImportError as exc:
                print("[esp32] {} - running without the buzzer".format(exc), file=sys.stderr)
        # Stage 12
        self.session_log = session_log
        self.session_started = False
        self.source_description = ""

        self.tracker = InvalidFrameTracker(self.validity_config.window_seconds)
        self.fps_counter = FPSCounter()
        self.frame_index = 0
        self.started_perf = time.perf_counter()
        self.started_wall = time.time()
        self.started_t: Optional[float] = None
        self.temporal_state: Optional[TemporalState] = None
        self.alert_status: Optional[AlertStatus] = None
        self.buzzer_word: Optional[str] = None
        self.last_result: Optional[FrameResult] = None

        # session-wide statistics (what run_demo used to collect inline)
        self.time_in_state: Counter = Counter()
        self.ears: List[float] = []
        self.ears_left: List[float] = []
        self.ears_right: List[float] = []
        self.mars: List[float] = []
        self.yaws: List[float] = []
        self.pitches: List[float] = []
        self.rolls: List[float] = []
        self.cnn_counts = {"left": Counter(), "right": Counter()}   # per-eye OPEN/CLOSED counts on VALID frames
        self.cnn_confidences: List[float] = []
        self.cnn_times: List[float] = []
        self.crops_total = 0
        self.crops_valid = 0
        self.eye_widths: List[float] = []

        # event stream for UIs (transitions, alerts, driver actions, link changes)
        self.events: List[Dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._transitions_seen = 0
        self._pending_alert_events: List[Dict[str, Any]] = []
        self._link_was_connected: Optional[bool] = None

    # -- configuration report -----------------------------------------------------------
    def config_dict(self) -> Dict[str, Any]:
        """Thresholds in force, for the session log and the dashboard."""
        tc = asdict(self.temporal.config)
        out = {"temporal": tc, "validity": asdict(self.validity_config), "eye": asdict(self.eye_config),
               "pose_method": self.pose_config.method,
               "cnn": (str(getattr(self.classifier, "path", "model")) if self.classifier is not None
                       else "off ({})".format(self.cnn_status or "disabled")),
               "serial_port": self.serial_port or "off"}
        if self.alerter is not None:
            ac = self.alerter.config
            out["alerts"] = {"beep_interval_s": ac.beep_interval_s, "voice_after_s": ac.voice_after_s,
                             "voice_closure_s": ac.voice_closure_s, "voice_cooldown_s": ac.voice_cooldown_s,
                             "dismiss_s": ac.dismiss_s, "mild_beep_on_entry": ac.mild_beep_on_entry,
                             "audio": ac.audio, "tts": ac.tts_backend}
        else:
            out["alerts"] = "off"
        return out

    def describe(self) -> List[str]:
        """Start-up banner lines (same text run_demo printed before Stage 12)."""
        vc, ec, tc = self.validity_config, self.eye_config, self.temporal.config
        lines = ["Pose     : method {}".format(self.pose_config.method),
                 "Validity : |yaw| <= {} deg, |pitch| {}, face >= {} px, eye >= {} px, edge margin {} px, "
                 "window {} s".format(vc.max_abs_yaw_deg,
                                      "off" if vc.max_abs_pitch_deg is None else "<= {} deg".format(vc.max_abs_pitch_deg),
                                      vc.min_face_width_px, vc.min_eye_width_px, vc.edge_margin_px, vc.window_seconds),
                 "Eyes     : {}x{} crops, scale {} x eye width, roll alignment {}, CLAHE {}".format(
                     ec.size, ec.size, ec.crop_scale, "on" if ec.align_roll else "off", "on" if ec.equalize else "off")]
        if self.classifier is not None:
            meta = self.classifier.payload.get("metadata", {})
            lines.append("CNN      : {} | classes {} | trained {} | test acc {}".format(
                self.classifier.payload.get("format"), "/".join(self.classifier.classes),
                self.classifier.payload.get("saved_at", "?"),
                "{:.4f}".format(meta["test_accuracy"]) if "test_accuracy" in meta else "n/a"))
        else:
            lines.append("CNN      : {}".format(self.cnn_status or "disabled"))
        lines.append("Temporal : {:.0f} s window | eyes {} | PERCLOS mild {:.2f}/{:.2f} drowsy {:.2f}/{:.2f} "
                     "(enter/exit) | microsleep {:.1f} s | yawn MAR >= {:.2f} for {:.1f} s | nod drop {:.0f} deg | "
                     "dwell up {:.0f} s down {:.0f} s".format(
                         tc.window_s, tc.fusion, tc.perclos_mild_enter, tc.perclos_mild_exit, tc.perclos_drowsy_enter,
                         tc.perclos_drowsy_exit, tc.microsleep_s, tc.mar_yawn_thr, tc.min_yawn_s, tc.nod_drop_deg,
                         tc.up_dwell_s, tc.down_dwell_s))
        if self.alerter is not None:
            ac = self.alerter.config
            lines.append("Alerts   : MILD -> visual{} | DROWSY -> beep every {:.0f} s | voice after {:.0f} s in "
                         "DROWSY or {:.1f} s closed eyes, repeat every {:.0f} s | key d mutes audio {:.0f} s | audio {} | "
                         "beep {} | tts {} | log {}".format(
                             " + soft beep" if ac.mild_beep_on_entry else "", ac.beep_interval_s, ac.voice_after_s,
                             ac.voice_closure_s, ac.voice_cooldown_s, ac.dismiss_s, "on" if ac.audio else "OFF (--mute)",
                             self.alerter.audio.beeper_name, self.alerter.audio.speaker.backend, self.alerter.log.path))
        else:
            lines.append("Alerts   : disabled (--no-alerts)")
        lines.append("ESP32    : {}".format(
            "port {} | words ALERT/MILD/DROWSY/CLEAR | heartbeat 1 s | d mute -> CLEAR".format(self.serial_port)
            if self.buzzer is not None else "disabled (--no-serial)"))
        if self.session_log is not None:
            lines.append("Session  : SQLite {} | metrics every {:.1f} s | events: transitions, alerts, driver, link".format(
                self.session_log.path, self.session_log.metrics_interval_s))
        else:
            lines.append("Session  : not logged (--no-db)")
        return lines

    # -- session ----------------------------------------------------------------------
    def start_session(self, source_description: str = "") -> Optional[int]:
        self.source_description = source_description or self.source_description
        if self.session_log is not None and not self.session_started:
            self.session_started = True
            return self.session_log.start(self.source_description, self.config_dict(), wall=self.started_wall)
        return self.session_log.session_id if self.session_log is not None else None

    # -- per frame ----------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.session_started:
            self.start_session()
        wall = time.time()
        self.frame_index += 1
        detector = self.detector

        face = detector.process(frame)
        feats = compute_features(face) if face is not None else None
        pose = estimate_pose(face, self.pose_config) if face is not None else None
        assessment = assess_frame(face, feats, pose, self.validity_config, detector.last_face_count)
        self.tracker.update(assessment)
        crops = extract_eye_crops(frame, face, self.eye_config) if face is not None else (None, None)
        for crop in crops:
            if crop is not None:
                self.crops_total += 1
                self.crops_valid += int(crop.valid)
                self.eye_widths.append(crop.eye_width_px)

        # Stage 8: eye-state CNN on both crops in one forward pass. Computed
        # whenever the crops exist so the tester can watch it; counted as a
        # result only on VALID frames.
        eye_states: Optional[List] = None
        cnn_ms = 0.0
        if self.classifier is not None and face is not None:
            eye_states, cnn_ms = self.classifier.predict_crops(crops)
            self.cnn_times.append(cnn_ms)
            if assessment.valid:
                for side, state in zip(("left", "right"), eye_states):
                    if state is not None:
                        self.cnn_counts[side][state[0]] += 1
                        self.cnn_confidences.append(state[1])

        # Stage 9: feed the temporal layer. Invalid frames go in flagged so the
        # window can count them; their measurements are ignored inside.
        closed_probs = [(st[1] if st[0] == "CLOSED" else 1.0 - st[1]) for st in (eye_states or []) if st]
        observation = Observation(
            t=assessment.time_s, valid=assessment.valid,
            ear=feats.ear_mean if feats is not None else None,
            mar=feats.mar if feats is not None else None,
            pitch_deg=pose.pitch_deg if (pose is not None and pose.ok) else None,
            cnn_closed_prob=float(np.mean(closed_probs)) if closed_probs else None)
        if self.started_t is None:
            self.started_t = observation.t
        previous_state = self.temporal_state.state if self.temporal_state is not None else None
        self.temporal_state = self.temporal.update(observation)
        if previous_state is not None:
            self.time_in_state[previous_state] += 1
        transitions = self.temporal.transitions[self._transitions_seen:]
        self._transitions_seen = len(self.temporal.transitions)
        for tr_t, from_state, to_state, why in transitions:
            if self.echo:
                print("[temporal] {:7.1f} s  {} -> {}  ({})".format(
                    tr_t - self.temporal._frames[0][0] if self.temporal._frames else 0.0, from_state, to_state, why))

        # Stage 10: alerts follow the state; update() never blocks (audio is on its own thread).
        self._pending_alert_events = []
        if self.alerter is not None:
            self.alert_status = self.alerter.update(self.temporal_state)
        alert_events = self._pending_alert_events
        # Stage 11
        if self.buzzer is not None:
            self.buzzer_word = word_for(self.temporal_state, self.alert_status)
            self.buzzer.set_state(self.buzzer_word)

        fps = self.fps_counter.tick()
        inference_ms = detector.stats.inference_ms[-1] if detector.stats.inference_ms else 0.0
        if feats is not None:
            self.ears.append(feats.ear_mean)
            self.ears_left.append(feats.ear_left)
            self.ears_right.append(feats.ear_right)
            self.mars.append(feats.mar)
        if pose is not None and pose.ok:
            self.yaws.append(pose.yaw_deg)
            self.pitches.append(pose.pitch_deg)
            self.rolls.append(pose.roll_deg)

        result = FrameResult(frame_index=self.frame_index, t=observation.t, wall=wall, face=face, feats=feats,
                             pose=pose, assessment=assessment, crops=crops, eye_states=eye_states, cnn_ms=cnn_ms,
                             observation=observation, temporal_state=self.temporal_state, alert_status=self.alert_status,
                             buzzer_word=self.buzzer_word, transitions=transitions, alert_events=alert_events,
                             fps=fps, inference_ms=inference_ms)
        self.last_result = result
        self._record(result)
        return result

    # -- driver actions -------------------------------------------------------------------
    def dismiss(self) -> None:
        """Mute the alert audio (Stage 10 'd'). The buzzer follows via word_for() -> CLEAR."""
        if self.alerter is None:
            return
        self.alerter.dismiss()
        self._event("driver", "mute", detail="alert audio muted {:.0f} s".format(self.alerter.config.dismiss_s))

    def reset(self) -> None:
        """Full reset (Stage 10 'r' / DISMISS button): alert cleared, state machine back to
        ALERT with an empty window. A still-drowsy driver is re-detected within seconds."""
        t = self.last_result.t if self.last_result is not None else 0.0
        if self.alerter is not None:
            self.alerter.reset()
        self.temporal.reset(t)
        if self.echo:
            print("[features] {:7.1f} s  alert reset by driver -> state ALERT, window cleared".format(
                t - self.started_t if self.started_t is not None else 0.0))
        self._event("driver", "reset", detail="alert cleared, state -> ALERT, window cleared")

    # -- shutdown -----------------------------------------------------------------------------
    def close(self) -> Dict[str, Any]:
        summary = self.summary()
        if self.buzzer is not None:
            self.buzzer.close()                       # sends CLEAR, then closes the port
        if self.alerter is not None:
            self.alerter.close()
        if self.session_log is not None and self.session_started:
            self.session_log.end(frames=self.frame_index, duration_s=summary["duration_s"], summary=summary)
            self.session_log.close()
        return summary

    # -- reports -------------------------------------------------------------------------------
    def metrics_row(self, r: FrameResult) -> Dict[str, Any]:
        """One flat dict of every live metric (the session log's metrics table,
        the dashboard's tiles)."""
        ts, feats, pose, a = r.temporal_state, r.feats, r.pose, r.alert_status
        pose_ok = pose is not None and pose.ok
        left = r.eye_states[0] if r.eye_states else None
        right = r.eye_states[1] if r.eye_states else None
        return {
            "state": ts.state, "sufficient": int(ts.sufficient), "valid": int(r.assessment.valid),
            "face_found": int(r.face is not None),
            "ear": feats.ear_mean if feats is not None else None,
            "ear_left": feats.ear_left if feats is not None else None,
            "ear_right": feats.ear_right if feats is not None else None,
            "mar": feats.mar if feats is not None else None,
            "yaw_deg": pose.yaw_deg if pose_ok else None, "pitch_deg": pose.pitch_deg if pose_ok else None,
            "roll_deg": pose.roll_deg if pose_ok else None,
            "cnn_closed_prob": r.observation.cnn_closed_prob,
            "cnn_left": left, "cnn_right": right,
            "perclos": ts.perclos, "blink_rate_per_min": ts.blink_rate_per_min, "mean_blink_s": ts.mean_blink_s,
            "closure_now_s": ts.closure_now_s, "longest_closure_s": ts.longest_closure_s,
            "yawn_count": ts.yawn_count, "yawn_rate_per_min": ts.yawn_rate_per_min, "nod_count": ts.nod_count,
            "invalid_rate": ts.invalid_rate, "fps": r.fps, "inference_ms": r.inference_ms, "cnn_ms": r.cnn_ms,
            "alert_level": a.level_name if a is not None else None,
            "alert_dismissed": int(a.dismissed) if a is not None else None,
            "buzzer_word": r.buzzer_word,
            "esp32_connected": int(self.buzzer.connected) if self.buzzer is not None else None,
        }

    def summary(self) -> Dict[str, Any]:
        """Session summary: what the console prints at exit and what the
        dashboard shows. Every number here is measured in this run."""
        elapsed = time.perf_counter() - self.started_perf
        stats = self.detector.stats
        total = sum(self.time_in_state.values()) or 1
        ts = self.temporal_state
        out: Dict[str, Any] = {
            "frames": self.frame_index, "duration_s": round(elapsed, 2),
            "fps_end_to_end": round(self.frame_index / elapsed, 2) if elapsed else 0.0,
            "face_frames": stats.faces, "face_rate": round(stats.detection_rate, 4),
            "inference_median_ms": round(stats.median_inference_ms, 2),
            "invalid_session_rate": round(self.tracker.session_rate, 4),
            "invalid_reasons": dict(self.tracker.reasons.most_common()),
            "time_in_state": {name: round(self.time_in_state[name] / total, 4) for name in ("ALERT", "MILD", "DROWSY")},
            "seconds_in_state": {name: round(self.time_in_state[name] / total * elapsed, 1)
                                 for name in ("ALERT", "MILD", "DROWSY")},
            "transitions": len(self.temporal.transitions),
            "transition_list": [(round(t - self.started_t, 1) if self.started_t is not None else None, a, b, why)
                                for t, a, b, why in self.temporal.transitions],
            "event_totals": dict(self.temporal.event_totals),
            "final_state": ts.state if ts is not None else None,
            "final_perclos": round(ts.perclos, 4) if ts is not None else None,
            "final_window": (ts.as_row() if ts is not None else None),
        }
        for name, values in (("ear", self.ears), ("mar", self.mars), ("yaw_deg", self.yaws),
                             ("pitch_deg", self.pitches), ("roll_deg", self.rolls)):
            if values:
                arr = np.asarray(values, dtype=float)
                out[name] = {"median": round(float(np.median(arr)), 4), "p5": round(float(np.percentile(arr, 5)), 4),
                             "p95": round(float(np.percentile(arr, 95)), 4), "std": round(float(arr.std()), 4)}
        if self.classifier is not None and self.cnn_times:
            out["cnn"] = {"left": dict(self.cnn_counts["left"]), "right": dict(self.cnn_counts["right"]),
                          "mean_confidence": round(float(np.mean(self.cnn_confidences)), 4) if self.cnn_confidences else None,
                          "median_ms": round(float(np.median(self.cnn_times)), 2)}
        if self.alerter is not None:
            counts = Counter(event for _, event, _ in self.alerter.events)
            out["alert_events"] = sum(counts.values())
            out["alert_counts"] = dict(counts)
            out["alert_log"] = str(self.alerter.log.path) if self.alerter.log.rows else None
        if self.buzzer is not None:
            st = self.buzzer.stats
            out["esp32"] = {"port": self.buzzer.port, "connects": st.connects, "disconnects": st.disconnects,
                            "sent": st.sent, "acks": st.acks, "errors": st.errors, "ready": st.ready_line}
        if self.session_log is not None and self.session_started:
            out["session_id"] = self.session_log.session_id
            out["session_db"] = str(self.session_log.path)
        return out

    def recent_events(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._events_lock:
            return list(self.events[-limit:])

    # -- internals -----------------------------------------------------------------------------
    def _on_alert_event(self, ts: TemporalState, event: str, level: int, detail: str) -> None:
        self._pending_alert_events.append({"t": ts.t, "event": event, "level": LEVEL_NAMES[level], "detail": detail,
                                           "state": ts.state, "perclos": ts.perclos, "closure_now_s": ts.closure_now_s})

    def _event(self, kind: str, event: str, detail: str = "", t: Optional[float] = None, wall: Optional[float] = None,
               **extra: Any) -> None:
        r = self.last_result
        t = t if t is not None else (r.t if r is not None else 0.0)
        wall = wall if wall is not None else time.time()
        ts = r.temporal_state if r is not None else None
        feats = r.feats if r is not None else None
        pose = r.pose if (r is not None and r.pose is not None and r.pose.ok) else None
        row = {"kind": kind, "event": event, "detail": detail, "wall": wall,
               "t": round(t - self.started_t, 3) if self.started_t is not None else 0.0,
               "state": ts.state if ts is not None else None,
               "perclos": ts.perclos if ts is not None else None,
               "closure_now_s": ts.closure_now_s if ts is not None else None}
        row.update(extra)
        with self._events_lock:
            self.events.append(row)
        if self.session_log is not None:
            self.session_log.log_event(
                kind, event, t=t, wall=wall, from_state=extra.get("from_state"), to_state=extra.get("to_state"),
                level=extra.get("level"), detail=detail, state=row["state"], perclos=row["perclos"],
                closure_now_s=row["closure_now_s"], ear=feats.ear_mean if feats is not None else None,
                mar=feats.mar if feats is not None else None, yaw_deg=pose.yaw_deg if pose is not None else None,
                pitch_deg=pose.pitch_deg if pose is not None else None)

    def _record(self, r: FrameResult) -> None:
        for t, from_state, to_state, why in r.transitions:
            self._event("transition", "{}->{}".format(from_state, to_state), detail=why, t=t, wall=r.wall,
                        from_state=from_state, to_state=to_state)
        for ev in r.alert_events:
            self._event("alert", ev["event"], detail=ev["detail"], t=ev["t"], wall=r.wall, level=ev["level"])
        if self.buzzer is not None and self.buzzer.connected != self._link_was_connected:
            if self._link_was_connected is not None or self.buzzer.connected:
                self._event("link", "connected" if self.buzzer.connected else "disconnected",
                            detail=self.buzzer.status_text(), t=r.t, wall=r.wall)
            self._link_was_connected = self.buzzer.connected
        if self.session_log is not None:
            row = self.metrics_row(r)
            self.session_log.log_metrics(r.t, row, wall=r.wall)


# --- building the pieces from plain settings -------------------------------------------------

DEFAULT_SETTINGS: Dict[str, Any] = {
    "device": "0", "width": 640, "height": 480, "gray": False,
    "no_cnn": False, "model": str(MODEL_PATH), "fusion": "fused",
    "perclos_mild": 0.15, "perclos_drowsy": 0.30, "microsleep": 1.5, "yawn_mar": 0.60, "window": 60.0,
    "pose_method": "matrix", "max_yaw": 30.0, "max_pitch": 0.0, "min_face_px": 80, "min_eye_px": 15.0,
    "eye_size": 64, "crop_scale": 1.5, "align": True,
    "alerts": True, "mute": False, "tts": "auto", "beep_interval": 3.0, "voice_after": 6.0, "voice_closure": 2.0,
    "voice_cooldown": 15.0, "dismiss": 30.0, "mild_beep": True, "alert_log": None,
    "serial": "auto", "db": "logs/sessions.db", "metrics_interval": 1.0,
    "mirror": True, "draw_landmarks": True, "max_frames": 0,
}


def build_configs(settings: Dict[str, Any]):
    """settings dict -> (camera, pose, validity, eye, temporal, alert) configs.
    The same derivations src.features.main uses."""
    s = {**DEFAULT_SETTINGS, **settings}
    device = int(s["device"]) if str(s["device"]).isdigit() else str(s["device"])
    camera = CameraConfig(device=device, width=int(s["width"]), height=int(s["height"]), flip_horizontal=False)
    pose = PoseConfig(method=s["pose_method"])
    validity = ValidityConfig(max_abs_yaw_deg=float(s["max_yaw"]),
                              max_abs_pitch_deg=float(s["max_pitch"]) if float(s["max_pitch"]) > 0 else None,
                              min_face_width_px=int(s["min_face_px"]), min_eye_width_px=float(s["min_eye_px"]),
                              window_seconds=float(s["window"]))
    eye = EyePreprocessConfig(size=int(s["eye_size"]), crop_scale=float(s["crop_scale"]), align_roll=bool(s["align"]),
                              min_eye_width_px=float(s["min_eye_px"]))
    temporal = TemporalConfig(
        window_s=float(s["window"]), fusion=s["fusion"], perclos_mild_enter=float(s["perclos_mild"]),
        perclos_mild_exit=round(float(s["perclos_mild"]) * 2 / 3, 3), perclos_drowsy_enter=float(s["perclos_drowsy"]),
        perclos_drowsy_exit=round(float(s["perclos_drowsy"]) * 0.73, 3), microsleep_s=float(s["microsleep"]),
        mar_yawn_thr=float(s["yawn_mar"]))
    alert = AlertConfig(beep_interval_s=float(s["beep_interval"]), voice_after_s=float(s["voice_after"]),
                        voice_closure_s=float(s["voice_closure"]), voice_cooldown_s=float(s["voice_cooldown"]),
                        dismiss_s=float(s["dismiss"]), mild_beep_on_entry=bool(s["mild_beep"]), tts_backend=s["tts"],
                        audio=not bool(s["mute"]), log_path=Path(s["alert_log"]) if s["alert_log"] else None)
    return s, camera, pose, validity, eye, temporal, alert


def load_classifier(settings: Dict[str, Any], eye: EyePreprocessConfig) -> Tuple[Optional[EyeStateClassifier], str]:
    """Same rules as src.features.main: a missing model or missing torch is a
    status message, a size mismatch is an error."""
    if settings.get("no_cnn"):
        return None, "disabled (--no-cnn)"
    try:
        classifier = EyeStateClassifier(Path(settings["model"]))
    except FileNotFoundError as exc:
        return None, "no model at {} - running without it".format(settings["model"])
    except ImportError:
        return None, "PyTorch missing - running without it"
    if classifier.size != eye.size:
        raise ValueError("model expects {0}x{0} crops but eye size is {1}".format(classifier.size, eye.size))
    if classifier.config.equalize != eye.equalize:
        eye.equalize = classifier.config.equalize            # the model decides how crops are prepared
    return classifier, ""


# --- threaded runner for UIs -----------------------------------------------------------------

@dataclass
class Snapshot:
    """What a UI needs per redraw. Plain values only - no OpenCV objects."""

    wall: float
    t: float
    elapsed_s: float
    frame_index: int
    jpeg: bytes
    state: str
    state_since_s: float
    sufficient: bool
    reasons: List[str]
    ear: Optional[float]
    ear_left: Optional[float]
    ear_right: Optional[float]
    mar: Optional[float]
    perclos: float
    blink_rate_per_min: float
    mean_blink_s: float
    closure_now_s: float
    longest_closure_s: float
    yawn_count: int
    yawn_rate_per_min: float
    nod_count: int
    yaw_deg: Optional[float]
    pitch_deg: Optional[float]
    roll_deg: Optional[float]
    pose_method: str
    face_found: bool
    faces_in_frame: int
    valid: bool
    invalid_reasons: List[str]
    invalid_rate_window: float
    invalid_rate_session: float
    fps: float
    inference_ms: float
    cnn_enabled: bool
    cnn_left: Optional[Tuple[str, float]]
    cnn_right: Optional[Tuple[str, float]]
    cnn_closed_prob: Optional[float]
    cnn_ms: float
    alert_level: str
    alert_message: str
    alert_dismissed: bool
    alert_dismiss_left_s: float
    beeps: int
    voices: int
    buzzer_word: Optional[str]
    esp32_connected: bool
    esp32_text: str
    session_id: Optional[int]
    db_path: Optional[str]


class PipelineWorker:
    """Runs the camera + pipeline on a daemon thread; UIs poll ``snapshot()``.

    The camera, detector, classifier and pipeline are all created *inside* the
    thread, so nothing OpenCV/MediaPipe-related is ever touched from a UI
    thread. ``start()`` returns at once; ``stop()`` joins.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dismiss = threading.Event()
        self._reset = threading.Event()
        self.settings: Dict[str, Any] = {}
        self.latest: Optional[Snapshot] = None
        self.error: Optional[str] = None
        self.pipeline: Optional[DrowsinessPipeline] = None
        self.history: Deque[Tuple[float, Optional[float], Optional[float], float, str, float]] = deque(maxlen=6000)
        self.final_summary: Optional[Dict[str, Any]] = None
        self.started_wall: Optional[float] = None
        self.stopped_wall: Optional[float] = None
        self.stop_reason: str = ""
        self.banner: List[str] = []

    # -- control ------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings: Dict[str, Any]) -> bool:
        if self.running:
            return False
        self.settings = {**DEFAULT_SETTINGS, **settings}
        self._stop.clear(); self._dismiss.clear(); self._reset.clear()
        with self._lock:
            self.latest = None
            self.history.clear()
        self.error = None
        self.final_summary = None
        self.stop_reason = ""
        self.banner = []
        self.started_wall = time.time()
        self.stopped_wall = None
        self._thread = threading.Thread(target=self._run, name="pipeline-worker", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def dismiss(self) -> None:
        self._dismiss.set()

    def reset(self) -> None:
        self._reset.set()

    def snapshot(self) -> Optional[Snapshot]:
        with self._lock:
            return self.latest

    def history_rows(self, max_points: int = 600) -> List[Tuple[float, Optional[float], Optional[float], float, str, float]]:
        with self._lock:
            rows = list(self.history)
        if len(rows) > max_points:
            step = len(rows) / max_points
            rows = [rows[int(i * step)] for i in range(max_points)]
        return rows

    def events(self, limit: int = 200) -> List[Dict[str, Any]]:
        p = self.pipeline
        return p.recent_events(limit) if p is not None else []

    def summary(self) -> Optional[Dict[str, Any]]:
        if self.final_summary is not None:
            return self.final_summary
        p = self.pipeline
        return p.summary() if (p is not None and self.running) else None

    # -- the thread ---------------------------------------------------------------------------
    def _run(self) -> None:
        pipeline: Optional[DrowsinessPipeline] = None
        source: Optional[FrameSource] = None
        detector: Optional[FaceLandmarkDetector] = None
        try:
            s, camera, pose, validity, eye, temporal, alert = build_configs(self.settings)
            classifier, cnn_status = load_classifier(s, eye)
            detector = FaceLandmarkDetector(LandmarkConfig(grayscale_input=bool(s["gray"])))
            source = create_source(camera)
            session_log = (SessionLogger(Path(s["db"]), metrics_interval_s=float(s["metrics_interval"]))
                           if s.get("db") else None)
            pipeline = DrowsinessPipeline(detector, classifier=classifier, pose_config=pose, validity_config=validity,
                                          eye_config=eye, temporal_config=temporal, alert_config=alert,
                                          alerts=bool(s["alerts"]), serial_port=s["serial"] or None,
                                          session_log=session_log, cnn_status=cnn_status, echo=False)
            self.pipeline = pipeline
            started = time.perf_counter()
            with source, detector:
                pipeline.start_session(source.description)
                self.banner = ["Camera   : {}".format(source.description)] + pipeline.describe()
                while not self._stop.is_set():
                    frame = source.read()
                    if frame is None:
                        self.stop_reason = "stream ended or camera lost"
                        break
                    result = pipeline.process(frame)
                    if self._dismiss.is_set():
                        self._dismiss.clear()
                        pipeline.dismiss()
                    if self._reset.is_set():
                        self._reset.clear()
                        pipeline.reset()
                    display = self._annotate(frame, result, pipeline, detector, s)
                    ok, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    snap = self._snapshot(result, pipeline, detector, buf.tobytes() if ok else b"",
                                          time.perf_counter() - started, classifier is not None)
                    with self._lock:
                        self.latest = snap
                        ts = result.temporal_state
                        self.history.append((round(snap.elapsed_s, 2), snap.ear, snap.mar, ts.perclos, ts.state,
                                             ts.closure_now_s))
                    if s["max_frames"] and result.frame_index >= int(s["max_frames"]):
                        self.stop_reason = "reached max_frames {}".format(s["max_frames"])
                        break
                if not self.stop_reason:
                    self.stop_reason = "stopped by user"
        except (CameraError, LandmarkModelError, ValueError) as exc:
            self.error = "{}: {}".format(type(exc).__name__, exc)
            self.stop_reason = "error"
        except Exception:                                   # keep the traceback for the dashboard
            self.error = traceback.format_exc()
            self.stop_reason = "error"
        finally:
            if pipeline is not None:
                try:
                    self.final_summary = pipeline.close()
                except Exception:
                    self.error = (self.error or "") + "\n" + traceback.format_exc()
            self.stopped_wall = time.time()

    @staticmethod
    def _annotate(frame: np.ndarray, r: FrameResult, pipeline: DrowsinessPipeline, detector: FaceLandmarkDetector,
                  s: Dict[str, Any]) -> np.ndarray:
        display = frame.copy()
        mirror = bool(s["mirror"])
        if r.face is not None and s["draw_landmarks"]:
            draw_landmarks(display, r.face, "contours")
            draw_feature_geometry(display, r.face)
            draw_eye_boxes(display, r.crops)
        if mirror:
            display = cv2.flip(display, 1)
        draw_ignored_faces(display, detector.last_ignored_boxes, mirror)
        if r.face is not None and r.pose is not None:
            draw_pose(display, r.face, r.pose, mirror)
        ts = r.temporal_state
        color = STATE_COLORS[ts.state] if ts.sufficient else (160, 160, 160)
        label = "{}  {:.0f} s".format(ts.state, max(0.0, ts.t - ts.state_since))
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        if not r.assessment.valid:
            reason = "INVALID: " + ", ".join(r.assessment.reasons)[:60]
            cv2.putText(display, reason, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, reason, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1, cv2.LINE_AA)
        if pipeline.alerter is not None:
            draw_alert_overlay(display, r.alert_status, pipeline.alerter.config.flash_hz)
        return display

    @staticmethod
    def _snapshot(r: FrameResult, pipeline: DrowsinessPipeline, detector: FaceLandmarkDetector, jpeg: bytes,
                  elapsed: float, cnn_enabled: bool) -> Snapshot:
        ts, feats, pose, a = r.temporal_state, r.feats, r.pose, r.alert_status
        pose_ok = pose is not None and pose.ok
        buzzer = pipeline.buzzer
        log = pipeline.session_log
        return Snapshot(
            wall=r.wall, t=r.t, elapsed_s=elapsed, frame_index=r.frame_index, jpeg=jpeg,
            state=ts.state, state_since_s=max(0.0, ts.t - ts.state_since), sufficient=ts.sufficient,
            reasons=list(ts.reasons),
            ear=feats.ear_mean if feats is not None else None, ear_left=feats.ear_left if feats is not None else None,
            ear_right=feats.ear_right if feats is not None else None, mar=feats.mar if feats is not None else None,
            perclos=ts.perclos, blink_rate_per_min=ts.blink_rate_per_min, mean_blink_s=ts.mean_blink_s,
            closure_now_s=ts.closure_now_s, longest_closure_s=ts.longest_closure_s, yawn_count=ts.yawn_count,
            yawn_rate_per_min=ts.yawn_rate_per_min, nod_count=ts.nod_count,
            yaw_deg=pose.yaw_deg if pose_ok else None, pitch_deg=pose.pitch_deg if pose_ok else None,
            roll_deg=pose.roll_deg if pose_ok else None, pose_method=pipeline.pose_config.method,
            face_found=r.face is not None, faces_in_frame=detector.last_face_count, valid=r.assessment.valid,
            invalid_reasons=list(r.assessment.reasons), invalid_rate_window=pipeline.tracker.window_rate,
            invalid_rate_session=pipeline.tracker.session_rate, fps=r.fps, inference_ms=r.inference_ms,
            cnn_enabled=cnn_enabled,
            cnn_left=tuple(r.eye_states[0]) if (r.eye_states and r.eye_states[0]) else None,
            cnn_right=tuple(r.eye_states[1]) if (r.eye_states and r.eye_states[1]) else None,
            cnn_closed_prob=r.observation.cnn_closed_prob, cnn_ms=r.cnn_ms,
            alert_level=a.level_name if a is not None else "OFF", alert_message=a.message if a is not None else "",
            alert_dismissed=a.dismissed if a is not None else False,
            alert_dismiss_left_s=(a.dismissed_until - a.t) if (a is not None and a.dismissed) else 0.0,
            beeps=a.beeps_this_episode if a is not None else 0, voices=a.voices_this_episode if a is not None else 0,
            buzzer_word=r.buzzer_word, esp32_connected=bool(buzzer.connected) if buzzer is not None else False,
            esp32_text=buzzer.status_text() if buzzer is not None else "ESP32 link off",
            session_id=log.session_id if log is not None else None, db_path=str(log.path) if log is not None else None)



# --- separate-process runner for UIs ------------------------------------------------------------

def _worker_process_main(settings: Dict[str, Any], cmd_q, out_q, snapshot_hz: float) -> None:
    """Child-process entry: run a PipelineWorker and stream its state to the parent."""
    import queue as _queue
    worker = PipelineWorker()
    worker.start(settings)
    last_frame, last_events, last_periodic, banner_sent = -1, 0, 0.0, False
    min_gap = 1.0 / max(snapshot_hz, 1.0)
    last_snap_sent = 0.0
    while True:
        try:
            while True:
                cmd = cmd_q.get_nowait()
                if cmd == "stop":
                    worker.stop()
                elif cmd == "dismiss":
                    worker.dismiss()
                elif cmd == "reset":
                    worker.reset()
        except _queue.Empty:
            pass
        if worker.banner and not banner_sent:
            out_q.put(("banner", list(worker.banner)))
            banner_sent = True
        now = time.time()
        snap = worker.snapshot()
        if snap is not None and snap.frame_index != last_frame and now - last_snap_sent >= min_gap:
            out_q.put(("snapshot", snap))
            last_frame, last_snap_sent = snap.frame_index, now
        events = worker.events(100000)
        if len(events) > last_events:
            out_q.put(("events", events[last_events:]))
            last_events = len(events)
        if now - last_periodic >= 1.0:
            last_periodic = now
            out_q.put(("history", worker.history_rows(300)))
            summary = worker.summary()
            if summary is not None:
                out_q.put(("summary", summary))
        if not worker.running:
            break
        time.sleep(0.02)
    if worker.banner and not banner_sent:
        out_q.put(("banner", list(worker.banner)))
    events = worker.events(100000)
    if len(events) > last_events:
        out_q.put(("events", events[last_events:]))
    out_q.put(("history", worker.history_rows(300)))
    out_q.put(("final", {"summary": worker.final_summary, "error": worker.error, "stop_reason": worker.stop_reason}))


class PipelineProcess:
    """Same interface as PipelineWorker, but the camera + pipeline run in a
    separate *process*. Measured reason (2026-09-13, this laptop): in one
    process a Python thread that hogs the interpreter lock starves MediaPipe -
    10 frames in 6 s against 32 FPS alone - and Streamlit's page reruns are
    exactly that kind of load (6 FPS / 118 ms inference in the first in-process
    dashboard run, 24 ms headless). A process boundary makes the dashboard
    unable to slow the detector. State flows parent-ward through a queue; the
    parent keeps a small reader thread that only stores what arrives."""

    def __init__(self, snapshot_hz: float = 10.0) -> None:
        import multiprocessing as mp
        self._mp = mp.get_context("spawn")
        self.snapshot_hz = snapshot_hz
        self._proc = None
        self._cmd_q = None
        self._out_q = None
        self._reader: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.settings: Dict[str, Any] = {}
        self.latest: Optional[Snapshot] = None
        self.error: Optional[str] = None
        self.stop_reason: str = ""
        self.banner: List[str] = []
        self.final_summary: Optional[Dict[str, Any]] = None
        self._summary: Optional[Dict[str, Any]] = None
        self._history: List[Tuple] = []
        self._events: List[Dict[str, Any]] = []
        self.started_wall: Optional[float] = None
        self.stopped_wall: Optional[float] = None
        self._finished = threading.Event()

    # -- control -------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive() and not self._finished.is_set()

    def start(self, settings: Dict[str, Any]) -> bool:
        if self.running:
            return False
        self.settings = {**DEFAULT_SETTINGS, **settings}
        with self._lock:
            self.latest, self._summary, self._history, self._events = None, None, [], []
        self.error, self.stop_reason, self.banner, self.final_summary = None, "", [], None
        self._finished.clear()
        self.started_wall, self.stopped_wall = time.time(), None
        self._cmd_q, self._out_q = self._mp.Queue(), self._mp.Queue()
        self._proc = self._mp.Process(target=_worker_process_main, name="pipeline-process",
                                      args=(self.settings, self._cmd_q, self._out_q, self.snapshot_hz), daemon=True)
        self._proc.start()
        self._reader = threading.Thread(target=self._read_loop, name="pipeline-process-reader", daemon=True)
        self._reader.start()
        return True

    def stop(self, timeout: float = 15.0) -> None:
        if self._proc is None:
            return
        if self._proc.is_alive():
            try:
                self._cmd_q.put("stop")
            except Exception:
                pass
        self._finished.wait(timeout)
        self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)
            if not self.error:
                self.error = "the detector process did not stop within {:.0f} s and was terminated".format(timeout)
        self._finished.set()
        self.stopped_wall = self.stopped_wall or time.time()

    def dismiss(self) -> None:
        if self.running:
            self._cmd_q.put("dismiss")

    def reset(self) -> None:
        if self.running:
            self._cmd_q.put("reset")

    # -- reads ------------------------------------------------------------------------------------
    def snapshot(self) -> Optional[Snapshot]:
        with self._lock:
            return self.latest

    def history_rows(self, max_points: int = 300) -> List[Tuple]:
        with self._lock:
            rows = list(self._history)
        return rows[-max_points:] if len(rows) > max_points else rows

    def events(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events[-limit:])

    def summary(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.final_summary if self.final_summary is not None else self._summary

    # -- reader thread -----------------------------------------------------------------------------
    def _read_loop(self) -> None:
        import queue as _queue
        proc, out_q = self._proc, self._out_q
        while True:
            try:
                kind, payload = out_q.get(timeout=0.5)
            except _queue.Empty:
                if not proc.is_alive():
                    if not self._finished.is_set():
                        self.error = self.error or "the detector process ended unexpectedly (exit code {})".format(
                            proc.exitcode)
                        self.stop_reason = self.stop_reason or "error"
                        self.stopped_wall = time.time()
                        self._finished.set()
                    return
                continue
            except (EOFError, OSError):
                self._finished.set()
                return
            with self._lock:
                if kind == "snapshot":
                    self.latest = payload
                elif kind == "events":
                    self._events.extend(payload)
                elif kind == "history":
                    self._history = list(payload)
                elif kind == "summary":
                    self._summary = payload
                elif kind == "banner":
                    self.banner = list(payload)
                elif kind == "final":
                    self.final_summary = payload.get("summary")
                    self.error = payload.get("error")
                    self.stop_reason = payload.get("stop_reason") or "stopped"
                    self.stopped_wall = time.time()
            if kind == "final":
                self._finished.set()
                return


# --- headless command line -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 12: run the shared pipeline headless (the dashboard's backend).")
    p.add_argument("--device", default="0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--fusion", choices=("cnn", "ear", "fused"), default="fused")
    p.add_argument("--perclos-mild", type=float, default=0.15)
    p.add_argument("--perclos-drowsy", type=float, default=0.30)
    p.add_argument("--microsleep", type=float, default=1.5)
    p.add_argument("--min-face-px", type=int, default=80)
    p.add_argument("--min-eye-px", type=float, default=15.0)
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--no-serial", action="store_true")
    p.add_argument("--serial", default="auto")
    p.add_argument("--db", default="logs/sessions.db")
    p.add_argument("--no-db", action="store_true")
    p.add_argument("--metrics-interval", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--print-every", type=float, default=1.0, help="Seconds between status lines")
    p.add_argument("--process", action="store_true", help="Run through PipelineProcess (what the dashboard uses)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    worker = PipelineProcess() if a.process else PipelineWorker()
    worker.start({"device": a.device, "width": a.width, "height": a.height, "no_cnn": a.no_cnn, "model": str(a.model),
                  "fusion": a.fusion, "perclos_mild": a.perclos_mild, "perclos_drowsy": a.perclos_drowsy,
                  "microsleep": a.microsleep, "min_face_px": a.min_face_px, "min_eye_px": a.min_eye_px,
                  "alerts": not a.no_alerts, "mute": a.mute, "serial": None if a.no_serial else a.serial,
                  "db": None if a.no_db else a.db, "metrics_interval": a.metrics_interval, "max_frames": a.max_frames})
    last_print = 0.0
    printed_banner = False
    try:
        while worker.running:
            time.sleep(0.05)
            if not printed_banner and worker.banner:
                for line in worker.banner:
                    print("[pipeline] " + line)
                printed_banner = True
            snap = worker.snapshot()
            if snap is not None and time.time() - last_print >= a.print_every:
                last_print = time.time()
                print("[pipeline] {:6.1f} s  frame {:5d}  {:<6} PERCLOS {:5.1%}  EAR {}  MAR {}  yaw {} pitch {}  "
                      "FPS {:4.1f}  invalid {:4.0%}  alert {:<6}  {}  {}".format(
                          snap.elapsed_s, snap.frame_index, snap.state, snap.perclos,
                          "{:.3f}".format(snap.ear) if snap.ear is not None else "  -  ",
                          "{:.3f}".format(snap.mar) if snap.mar is not None else "  -  ",
                          "{:+.0f}".format(snap.yaw_deg) if snap.yaw_deg is not None else "  -",
                          "{:+.0f}".format(snap.pitch_deg) if snap.pitch_deg is not None else "  -",
                          snap.fps, snap.invalid_rate_window, snap.alert_level,
                          snap.buzzer_word or "", snap.esp32_text))
    except KeyboardInterrupt:
        print("\n[pipeline] interrupted")
        worker.stop()
    if worker.error:
        print("[pipeline] ERROR: {}".format(worker.error), file=sys.stderr)
    print("[pipeline] stopped: {}".format(worker.stop_reason))
    for ev in worker.events():
        print("[pipeline] event  t={:7.1f}s  {:<10} {:<14} {}".format(ev["t"], ev["kind"], ev["event"], ev["detail"]))
    print("[pipeline] summary: {}".format(json.dumps(worker.summary(), indent=1, default=str)))
    return 1 if worker.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
