"""Per-frame geometric features and frame validity (Stages 3 and 4).

Pipeline slice implemented here::

    FaceLandmarks (Stage 2) -> EAR (per eye) + MAR          [Stage 3]
                            -> head pose (src.headpose)     [Stage 4]
                            -> frame validity + invalid-frame rate [Stage 4]

Eye Aspect Ratio  (Soukupova & Cech, 2016)
------------------------------------------
Six landmarks per eye, p1..p6, in pixel coordinates::

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

p1 and p4 are the eye corners (horizontal width); p2, p3 lie on the upper lid
and p6, p5 directly below them on the lower lid. Two vertical lid gaps are
averaged and divided by the width, so the value is independent of the
distance to the camera and of in-plane head roll. It falls towards zero as the
eye closes.

MediaPipe face-mesh indices (subject's anatomy, un-mirrored frame):

    right eye  p1..p6 = 33, 160, 158, 133, 153, 144
    left eye   p1..p6 = 362, 385, 387, 263, 373, 380

Mouth Aspect Ratio
------------------
Mean of three inner-lip vertical gaps divided by the mouth width::

    MAR = mean(|p82 - p87|, |p13 - p14|, |p312 - p317|) / |p61 - p291|

13/14 are the inner-lip centres, 82/87 and 312/317 one step either side, and
61/291 are the mouth corners. Three pairs instead of one average out landmark
jitter. The value is ~0 with the lips touching and rises as the mouth opens.

Frame validity (Stage 4)
------------------------
A frame is INVALID - its EAR/MAR must not be interpreted as eye or mouth state
- when any of these hold:

* no face (or faces present but none inside the driver zone);
* the face is not reliably measurable: too small, partly outside the frame,
  degenerate landmark geometry, or eyes too narrow in pixels for EAR to be
  precise;
* head pose unavailable, or |yaw| beyond the configured limit (EAR collapses
  or inflates under yaw - see the Stage 3 live test). An optional |pitch|
  limit exists but is off by default: how pitch is treated (nods versus
  looking down) is Stage 9's decision.

``InvalidFrameTracker`` keeps the invalid-frame rate over a rolling window and
for the session, with a histogram of reasons. Every threshold here is an
INITIAL, configurable value - none is tuned yet. Record real sessions with
``--record`` and set them from measured distributions.

Run directly for a live demo::

    python -m src.features                       # live camera
    python -m src.features --record data/ear.csv # also log per-frame values
    python -m src.features --self-test           # no camera needed
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.capture import (CameraConfig, CameraError, FPSCounter, FrameSource,
                         create_source, save_snapshot)
from src.eye_cnn import (EyePreprocessConfig, draw_eye_boxes, draw_eye_panel,
                         extract_eye_crops, save_eye_crops)
from src.headpose import POSE_METHODS, HeadPose, PoseConfig, draw_pose, estimate_pose
from src.landmarks import (DRAW_MODES, FaceLandmarkDetector, FaceLandmarks,
                           LandmarkConfig, LandmarkModelError, draw_driver_zone,
                           draw_ignored_faces, draw_landmarks)

# --- landmark index sets (MediaPipe canonical topology) ---------------------
# Order matters: p1..p6 for the EAR formula.
RIGHT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
LEFT_EYE: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

MOUTH_CORNERS: Tuple[int, int] = (61, 291)          # subject's right, left corner
MOUTH_UPPER_INNER: Tuple[int, ...] = (82, 13, 312)  # right of centre, centre, left
MOUTH_LOWER_INNER: Tuple[int, ...] = (87, 14, 317)  # directly below the three above

FEATURE_LANDMARKS: Tuple[int, ...] = (RIGHT_EYE + LEFT_EYE + MOUTH_CORNERS
                                      + MOUTH_UPPER_INNER + MOUTH_LOWER_INNER)


# --- formulas --------------------------------------------------------------

def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR from a (6, 2) array of pixel points ordered p1..p6.

    Returns NaN when the eye width is degenerate (should not happen with a
    detected face, but downstream code must never divide by zero).
    """
    width = float(np.linalg.norm(points[0] - points[3]))
    if width < 1e-6:
        return math.nan
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    return float(vertical / (2.0 * width))


def mouth_aspect_ratio(upper: np.ndarray, lower: np.ndarray, corners: np.ndarray) -> float:
    """MAR from matching (k, 2) upper/lower inner-lip points and (2, 2) corners."""
    width = float(np.linalg.norm(corners[0] - corners[1]))
    if width < 1e-6:
        return math.nan
    gaps = np.linalg.norm(upper - lower, axis=1)
    return float(gaps.mean() / width)


# --- output ------------------------------------------------------------------

@dataclass
class GeometricFeatures:
    """Per-frame measurements. Raw values, no interpretation attached."""

    ear_left: float          # subject's left eye
    ear_right: float         # subject's right eye
    ear_mean: float          # mean of the two (or the valid one if one is NaN)
    mar: float
    eye_width_left_px: float
    eye_width_right_px: float
    mouth_width_px: float
    timestamp_ms: int

    @property
    def valid(self) -> bool:
        return not (math.isnan(self.ear_mean) or math.isnan(self.mar))

    @classmethod
    def csv_fields(cls) -> List[str]:
        return [f.name for f in fields(cls)]


def compute_features(face: FaceLandmarks) -> GeometricFeatures:
    """Compute EAR (both eyes) and MAR from one frame's landmarks."""
    left = face.points(LEFT_EYE)
    right = face.points(RIGHT_EYE)
    upper = face.points(MOUTH_UPPER_INNER)
    lower = face.points(MOUTH_LOWER_INNER)
    corners = face.points(MOUTH_CORNERS)

    ear_left = eye_aspect_ratio(left)
    ear_right = eye_aspect_ratio(right)
    both = [v for v in (ear_left, ear_right) if not math.isnan(v)]
    ear_mean = float(np.mean(both)) if both else math.nan

    return GeometricFeatures(
        ear_left=ear_left,
        ear_right=ear_right,
        ear_mean=ear_mean,
        mar=mouth_aspect_ratio(upper, lower, corners),
        eye_width_left_px=float(np.linalg.norm(left[0] - left[3])),
        eye_width_right_px=float(np.linalg.norm(right[0] - right[3])),
        mouth_width_px=float(np.linalg.norm(corners[0] - corners[1])),
        timestamp_ms=face.timestamp_ms,
    )


# --- frame validity (Stage 4) ------------------------------------------------

@dataclass
class ValidityConfig:
    """Frame validity thresholds. INITIAL VALUES - configurable, not tuned.

    Set them from measured data: record sessions with ``--record`` and look at
    the yaw, face-width and eye-width distributions before changing anything.
    """

    max_abs_yaw_deg: float = 30.0                 # beyond this EAR/MAR are not interpreted
    max_abs_pitch_deg: Optional[float] = None     # off: Stage 9 decides how pitch is treated
    min_face_width_px: int = 80                   # smaller faces give coarse landmarks
    min_eye_width_px: float = 15.0                # EAR precision collapses on narrow eyes
    edge_margin_px: int = 4                       # landmarks this close to the border = face partly out
    window_seconds: float = 60.0                  # rolling window for the invalid-frame rate


@dataclass
class FrameAssessment:
    """Whether this frame may be interpreted, and why not if it may not."""

    valid: bool
    keys: List[str]            # machine-readable reason keys, for counting
    reasons: List[str]         # human-readable reasons, for display
    face_found: bool
    features: Optional[GeometricFeatures]
    pose: Optional[HeadPose]
    time_s: float              # monotonic time of assessment


def assess_frame(face: Optional[FaceLandmarks], feats: Optional[GeometricFeatures],
                 pose: Optional[HeadPose], config: Optional[ValidityConfig] = None,
                 faces_in_frame: int = 0, now: Optional[float] = None) -> FrameAssessment:
    """Decide whether a frame is VALID for interpretation. Pure function."""
    config = config or ValidityConfig()
    now = time.monotonic() if now is None else now
    keys: List[str] = []
    reasons: List[str] = []

    if face is None:
        if faces_in_frame > 0:
            keys.append("no_driver_in_zone")
            reasons.append("{} face(s) but none in driver zone".format(faces_in_frame))
        else:
            keys.append("no_face")
            reasons.append("no face")
        return FrameAssessment(False, keys, reasons, False, None, None, now)

    width, height = face.frame_size
    xs = face.normalized[:, 0] * width
    ys = face.normalized[:, 1] * height
    face_width_px = float(xs.max() - xs.min())
    if face_width_px < config.min_face_width_px:
        keys.append("face_too_small")
        reasons.append("face too small ({:.0f} px < {})".format(face_width_px, config.min_face_width_px))
    m = config.edge_margin_px
    if xs.min() < m or ys.min() < m or xs.max() > width - 1 - m or ys.max() > height - 1 - m:
        keys.append("face_at_edge")
        reasons.append("face partly out of frame")

    if feats is None or not feats.valid:
        keys.append("degenerate_landmarks")
        reasons.append("degenerate landmarks")
    else:
        narrowest = min(feats.eye_width_left_px, feats.eye_width_right_px)
        if narrowest < config.min_eye_width_px:
            keys.append("eye_too_small")
            reasons.append("eye too narrow ({:.0f} px < {:.0f})".format(narrowest, config.min_eye_width_px))

    if pose is None or not pose.ok:
        keys.append("pose_unavailable")
        reasons.append("head pose unavailable")
    else:
        if abs(pose.yaw_deg) > config.max_abs_yaw_deg:
            keys.append("yaw")
            reasons.append("yaw {:+.0f} deg beyond +/-{:.0f}".format(pose.yaw_deg, config.max_abs_yaw_deg))
        if config.max_abs_pitch_deg is not None and abs(pose.pitch_deg) > config.max_abs_pitch_deg:
            keys.append("pitch")
            reasons.append("pitch {:+.0f} deg beyond +/-{:.0f}".format(pose.pitch_deg, config.max_abs_pitch_deg))

    return FrameAssessment(not keys, keys, reasons, True, feats, pose, now)


class InvalidFrameTracker:
    """Invalid-frame rate over a rolling time window and over the session."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._window: Deque[Tuple[float, bool]] = deque()
        self.total = 0
        self.invalid = 0
        self.reasons: Counter = Counter()

    def update(self, assessment: FrameAssessment) -> None:
        self._window.append((assessment.time_s, assessment.valid))
        cutoff = assessment.time_s - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()
        self.total += 1
        if not assessment.valid:
            self.invalid += 1
            self.reasons.update(assessment.keys)

    @property
    def window_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for _, valid in self._window if not valid) / len(self._window)

    @property
    def session_rate(self) -> float:
        return self.invalid / self.total if self.total else 0.0


# --- recording ---------------------------------------------------------------

class FeatureRecorder:
    """Writes one CSV row per frame. Development tool: collecting real EAR/MAR/
    pose distributions is how Stage 9's thresholds will be set."""

    POSE_FIELDS = ["yaw_deg", "pitch_deg", "roll_deg", "pose_method"]
    VALIDITY_FIELDS = ["valid", "invalid_reasons"]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer = None
        self.rows = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame", "face_found", "inference_ms"] + GeometricFeatures.csv_fields()
                              + self.POSE_FIELDS + self.VALIDITY_FIELDS)

    def write(self, frame_index: int, inference_ms: float, feats: Optional[GeometricFeatures],
              pose: Optional[HeadPose] = None, assessment: Optional[FrameAssessment] = None) -> None:
        values = ([getattr(feats, name) for name in GeometricFeatures.csv_fields()]
                  if feats is not None else [""] * len(GeometricFeatures.csv_fields()))
        pose_values = ([round(pose.yaw_deg, 2), round(pose.pitch_deg, 2), round(pose.roll_deg, 2), pose.method]
                       if pose is not None and pose.ok else ["", "", "", ""])
        validity = ([int(assessment.valid), ";".join(assessment.keys)] if assessment is not None else ["", ""])
        self._writer.writerow([frame_index, int(feats is not None), round(inference_ms, 2)]
                              + values + pose_values + validity)
        self.rows += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# --- drawing -----------------------------------------------------------------

COLOR_EYE = (0, 255, 255)      # yellow
COLOR_MOUTH = (255, 0, 255)    # magenta
COLOR_TEXT = (255, 255, 255)
COLOR_OK = (0, 255, 0)
COLOR_BAD = (0, 0, 255)
COLOR_WARN = (0, 200, 255)
COLOR_MUTED = (150, 150, 150)
COLOR_POSE_TEXT = (255, 160, 0)

EAR_BAR_MAX = 0.5   # display scale only - not a threshold
MAR_BAR_MAX = 1.0


def draw_feature_geometry(frame: np.ndarray, face: FaceLandmarks) -> None:
    """Draw exactly the points and distances the formulas use, on the raw frame."""
    px = face.pixels

    for eye in (LEFT_EYE, RIGHT_EYE):
        p = [tuple(int(v) for v in px[i]) for i in eye]
        cv2.line(frame, p[0], p[3], COLOR_EYE, 1, cv2.LINE_AA)  # width  p1-p4
        cv2.line(frame, p[1], p[5], COLOR_EYE, 1, cv2.LINE_AA)  # gap    p2-p6
        cv2.line(frame, p[2], p[4], COLOR_EYE, 1, cv2.LINE_AA)  # gap    p3-p5
        for point in p:
            cv2.circle(frame, point, 2, COLOR_EYE, -1, cv2.LINE_AA)

    corners = [tuple(int(v) for v in px[i]) for i in MOUTH_CORNERS]
    cv2.line(frame, corners[0], corners[1], COLOR_MOUTH, 1, cv2.LINE_AA)
    for up, low in zip(MOUTH_UPPER_INNER, MOUTH_LOWER_INNER):
        a = tuple(int(v) for v in px[up])
        b = tuple(int(v) for v in px[low])
        cv2.line(frame, a, b, COLOR_MOUTH, 1, cv2.LINE_AA)
        cv2.circle(frame, a, 2, COLOR_MOUTH, -1, cv2.LINE_AA)
        cv2.circle(frame, b, 2, COLOR_MOUTH, -1, cv2.LINE_AA)
    for point in corners:
        cv2.circle(frame, point, 2, COLOR_MOUTH, -1, cv2.LINE_AA)


def _fmt(value: float) -> str:
    return "  -- " if math.isnan(value) else "{:5.3f}".format(value)


def _bar(frame: np.ndarray, x: int, y: int, value: float, vmax: float, color, width: int = 150) -> None:
    cv2.rectangle(frame, (x, y - 10), (x + width, y), (60, 60, 60), -1)
    if not math.isnan(value):
        fill = int(min(max(value / vmax, 0.0), 1.0) * width)
        cv2.rectangle(frame, (x, y - 10), (x + fill, y), color, -1)
    cv2.rectangle(frame, (x, y - 10), (x + width, y), (160, 160, 160), 1)


def _trace(frame: np.ndarray, values: Sequence[float], x: int, y: int, w: int, h: int,
           vmax: float, color, label: str) -> None:
    """Sparkline of recent values. Display aid only - it decides nothing."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        if math.isnan(v):
            continue
        px = x + int(i * (w - 1) / max(1, n - 1)) if n > 1 else x
        py = y + h - 1 - int(min(max(v / vmax, 0.0), 1.0) * (h - 1))
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x + 4, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_feature_hud(frame: np.ndarray, fps: float, inference_ms: float,
                     feats: Optional[GeometricFeatures], face: Optional[FaceLandmarks],
                     mirror: bool, ear_trace: Sequence[float], mar_trace: Sequence[float],
                     pose: Optional[HeadPose] = None, assessment: Optional[FrameAssessment] = None,
                     tracker: Optional[InvalidFrameTracker] = None, face_count: int = 0) -> None:
    """Status overlay. Call after any display mirroring so text reads normally."""
    height, width = frame.shape[:2]
    valid = assessment.valid if assessment is not None else True
    value_color_eye = COLOR_EYE if valid else COLOR_MUTED
    value_color_mouth = COLOR_MOUTH if valid else COLOR_MUTED

    def put(text, x, y, color=COLOR_TEXT, scale=0.6):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    put("Stage 5 - EAR / MAR / head pose / validity / eye crops", 10, 24)
    put("FPS {:5.1f}   inference {:5.1f} ms".format(fps, inference_ms), 10, 46, COLOR_OK)

    if feats is None:
        put("NO FACE DETECTED" if face_count == 0 else
            "{} FACE(S) OUTSIDE DRIVER ZONE".format(face_count), 10, 72, COLOR_BAD)
    else:
        put("EAR  L {}   R {}   mean {}{}".format(_fmt(feats.ear_left), _fmt(feats.ear_right),
                                                   _fmt(feats.ear_mean), "" if valid else "   (not interpreted)"),
            10, 72, value_color_eye)
        _bar(frame, 10, 90, feats.ear_mean, EAR_BAR_MAX, value_color_eye)
        put("MAR  {}".format(_fmt(feats.mar)), 10, 114, value_color_mouth)
        _bar(frame, 10, 132, feats.mar, MAR_BAR_MAX, value_color_mouth)

        # Per-eye labels beside the eyes, placed after mirroring so they read
        # correctly. L/R are the subject's own left/right.
        if face is not None:
            for label, idx in (("L", LEFT_EYE[3]), ("R", RIGHT_EYE[0])):
                x, y = face.pixel(idx)
                if mirror:
                    x = width - 1 - x
                put(label, x + (6 if (label == "L") == mirror else -18), y - 8, COLOR_EYE, 0.5)

    if face is not None and face.faces_in_frame > 1:
        put("{} faces - measuring driver only ({})".format(face.faces_in_frame, face.selection),
            10, 156, COLOR_WARN, 0.55)

    if pose is not None and pose.ok:
        put("yaw {:+5.0f}   pitch {:+5.0f}   roll {:+5.0f}   ({}{})".format(
            pose.yaw_deg, pose.pitch_deg, pose.roll_deg, pose.method,
            ", ~{:.0f} cm".format(pose.distance_cm) if pose.distance_cm else ""), 10, 178, COLOR_POSE_TEXT)
    elif feats is not None:
        put("head pose unavailable", 10, 178, COLOR_MUTED)

    if assessment is not None:
        if assessment.valid:
            put("VALID FRAME", 10, 202, COLOR_OK)
        else:
            put("INVALID: " + "; ".join(assessment.reasons), 10, 202, COLOR_BAD, 0.55)
    if tracker is not None:
        put("invalid frames {:4.0%} ({:.0f} s window)   {:4.0%} session".format(
            tracker.window_rate, tracker.window_seconds, tracker.session_rate), 10, 224, (200, 200, 200), 0.5)

    _trace(frame, ear_trace, width - 210, height - 130, 200, 50, EAR_BAR_MAX, COLOR_EYE, "EAR mean")
    _trace(frame, mar_trace, width - 210, height - 72, 200, 50, MAR_BAR_MAX, COLOR_MOUTH, "MAR")

    put("q quit  m mesh  g gray  p pose  z zone  c crops  e save eyes  s snapshot", 10, height - 12,
        (200, 200, 200), 0.5)


# --- live demo ---------------------------------------------------------------

WINDOW_NAME = "Drowsiness Detection - Stage 5 (Features / Pose / Validity / Eye Crops)"
TRACE_LENGTH = 200
CROP_DIR_RAW = Path(__file__).resolve().parent.parent / "data" / "eye_crops" / "raw"


def _describe(name: str, values: List[float]) -> str:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return "  {:<10} no valid values".format(name)
    return "  {:<10} min {:+.3f}  p5 {:+.3f}  median {:+.3f}  p95 {:+.3f}  max {:+.3f}  std {:.4f}".format(
        name, arr.min(), np.percentile(arr, 5), np.median(arr), np.percentile(arr, 95),
        arr.max(), arr.std())


def run_demo(source: FrameSource, detector: FaceLandmarkDetector, mode: str = "contours",
             mirror: bool = True, show_window: bool = True, max_frames: int = 0,
             record: Optional[Path] = None, pose_config: Optional[PoseConfig] = None,
             validity_config: Optional[ValidityConfig] = None, show_zone: bool = True,
             eye_config: Optional[EyePreprocessConfig] = None, show_crops: bool = True,
             dump_crops: Optional[Path] = None, dump_every: int = 30) -> int:
    pose_config = pose_config or PoseConfig()
    validity_config = validity_config or ValidityConfig()
    eye_config = eye_config or EyePreprocessConfig()
    tracker = InvalidFrameTracker(validity_config.window_seconds)
    crops_total = 0
    crops_valid = 0
    eye_widths: List[float] = []
    dumped = 0
    fps_counter = FPSCounter()
    mode_index = DRAW_MODES.index(mode)
    ear_trace: Deque[float] = deque(maxlen=TRACE_LENGTH)
    mar_trace: Deque[float] = deque(maxlen=TRACE_LENGTH)
    ears: List[float] = []
    ears_left: List[float] = []
    ears_right: List[float] = []
    mars: List[float] = []
    yaws: List[float] = []
    pitches: List[float] = []
    rolls: List[float] = []
    recorder = FeatureRecorder(record) if record else None
    frame_index = 0
    started = time.perf_counter()

    with source, detector:
        if recorder:
            recorder.open()
            print("[features] Recording per-frame values to {}".format(recorder.path))
        print("[features] Camera   : {}".format(source.description))
        print("[features] Pose     : method {}".format(pose_config.method))
        print("[features] Validity : |yaw| <= {} deg, |pitch| {}, face >= {} px, eye >= {} px, "
              "edge margin {} px, window {} s".format(
                  validity_config.max_abs_yaw_deg,
                  "off" if validity_config.max_abs_pitch_deg is None
                  else "<= {} deg".format(validity_config.max_abs_pitch_deg),
                  validity_config.min_face_width_px, validity_config.min_eye_width_px,
                  validity_config.edge_margin_px, validity_config.window_seconds))
        print("[features] Eyes     : {}x{} crops, scale {} x eye width, roll alignment {}, CLAHE {}".format(
            eye_config.size, eye_config.size, eye_config.crop_scale,
            "on" if eye_config.align_roll else "off", "on" if eye_config.equalize else "off"))
        if dump_crops:
            print("[features] Dumping valid eye crops every {} frames to {}".format(dump_every, dump_crops))
        print("[features] Keys     : q/ESC quit | m mesh mode | g gray input | p pose method | "
              "z driver zone | c crop panel | e save eye crops | s snapshot")
        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = source.read()
            if frame is None:
                print("[features] No frame received - stream ended or camera lost.")
                break
            frame_index += 1

            face = detector.process(frame)
            feats = compute_features(face) if face is not None else None
            pose = estimate_pose(face, pose_config) if face is not None else None
            assessment = assess_frame(face, feats, pose, validity_config, detector.last_face_count)
            tracker.update(assessment)
            crops = extract_eye_crops(frame, face, eye_config) if face is not None else (None, None)
            for crop in crops:
                if crop is not None:
                    crops_total += 1
                    crops_valid += int(crop.valid)
                    eye_widths.append(crop.eye_width_px)
            if (dump_crops and frame_index % dump_every == 0 and assessment.valid
                    and all(c is not None and c.valid for c in crops)):
                save_eye_crops(crops, dump_crops, tag="f{:06d}".format(frame_index))
                dumped += 2
            fps = fps_counter.tick()
            inference_ms = detector.stats.inference_ms[-1] if detector.stats.inference_ms else 0.0

            if feats is not None:
                ears.append(feats.ear_mean)
                ears_left.append(feats.ear_left)
                ears_right.append(feats.ear_right)
                mars.append(feats.mar)
            ear_trace.append(feats.ear_mean if (feats is not None and assessment.valid) else math.nan)
            mar_trace.append(feats.mar if (feats is not None and assessment.valid) else math.nan)
            if pose is not None and pose.ok:
                yaws.append(pose.yaw_deg)
                pitches.append(pose.pitch_deg)
                rolls.append(pose.roll_deg)
            if recorder:
                recorder.write(frame_index, inference_ms, feats, pose, assessment)

            if show_window:
                display = frame.copy()
                if show_zone:
                    draw_driver_zone(display, detector.config)
                if face is not None:
                    draw_landmarks(display, face, DRAW_MODES[mode_index])
                    draw_feature_geometry(display, face)
                    draw_eye_boxes(display, crops)
                if mirror:
                    display = cv2.flip(display, 1)
                draw_ignored_faces(display, detector.last_ignored_boxes, mirror)
                if face is not None and pose is not None:
                    draw_pose(display, face, pose, mirror)
                draw_feature_hud(display, fps, inference_ms, feats, face, mirror, ear_trace, mar_trace,
                                 pose, assessment, tracker, detector.last_face_count)
                if show_crops:
                    draw_eye_panel(display, crops, 10, display.shape[0] - 140)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[features] Quit key pressed.")
                    break
                if key == ord("m"):
                    mode_index = (mode_index + 1) % len(DRAW_MODES)
                if key == ord("g"):
                    detector.config.grayscale_input = not detector.config.grayscale_input
                if key == ord("p"):
                    pose_config.method = POSE_METHODS[(POSE_METHODS.index(pose_config.method) + 1)
                                                      % len(POSE_METHODS)]
                    print("[features] pose method -> {}".format(pose_config.method))
                if key == ord("z"):
                    show_zone = not show_zone
                if key == ord("c"):
                    show_crops = not show_crops
                if key == ord("e"):
                    if all(c is not None for c in crops):
                        written = save_eye_crops(crops, tag="manual")
                        written += save_eye_crops(crops, CROP_DIR_RAW, tag="manual", raw=True)
                        print("[features] Saved eye crops: {}".format(", ".join(p.name for p in written)))
                    else:
                        print("[features] No eye crops to save (no face).")
                if key == ord("s"):
                    print("[features] Saved {}".format(save_snapshot(display)))
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[features] Preview window closed.")
                    break

            if max_frames and frame_index >= max_frames:
                print("[features] Reached --max-frames {}.".format(max_frames))
                break

    if recorder:
        recorder.close()
        print("[features] Wrote {} rows to {}".format(recorder.rows, recorder.path))
    if show_window:
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    elapsed = time.perf_counter() - started
    stats = detector.stats
    print("[features] {} frames in {:.1f}s -> {:.1f} FPS end to end; face in {}/{} ({:.1%}); "
          "inference median {:.1f} ms".format(frame_index, elapsed, frame_index / elapsed if elapsed else 0.0,
                                              stats.faces, stats.frames, stats.detection_rate,
                                              stats.median_inference_ms))
    print("[features] Raw measured values over frames with a face (no thresholds applied):")
    print(_describe("EAR mean", ears))
    print(_describe("EAR left", ears_left))
    print(_describe("EAR right", ears_right))
    print(_describe("MAR", mars))
    print(_describe("yaw deg", yaws))
    print(_describe("pitch deg", pitches))
    print(_describe("roll deg", rolls))
    print("[features] Invalid frames: {}/{} = {:.1%} of the session ({:.1%} in the last {:.0f} s window)".format(
        tracker.invalid, tracker.total, tracker.session_rate, tracker.window_rate, tracker.window_seconds))
    if tracker.reasons:
        print("[features] Invalid-frame reasons: " + ", ".join(
            "{} x{}".format(key, count) for key, count in tracker.reasons.most_common()))
    if crops_total:
        widths = np.array(eye_widths)
        print("[features] Eye crops: {} produced, {} geometrically valid ({:.1%}); eye width px "
              "min {:.0f} median {:.0f} max {:.0f}; raw crop side median {:.0f} px -> {}x{}".format(
                  crops_total, crops_valid, crops_valid / crops_total, widths.min(), np.median(widths),
                  widths.max(), np.median(widths) * eye_config.crop_scale, eye_config.size, eye_config.size))
    if dump_crops:
        print("[features] Dumped {} eye crop files to {}".format(dumped, dump_crops))
    return 0


# --- self-test ---------------------------------------------------------------

def _fake_face(pixel_points: dict, size=(640, 480), default=0.5) -> FaceLandmarks:
    """Build a FaceLandmarks whose chosen indices sit at given pixel positions."""
    normalized = np.zeros((478, 3), dtype=np.float32)
    normalized[:, :2] = default
    for idx, (x, y) in pixel_points.items():
        normalized[idx, 0] = x / size[0]
        normalized[idx, 1] = y / size[1]
    return FaceLandmarks(normalized, size, 0, 0.0)


def _box_face(x0: float, y0: float, x1: float, y1: float, size=(640, 480)) -> FaceLandmarks:
    """A face whose landmarks fill the given pixel box (used for validity tests)."""
    normalized = np.zeros((478, 3), dtype=np.float32)
    normalized[:, 0] = np.linspace(x0, x1, 478) / size[0]
    normalized[:, 1] = (y0 + (y1 - y0) * 0.5 * (1 + np.sin(np.linspace(0, 12, 478)))) / size[1]
    return FaceLandmarks(normalized, size, 0, 0.0)


def self_test() -> int:
    """Verifies the formulas and the validity gate on known inputs. No camera or model."""
    all_idx = FEATURE_LANDMARKS
    assert len(set(all_idx)) == len(all_idx), "duplicate landmark index in feature sets"
    assert all(0 <= i < 478 for i in all_idx), "landmark index out of range"
    print("[self-test] {} distinct landmark indices, all within 0..477".format(len(all_idx)))

    # Synthetic eye: width 10, both lid gaps 2  ->  EAR = (2 + 2) / (2 * 10) = 0.2
    eye = np.array([[0, 0], [3, -1], [7, -1], [10, 0], [7, 1], [3, 1]], dtype=np.float32)
    assert abs(eye_aspect_ratio(eye) - 0.2) < 1e-6, eye_aspect_ratio(eye)
    closed = eye.copy()
    closed[:, 1] = 0
    assert eye_aspect_ratio(closed) == 0.0, "closed eye must give EAR 0"
    assert abs(eye_aspect_ratio(eye * 3.7) - 0.2) < 1e-6, "EAR must be scale invariant"
    theta = math.radians(33)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    assert abs(eye_aspect_ratio(eye @ rot.T) - 0.2) < 1e-5, "EAR must be rotation invariant"
    degenerate = eye.copy()
    degenerate[3] = degenerate[0]
    assert math.isnan(eye_aspect_ratio(degenerate)), "zero width must give NaN, not a crash"
    print("[self-test] EAR: value, closed=0, scale and rotation invariance, NaN on zero width OK")

    # Synthetic mouth: width 20, three gaps of 4  ->  MAR = 4 / 20 = 0.2
    corners = np.array([[0, 0], [20, 0]], dtype=np.float32)
    upper = np.array([[7, -2], [10, -2], [13, -2]], dtype=np.float32)
    lower = np.array([[7, 2], [10, 2], [13, 2]], dtype=np.float32)
    assert abs(mouth_aspect_ratio(upper, lower, corners) - 0.2) < 1e-6
    assert mouth_aspect_ratio(upper, upper, corners) == 0.0, "touching lips must give MAR 0"
    assert math.isnan(mouth_aspect_ratio(upper, lower, np.zeros((2, 2), np.float32)))
    print("[self-test] MAR: value, closed=0, NaN on zero width OK")

    # End to end through compute_features with a fabricated FaceLandmarks:
    # left eye open (EAR 0.2), right eye closed (EAR 0), mouth MAR 0.2.
    pts = {}
    for k, (x, y) in enumerate(eye + np.array([400, 200], np.float32)):
        pts[LEFT_EYE[k]] = (float(x), float(y))
    for k, (x, y) in enumerate(closed + np.array([200, 200], np.float32)):
        pts[RIGHT_EYE[k]] = (float(x), float(y))
    for k, (x, y) in enumerate(corners + np.array([300, 350], np.float32)):
        pts[MOUTH_CORNERS[k]] = (float(x), float(y))
    for k in range(3):
        pts[MOUTH_UPPER_INNER[k]] = tuple(map(float, upper[k] + np.array([300, 350], np.float32)))
        pts[MOUTH_LOWER_INNER[k]] = tuple(map(float, lower[k] + np.array([300, 350], np.float32)))
    feats = compute_features(_fake_face(pts))
    assert abs(feats.ear_left - 0.2) < 1e-3, feats.ear_left
    assert abs(feats.ear_right - 0.0) < 1e-3, feats.ear_right
    assert abs(feats.ear_mean - 0.1) < 1e-3, feats.ear_mean
    assert abs(feats.mar - 0.2) < 1e-3, feats.mar
    assert feats.valid
    print("[self-test] compute_features end to end OK (L 0.2, R 0.0, mean 0.1, MAR 0.2)")

    # --- validity gate ---
    cfg = ValidityConfig()
    good_pose = HeadPose(5.0, -3.0, 1.0, "test")
    good_feats = GeometricFeatures(0.3, 0.3, 0.3, 0.05, 40.0, 40.0, 60.0, 0)
    ok_face = _box_face(220, 140, 420, 340)

    a = assess_frame(None, None, None, cfg, faces_in_frame=0, now=0.0)
    assert not a.valid and a.keys == ["no_face"], a
    a = assess_frame(None, None, None, cfg, faces_in_frame=2, now=0.0)
    assert not a.valid and a.keys == ["no_driver_in_zone"], a
    a = assess_frame(ok_face, good_feats, good_pose, cfg, now=0.0)
    assert a.valid and a.keys == [], a
    a = assess_frame(ok_face, good_feats, HeadPose(41.0, 0.0, 0.0, "test"), cfg, now=0.0)
    assert not a.valid and a.keys == ["yaw"], a
    a = assess_frame(ok_face, good_feats, HeadPose(-31.0, 0.0, 0.0, "test"), cfg, now=0.0)
    assert not a.valid and a.keys == ["yaw"], "negative yaw beyond the limit must also be invalid"
    a = assess_frame(ok_face, good_feats, HeadPose(0.0, -50.0, 0.0, "test"), cfg, now=0.0)
    assert a.valid, "pitch limit is off by default"
    a = assess_frame(ok_face, good_feats, HeadPose(0.0, -50.0, 0.0, "test"),
                     ValidityConfig(max_abs_pitch_deg=40.0), now=0.0)
    assert not a.valid and a.keys == ["pitch"], a
    a = assess_frame(_box_face(300, 200, 350, 250), good_feats, good_pose, cfg, now=0.0)
    assert not a.valid and "face_too_small" in a.keys, a
    a = assess_frame(_box_face(500, 140, 639, 340), good_feats, good_pose, cfg, now=0.0)
    assert not a.valid and "face_at_edge" in a.keys, a
    a = assess_frame(ok_face, GeometricFeatures(0.3, 0.3, 0.3, 0.05, 12.0, 40.0, 60.0, 0), good_pose, cfg, now=0.0)
    assert not a.valid and a.keys == ["eye_too_small"], a
    a = assess_frame(ok_face, GeometricFeatures(math.nan, 0.3, 0.3, math.nan, 40.0, 40.0, 60.0, 0), good_pose, cfg, now=0.0)
    assert not a.valid and a.keys == ["degenerate_landmarks"], a
    a = assess_frame(ok_face, good_feats, None, cfg, now=0.0)
    assert not a.valid and a.keys == ["pose_unavailable"], a
    a = assess_frame(_box_face(600, 400, 639, 479), good_feats, HeadPose(60.0, 0.0, 0.0, "test"), cfg, now=0.0)
    assert set(a.keys) == {"face_too_small", "face_at_edge", "yaw"}, "all reasons must be reported together"
    print("[self-test] validity gate OK: no face, zone, yaw both signs, pitch on/off, size, edge, eye, NaN, pose")

    # --- invalid-frame tracker ---
    tracker = InvalidFrameTracker(window_seconds=10.0)
    for t in range(20):   # t = 0..19 s, invalid on even seconds
        valid = t % 2 == 1
        tracker.update(FrameAssessment(valid, [] if valid else ["yaw"], [], True, None, None, float(t)))
    assert tracker.total == 20 and tracker.invalid == 10
    assert abs(tracker.session_rate - 0.5) < 1e-9
    # window keeps t = 9..19 (11 frames); the even ones 10..18 are the 5 invalid
    assert len(tracker._window) == 11, len(tracker._window)
    assert abs(tracker.window_rate - 5 / 11) < 1e-9, tracker.window_rate
    for t in range(20, 30):
        tracker.update(FrameAssessment(True, [], [], True, None, None, float(t)))
    assert tracker.window_rate == 0.0, "old invalid frames must leave the window"
    assert abs(tracker.session_rate - 10 / 30) < 1e-9
    assert tracker.reasons == Counter({"yaw": 10}), tracker.reasons
    print("[self-test] invalid-frame tracker OK: window trimming, session rate, reason histogram")

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    face = _fake_face(pts)
    draw_feature_geometry(canvas, face)
    draw_feature_hud(canvas, 20.0, 8.0, feats, face, True, [0.3, 0.1, math.nan, 0.3], [0.05, 0.6],
                     good_pose, assess_frame(ok_face, good_feats, good_pose, cfg, now=0.0), tracker, 1)
    draw_feature_hud(canvas, 20.0, 8.0, feats, face, True, [], [],
                     HeadPose(45.0, 0.0, 0.0, "pnp"),
                     assess_frame(ok_face, good_feats, HeadPose(45.0, 0.0, 0.0, "pnp"), cfg, now=0.0), tracker, 1)
    draw_feature_hud(canvas, 20.0, 8.0, None, None, True, [], [], None,
                     assess_frame(None, None, None, cfg, now=0.0), tracker, 0)
    assert canvas.any()
    print("[self-test] drawing OK (valid, invalid, no face)")

    print("[self-test] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 4: live EAR / MAR, head pose and frame validity from facial landmarks.")
    parser.add_argument("--device", default="0", help="Camera index or video path (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mode", choices=DRAW_MODES, default="contours", help="Landmark draw style")
    parser.add_argument("--gray", action="store_true", help="Feed grayscale frames to MediaPipe")
    parser.add_argument("--pose-method", choices=POSE_METHODS, default="matrix",
                        help="Head-pose estimator (default matrix; pnp is a cross-check; 'p' key cycles live)")
    parser.add_argument("--max-yaw", type=float, default=30.0,
                        help="|yaw| in degrees beyond which a frame is INVALID (default 30, untuned)")
    parser.add_argument("--max-pitch", type=float, default=0.0,
                        help="|pitch| limit in degrees; 0 = off (default off)")
    parser.add_argument("--min-face-px", type=int, default=80, help="Minimum face width in px (default 80)")
    parser.add_argument("--min-eye-px", type=float, default=15.0, help="Minimum eye width in px (default 15)")
    parser.add_argument("--window", type=float, default=60.0,
                        help="Rolling window in seconds for the invalid-frame rate (default 60)")
    parser.add_argument("--eye-size", type=int, default=64, help="Eye crop output size in px (default 64)")
    parser.add_argument("--crop-scale", type=float, default=1.5,
                        help="Crop side as a multiple of the eye-corner distance (default 1.5, verify vs MRL)")
    parser.add_argument("--no-align", action="store_true", help="Do not rotate crops to level the eye")
    parser.add_argument("--no-crops", action="store_true", help="Start with the eye-crop panel hidden ('c' toggles)")
    parser.add_argument("--dump-crops", type=Path, default=None,
                        help="Save valid eye crops from VALID frames to this folder (e.g. data/eye_crops/session1)")
    parser.add_argument("--dump-every", type=int, default=30, help="With --dump-crops: every N frames (default 30)")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the display")
    parser.add_argument("--no-zone", action="store_true",
                        help="Start with the driver-zone ellipse hidden ('z' key toggles it live)")
    parser.add_argument("--no-window", action="store_true", help="Headless: statistics only")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = until quit)")
    parser.add_argument("--record", type=Path, default=None,
                        help="Write per-frame values to this CSV (e.g. data/features.csv)")
    parser.add_argument("--self-test", action="store_true", help="Check formulas and validity gate, no camera")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    device = int(args.device) if args.device.isdigit() else args.device
    camera = CameraConfig(device=device, width=args.width, height=args.height, flip_horizontal=False)
    detector = FaceLandmarkDetector(LandmarkConfig(grayscale_input=args.gray))
    pose_config = PoseConfig(method=args.pose_method)
    validity_config = ValidityConfig(
        max_abs_yaw_deg=args.max_yaw,
        max_abs_pitch_deg=args.max_pitch if args.max_pitch > 0 else None,
        min_face_width_px=args.min_face_px,
        min_eye_width_px=args.min_eye_px,
        window_seconds=args.window,
    )
    eye_config = EyePreprocessConfig(size=args.eye_size, crop_scale=args.crop_scale,
                                     align_roll=not args.no_align, min_eye_width_px=args.min_eye_px)
    try:
        return run_demo(create_source(camera), detector, mode=args.mode, mirror=not args.no_mirror,
                        show_window=not args.no_window, max_frames=args.max_frames, record=args.record,
                        pose_config=pose_config, validity_config=validity_config,
                        show_zone=not args.no_zone, eye_config=eye_config, show_crops=not args.no_crops,
                        dump_crops=args.dump_crops, dump_every=args.dump_every)
    except (CameraError, LandmarkModelError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[features] Interrupted by user.")
        cv2.destroyAllWindows()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
