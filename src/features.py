"""Geometric features from facial landmarks: EAR and MAR (Stage 3).

Pipeline slice implemented here::

    FaceLandmarks (Stage 2) -> Eye Aspect Ratio (per eye) + Mouth Aspect Ratio

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

Scope
-----
This module only *measures*. It applies no thresholds and keeps no history;
deciding what an EAR value means (blink, closure, drowsiness) is Stage 9's job
and must be based on measured data. Pixel coordinates are used deliberately:
normalised x/y have different scales on a 4:3 frame and would distort the
ratios.

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
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.capture import (CameraConfig, CameraError, FPSCounter, FrameSource,
                         create_source, save_snapshot)
from src.landmarks import (DRAW_MODES, FaceLandmarkDetector, FaceLandmarks,
                           LandmarkConfig, LandmarkModelError, draw_landmarks)

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


# --- recording ---------------------------------------------------------------

class FeatureRecorder:
    """Writes one CSV row per frame. Development tool: collecting real EAR/MAR
    distributions for open/closed eyes is how Stage 9's thresholds will be set."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer = None
        self.rows = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame", "face_found", "inference_ms"] + GeometricFeatures.csv_fields())

    def write(self, frame_index: int, inference_ms: float, feats: Optional[GeometricFeatures]) -> None:
        values = ([getattr(feats, name) for name in GeometricFeatures.csv_fields()]
                  if feats is not None else [""] * len(GeometricFeatures.csv_fields()))
        self._writer.writerow([frame_index, int(feats is not None), round(inference_ms, 2)] + values)
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
                     mirror: bool, ear_trace: Sequence[float], mar_trace: Sequence[float]) -> None:
    """Status overlay. Call after any display mirroring so text reads normally."""
    height, width = frame.shape[:2]

    def put(text, x, y, color=COLOR_TEXT, scale=0.6):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    put("Stage 3 - EAR / MAR", 10, 24)
    put("FPS {:5.1f}   inference {:5.1f} ms".format(fps, inference_ms), 10, 46, COLOR_OK)

    if feats is None:
        put("NO FACE DETECTED", 10, 72, COLOR_BAD)
    else:
        put("EAR  L {}   R {}   mean {}".format(_fmt(feats.ear_left), _fmt(feats.ear_right),
                                                 _fmt(feats.ear_mean)), 10, 72, COLOR_EYE)
        _bar(frame, 10, 90, feats.ear_mean, EAR_BAR_MAX, COLOR_EYE)
        put("MAR  {}".format(_fmt(feats.mar)), 10, 114, COLOR_MOUTH)
        _bar(frame, 10, 132, feats.mar, MAR_BAR_MAX, COLOR_MOUTH)

        # Per-eye labels beside the eyes, placed after mirroring so they read
        # correctly. L/R are the subject's own left/right.
        if face is not None:
            for label, idx in (("L", LEFT_EYE[3]), ("R", RIGHT_EYE[0])):
                x, y = face.pixel(idx)
                if mirror:
                    x = width - 1 - x
                put(label, x + (6 if (label == "L") == mirror else -18), y - 8, COLOR_EYE, 0.5)

    _trace(frame, ear_trace, width - 210, height - 130, 200, 50, EAR_BAR_MAX, COLOR_EYE, "EAR mean")
    _trace(frame, mar_trace, width - 210, height - 72, 200, 50, MAR_BAR_MAX, COLOR_MOUTH, "MAR")

    put("q=quit  m=mesh mode  g=gray input  s=snapshot", 10, height - 12, (200, 200, 200), 0.5)


# --- live demo ---------------------------------------------------------------

WINDOW_NAME = "Drowsiness Detection - Stage 3 (EAR / MAR)"
TRACE_LENGTH = 200


def _describe(name: str, values: List[float]) -> str:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return "  {:<8} no valid values".format(name)
    return "  {:<8} min {:.3f}  p5 {:.3f}  median {:.3f}  p95 {:.3f}  max {:.3f}  std {:.4f}".format(
        name, arr.min(), np.percentile(arr, 5), np.median(arr), np.percentile(arr, 95),
        arr.max(), arr.std())


def run_demo(source: FrameSource, detector: FaceLandmarkDetector, mode: str = "contours",
             mirror: bool = True, show_window: bool = True, max_frames: int = 0,
             record: Optional[Path] = None) -> int:
    fps_counter = FPSCounter()
    mode_index = DRAW_MODES.index(mode)
    ear_trace: Deque[float] = deque(maxlen=TRACE_LENGTH)
    mar_trace: Deque[float] = deque(maxlen=TRACE_LENGTH)
    ears: List[float] = []
    ears_left: List[float] = []
    ears_right: List[float] = []
    mars: List[float] = []
    recorder = FeatureRecorder(record) if record else None
    frame_index = 0
    started = time.perf_counter()

    with source, detector:
        if recorder:
            recorder.open()
            print("[features] Recording per-frame values to {}".format(recorder.path))
        print("[features] Camera : {}".format(source.description))
        print("[features] Keys   : q/ESC quit | m mesh mode | g gray input | s snapshot")
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
            fps = fps_counter.tick()
            inference_ms = detector.stats.inference_ms[-1] if detector.stats.inference_ms else 0.0

            if feats is not None:
                ears.append(feats.ear_mean)
                ears_left.append(feats.ear_left)
                ears_right.append(feats.ear_right)
                mars.append(feats.mar)
                ear_trace.append(feats.ear_mean)
                mar_trace.append(feats.mar)
            else:
                ear_trace.append(math.nan)
                mar_trace.append(math.nan)
            if recorder:
                recorder.write(frame_index, inference_ms, feats)

            if show_window:
                display = frame.copy()
                if face is not None:
                    draw_landmarks(display, face, DRAW_MODES[mode_index])
                    draw_feature_geometry(display, face)
                if mirror:
                    display = cv2.flip(display, 1)
                draw_feature_hud(display, fps, inference_ms, feats, face, mirror, ear_trace, mar_trace)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[features] Quit key pressed.")
                    break
                if key == ord("m"):
                    mode_index = (mode_index + 1) % len(DRAW_MODES)
                if key == ord("g"):
                    detector.config.grayscale_input = not detector.config.grayscale_input
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
    return 0


# --- self-test ---------------------------------------------------------------

def _fake_face(pixel_points: dict, size=(640, 480)) -> FaceLandmarks:
    """Build a FaceLandmarks whose chosen indices sit at given pixel positions."""
    normalized = np.zeros((478, 3), dtype=np.float32)
    normalized[:, :2] = 0.5
    for idx, (x, y) in pixel_points.items():
        normalized[idx, 0] = x / size[0]
        normalized[idx, 1] = y / size[1]
    return FaceLandmarks(normalized, size, 0, 0.0)


def self_test() -> int:
    """Verifies the formulas on known geometry. Needs no camera or model."""
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

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    face = _fake_face(pts)
    draw_feature_geometry(canvas, face)
    draw_feature_hud(canvas, 20.0, 8.0, feats, face, True, [0.3, 0.1, math.nan, 0.3], [0.05, 0.6])
    draw_feature_hud(canvas, 20.0, 8.0, None, None, True, [], [])
    assert canvas.any()
    print("[self-test] drawing OK (with face, without face, NaN in trace)")

    print("[self-test] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: live EAR and MAR from facial landmarks.")
    parser.add_argument("--device", default="0", help="Camera index or video path (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mode", choices=DRAW_MODES, default="contours", help="Landmark draw style")
    parser.add_argument("--gray", action="store_true", help="Feed grayscale frames to MediaPipe")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the display")
    parser.add_argument("--no-window", action="store_true", help="Headless: statistics only")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = until quit)")
    parser.add_argument("--record", type=Path, default=None,
                        help="Write per-frame EAR/MAR values to this CSV (e.g. data/features.csv)")
    parser.add_argument("--self-test", action="store_true", help="Check the formulas, no camera needed")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    device = int(args.device) if args.device.isdigit() else args.device
    camera = CameraConfig(device=device, width=args.width, height=args.height, flip_horizontal=False)
    detector = FaceLandmarkDetector(LandmarkConfig(grayscale_input=args.gray))
    try:
        return run_demo(create_source(camera), detector, mode=args.mode, mirror=not args.no_mirror,
                        show_window=not args.no_window, max_frames=args.max_frames, record=args.record)
    except (CameraError, LandmarkModelError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[features] Interrupted by user.")
        cv2.destroyAllWindows()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
