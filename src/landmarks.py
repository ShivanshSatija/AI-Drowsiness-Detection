"""Facial landmark detection with MediaPipe Face Landmarker (Stage 2).

Pipeline slice implemented here::

    frame (BGR) -> RGB -> MediaPipe FaceLandmarker -> FaceLandmarks | None

MediaPipe 0.10.35 ships only the Tasks API; the legacy ``mp.solutions.face_mesh``
module no longer exists. The Tasks ``FaceLandmarker`` is used in VIDEO mode,
which tracks the face between frames instead of re-detecting it every time.

Output contract for later stages
--------------------------------
``FaceLandmarkDetector.process(frame)`` returns a ``FaceLandmarks`` object or
``None`` when no face is present. ``FaceLandmarks.normalized`` is a (478, 3)
array in MediaPipe's canonical face-mesh topology:

* index meaning is fixed and documented by MediaPipe (e.g. 33/133 are the
  corners of the right eye, 362/263 the corners of the left eye, 468-477 the
  two irises). Stage 3 (EAR/MAR), Stage 4 (head pose) and Stage 5 (eye crops)
  select their landmarks by these indices.
* LEFT/RIGHT follow the *subject's* anatomy. This is only true when detection
  runs on the un-mirrored camera frame, so this module never flips the input;
  mirroring is applied to the display image afterwards, if at all.
* x, y are normalised to [0, 1] of the frame; ``pixels`` converts to integers.

Run directly for a live demo::

    python -m src.landmarks               # live camera, contours drawn
    python -m src.landmarks --mode mesh   # full tesselation
    python -m src.landmarks --gray        # feed grayscale frames (IR-camera rehearsal)
    python -m src.landmarks --self-test   # no camera needed
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from src.capture import (PROJECT_ROOT, CameraConfig, CameraError, FPSCounter,
                         FrameSource, create_source, save_snapshot)

# --- model -------------------------------------------------------------------
# Official MediaPipe model. Committed to models/ so a fresh clone runs offline;
# the hash is checked on load so a corrupted or substituted file is noticed.
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
NUM_LANDMARKS = 478  # 468 mesh points + 10 iris points


class LandmarkModelError(RuntimeError):
    """The face landmarker model file is missing or unusable."""


def ensure_model(path: Path = MODEL_PATH, verify_hash: bool = True) -> Path:
    """Confirm the model exists (and matches the expected hash)."""
    if not path.exists():
        raise LandmarkModelError(
            "Face landmarker model not found at {}\n"
            "Download it from:\n  {}\n"
            "and place it at that path (about 3.7 MB).".format(path, MODEL_URL))
    if verify_hash:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            print("[landmarks] WARNING: model hash {}... differs from the expected "
                  "{}... - a different model version is in use.".format(
                      digest[:12], MODEL_SHA256[:12]), file=sys.stderr)
    return path


# --- configuration -----------------------------------------------------------

@dataclass
class LandmarkConfig:
    """Detector settings. Confidence values are MediaPipe's defaults and are
    starting points, not tuned values."""

    model_path: Path = MODEL_PATH
    num_faces: int = 1                      # driver monitoring: one face
    min_detection_confidence: float = 0.5   # initial face detection
    min_presence_confidence: float = 0.5    # is a face still present
    min_tracking_confidence: float = 0.5    # frame-to-frame tracking
    grayscale_input: bool = False           # collapse to gray, then back to 3 channels
    verify_model_hash: bool = True


# --- output ------------------------------------------------------------------

@dataclass
class FaceLandmarks:
    """Landmarks for one face, ready for the geometric stages to consume."""

    normalized: np.ndarray            # (478, 3) float32: x, y in [0,1]; z relative
    frame_size: Tuple[int, int]       # (width, height) of the frame analysed
    timestamp_ms: int                 # monotonically increasing frame timestamp
    inference_ms: float               # time spent inside MediaPipe for this frame

    @property
    def count(self) -> int:
        return int(self.normalized.shape[0])

    @property
    def pixels(self) -> np.ndarray:
        """(N, 2) int32 pixel coordinates in the analysed (un-mirrored) frame."""
        width, height = self.frame_size
        scaled = self.normalized[:, :2] * np.array([width, height], dtype=np.float32)
        return np.rint(scaled).astype(np.int32)

    def pixel(self, index: int) -> Tuple[int, int]:
        x, y = self.pixels[index]
        return int(x), int(y)

    def points(self, indices: Sequence[int]) -> np.ndarray:
        """(k, 2) float32 sub-pixel coordinates for the given landmark indices.

        Later stages compute distances from these, so they stay unrounded.
        """
        width, height = self.frame_size
        return self.normalized[list(indices), :2] * np.array([width, height], dtype=np.float32)

    def bounding_box(self, indices: Optional[Sequence[int]] = None,
                     margin: int = 0) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) pixel box around some or all landmarks, clipped to frame."""
        pts = self.pixels if indices is None else self.pixels[list(indices)]
        width, height = self.frame_size
        x0 = max(0, int(pts[:, 0].min()) - margin)
        y0 = max(0, int(pts[:, 1].min()) - margin)
        x1 = min(width - 1, int(pts[:, 0].max()) + margin)
        y1 = min(height - 1, int(pts[:, 1].max()) + margin)
        return x0, y0, x1, y1


# --- detector ----------------------------------------------------------------

@dataclass
class DetectorStats:
    frames: int = 0
    faces: int = 0
    inference_ms: List[float] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.faces / self.frames if self.frames else 0.0

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean(self.inference_ms)) if self.inference_ms else 0.0

    @property
    def median_inference_ms(self) -> float:
        return float(np.median(self.inference_ms)) if self.inference_ms else 0.0


class FaceLandmarkDetector:
    """Wraps MediaPipe FaceLandmarker (VIDEO mode) behind a small, stable API."""

    def __init__(self, config: Optional[LandmarkConfig] = None) -> None:
        self.config = config or LandmarkConfig()
        self._landmarker: Optional[vision.FaceLandmarker] = None
        self._start = time.monotonic()
        self._last_timestamp_ms = -1
        self.stats = DetectorStats()

    def open(self) -> None:
        if self._landmarker is not None:
            return
        model = ensure_model(self.config.model_path, self.config.verify_model_hash)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.config.num_faces,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.monotonic()
        self._last_timestamp_ms = -1

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "FaceLandmarkDetector":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _next_timestamp_ms(self) -> int:
        # VIDEO mode insists on strictly increasing timestamps; two frames in
        # the same millisecond would otherwise raise inside MediaPipe.
        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def _prepare_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.config.grayscale_input:
            # Rehearsal for the NoIR/IR camera, whose frames carry no colour:
            # collapse to one channel, then replicate so the model still sees
            # the 3-channel input it expects.
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarks]:
        """Detect landmarks in one frame. Returns None when no face is found."""
        if self._landmarker is None:
            raise RuntimeError("process() called before open()")

        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(self._prepare_rgb(frame_bgr))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = self._next_timestamp_ms()

        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        self.stats.frames += 1
        self.stats.inference_ms.append(inference_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        normalized = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        self.stats.faces += 1
        return FaceLandmarks(normalized=normalized, frame_size=(width, height),
                             timestamp_ms=timestamp_ms, inference_ms=inference_ms)


# --- drawing -----------------------------------------------------------------
# Connection sets come from MediaPipe's own topology tables, converted once to
# index arrays so a whole set is drawn with a single cv2.polylines call.

def _index_arrays(connections) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.array([c.start for c in connections], dtype=np.int32)
    ends = np.array([c.end for c in connections], dtype=np.int32)
    return starts, ends


_C = FaceLandmarksConnections
CONNECTION_SETS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "contours": _index_arrays(_C.FACE_LANDMARKS_CONTOURS),
    "tesselation": _index_arrays(_C.FACE_LANDMARKS_TESSELATION),
    "left_iris": _index_arrays(_C.FACE_LANDMARKS_LEFT_IRIS),
    "right_iris": _index_arrays(_C.FACE_LANDMARKS_RIGHT_IRIS),
}

DRAW_MODES = ("contours", "mesh", "points", "off")

COLOR_CONTOUR = (0, 255, 0)
COLOR_MESH = (90, 90, 90)
COLOR_IRIS = (255, 200, 0)
COLOR_POINT = (0, 220, 255)


def _draw_set(frame: np.ndarray, pixels: np.ndarray, name: str, color, thickness: int) -> None:
    starts, ends = CONNECTION_SETS[name]
    segments = np.stack([pixels[starts], pixels[ends]], axis=1)  # (N, 2, 2)
    cv2.polylines(frame, segments, False, color, thickness, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, face: FaceLandmarks, mode: str = "contours") -> None:
    """Draw landmarks onto ``frame`` in place. ``frame`` must be the analysed frame
    (same size and orientation as the one passed to ``process``)."""
    if mode == "off":
        return
    pixels = face.pixels
    if mode == "mesh":
        _draw_set(frame, pixels, "tesselation", COLOR_MESH, 1)
        _draw_set(frame, pixels, "contours", COLOR_CONTOUR, 1)
    elif mode == "contours":
        _draw_set(frame, pixels, "contours", COLOR_CONTOUR, 1)
    elif mode == "points":
        for x, y in pixels:
            cv2.circle(frame, (int(x), int(y)), 1, COLOR_POINT, -1, cv2.LINE_AA)
    if face.count >= NUM_LANDMARKS and mode != "points":
        _draw_set(frame, pixels, "left_iris", COLOR_IRIS, 1)
        _draw_set(frame, pixels, "right_iris", COLOR_IRIS, 1)


def draw_hud(frame: np.ndarray, fps: float, face: Optional[FaceLandmarks],
             stats: DetectorStats, mode: str, gray: bool) -> None:
    """Status overlay. Call *after* any display mirroring so text reads normally."""
    height = frame.shape[0]
    lines = [
        ("Stage 2 - MediaPipe Face Landmarker", (255, 255, 255)),
        ("FPS {:5.1f}   inference {:5.1f} ms".format(
            fps, face.inference_ms if face else stats.inference_ms[-1] if stats.inference_ms else 0.0),
         (0, 255, 0)),
    ]
    if face is not None:
        lines.append(("FACE FOUND  {} landmarks".format(face.count), (0, 255, 0)))
    else:
        lines.append(("NO FACE DETECTED", (0, 0, 255)))
    lines.append(("detection rate {:5.1%} over {} frames".format(stats.detection_rate, stats.frames),
                  (200, 200, 200)))
    lines.append(("draw: {}   gray input: {}".format(mode, "ON" if gray else "off"),
                  (200, 200, 200)))

    y = 24
    for text, color in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 22

    cv2.putText(frame, "q=quit  m=draw mode  g=gray input  s=snapshot", (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# --- live demo ---------------------------------------------------------------

WINDOW_NAME = "Drowsiness Detection - Stage 2 (Face Landmarks)"


def run_demo(source: FrameSource, detector: FaceLandmarkDetector, mode: str = "contours",
             mirror: bool = True, show_window: bool = True, max_frames: int = 0) -> int:
    fps_counter = FPSCounter()
    mode_index = DRAW_MODES.index(mode)
    started = time.perf_counter()

    with source, detector:
        print("[landmarks] Camera : {}".format(source.description))
        print("[landmarks] Model  : {}".format(detector.config.model_path.name))
        print("[landmarks] Keys   : q/ESC quit | m draw mode | g gray input | s snapshot")
        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = source.read()
            if frame is None:
                print("[landmarks] No frame received - stream ended or camera lost.")
                break

            face = detector.process(frame)  # always on the un-mirrored frame
            fps = fps_counter.tick()

            if show_window:
                display = frame.copy()
                if face is not None:
                    draw_landmarks(display, face, DRAW_MODES[mode_index])
                if mirror:
                    display = cv2.flip(display, 1)
                draw_hud(display, fps, face, detector.stats, DRAW_MODES[mode_index],
                         detector.config.grayscale_input)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[landmarks] Quit key pressed.")
                    break
                if key == ord("m"):
                    mode_index = (mode_index + 1) % len(DRAW_MODES)
                if key == ord("g"):
                    detector.config.grayscale_input = not detector.config.grayscale_input
                if key == ord("s"):
                    print("[landmarks] Saved {}".format(save_snapshot(display)))
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[landmarks] Preview window closed.")
                    break

            if max_frames and detector.stats.frames >= max_frames:
                print("[landmarks] Reached --max-frames {}.".format(max_frames))
                break

    if show_window:
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    elapsed = time.perf_counter() - started
    stats = detector.stats
    print("[landmarks] {} frames in {:.1f}s -> {:.1f} FPS end to end".format(
        stats.frames, elapsed, stats.frames / elapsed if elapsed else 0.0))
    print("[landmarks] face detected in {}/{} frames ({:.1%})".format(
        stats.faces, stats.frames, stats.detection_rate))
    print("[landmarks] inference: mean {:.1f} ms, median {:.1f} ms".format(
        stats.mean_inference_ms, stats.median_inference_ms))
    return 0


# --- self-test ---------------------------------------------------------------

def self_test() -> int:
    """Checks that need no camera: model integrity, model load, no-face path,
    coordinate helpers, and the drawing code."""
    print("[self-test] mediapipe {} | OpenCV {} | NumPy {}".format(
        mp.__version__, cv2.__version__, np.__version__))

    ensure_model()
    print("[self-test] model present, hash verified")

    with FaceLandmarkDetector() as detector:
        # Synthetic frames contain no face: every result must be None, without
        # any exception, and the timestamps must keep MediaPipe happy.
        blank = np.full((480, 640, 3), 40, dtype=np.uint8)
        results = [detector.process(blank) for _ in range(5)]
        assert all(r is None for r in results), "synthetic frame produced a face"
        print("[self-test] no-face path OK on 5 synthetic frames "
              "(mean inference {:.1f} ms)".format(detector.stats.mean_inference_ms))

        detector.config.grayscale_input = True
        assert detector.process(blank) is None
        print("[self-test] grayscale input path OK")

    # Coordinate helpers on a fabricated landmark set.
    normalized = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    normalized[:, 0] = np.linspace(0.1, 0.9, NUM_LANDMARKS)
    normalized[:, 1] = 0.5
    face = FaceLandmarks(normalized, (640, 480), 0, 0.0)
    assert face.pixel(0) == (64, 240), face.pixel(0)
    assert face.pixel(NUM_LANDMARKS - 1) == (576, 240), face.pixel(NUM_LANDMARKS - 1)
    assert face.points([0]).dtype == np.float32
    assert face.bounding_box(margin=5) == (59, 235, 581, 245), face.bounding_box(margin=5)
    print("[self-test] pixel/points/bounding_box helpers OK")

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    for mode in DRAW_MODES:
        draw_landmarks(canvas, face, mode)
    draw_hud(canvas, 30.0, face, DetectorStats(frames=1, faces=1, inference_ms=[5.0]),
             "contours", False)
    assert canvas.any(), "drawing produced an empty image"
    print("[self-test] drawing OK for modes {}".format(", ".join(DRAW_MODES)))

    print("[self-test] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: live facial landmarks with MediaPipe.")
    parser.add_argument("--device", default="0", help="Camera index or video path (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mode", choices=DRAW_MODES, default="contours", help="Initial draw style")
    parser.add_argument("--gray", action="store_true",
                        help="Feed grayscale frames to the model (IR camera rehearsal)")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the display")
    parser.add_argument("--no-window", action="store_true",
                        help="Headless: run detection and print statistics only")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after this many frames (0 = run until quit)")
    parser.add_argument("--self-test", action="store_true", help="Run checks that need no camera")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()

    device = int(args.device) if args.device.isdigit() else args.device
    # flip_horizontal=False: detection must see the true frame so that
    # left/right landmarks match the subject's anatomy. Mirroring is display-only.
    camera = CameraConfig(device=device, width=args.width, height=args.height,
                          flip_horizontal=False)
    detector = FaceLandmarkDetector(LandmarkConfig(grayscale_input=args.gray))

    try:
        return run_demo(create_source(camera), detector, mode=args.mode,
                        mirror=not args.no_mirror, show_window=not args.no_window,
                        max_frames=args.max_frames)
    except (CameraError, LandmarkModelError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[landmarks] Interrupted by user.")
        cv2.destroyAllWindows()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
