"""Camera capture layer for the drowsiness detection system.

Stage 1 scope: open a camera, deliver BGR frames, show a live preview.

Everything downstream (MediaPipe, EAR/MAR, CNN, temporal analysis) will consume
frames through the ``FrameSource`` interface defined here, so the physical
camera can later be swapped -- laptop webcam -> USB webcam -> modified NoIR
webcam -- without touching any other module.

Run directly for a camera test::

    python -m src.capture                 # preview the default camera
    python -m src.capture --list          # probe which camera indices exist
    python -m src.capture --device 1      # preview the second camera
    python -m src.capture --self-test     # no hardware needed
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

# Project root = parent of src/. Used so snapshots land in the repo's data/
# folder no matter which directory the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"

# Preferred cv2 capture backends, in the order we try them.
# DirectShow first on Windows: MSMF is often slow to open (several seconds)
# and mis-reports properties on some laptop webcams.
if platform.system() == "Windows":
    _BACKEND_ORDER = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
else:
    _BACKEND_ORDER = [cv2.CAP_ANY]

_BACKEND_NAMES = {
    cv2.CAP_ANY: "ANY",
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_V4L2: "V4L2",
}


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or stops delivering frames."""


@dataclass
class CameraConfig:
    """All camera settings in one place.

    ``device`` is either an integer index (0 = default webcam) or a string:
    a video file path or a stream URL. Later stages change values here, not
    the code that reads frames.
    """

    device: Union[int, str] = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_horizontal: bool = True   # mirror view feels natural to the driver
    warmup_frames: int = 5         # frames discarded while auto-exposure settles
    max_read_retries: int = 5      # tolerate occasional dropped USB frames
    backend: Optional[int] = None  # None = try _BACKEND_ORDER automatically


class FrameSource(ABC):
    """Interface that every camera/video source implements.

    This abstraction is the point of Stage 1: Stage 13 swaps in the modified
    NoIR webcam by constructing a different source, and the rest of the
    pipeline stays unchanged.
    """

    @abstractmethod
    def open(self) -> None:
        """Acquire the device. Raises CameraError on failure."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if the stream has ended."""

    @abstractmethod
    def release(self) -> None:
        """Release the device. Safe to call more than once."""

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...

    @property
    def description(self) -> str:
        return self.__class__.__name__

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the source ends. Convenience for later stages."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class WebcamSource(FrameSource):
    """OpenCV-backed source: USB/laptop webcam, video file, or stream URL."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend_used: Optional[int] = None
        self._is_file = isinstance(self.config.device, str) and Path(
            str(self.config.device)
        ).exists()

    def open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return

        device = self.config.device
        if self.config.backend is not None:
            backends = [self.config.backend]
        elif isinstance(device, str):
            backends = [cv2.CAP_ANY]
        else:
            backends = _BACKEND_ORDER

        tried = []
        for backend in backends:
            cap = cv2.VideoCapture(device, backend)
            if cap.isOpened():
                self._cap = cap
                self._backend_used = backend
                break
            cap.release()
            tried.append(_BACKEND_NAMES.get(backend, str(backend)))

        if self._cap is None:
            raise CameraError(
                "Could not open camera device {!r} (tried backends: {}). "
                "Is another app using the camera, or is the index wrong? "
                "Run 'python -m src.capture --list' to see available devices.".format(
                    device, ", ".join(tried)
                )
            )

        if not self._is_file:
            self._apply_settings()
            self._warmup()

    def _apply_settings(self) -> None:
        """Request resolution/FPS. Drivers may ignore or clamp these values."""
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # A small buffer keeps the preview close to real time instead of
        # replaying a queue of stale frames after a slow processing step.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _warmup(self) -> None:
        assert self._cap is not None
        for _ in range(max(0, self.config.warmup_frames)):
            self._cap.read()

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise CameraError("read() called before open()")

        for attempt in range(self.config.max_read_retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                if self.config.flip_horizontal and not self._is_file:
                    frame = cv2.flip(frame, 1)
                return frame
            if self._is_file:
                return None  # end of video file
            time.sleep(0.01 * (attempt + 1))

        return None  # camera unplugged or driver stalled

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Resolution the driver actually gave us (may differ from requested)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def reported_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    @property
    def description(self) -> str:
        backend = _BACKEND_NAMES.get(self._backend_used, "?")
        width, height = self.actual_resolution
        return "WebcamSource(device={!r}, backend={}, {}x{})".format(
            self.config.device, backend, width, height
        )


class SyntheticSource(FrameSource):
    """Generated frames -- lets the pipeline run with no camera attached.

    Used by ``--self-test`` now, and useful later for debugging on a machine
    without the hardware.
    """

    def __init__(self, config: Optional[CameraConfig] = None, n_frames: int = 0) -> None:
        self.config = config or CameraConfig()
        self.n_frames = n_frames  # 0 = unlimited
        self._index = 0
        self._open = False

    def open(self) -> None:
        self._open = True
        self._index = 0

    def read(self) -> Optional[np.ndarray]:
        if not self._open:
            raise CameraError("read() called before open()")
        if self.n_frames and self._index >= self.n_frames:
            return None

        height, width = self.config.height, self.config.width
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = int((width - 80) * (0.5 + 0.5 * np.sin(self._index / 15.0)))
        cv2.circle(frame, (x + 40, height // 2), 40, (0, 200, 255), -1)
        cv2.putText(frame, "SYNTHETIC {:04d}".format(self._index), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._index += 1
        return frame

    def release(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def description(self) -> str:
        return "SyntheticSource({}x{})".format(self.config.width, self.config.height)


def create_source(config: Optional[CameraConfig] = None,
                  synthetic: bool = False) -> FrameSource:
    """Factory used by every later stage. The source is chosen here, once."""
    config = config or CameraConfig()
    return SyntheticSource(config) if synthetic else WebcamSource(config)


def list_cameras(max_index: int = 5) -> List[dict]:
    """Probe camera indices 0..max_index-1 and report which ones deliver frames.

    Useful when the USB webcam is added in Stage 13 and we need its index.
    """
    found = []
    for index in range(max_index):
        for backend in _BACKEND_ORDER:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                found.append({
                    "index": index,
                    "backend": _BACKEND_NAMES.get(backend, str(backend)),
                    "resolution": (frame.shape[1], frame.shape[0]),
                })
                break
    return found


class FPSCounter:
    """Rolling-average FPS over the last N frame intervals."""

    def __init__(self, window: int = 30) -> None:
        self._intervals: deque = deque(maxlen=window)
        self._last: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            self._intervals.append(now - self._last)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._intervals:
            return 0.0
        mean = sum(self._intervals) / len(self._intervals)
        return 1.0 / mean if mean > 0 else 0.0


def save_snapshot(frame: np.ndarray, directory: Path = SNAPSHOT_DIR) -> Path:
    """Write a timestamped PNG. Used to record baseline camera image quality."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "snapshot_{:%Y%m%d_%H%M%S_%f}.png".format(datetime.now())
    cv2.imwrite(str(path), frame)
    return path


WINDOW_NAME = "Drowsiness Detection - Stage 1 (Camera Test)"


def run_preview(source: FrameSource, show_overlay: bool = True) -> int:
    """Live preview loop. Returns a process exit code."""
    fps_counter = FPSCounter()
    frame_count = 0
    started = time.perf_counter()

    with source:
        print("[capture] Opened: {}".format(source.description))
        print("[capture] Keys:  q / ESC = quit   s = save snapshot")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = source.read()
            if frame is None:
                print("[capture] No frame received - stream ended or camera lost.")
                break

            frame_count += 1
            fps = fps_counter.tick()

            display = frame.copy()
            if show_overlay:
                height, width = display.shape[:2]
                cv2.putText(display, "{}x{}  FPS: {:5.1f}".format(width, height, fps),
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "q=quit  s=snapshot", (10, height - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                print("[capture] Quit key pressed.")
                break
            if key == ord("s"):
                print("[capture] Saved {}".format(save_snapshot(frame)))

            # User closed the window with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[capture] Preview window closed.")
                break

    cv2.destroyAllWindows()
    for _ in range(4):  # flush pending GUI events so the window really closes
        cv2.waitKey(1)

    elapsed = time.perf_counter() - started
    average = frame_count / elapsed if elapsed > 0 else 0.0
    print("[capture] {} frames in {:.1f}s  (average {:.1f} FPS)".format(
        frame_count, elapsed, average))
    return 0


def self_test() -> int:
    """Headless check that the capture layer works without any camera."""
    print("[self-test] OpenCV {} | NumPy {}".format(cv2.__version__, np.__version__))

    config = CameraConfig(width=320, height=240)
    with create_source(config, synthetic=True) as source:
        frames = [source.read() for _ in range(5)]
    assert all(f is not None and f.shape == (240, 320, 3) for f in frames), \
        "synthetic source returned an unexpected frame shape"

    with SyntheticSource(config, n_frames=3) as limited:
        assert len(list(limited.frames())) == 3, "frames() did not stop at n_frames"

    counter = FPSCounter()
    for _ in range(3):
        counter.tick()
        time.sleep(0.01)
    assert counter.fps > 0, "FPS counter did not produce a value"

    print("[self-test] PASS - frame source, shapes, iteration and FPS counter OK.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1 camera test for the drowsiness detection system.")
    parser.add_argument("--device", default="0",
                        help="Camera index (0, 1, ...) or a video file path. Default: 0")
    parser.add_argument("--width", type=int, default=640,
                        help="Requested frame width (default 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Requested frame height (default 480)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Requested camera FPS (default 30)")
    parser.add_argument("--no-flip", action="store_true",
                        help="Do not mirror the image horizontally")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto",
                        help="Force an OpenCV capture backend (Windows troubleshooting)")
    parser.add_argument("--list", action="store_true",
                        help="List working camera indices and exit")
    parser.add_argument("--self-test", action="store_true",
                        help="Run a headless check that needs no camera")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return self_test()

    if args.list:
        cameras = list_cameras()
        if not cameras:
            print("No cameras found. Check Windows camera privacy settings "
                  "and close other apps that may hold the camera.")
            return 1
        print("Available cameras:")
        for cam in cameras:
            print("  index {}  backend {}  {}x{}".format(
                cam["index"], cam["backend"], cam["resolution"][0], cam["resolution"][1]))
        return 0

    device: Union[int, str] = int(args.device) if args.device.isdigit() else args.device
    backend_map = {"auto": None, "dshow": cv2.CAP_DSHOW,
                   "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}

    config = CameraConfig(
        device=device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip_horizontal=not args.no_flip,
        backend=backend_map[args.backend],
    )

    try:
        return run_preview(create_source(config))
    except CameraError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[capture] Interrupted by user.")
        cv2.destroyAllWindows()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
