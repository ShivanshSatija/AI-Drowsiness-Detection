"""Repeatable acceptance + baseline test for the camera capture path.

Stage 1 scope only: this measures the camera and the capture layer. It does not
detect faces, eyes or drowsiness, and it loads no models.

Two jobs:

1. **Acceptance test** -- prove the real camera actually works: it opens, it
   delivers frames of the right shape, the image is live rather than black or
   frozen, frames are not dropped, and the device can be released and reopened
   without leaking a handle.

2. **Baseline record** -- write every measurement to a JSON file under
   ``evaluation/results/`` so the same run can be repeated later and compared.
   This is what Stage 13 needs: run it once before removing the IR-cut filter,
   again after the NoIR conversion, and again with the 850 nm illuminator on,
   then diff the results.

The image statistics are chosen for that later comparison:

* ``mean_brightness``      -- how much light reaches the sensor
* ``channel_means``        -- B/G/R balance; removing the IR-cut filter makes
                              the channels converge as IR leaks into all three
* ``focus_laplacian_var``  -- sharpness, for refocusing after reassembly
* ``saturated_fraction``   -- pixels at/near 255, i.e. IR hotspots
* ``grid_brightness``      -- 3x3 map, i.e. how evenly the scene is lit

Usage::

    python evaluation/camera_baseline.py --label unmodified-laptop-webcam
    python evaluation/camera_baseline.py --label noir-ir-dark --notes "IR on, lights off"
    python evaluation/camera_baseline.py --compare results/a.json results/b.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Import the project's capture layer without hard-coding an absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.capture import (CameraConfig, CameraError, WebcamSource,  # noqa: E402
                         list_cameras, save_snapshot)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    """Collects checks and measurements, then prints and serialises them."""

    def __init__(self, label: str, notes: str) -> None:
        self.label = label
        self.notes = notes
        self.checks: List[dict] = []
        self.data: dict = {}

    def check(self, name: str, ok: bool, detail: str = "",
              severity_if_bad: str = FAIL) -> bool:
        status = PASS if ok else severity_if_bad
        self.checks.append({"name": name, "status": status, "detail": detail})
        print("  [{}] {}{}".format(status, name, " - " + detail if detail else ""))
        return ok

    def record(self, key: str, value) -> None:
        self.data[key] = value

    @property
    def failed(self) -> List[str]:
        return [c["name"] for c in self.checks if c["status"] == FAIL]

    @property
    def warned(self) -> List[str]:
        return [c["name"] for c in self.checks if c["status"] == WARN]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "notes": self.notes,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "measurements": self.data,
            "checks": self.checks,
            "verdict": FAIL if self.failed else (WARN if self.warned else PASS),
        }

    def save(self, directory: Path = RESULTS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.label)
        path = directory / "camera_baseline_{}_{:%Y%m%d_%H%M%S}.json".format(
            safe, datetime.now())
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def frame_statistics(frame: np.ndarray) -> dict:
    """Image statistics that make a NoIR/IR change visible in the numbers."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # 3x3 brightness map: uneven IR illumination shows up as a bright centre
    # cell surrounded by dark corners.
    grid = []
    for row in range(3):
        cells = []
        for col in range(3):
            cell = gray[row * height // 3:(row + 1) * height // 3,
                        col * width // 3:(col + 1) * width // 3]
            cells.append(round(float(cell.mean()), 1))
        grid.append(cells)

    flat = grid[0] + grid[1] + grid[2]
    means = frame.reshape(-1, 3).mean(axis=0)  # BGR order

    return {
        "mean_brightness": round(float(gray.mean()), 2),
        "std_brightness": round(float(gray.std()), 2),
        "min_brightness": int(gray.min()),
        "max_brightness": int(gray.max()),
        "channel_means": {"blue": round(float(means[0]), 2),
                          "green": round(float(means[1]), 2),
                          "red": round(float(means[2]), 2)},
        "focus_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "saturated_fraction": round(float((gray >= 250).mean()), 5),
        "dark_fraction": round(float((gray <= 5).mean()), 5),
        "grid_brightness": grid,
        "grid_uniformity_ratio": round(min(flat) / max(flat), 3) if max(flat) > 0 else 0.0,
    }


def run_baseline(args: argparse.Namespace) -> Report:
    report = Report(args.label, args.notes)
    print("Camera baseline: label={!r}".format(args.label))
    if args.notes:
        print("Notes: {}".format(args.notes))
    print("Environment: Python {} | OpenCV {} | NumPy {}".format(
        platform.python_version(), cv2.__version__, np.__version__))

    # --- 1. enumeration --------------------------------------------------
    print("\n[1/6] Enumerating cameras")
    t0 = time.perf_counter()
    cameras = list_cameras(max_index=args.max_index)
    print("  probed indices 0..{} in {:.1f}s".format(args.max_index - 1,
                                                     time.perf_counter() - t0))
    for cam in cameras:
        print("    index {}  backend {}  {}x{}".format(
            cam["index"], cam["backend"], cam["resolution"][0], cam["resolution"][1]))
    report.record("cameras_detected", cameras)
    report.check("at least one camera detected", len(cameras) > 0)
    if not cameras:
        return report

    # --- 2. open ---------------------------------------------------------
    print("\n[2/6] Opening device {}".format(args.device))
    config = CameraConfig(device=args.device, width=args.width,
                          height=args.height, fps=args.fps)
    source = WebcamSource(config)
    t0 = time.perf_counter()
    try:
        source.open()
    except CameraError as exc:
        report.check("camera opens", False, str(exc))
        return report
    open_time = time.perf_counter() - t0

    resolution = source.actual_resolution
    print("  {}".format(source.description))
    print("  open time {:.2f}s | driver-reported FPS {:.1f}".format(
        open_time, source.reported_fps))
    report.record("open_time_s", round(open_time, 3))
    report.record("backend", source.description.split("backend=")[-1].split(",")[0])
    report.record("resolution", list(resolution))
    report.record("driver_reported_fps", round(source.reported_fps, 2))
    report.check("camera opens", True)
    report.check("open time under 5s", open_time < 5.0, "{:.2f}s".format(open_time))
    report.check("resolution matches request",
                 resolution == (args.width, args.height),
                 "got {}x{}, requested {}x{}".format(resolution[0], resolution[1],
                                                     args.width, args.height),
                 severity_if_bad=WARN)

    # --- 3. capture ------------------------------------------------------
    print("\n[3/6] Capturing {} frames".format(args.frames))
    latencies, brightness_series, dropped = [], [], 0
    first_frame = last_frame = None
    t_start = time.perf_counter()
    for _ in range(args.frames):
        t = time.perf_counter()
        frame = source.read()
        latencies.append((time.perf_counter() - t) * 1000.0)
        if frame is None:
            dropped += 1
            continue
        first_frame = first_frame if first_frame is not None else frame
        last_frame = frame
        brightness_series.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    elapsed = time.perf_counter() - t_start

    captured = args.frames - dropped
    if captured == 0:
        report.check("frames captured", False, "every read returned None")
        source.release()
        return report

    measured_fps = captured / elapsed
    latencies_sorted = sorted(latencies)
    percentile = lambda p: latencies_sorted[min(len(latencies_sorted) - 1,
                                                int(len(latencies_sorted) * p))]
    drop_rate = dropped / args.frames

    print("  {} frames in {:.2f}s -> {:.1f} FPS".format(captured, elapsed, measured_fps))
    print("  dropped {} ({:.1%})".format(dropped, drop_rate))
    print("  read latency: median {:.1f} ms | p95 {:.1f} ms | max {:.1f} ms".format(
        percentile(0.50), percentile(0.95), latencies_sorted[-1]))

    report.record("frames_requested", args.frames)
    report.record("frames_captured", captured)
    report.record("dropped_frames", dropped)
    report.record("drop_rate", round(drop_rate, 4))
    report.record("elapsed_s", round(elapsed, 3))
    report.record("measured_fps", round(measured_fps, 2))
    report.record("latency_ms", {"median": round(percentile(0.50), 2),
                                 "p95": round(percentile(0.95), 2),
                                 "max": round(latencies_sorted[-1], 2)})

    report.check("frame shape is {}x{}x3".format(args.height, args.width),
                 first_frame.shape == (args.height, args.width, 3),
                 str(first_frame.shape), severity_if_bad=WARN)
    report.check("drop rate at or under 2%", drop_rate <= 0.02,
                 "{:.1%}".format(drop_rate))
    report.check("measured FPS at least {}".format(args.min_fps),
                 measured_fps >= args.min_fps, "{:.1f} FPS".format(measured_fps))
    report.check("image is live, not a frozen still",
                 float(np.std(brightness_series)) > 0.0,
                 "brightness std over time {:.3f}".format(float(np.std(brightness_series))))

    # --- 4. image statistics --------------------------------------------
    print("\n[4/6] Image statistics (last frame)")
    stats = frame_statistics(last_frame)
    report.record("image_stats", stats)
    print("  mean brightness {:.1f} | std {:.1f} | min {} | max {}".format(
        stats["mean_brightness"], stats["std_brightness"],
        stats["min_brightness"], stats["max_brightness"]))
    print("  channel means  B {blue} | G {green} | R {red}".format(**stats["channel_means"]))
    print("  focus (Laplacian var) {:.1f}".format(stats["focus_laplacian_var"]))
    print("  saturated pixels {:.2%} | very dark pixels {:.2%}".format(
        stats["saturated_fraction"], stats["dark_fraction"]))
    print("  3x3 brightness map (illumination evenness):")
    for row in stats["grid_brightness"]:
        print("    {:>7.1f} {:>7.1f} {:>7.1f}".format(*row))
    print("  uniformity ratio (dimmest/brightest cell) {:.3f}".format(
        stats["grid_uniformity_ratio"]))

    # Not a pass/fail in normal light: a dark room with the IR illuminator off
    # is *supposed* to be dark. Reported so the operator can judge it.
    report.check("image is not black", stats["mean_brightness"] > 10,
                 "mean brightness {:.1f}".format(stats["mean_brightness"]),
                 severity_if_bad=WARN)

    # --- 5. optional sample frame ---------------------------------------
    if args.save_frame:
        path = save_snapshot(last_frame)
        report.record("sample_frame", str(path.relative_to(PROJECT_ROOT)))
        print("\n[5/6] Sample frame saved: {}".format(path))
        print("      (data/snapshots/ is git-ignored - it may contain your face)")
    else:
        print("\n[5/6] Sample frame not saved (pass --save-frame to keep one)")

    # --- 6. release and reopen ------------------------------------------
    print("\n[6/6] Release, then reopen {} times (handle-leak check)".format(args.reopens))
    source.release()
    report.check("released cleanly", not source.is_open)
    reopen_times = []
    for i in range(1, args.reopens + 1):
        probe = WebcamSource(CameraConfig(device=args.device))
        t0 = time.perf_counter()
        ok = False
        try:
            probe.open()
            ok = probe.read() is not None
            reopen_times.append(round(time.perf_counter() - t0, 3))
            print("  reopen {}: OK in {:.2f}s".format(i, reopen_times[-1]))
        except CameraError as exc:
            print("  reopen {}: FAILED - {}".format(i, exc))
        finally:
            probe.release()
        report.check("reopen cycle {}".format(i), ok)
    report.record("reopen_times_s", reopen_times)

    return report


def print_verdict(report: Report, saved: Path) -> int:
    print("\n" + "=" * 66)
    verdict = report.to_dict()["verdict"]
    print("VERDICT: {}   ({} checks, {} failed, {} warnings)".format(
        verdict, len(report.checks), len(report.failed), len(report.warned)))
    if report.failed:
        print("  failed: {}".format(", ".join(report.failed)))
    if report.warned:
        print("  warnings: {}".format(", ".join(report.warned)))
    print("Saved: {}".format(saved))
    print("=" * 66)
    return 1 if report.failed else 0


def resolve_result(path_str: str) -> Path:
    """Accept a full path, or just a filename living in evaluation/results/."""
    path = Path(path_str)
    if path.exists():
        return path
    fallback = RESULTS_DIR / path.name
    if fallback.exists():
        return fallback
    raise SystemExit("Result file not found: {}\n  (also tried {})".format(path, fallback))


def compare(path_a: Path, path_b: Path) -> int:
    """Side-by-side diff of two baseline runs -- the Stage 13 before/after."""
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))

    print("A: {:<28} {}".format(a["label"], a["timestamp"]))
    print("B: {:<28} {}".format(b["label"], b["timestamp"]))
    if a.get("notes") or b.get("notes"):
        print("A notes: {}\nB notes: {}".format(a.get("notes", ""), b.get("notes", "")))
    print()

    rows = [
        ("measured FPS", ("measurements", "measured_fps")),
        ("drop rate", ("measurements", "drop_rate")),
        ("open time (s)", ("measurements", "open_time_s")),
        ("median latency (ms)", ("measurements", "latency_ms", "median")),
        ("mean brightness", ("measurements", "image_stats", "mean_brightness")),
        ("std brightness", ("measurements", "image_stats", "std_brightness")),
        ("blue mean", ("measurements", "image_stats", "channel_means", "blue")),
        ("green mean", ("measurements", "image_stats", "channel_means", "green")),
        ("red mean", ("measurements", "image_stats", "channel_means", "red")),
        ("focus (Laplacian)", ("measurements", "image_stats", "focus_laplacian_var")),
        ("saturated fraction", ("measurements", "image_stats", "saturated_fraction")),
        ("uniformity ratio", ("measurements", "image_stats", "grid_uniformity_ratio")),
    ]

    def dig(source: dict, keys):
        for key in keys:
            if not isinstance(source, dict) or key not in source:
                return None
            source = source[key]
        return source

    print("{:<22} {:>12} {:>12} {:>12}".format("metric", "A", "B", "change"))
    print("-" * 60)
    for title, keys in rows:
        va, vb = dig(a, keys), dig(b, keys)
        if va is None or vb is None:
            continue
        delta = "{:+.3f}".format(vb - va) if isinstance(va, (int, float)) else ""
        print("{:<22} {:>12} {:>12} {:>12}".format(title, va, vb, delta))
    print("-" * 60)
    print("verdict          A={}  B={}".format(a.get("verdict"), b.get("verdict")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeatable camera acceptance test and baseline record (Stage 1 scope).")
    parser.add_argument("--label", default="unlabelled",
                        help="Name for this run, e.g. unmodified-laptop-webcam, noir-ir-dark")
    parser.add_argument("--notes", default="",
                        help="Free text: lighting, distance, glasses, IR on/off")
    parser.add_argument("--device", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument("--frames", type=int, default=120,
                        help="Frames to capture for the measurement (default 120)")
    parser.add_argument("--reopens", type=int, default=3,
                        help="Release/reopen cycles for the handle-leak check")
    parser.add_argument("--max-index", type=int, default=3,
                        help="Highest camera index to probe during enumeration")
    parser.add_argument("--min-fps", type=float, default=10.0,
                        help="FPS below which the run is marked FAIL")
    parser.add_argument("--save-frame", action="store_true",
                        help="Keep one sample frame in data/snapshots/ (git-ignored)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the JSON result file")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Compare two saved baseline files and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.compare:
        return compare(resolve_result(args.compare[0]), resolve_result(args.compare[1]))

    report = run_baseline(args)
    saved = Path("(not saved)")
    if not args.no_save:
        saved = report.save()
    return print_verdict(report, saved)


if __name__ == "__main__":
    raise SystemExit(main())
