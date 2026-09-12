"""Two people in view: does the detector keep measuring the driver?

Builds a synthetic video from one frontal face photo pasted at two sizes, then
checks which face ``FaceLandmarkDetector`` returns in each phase:

    A   0- 99  driver alone
    B 100-199  a second (smaller) face appears at the right
    C 200-299  the second face slides onto the driver anchor itself
    D 300-349  driver gone, second face alone inside the zone
    E 350-399  second face alone, far corner, outside the zone

The anchor is deliberately placed nearer the second face's final position so
that phases B and C separate the continuity lock from the nearest-anchor rule.

Any frontal portrait works. The project was verified with MediaPipe's own test
image (https://storage.googleapis.com/mediapipe-assets/business-person.png),
which is not redistributed here::

    python evaluation/two_face_test.py --portrait path/to/portrait.png

Exit code 0 when every check passes. Nothing here depends on a camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.landmarks import FaceLandmarkDetector, LandmarkConfig, face_center  # noqa: E402

W, H = 640, 480
ANCHOR = (0.58, 0.62)
DRIVER_POS = (0.27, 0.45)
SECOND_POS = (0.85, 0.60)
OUTSIDE_POS = (0.15, 0.20)


def locate_face(image: np.ndarray) -> Tuple[int, int]:
    """Pixel centre of the face in a still image, found with the same detector."""
    frame = cv2.resize(image, (W, H))
    with FaceLandmarkDetector(LandmarkConfig(driver_zone_radius=2.0)) as det:
        face = None
        for _ in range(3):  # VIDEO mode may need a frame or two
            face = det.process(frame) or face
    if face is None:
        raise SystemExit("No face found in the portrait - use a clear frontal photo.")
    cx, cy = face_center(face.normalized)
    return int(cx * image.shape[1]), int(cy * image.shape[0])


def scaled(image: np.ndarray, face_xy: Tuple[int, int], height: int):
    scale = height / image.shape[0]
    img = cv2.resize(image, (int(image.shape[1] * scale), height))
    return img, (int(face_xy[0] * scale), int(face_xy[1] * scale))


def paste(canvas: np.ndarray, img: np.ndarray, face_offset: Tuple[int, int], pos: Tuple[float, float]) -> None:
    """Paste img so that its face centre lands at normalised position pos."""
    x0, y0 = int(pos[0] * W) - face_offset[0], int(pos[1] * H) - face_offset[1]
    h, w = img.shape[:2]
    xs, ys, xe, ye = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xe > xs and ye > ys:
        canvas[ys:ye, xs:xe] = img[ys - y0:ye - y0, xs - x0:xe - x0]


def build_frames(portrait: np.ndarray) -> List[np.ndarray]:
    face_xy = locate_face(portrait)
    # Face sizes must stay within what MediaPipe's short-range detector finds:
    # a 240-px-tall portrait (about 70 px face) was not detected at all.
    big, big_face = scaled(portrait, face_xy, 400)
    small, small_face = scaled(portrait, face_xy, 360)
    frames = []
    for i in range(400):
        canvas = np.full((H, W, 3), 60, dtype=np.uint8)
        if i < 300:
            paste(canvas, big, big_face, DRIVER_POS)
        if 100 <= i < 200:
            paste(canvas, small, small_face, SECOND_POS)
        elif 200 <= i < 300:
            t = (i - 200) / 99.0
            pos = (SECOND_POS[0] + (ANCHOR[0] - SECOND_POS[0]) * t,
                   SECOND_POS[1] + (ANCHOR[1] - SECOND_POS[1]) * t)
            paste(canvas, small, small_face, pos)
        elif 300 <= i < 350:
            paste(canvas, small, small_face, ANCHOR)
        elif i >= 350:
            paste(canvas, small, small_face, OUTSIDE_POS)
        frames.append(canvas)
    return frames


class Checks:
    def __init__(self) -> None:
        self.failed: List[str] = []

    def __call__(self, label: str, ok: bool, detail: str = "") -> None:
        print("  [{}] {}{}".format("PASS" if ok else "FAIL", label, " - " + detail if detail else ""))
        if not ok:
            self.failed.append(label)


def near(center: Optional[np.ndarray], target, tol: float = 0.08) -> bool:
    return center is not None and float(np.linalg.norm(center - np.asarray(target))) <= tol


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Two-face driver-selection test (no camera needed).")
    parser.add_argument("--portrait", type=Path, required=True, help="A clear frontal face photo")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "two_faces.avi",
                        help="Where to write the synthetic video (default data/two_faces.avi, git-ignored)")
    args = parser.parse_args(argv)

    portrait = cv2.imread(str(args.portrait))
    if portrait is None:
        raise SystemExit("Could not read {}".format(args.portrait))

    frames = build_frames(portrait)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    print("synthetic video: {} frames -> {}".format(len(frames), args.out))
    print("replay it in the live demo:  python -m src.features --device {}".format(args.out))

    records = []  # (face count, selected centre or None, selection reason)
    with FaceLandmarkDetector(LandmarkConfig(driver_anchor=ANCHOR)) as det:
        for f in frames:
            face = det.process(f)
            records.append((det.last_face_count,
                            face_center(face.normalized) if face is not None else None,
                            face.selection if face is not None else None))
        stats = det.stats
    print("inference median {:.1f} ms | two-face frames {} | no-driver frames {} | duplicates removed {}".format(
        stats.median_inference_ms, stats.multi_face_frames, stats.no_driver_frames, stats.duplicate_detections))

    check = Checks()

    def phase(name, a, b, expect_count, expect_target, expect_reason):
        recs = records[a:b]
        count_ok = sum(1 for c, _, _ in recs if c == expect_count) / len(recs)
        check("{}: {} face(s) seen".format(name, expect_count), count_ok >= 0.9, "{:.0%}".format(count_ok))
        selected = [r for r in recs if r[1] is not None]
        if expect_target is None:
            none_rate = 1 - len(selected) / len(recs)
            check("{}: no driver returned".format(name), none_rate >= 0.9, "{:.0%}".format(none_rate))
            return
        on_target = sum(1 for r in selected if near(r[1], expect_target)) / max(1, len(selected))
        reasons = sum(1 for r in selected if r[2] == expect_reason) / max(1, len(selected))
        check("{}: driver returned".format(name), len(selected) / len(recs) >= 0.9)
        check("{}: the expected face is selected".format(name), on_target >= 0.95, "{:.0%}".format(on_target))
        check("{}: reason '{}'".format(name, expect_reason), reasons >= 0.9, "{:.0%}".format(reasons))

    print("\nA  driver alone")
    phase("A", 0, 100, 1, DRIVER_POS, "single")
    print("B  second face appears at the right")
    phase("B", 100, 200, 2, DRIVER_POS, "lock")
    print("C  second face slides onto the anchor")
    phase("C", 200, 300, 2, DRIVER_POS, "lock")
    print("D  driver gone, second face alone inside the zone")
    phase("D", 300, 350, 1, ANCHOR, "single")
    print("E  second face alone, outside the zone")
    phase("E", 350, 400, 1, None, None)

    print("\nfresh start with both faces present (no lock yet)")
    with FaceLandmarkDetector(LandmarkConfig(driver_anchor=ANCHOR)) as fresh:
        picks = []
        for f in frames[150:160]:
            face = fresh.process(f)
            picks.append(None if face is None else (face.selection, near(face_center(face.normalized), SECOND_POS, 0.1)))
    check("first pick is the face nearest the anchor", picks[0] is not None and picks[0] == ("nearest anchor", True))
    check("then kept by the lock", all(p is not None and p[0] == "lock" and p[1] for p in picks[1:]))

    print("\n" + "=" * 60)
    if check.failed:
        print("{} CHECK(S) FAILED: {}".format(len(check.failed), check.failed))
        return 1
    print("ALL TWO-FACE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
