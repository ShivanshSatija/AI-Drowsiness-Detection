"""Eye-region extraction and preprocessing for the eye-state CNN (Stage 5).

This module owns everything the CNN will ever see. Stage 5 provides the
preprocessing; Stages 7-8 add the network, its training utilities and live
inference to this same file, so training and inference share one import.

Pipeline for one eye::

    frame (BGR) + landmarks
        -> square crop centred on the eye, side = crop_scale x eye-corner distance,
           rotated so the eye corners are horizontal, edge-replicated if it
           leaves the frame                                  (crop_eye)
        -> grayscale                                          (to_grayscale)
        -> resize to size x size                              (resize_eye)
        -> float32, per-image standardisation: mean 0, std 1 (normalize_eye)

``preprocess_eye_image`` applies the last three steps to ANY eye image - a live
crop or an MRL dataset file - and is the single function both the training
pipeline (Stage 6/7) and live inference (Stage 8) must call. That is what
guarantees there is no train/inference mismatch: MRL images are already
grayscale eye crops, so on them the same function simply skips the colour
conversion.

Design choices that bind later stages
-------------------------------------
* No left/right flipping. The MRL Eye Dataset does not label which eye an
  image shows, so the classifier has to be side-agnostic; training
  augmentation adds horizontal flips instead.
* ``crop_scale`` (1.5) and ``size`` (64) are INITIAL values. MRL images are
  tight, roughly square eye crops; Stage 6 compares real MRL samples with live
  crops saved by this module (``e`` key / ``--dump-crops``) and adjusts the
  scale so both look alike before any training happens.
* Per-image standardisation (rather than dataset mean/std) removes global
  brightness and contrast differences between MRL's infrared sensors and our
  webcam / NoIR camera. Optional CLAHE (``equalize``) exists for Stage 7's
  ablation and is off by default.
* Validity here is geometric only (eye too narrow, crop partly outside the
  frame). Stage 4's frame gate is separate; Stage 8 requires both.

Run directly::

    python -m src.eye_cnn --self-test           # no camera needed
    python -m src.eye_cnn --image eye.png       # preprocess one file, print shapes/stats
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.capture import PROJECT_ROOT
from src.features import LEFT_EYE, RIGHT_EYE

CROP_DIR = PROJECT_ROOT / "data" / "eye_crops"


# --- configuration -----------------------------------------------------------

@dataclass
class EyePreprocessConfig:
    """Shared by live extraction and the training pipeline. Change values here,
    never in only one place."""

    size: int = 64                  # output side in pixels (square)
    crop_scale: float = 1.5         # crop side = crop_scale x eye-corner distance  (verify vs MRL, Stage 6)
    align_roll: bool = True         # rotate so the eye corners are horizontal before cropping
    equalize: bool = False          # CLAHE before standardisation (Stage 7 ablation option)
    min_eye_width_px: float = 15.0  # narrower eyes give unusable crops


# --- pure image preprocessing (identical for training and inference) --------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """uint8 single-channel image from BGR, BGRA or already-gray input."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def resize_eye(gray: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. INTER_AREA when shrinking (no aliasing),
    INTER_LINEAR when enlarging. Deterministic, so both pipelines agree."""
    if gray.shape[0] == size and gray.shape[1] == size:
        return gray.copy()
    shrinking = gray.shape[0] > size or gray.shape[1] > size
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def normalize_eye(gray_resized: np.ndarray, equalize: bool = False) -> np.ndarray:
    """float32 array with mean 0 and std 1 (zeros for a flat image)."""
    img = gray_resized
    if equalize:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)
    x = img.astype(np.float32) / 255.0
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(x.mean())) / std).astype(np.float32)


def preprocess_eye_image(image: np.ndarray,
                         config: Optional[EyePreprocessConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """THE shared entry point.

    Returns (gray_resized uint8 (size, size), tensor float32 (size, size)).
    Call it on MRL files when building the training set and on live crops at
    inference; nothing else may prepare CNN input.
    """
    config = config or EyePreprocessConfig()
    gray = resize_eye(to_grayscale(image), config.size)
    return gray, normalize_eye(gray, config.equalize)


# --- landmark-based extraction ----------------------------------------------

@dataclass
class EyeCrop:
    side: str                              # "left" or "right" - the subject's own
    raw: np.ndarray                        # square gray crop before resizing (for saving / inspection)
    gray: np.ndarray                       # (size, size) uint8 - what the development display shows
    tensor: np.ndarray                     # (size, size) float32 standardised - what the CNN will see
    corners: np.ndarray                    # (4, 2) crop square corners in the un-mirrored frame
    center: Tuple[float, float]            # eye centre in frame pixels
    eye_width_px: float                    # corner-to-corner distance
    angle_deg: float                       # eye-corner line angle in the image (roll)
    valid: bool
    reason: str = ""


def eye_geometry(face, indices: Tuple[int, ...]) -> Tuple[np.ndarray, float, float]:
    """(centre xy, corner distance px, corner-line angle deg) from the six EAR landmarks."""
    pts = face.points(indices)                     # (6, 2) float32, p1..p6
    center = pts.mean(axis=0)
    corner = pts[3] - pts[0]                       # p4 - p1, points towards +x for both eyes
    width = float(np.linalg.norm(corner))
    angle = math.degrees(math.atan2(float(corner[1]), float(corner[0])))
    return center, width, angle


def crop_eye(gray_frame: np.ndarray, face, indices: Tuple[int, ...], side: str,
             config: Optional[EyePreprocessConfig] = None) -> EyeCrop:
    """Crop one eye from a grayscale frame using its landmarks."""
    config = config or EyePreprocessConfig()
    height, width = gray_frame.shape[:2]
    center, eye_width, angle = eye_geometry(face, indices)

    side_px = max(4, int(round(config.crop_scale * max(eye_width, 1.0))))
    rotation = angle if config.align_roll else 0.0
    # Rotate about the eye centre so the corners are level, then shift so the
    # centre lands in the middle of a side_px x side_px output: one warp does
    # rotation, cropping and edge padding together.
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rotation, 1.0)
    matrix[0, 2] += side_px / 2.0 - float(center[0])
    matrix[1, 2] += side_px / 2.0 - float(center[1])
    raw = cv2.warpAffine(gray_frame, matrix, (side_px, side_px),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    inverse = cv2.invertAffineTransform(matrix)
    unit = np.array([[0, 0], [side_px, 0], [side_px, side_px], [0, side_px]], dtype=np.float32)
    corners = cv2.transform(unit.reshape(1, 4, 2), inverse).reshape(4, 2)

    reasons = []
    if eye_width < config.min_eye_width_px:
        reasons.append("eye too narrow ({:.0f} px < {:.0f})".format(eye_width, config.min_eye_width_px))
    if (corners[:, 0].min() < 0 or corners[:, 1].min() < 0
            or corners[:, 0].max() > width - 1 or corners[:, 1].max() > height - 1):
        reasons.append("crop partly outside frame")

    gray, tensor = preprocess_eye_image(raw, config)
    return EyeCrop(side=side, raw=raw, gray=gray, tensor=tensor, corners=corners,
                   center=(float(center[0]), float(center[1])), eye_width_px=eye_width,
                   angle_deg=angle, valid=not reasons, reason="; ".join(reasons))


def extract_eye_crops(frame_bgr: np.ndarray, face,
                      config: Optional[EyePreprocessConfig] = None
                      ) -> Tuple[Optional[EyeCrop], Optional[EyeCrop]]:
    """(left, right) crops for the subject's eyes, or (None, None) without a face."""
    if face is None:
        return None, None
    config = config or EyePreprocessConfig()
    gray_frame = to_grayscale(frame_bgr)
    left = crop_eye(gray_frame, face, LEFT_EYE, "left", config)
    right = crop_eye(gray_frame, face, RIGHT_EYE, "right", config)
    return left, right


def save_eye_crops(crops: Tuple[Optional[EyeCrop], Optional[EyeCrop]], directory: Path = CROP_DIR,
                   tag: str = "", raw: bool = False) -> List[Path]:
    """Write the (size x size) gray crops - or the un-resized raw crops - as PNGs.

    Files are named eye_<side>_<timestamp>[_<tag>].png. They are the material
    for the Stage 6 comparison against MRL samples and for a self-collected
    test set; data/ is git-ignored.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = "{:%Y%m%d_%H%M%S_%f}".format(datetime.now())
    written = []
    for crop in crops:
        if crop is None:
            continue
        name = "eye_{}_{}{}.png".format(crop.side, stamp, "_" + tag if tag else "")
        path = directory / name
        cv2.imwrite(str(path), crop.raw if raw else crop.gray)
        written.append(path)
    return written


# --- development display -----------------------------------------------------

COLOR_CROP_OK = (0, 255, 255)
COLOR_CROP_BAD = (0, 0, 255)


def draw_eye_boxes(frame: np.ndarray, crops: Tuple[Optional[EyeCrop], Optional[EyeCrop]]) -> None:
    """Outline each crop square on the raw (un-mirrored) frame."""
    for crop in crops:
        if crop is None:
            continue
        pts = np.rint(crop.corners).astype(np.int32).reshape(1, 4, 2)
        cv2.polylines(frame, pts, True, COLOR_CROP_OK if crop.valid else COLOR_CROP_BAD, 1, cv2.LINE_AA)


def draw_eye_panel(display: np.ndarray, crops: Tuple[Optional[EyeCrop], Optional[EyeCrop]],
                   x: int, y: int, tile: int = 96) -> None:
    """Show both preprocessed crops enlarged, left eye first (matches the
    mirrored view where the subject's left eye appears on the left). Call
    after mirroring; the tiles themselves are never mirrored."""
    for i, crop in enumerate(crops):
        x0 = x + i * (tile + 10)
        cv2.rectangle(display, (x0 - 1, y - 1), (x0 + tile, y + tile), (60, 60, 60), -1)
        if crop is None:
            cv2.putText(display, "no eye", (x0 + 14, y + tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (140, 140, 140), 1, cv2.LINE_AA)
            continue
        big = cv2.resize(crop.gray, (tile, tile), interpolation=cv2.INTER_NEAREST)
        display[y:y + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        color = COLOR_CROP_OK if crop.valid else COLOR_CROP_BAD
        cv2.rectangle(display, (x0 - 1, y - 1), (x0 + tile, y + tile), color, 1)
        label = "{} {:.0f}px".format("L" if crop.side == "left" else "R", crop.eye_width_px)
        cv2.putText(display, label, (x0 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, label, (x0 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if not crop.valid:
            cv2.putText(display, "invalid", (x0 + 3, y + tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLOR_CROP_BAD, 1, cv2.LINE_AA)


# --- self-test ---------------------------------------------------------------

def _synthetic_eye_frame(center: Tuple[float, float], half_width: float, angle_deg: float,
                         size=(640, 480)):
    """A dark frame with a bright ellipse 'eye' and matching six landmarks."""
    from src.landmarks import NUM_LANDMARKS, FaceLandmarks

    frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
    cv2.ellipse(frame, (int(round(center[0])), int(round(center[1]))),
                (int(half_width), int(half_width * 0.4)), angle_deg, 0, 360, (230, 230, 230), -1)
    a = math.radians(angle_deg)
    along = np.array([math.cos(a), math.sin(a)])
    up = np.array([math.sin(a), -math.cos(a)])           # perpendicular, towards the image top
    c = np.asarray(center, dtype=np.float64)
    pts = {
        0: c - along * half_width,                        # p1 corner
        1: c - along * half_width * 0.4 + up * half_width * 0.3,   # p2 upper lid
        2: c + along * half_width * 0.4 + up * half_width * 0.3,   # p3 upper lid
        3: c + along * half_width,                        # p4 corner
        4: c + along * half_width * 0.4 - up * half_width * 0.3,   # p5 lower lid
        5: c - along * half_width * 0.4 - up * half_width * 0.3,   # p6 lower lid
    }
    normalized = np.full((NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
    for eye in (LEFT_EYE, RIGHT_EYE):
        for k, idx in enumerate(eye):
            normalized[idx, 0] = pts[k][0] / size[0]
            normalized[idx, 1] = pts[k][1] / size[1]
    return frame, FaceLandmarks(normalized, size, 0, 0.0)


def _ellipse_orientation(gray: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Orientation (deg) and centroid of the bright blob in a crop, via moments."""
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    m = cv2.moments(mask, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    theta = 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    return theta, (cx, cy)


def self_test() -> int:
    config = EyePreprocessConfig()

    # 1. Pure preprocessing on an MRL-like file: shapes, dtypes, normalisation.
    mrl_like = np.random.RandomState(0).randint(0, 256, (80, 86), dtype=np.uint8)
    gray, tensor = preprocess_eye_image(mrl_like, config)
    assert gray.shape == (64, 64) and gray.dtype == np.uint8, (gray.shape, gray.dtype)
    assert tensor.shape == (64, 64) and tensor.dtype == np.float32
    assert abs(float(tensor.mean())) < 1e-4 and abs(float(tensor.std()) - 1.0) < 1e-3, (tensor.mean(), tensor.std())
    flat_gray, flat_tensor = preprocess_eye_image(np.full((30, 30), 77, np.uint8), config)
    assert not flat_tensor.any(), "a flat image must standardise to zeros, not NaN"
    bgr_gray, _ = preprocess_eye_image(cv2.cvtColor(mrl_like, cv2.COLOR_GRAY2BGR), config)
    assert np.array_equal(bgr_gray, gray), "BGR input must give the same gray as direct gray input"
    up_gray, _ = preprocess_eye_image(np.random.RandomState(1).randint(0, 256, (20, 20), dtype=np.uint8), config)
    assert up_gray.shape == (64, 64), "small images must be enlarged to the same size"
    print("[self-test] preprocess_eye_image: 64x64 uint8 + standardised float32, flat image -> zeros, "
          "BGR == gray, enlarging OK")

    # 2. Landmark crop on a synthetic tilted eye: centred, level after alignment.
    frame, face = _synthetic_eye_frame((400.0, 200.0), 30.0, 20.0)
    left, right = extract_eye_crops(frame, face, config)
    assert left is not None and right is not None
    assert left.valid and left.reason == "", left.reason
    assert left.gray.shape == (64, 64) and left.tensor.shape == (64, 64)
    assert left.raw.shape == (90, 90), left.raw.shape           # 1.5 x 60 px corner distance
    assert abs(left.eye_width_px - 60.0) < 0.5 and abs(left.angle_deg - 20.0) < 0.5
    theta, (cx, cy) = _ellipse_orientation(left.gray)
    assert abs(theta) < 3.0, "after alignment the eye should be level, got {:.1f} deg".format(theta)
    assert abs(cx - 32) < 2 and abs(cy - 32) < 2, "eye should be centred in the crop, got ({:.1f}, {:.1f})".format(cx, cy)
    unaligned = crop_eye(to_grayscale(frame), face, LEFT_EYE, "left",
                         EyePreprocessConfig(align_roll=False))
    theta_u, _ = _ellipse_orientation(unaligned.gray)
    assert abs(abs(theta_u) - 20.0) < 3.0, "without alignment the tilt must remain, got {:.1f}".format(theta_u)
    print("[self-test] crop: 90 px raw -> 64 px, eye centred within 2 px, tilt 20 deg -> {:.1f} deg aligned "
          "/ {:.1f} deg unaligned".format(theta, theta_u))

    # 3. Train/inference contract: the shared function on the raw crop gives the
    #    exact tensor the live path produced.
    gray2, tensor2 = preprocess_eye_image(left.raw, config)
    assert np.array_equal(gray2, left.gray) and np.array_equal(tensor2, left.tensor)
    print("[self-test] contract: preprocess_eye_image(raw crop) == live crop tensor, bit for bit")

    # 4. Invalid geometry is flagged but still produces an image; no face -> (None, None).
    edge_frame, edge_face = _synthetic_eye_frame((12.0, 12.0), 30.0, 0.0)
    edge, _ = extract_eye_crops(edge_frame, edge_face, config)
    assert not edge.valid and "outside" in edge.reason and edge.gray.shape == (64, 64), edge.reason
    tiny_frame, tiny_face = _synthetic_eye_frame((300.0, 200.0), 5.0, 0.0)
    tiny, _ = extract_eye_crops(tiny_frame, tiny_face, config)
    assert not tiny.valid and "narrow" in tiny.reason, tiny.reason
    assert extract_eye_crops(frame, None, config) == (None, None)
    print("[self-test] invalid handling: edge crop -> '{}', tiny eye -> '{}', no face -> (None, None)".format(
        edge.reason, tiny.reason))

    # 5. Drawing and saving do not crash; saved file round-trips through the shared function.
    display = frame.copy()
    draw_eye_boxes(display, (left, right))
    draw_eye_panel(display, (left, None), 10, 300)
    draw_eye_panel(display, (edge, tiny), 250, 300)
    assert display.any()
    out_dir = PROJECT_ROOT / "data" / "eye_crops" / "_selftest"
    paths = save_eye_crops((left, right), out_dir, tag="selftest")
    reloaded = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(reloaded, left.gray), "saved crop must reload identically"
    _, tensor3 = preprocess_eye_image(reloaded, config)
    assert np.array_equal(tensor3, left.tensor), "a saved crop must preprocess to the same tensor"
    for p in paths:
        p.unlink()
    out_dir.rmdir()
    print("[self-test] drawing OK; saved crops reload and re-preprocess identically")

    print("[self-test] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5: eye crop preprocessing (shared with training).")
    parser.add_argument("--self-test", action="store_true", help="Run checks that need no camera")
    parser.add_argument("--image", type=Path, help="Preprocess one eye image file and report shapes/statistics")
    parser.add_argument("--out", type=Path, help="With --image: write the resized gray result here")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--equalize", action="store_true", help="Apply CLAHE before standardisation")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if args.image:
        image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
        if image is None:
            print("ERROR: could not read {}".format(args.image), file=sys.stderr)
            return 1
        config = EyePreprocessConfig(size=args.size, equalize=args.equalize)
        t0 = time.perf_counter()
        gray, tensor = preprocess_eye_image(image, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print("input  : {} {} {}".format(args.image.name, image.shape, image.dtype))
        print("output : gray {} {} | tensor {} {} mean {:+.4f} std {:.4f} | {:.2f} ms".format(
            gray.shape, gray.dtype, tensor.shape, tensor.dtype, float(tensor.mean()), float(tensor.std()),
            elapsed_ms))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out), gray)
            print("wrote   : {}".format(args.out))
        return 0
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
