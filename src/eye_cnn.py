"""Eye-region extraction, preprocessing and the eye-state CNN (Stages 5, 7, 8).

This module owns everything the CNN will ever see - and the CNN itself.

Stage 5 - preprocessing (no PyTorch needed)::

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

Stage 7 - model (PyTorch, imported lazily so Stage 5 code runs without it):

* ``augment_eye``      training-time augmentation on the uint8 image, applied
                       BEFORE standardisation so low-light noise keeps a realistic
                       signal-to-noise ratio; includes flips (the dataset does not
                       label eye side), geometry jitter, gamma darkening, noise,
                       blur and specular spots (glasses reflections)
* ``EyeStateCNN``      small CNN, 1 x size x size in, 2 logits out
* ``save_model`` / ``load_model``  checkpoint file that carries its own
                       preprocessing config and class names, so Stage 8 can
                       refuse a mismatched model instead of silently mis-predicting
* ``EyeStateClassifier``  what the live pipeline will call in Stage 8

Class convention: index 0 = CLOSED, 1 = OPEN - the same coding the MRL Eye
Dataset uses for its eye-state field.

Design choices that bind later stages
-------------------------------------
* No left/right flipping at inference; horizontal flip is an augmentation.
* ``crop_scale`` (1.5) and ``size`` (64) are INITIAL values, checked against
  real MRL samples in Stage 6 before any training happens.
* Validity here is geometric only (eye too narrow, crop partly outside the
  frame). Stage 4's frame gate is separate; Stage 8 requires both.

Run directly::

    python -m src.eye_cnn --self-test           # no camera needed
    python -m src.eye_cnn --image eye.png       # preprocess one file, print shapes/stats
    python -m src.eye_cnn --predict eye.png     # classify one file with models/eye_cnn.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.capture import PROJECT_ROOT

CROP_DIR = PROJECT_ROOT / "data" / "eye_crops"
MODEL_PATH = PROJECT_ROOT / "models" / "eye_cnn.pt"
CLASSES: Tuple[str, str] = ("CLOSED", "OPEN")   # index 0 = closed, 1 = open (MRL coding)
CHECKPOINT_FORMAT = "eye_cnn_v1"


# =============================================================================
# Stage 5 - preprocessing
# =============================================================================

@dataclass
class EyePreprocessConfig:
    """Shared by live extraction and the training pipeline. Change values here,
    never in only one place."""

    size: int = 64                  # output side in pixels (square)
    crop_scale: float = 1.5         # crop side = crop_scale x eye-corner distance  (verify vs MRL, Stage 6)
    align_roll: bool = True         # rotate so the eye corners are horizontal before cropping
    equalize: bool = False          # CLAHE before standardisation (Stage 7 ablation option)
    min_eye_width_px: float = 15.0  # narrower eyes give unusable crops


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
    # Imported here so that importing this module (e.g. in Colab for training)
    # does not pull in MediaPipe through src.features.
    from src.features import LEFT_EYE, RIGHT_EYE

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


COLOR_OPEN = (0, 255, 0)
COLOR_CLOSED = (0, 140, 255)


def state_color(label: Optional[str]) -> Tuple[int, int, int]:
    return COLOR_OPEN if label == "OPEN" else COLOR_CLOSED if label == "CLOSED" else (150, 150, 150)


def draw_eye_panel(display: np.ndarray, crops: Tuple[Optional[EyeCrop], Optional[EyeCrop]],
                   x: int, y: int, tile: int = 96,
                   states: Optional[Sequence[Optional[Tuple[str, float]]]] = None) -> None:
    """Show both preprocessed crops enlarged, left eye first (matches the
    mirrored view where the subject's left eye appears on the left). Call
    after mirroring; the tiles themselves are never mirrored. ``states`` are
    the CNN's (label, confidence) per eye when a model is loaded (Stage 8)."""
    for i, crop in enumerate(crops):
        x0 = x + i * (tile + 10)
        cv2.rectangle(display, (x0 - 1, y - 1), (x0 + tile, y + tile), (60, 60, 60), -1)
        if crop is None:
            cv2.putText(display, "no eye", (x0 + 14, y + tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (140, 140, 140), 1, cv2.LINE_AA)
            continue
        big = cv2.resize(crop.gray, (tile, tile), interpolation=cv2.INTER_NEAREST)
        display[y:y + tile, x0:x0 + tile] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        state = states[i] if states is not None and i < len(states) else None
        color = (state_color(state[0]) if state else COLOR_CROP_OK) if crop.valid else COLOR_CROP_BAD
        cv2.rectangle(display, (x0 - 1, y - 1), (x0 + tile, y + tile), color, 2 if state else 1)
        label = "{} {:.0f}px".format("L" if crop.side == "left" else "R", crop.eye_width_px)
        cv2.putText(display, label, (x0 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, label, (x0 + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if state:
            text = "{} {:.2f}".format(state[0], state[1])
            cv2.putText(display, text, (x0 + 3, y + tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(display, text, (x0 + 3, y + tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
        elif not crop.valid:
            cv2.putText(display, "invalid", (x0 + 3, y + tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLOR_CROP_BAD, 1, cv2.LINE_AA)


# =============================================================================
# Stage 7 - augmentation (numpy / OpenCV only, so it runs anywhere)
# =============================================================================

@dataclass
class AugmentConfig:
    """Training-time augmentation of a uint8 eye image. Applied BEFORE
    standardisation. Initial values; Stage 7 may tune them."""

    hflip_p: float = 0.5                          # dataset does not label eye side -> side-agnostic model
    max_rotation_deg: float = 10.0                # residual roll after alignment
    scale_range: Tuple[float, float] = (0.85, 1.15)   # absorbs crop_scale mismatch vs MRL framing
    max_shift_px: int = 4
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    brightness_range: Tuple[float, float] = (-40.0, 40.0)
    gamma_range: Tuple[float, float] = (0.7, 2.2)     # > 1 darkens
    dark_p: float = 0.3                           # low-light rehearsal: extra strong darkening ...
    dark_scale_range: Tuple[float, float] = (0.25, 0.6)   # ... multiply intensities by this ...
    noise_sigma_range: Tuple[float, float] = (0.0, 14.0)  # ... then add sensor noise (before standardisation)
    blur_p: float = 0.3
    blur_sigma_range: Tuple[float, float] = (0.3, 1.2)
    specular_p: float = 0.15                      # bright spot / streak, like a glasses reflection
    cutout_p: float = 0.1                         # small dark occluder, like a frame edge
    cutout_size_px: int = 12


def augment_eye(gray: np.ndarray, rng: np.random.Generator,
                config: Optional[AugmentConfig] = None) -> np.ndarray:
    """Return an augmented uint8 copy of a (H, W) uint8 eye image.

    Order: flip -> rotation/scale/shift (one warp) -> contrast/brightness ->
    gamma -> optional strong darkening -> optional specular spot / cutout ->
    optional blur -> additive noise. Standardisation happens afterwards in
    ``normalize_eye``; applying darkening and noise here, before it, is what
    gives the network realistically low signal-to-noise inputs.
    """
    config = config or AugmentConfig()
    h, w = gray.shape[:2]
    img = gray[:, ::-1] if rng.random() < config.hflip_p else gray
    img = np.ascontiguousarray(img)

    angle = rng.uniform(-config.max_rotation_deg, config.max_rotation_deg)
    scale = rng.uniform(*config.scale_range)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += rng.uniform(-config.max_shift_px, config.max_shift_px)
    matrix[1, 2] += rng.uniform(-config.max_shift_px, config.max_shift_px)
    img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    x = img.astype(np.float32)
    x = (x - 128.0) * rng.uniform(*config.contrast_range) + 128.0 + rng.uniform(*config.brightness_range)
    x = np.clip(x, 0.0, 255.0)
    x = 255.0 * np.power(x / 255.0, rng.uniform(*config.gamma_range))
    if rng.random() < config.dark_p:
        x = x * rng.uniform(*config.dark_scale_range)

    if rng.random() < config.specular_p:
        overlay = np.zeros((h, w), dtype=np.float32)
        cx, cy = int(rng.uniform(0.15, 0.85) * w), int(rng.uniform(0.15, 0.85) * h)
        axes = (int(rng.uniform(1, 5)), int(rng.uniform(1, 9)))
        cv2.ellipse(overlay, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), 1.0)
        x = x + overlay * rng.uniform(120.0, 255.0)
    if rng.random() < config.cutout_p:
        s = config.cutout_size_px
        cx, cy = int(rng.uniform(0, w - s)), int(rng.uniform(0, h - s))
        x[cy:cy + s, cx:cx + s] = rng.uniform(0.0, 60.0)

    if rng.random() < config.blur_p:
        x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(*config.blur_sigma_range)))
    sigma = rng.uniform(*config.noise_sigma_range)
    if sigma > 0:
        x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


# =============================================================================
# Stage 7 - model, checkpoint format, classifier (PyTorch imported lazily)
# =============================================================================

def _torch():
    try:
        import torch  # noqa: WPS433
        import torch.nn as nn  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for the eye-state CNN: pip install torch "
                          "(CPU wheels: --index-url https://download.pytorch.org/whl/cpu)") from exc
    return torch, nn


def build_model(width: int = 32, dropout: float = 0.3, num_classes: int = 2):
    """EyeStateCNN: four double-conv blocks (w, 2w, 4w, 4w channels) with
    BatchNorm and ReLU, max-pooling between the first three, global average
    pooling, dropout, linear head. Parameter count scales with width^2:
    about 146 k at width 16, 328 k at width 24, 583 k at width 32. The
    self-test prints the measured CPU latency for a two-eye batch so the
    width can be chosen against the live frame budget (Stage 4: ~38 ms free)."""
    torch, nn = _torch()

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class EyeStateCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.width, self.dropout_p, self.num_classes = width, dropout, num_classes
            self.features = nn.Sequential(
                block(1, width), nn.MaxPool2d(2),            # size    -> size/2
                block(width, 2 * width), nn.MaxPool2d(2),    # size/2  -> size/4
                block(2 * width, 4 * width), nn.MaxPool2d(2),  # size/4 -> size/8
                block(4 * width, 4 * width),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * width, num_classes))

        def forward(self, x):
            return self.head(self.pool(self.features(x)))

    return EyeStateCNN()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path: Path, preprocess: EyePreprocessConfig,
               metadata: Optional[Dict] = None) -> Path:
    """Write a self-describing checkpoint: weights + architecture + the exact
    preprocessing config the model was trained with + class names + metadata.
    Only plain Python types go into metadata so it loads with weights_only=True."""
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        "arch": {"width": int(model.width), "dropout": float(model.dropout_p),
                 "num_classes": int(model.num_classes)},
        "classes": list(CLASSES),
        "preprocess": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(preprocess).items()},
        "metadata": dict(metadata or {}),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, str(path))
    return path


def load_model(path: Path = MODEL_PATH, device: str = "cpu"):
    """Return (model in eval mode on device, payload dict). Refuses unknown formats."""
    torch, _ = _torch()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("No trained eye-state model at {} - train it in Stage 7 first.".format(path))
    try:
        payload = torch.load(str(path), map_location=device, weights_only=True)
    except Exception:  # older torch without weights_only, or a checkpoint with extra types
        payload = torch.load(str(path), map_location=device)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unexpected checkpoint format {!r} in {}".format(payload.get("format"), path))
    arch = payload["arch"]
    model = build_model(width=arch["width"], dropout=arch["dropout"], num_classes=arch["num_classes"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


class EyeStateClassifier:
    """Loads models/eye_cnn.pt and classifies preprocessed eye tensors.

    ``predict`` takes float32 arrays of shape (N, size, size) as produced by
    ``preprocess_eye_image`` / ``EyeCrop.tensor`` and returns (labels, probs):
    labels are indices into CLASSES, probs is (N, 2). Stage 8 wires this to the
    live crops; nothing here touches the camera.
    """

    def __init__(self, path: Path = MODEL_PATH, device: str = "cpu") -> None:
        self.torch, _ = _torch()
        self.model, self.payload = load_model(path, device)
        self.device = device
        self.classes = tuple(self.payload["classes"])
        pre = dict(self.payload["preprocess"])
        pre = {k: (tuple(v) if isinstance(v, list) else v) for k, v in pre.items()}
        self.config = EyePreprocessConfig(**pre)
        self.size = self.config.size

    def predict(self, tensors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(tensors, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        if x.shape[1:] != (self.size, self.size):
            raise ValueError("expected tensors of shape (N, {0}, {0}), got {1}".format(self.size, x.shape))
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x[:, None]).to(self.device))
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()
        return probs.argmax(axis=1), probs

    def predict_image(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify any eye image file/array through the shared preprocessing."""
        _, tensor = preprocess_eye_image(image, self.config)
        labels, probs = self.predict(tensor[None])
        return self.classes[int(labels[0])], float(probs[0, labels[0]])

    def predict_crop(self, crop: EyeCrop) -> Tuple[str, float]:
        labels, probs = self.predict(crop.tensor[None])
        return self.classes[int(labels[0])], float(probs[0, labels[0]])

    def predict_crops(self, crops: Sequence[Optional[EyeCrop]]
                      ) -> Tuple[List[Optional[Tuple[str, float]]], float]:
        """Classify a (left, right) pair in ONE forward pass.

        Returns ([left, right], inference_ms) where each entry is
        (label, confidence) or None when that crop is missing or geometrically
        invalid. Stage 8's live loop calls this once per frame.
        """
        tensors, index = [], []
        for i, crop in enumerate(crops):
            if crop is not None and crop.valid:
                tensors.append(crop.tensor)
                index.append(i)
        results: List[Optional[Tuple[str, float]]] = [None] * len(crops)
        if not tensors:
            return results, 0.0
        t0 = time.perf_counter()
        labels, probs = self.predict(np.stack(tensors))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        for k, i in enumerate(index):
            results[i] = (self.classes[int(labels[k])], float(probs[k, labels[k]]))
        return results, elapsed_ms


# =============================================================================
# self-test
# =============================================================================

def _synthetic_eye_frame(center: Tuple[float, float], half_width: float, angle_deg: float,
                         size=(640, 480)):
    """A dark frame with a bright ellipse 'eye' and matching six landmarks."""
    from src.features import LEFT_EYE, RIGHT_EYE
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
    from src.features import LEFT_EYE

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

    # 6. Augmentation: shape/dtype preserved, deterministic under a seed, actually changes the image,
    #    and its low-light branch lowers the mean intensity.
    rng = np.random.default_rng(7)
    aug = augment_eye(left.gray, rng)
    assert aug.shape == left.gray.shape and aug.dtype == np.uint8
    assert not np.array_equal(aug, left.gray), "augmentation must change the image"
    assert np.array_equal(augment_eye(left.gray, np.random.default_rng(7)), aug), "same seed -> same result"
    dark_cfg = AugmentConfig(dark_p=1.0, dark_scale_range=(0.3, 0.3), gamma_range=(1.0, 1.0),
                             contrast_range=(1.0, 1.0), brightness_range=(0.0, 0.0), noise_sigma_range=(0.0, 0.0),
                             specular_p=0.0, cutout_p=0.0, blur_p=0.0, hflip_p=0.0, max_rotation_deg=0.0,
                             scale_range=(1.0, 1.0), max_shift_px=0)
    dark = augment_eye(left.gray, np.random.default_rng(0), dark_cfg)
    assert abs(float(dark.mean()) - 0.3 * float(left.gray.mean())) < 2.0, (dark.mean(), left.gray.mean())
    _, dark_tensor = preprocess_eye_image(dark, config)
    assert abs(float(dark_tensor.std()) - 1.0) < 1e-3, "standardisation must restore unit variance"
    print("[self-test] augmentation OK: uint8 preserved, seeded determinism, dark branch scales intensity by 0.3")

    # 7. Model, checkpoint round trip and classifier (skipped if PyTorch is absent).
    try:
        torch, _ = _torch()
    except ImportError as exc:
        print("[self-test] PyTorch not installed - model checks skipped ({})".format(exc))
        print("[self-test] PASS (preprocessing only)")
        return 0
    torch.manual_seed(0)
    model = build_model(width=8)
    logits = model(torch.zeros(3, 1, 64, 64))
    assert logits.shape == (3, 2), logits.shape
    full = build_model()
    n_params = count_parameters(full)
    assert count_parameters(build_model(width=16)) < n_params < 1_000_000, n_params
    full.eval()
    two_eyes = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        for _ in range(5):
            full(two_eyes)
        t0 = time.perf_counter()
        for _ in range(20):
            full(two_eyes)
    latency_ms = (time.perf_counter() - t0) / 20 * 1000.0
    print("[self-test] CPU latency, width {} two-eye batch: {:.1f} ms ({} threads)".format(
        full.width, latency_ms, torch.get_num_threads()))
    ckpt_dir = PROJECT_ROOT / "data" / "_selftest_ckpt"
    ckpt = save_model(model, ckpt_dir / "eye_cnn_selftest.pt", config, {"note": "self-test", "epochs": 0})
    reloaded_model, payload = load_model(ckpt)
    assert payload["classes"] == list(CLASSES) and payload["preprocess"]["size"] == 64
    x = torch.randn(2, 1, 64, 64)
    model.eval()
    assert torch.allclose(model(x), reloaded_model(x), atol=1e-6), "reloaded model must reproduce outputs"
    clf = EyeStateClassifier(ckpt)
    labels, probs = clf.predict(np.stack([left.tensor, right.tensor]))
    assert labels.shape == (2,) and probs.shape == (2, 2) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    label, conf = clf.predict_crop(left)
    assert label in CLASSES and 0.0 <= conf <= 1.0
    label_i, conf_i = clf.predict_image(mrl_like)
    assert label_i in CLASSES
    ckpt.unlink()
    ckpt_dir.rmdir()
    print("[self-test] model OK: forward (3,2); full model {:,} params; checkpoint round trip exact; "
          "classifier predicts crops and files".format(n_params))

    print("[self-test] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eye crop preprocessing (Stage 5) and eye-state CNN (Stage 7).")
    parser.add_argument("--self-test", action="store_true", help="Run checks that need no camera")
    parser.add_argument("--image", type=Path, help="Preprocess one eye image file and report shapes/statistics")
    parser.add_argument("--predict", type=Path, help="Classify one eye image file with the trained model")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Checkpoint for --predict")
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
    if args.predict:
        image = cv2.imread(str(args.predict), cv2.IMREAD_UNCHANGED)
        if image is None:
            print("ERROR: could not read {}".format(args.predict), file=sys.stderr)
            return 1
        clf = EyeStateClassifier(args.model)
        t0 = time.perf_counter()
        label, conf = clf.predict_image(image)
        print("{}: {} ({:.1%}) in {:.1f} ms | model {} trained {}".format(
            args.predict.name, label, conf, (time.perf_counter() - t0) * 1000.0,
            args.model.name, clf.payload.get("saved_at", "?")))
        return 0
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
