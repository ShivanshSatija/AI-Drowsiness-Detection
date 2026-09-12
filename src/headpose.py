"""Head-pose estimation from facial landmarks (Stage 4).

Two independent estimators, so each can be checked against the other:

``matrix`` (default)
    The 4x4 facial transformation matrix MediaPipe fits to its canonical face
    model using all 468 points. Needs ``output_transformation_matrix=True`` in
    ``LandmarkConfig`` (the default).

``pnp`` (cross-check)
    ``cv2.solvePnP`` on six landmarks against a generic 3-D face model with an
    approximate pinhole camera: focal length = frame width, principal point =
    frame centre, no distortion. Classic and fully explainable, but measured
    to be unreliable for gating: on twelve labelled real frames its yaw agreed
    with ``matrix`` in sign and ordering (r = 0.96) yet read +34 and +39 deg
    on two frontal frames with the mouth wide open, because the generic model
    assumes a closed mouth. ``matrix`` read +7 and +10 deg on the same frames.

Angle conventions, identical for both methods:

    yaw   > 0   face turned towards the subject's LEFT
                (nose moves to image-right in the un-mirrored frame)
    pitch > 0   face tilted UP
    roll  > 0   head tilted towards the subject's LEFT shoulder

Yaw and pitch are the direction the face points; roll is the remaining
rotation about that facing axis. Every rotation decomposes exactly as
``R = Ry(-yaw) Rx(-pitch) FRONTAL Rz(-roll)``, so all three are recovered
without approximation. That is what a validity gate and, later, nod detection
need.

Sign and scale were verified empirically on mirrored, rotated and
perspective-warped test images (see the Stage 4 notes in README.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.landmarks import FaceLandmarks

# --- generic 3-D face model for solvePnP --------------------------------------
# The widely used six-point generic model (Mallick, 2016) is in arbitrary
# units; it is rescaled here by 1/4.6 so the outer-eye-corner distance is
# 98 mm, a typical adult value, which makes the recovered translation a real
# distance in mm. Scale does not affect the angles. Model axes: x towards the
# subject's left (image-right when the frame is not mirrored), y up, z towards
# the camera. Nose tip at the origin.
PNP_LANDMARKS: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)
PNP_MODEL_MM = np.array([
    [0.0, 0.0, 0.0],           # 1   nose tip
    [0.0, -330.0, -65.0],      # 152 chin
    [-225.0, 170.0, -135.0],   # 33  outer corner, subject's right eye
    [225.0, 170.0, -135.0],    # 263 outer corner, subject's left eye
    [-150.0, -150.0, -125.0],  # 61  subject's right mouth corner
    [150.0, -150.0, -125.0],   # 291 subject's left mouth corner
], dtype=np.float64) / 4.6

POSE_METHODS = ("pnp", "matrix")


@dataclass
class PoseConfig:
    method: str = "matrix"
    focal_length_factor: float = 1.0   # pnp only: fx = fy = factor * frame width (approximate webcam)


@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    method: str
    distance_cm: Optional[float] = None   # informational only; intrinsics are approximate

    @property
    def ok(self) -> bool:
        return not any(math.isnan(v) for v in (self.yaw_deg, self.pitch_deg, self.roll_deg))


def camera_matrix(width: int, height: int, focal_length_factor: float = 1.0) -> np.ndarray:
    f = focal_length_factor * width
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# Model axes expressed in camera axes for a face looking straight at the camera:
# model x (subject's left) = camera x; model y (up) = camera -y; model z
# (towards the camera) = camera -z.
FRONTAL = np.diag([1.0, -1.0, -1.0])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def direction_angles(rotation: np.ndarray) -> Tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a model->camera rotation matrix.

    Model axes: x subject's-left, y up, z towards the camera. Camera axes
    (OpenCV): x right, y down, z forward, so a frontal face has forward
    (0, 0, -1) in camera coordinates. Yaw and pitch are the direction of that
    forward vector; roll is the rotation left about the facing axis once yaw
    and pitch are removed. The decomposition
    ``R = Ry(-yaw) Rx(-pitch) FRONTAL Rz(-roll)`` is exact for any rotation.
    """
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    yaw = math.atan2(forward[0], -forward[2])
    pitch = math.atan2(-forward[1], math.hypot(forward[0], forward[2]))
    facing = _rot_y(-yaw) @ _rot_x(-pitch) @ FRONTAL
    residual = facing.T @ rotation          # what remains: Rz(-roll)
    roll = -math.atan2(residual[1, 0], residual[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def estimate_pose_pnp(face: FaceLandmarks, config: Optional[PoseConfig] = None) -> HeadPose:
    config = config or PoseConfig()
    width, height = face.frame_size
    image_points = face.points(PNP_LANDMARKS).astype(np.float64)
    # Coincident or collinear points make solvePnP throw rather than fail
    # cleanly; degenerate landmarks must never crash the pipeline.
    if np.ptp(image_points, axis=0).min() < 1.0:
        return HeadPose(math.nan, math.nan, math.nan, "pnp")
    try:
        ok, rvec, tvec = cv2.solvePnP(PNP_MODEL_MM, image_points,
                                      camera_matrix(width, height, config.focal_length_factor),
                                      np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
    if not ok:
        return HeadPose(math.nan, math.nan, math.nan, "pnp")
    rotation, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = direction_angles(rotation)
    return HeadPose(yaw, pitch, roll, "pnp", distance_cm=float(tvec[2, 0]) / 10.0)


def pose_from_matrix(matrix: np.ndarray) -> HeadPose:
    """Angles from MediaPipe's facial transformation matrix.

    MediaPipe uses an OpenGL-style camera (x right, y up, z towards the viewer)
    and a canonical model that faces +z, so its axes are converted to the
    OpenCV convention used by ``direction_angles`` by negating y and z.
    """
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    flip = np.diag([1.0, -1.0, -1.0])
    yaw, pitch, roll = direction_angles(flip @ rotation)
    translation = np.asarray(matrix, dtype=np.float64)[:3, 3]
    return HeadPose(yaw, pitch, roll, "matrix", distance_cm=float(abs(translation[2])))


def estimate_pose(face: FaceLandmarks, config: Optional[PoseConfig] = None) -> Optional[HeadPose]:
    """Head pose for one face with the configured method; None if unavailable."""
    config = config or PoseConfig()
    if config.method == "matrix":
        if face.transformation_matrix is None:
            return None
        return pose_from_matrix(face.transformation_matrix)
    if config.method == "pnp":
        return estimate_pose_pnp(face, config)
    raise ValueError("unknown pose method {!r}; choose from {}".format(config.method, POSE_METHODS))


# --- drawing -----------------------------------------------------------------

COLOR_POSE = (255, 160, 0)


def draw_pose(frame: np.ndarray, face: FaceLandmarks, pose: HeadPose, mirror: bool = False,
              length: int = 90) -> None:
    """Arrow from the nose tip in the facing direction. Call after mirroring."""
    if not pose.ok:
        return
    width = frame.shape[1]
    x, y = face.pixel(1)
    if mirror:
        x = width - 1 - x
    dx = math.sin(math.radians(pose.yaw_deg)) * length
    dy = -math.sin(math.radians(pose.pitch_deg)) * length
    if mirror:
        dx = -dx
    cv2.arrowedLine(frame, (x, y), (int(x + dx), int(y + dy)), COLOR_POSE, 2, cv2.LINE_AA, tipLength=0.25)


# --- self-test ---------------------------------------------------------------

def _rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Model->camera rotation with the given angles (for tests): the inverse of
    ``direction_angles``. Camera y points down, so rotations about camera axes
    are negated to keep +yaw = image-right and +pitch = up."""
    return (_rot_y(math.radians(-yaw_deg)) @ _rot_x(math.radians(-pitch_deg)) @ FRONTAL
            @ _rot_z(math.radians(-roll_deg)))


def self_test() -> int:
    from src.landmarks import NUM_LANDMARKS

    # 1. Angle extraction is exact for known rotations, single-axis and combined.
    for yaw, pitch, roll in [(0, 0, 0), (25, 0, 0), (-40, 0, 0), (0, 15, 0), (0, -30, 0),
                             (0, 0, 12), (0, 0, -20), (20, -10, 5), (-35, 20, -8), (60, 25, -30)]:
        got = direction_angles(_rotation(yaw, pitch, roll))
        assert all(abs(g - e) < 1e-6 for g, e in zip(got, (yaw, pitch, roll))), (yaw, pitch, roll, got)
    print("[self-test] direction_angles recovers yaw/pitch/roll exactly for 10 known rotations")

    # 2. solvePnP round trip: project the model with a known pose, then recover it.
    width, height = 640, 480
    cam = camera_matrix(width, height)
    for yaw, pitch, roll in [(0, 0, 0), (30, 0, 0), (-30, 0, 0), (0, -20, 0), (10, -20, 8)]:
        rotation = _rotation(yaw, pitch, roll)
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = np.array([[0.0], [0.0], [600.0]])   # 60 cm in front of the camera
        projected, _ = cv2.projectPoints(PNP_MODEL_MM, rvec, tvec, cam, np.zeros((4, 1)))
        normalized = np.full((NUM_LANDMARKS, 3), 0.5, dtype=np.float32)
        for idx, (px, py) in zip(PNP_LANDMARKS, projected.reshape(-1, 2)):
            normalized[idx, 0] = px / width
            normalized[idx, 1] = py / height
        face = FaceLandmarks(normalized, (width, height), 0, 0.0)
        pose = estimate_pose_pnp(face)
        assert pose.ok
        expected = direction_angles(rotation)   # what this rotation truly reads as
        err = max(abs(pose.yaw_deg - expected[0]), abs(pose.pitch_deg - expected[1]),
                  abs(pose.roll_deg - expected[2]))
        assert err < 0.5, "pnp round trip error {:.2f} deg at {}".format(err, (yaw, pitch, roll))
        assert abs(pose.distance_cm - 60.0) < 1.0, pose.distance_cm
    print("[self-test] solvePnP round trip within 0.5 deg for 5 poses, distance within 1 cm")

    # 3. matrix path: an OpenGL-convention frontal matrix gives zero angles, and a
    #    yawed one the right sign.
    frontal = np.eye(4)
    pose = pose_from_matrix(frontal)
    assert max(abs(pose.yaw_deg), abs(pose.pitch_deg), abs(pose.roll_deg)) < 1e-9
    a = math.radians(20)
    yawed = np.eye(4)
    yawed[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    pose = pose_from_matrix(yawed)
    assert abs(abs(pose.yaw_deg) - 20) < 1e-6 and abs(pose.pitch_deg) < 1e-6, pose
    print("[self-test] matrix path: frontal -> 0 deg, 20 deg yaw -> |yaw| 20 deg")

    # 4. degenerate input does not crash.
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    face = FaceLandmarks(np.full((NUM_LANDMARKS, 3), 0.5, dtype=np.float32), (640, 480), 0, 0.0)
    pose = estimate_pose_pnp(face)
    draw_pose(canvas, face, pose, mirror=True)
    draw_pose(canvas, face, HeadPose(15, -10, 0, "pnp"), mirror=True)
    assert canvas.any()
    print("[self-test] degenerate landmarks handled; pose arrow drawn")

    print("[self-test] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
