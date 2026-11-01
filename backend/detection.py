"""Multimodal facial behavior analysis using MediaPipe Face Landmarker."""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from config import (
    MEDIA_PIPE_LEFT_EYE,
    MEDIA_PIPE_MOUTH,
    MEDIA_PIPE_POSE,
    MEDIA_PIPE_RIGHT_EYE,
    OVERLAY_LANDMARKS,
    AppConfig,
)


@dataclass
class FaceMetrics:
    face_detected: bool
    confidence: float
    ear: float | None
    left_ear: float | None
    right_ear: float | None
    mar: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    roll_deg: float | None
    landmarks_px: list[list[float]] = field(default_factory=list)
    overlay_landmarks: list[list[float]] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    message: str = ""

    @property
    def reliable(self) -> bool:
        return self.face_detected and self.ear is not None and self.confidence > 0


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def aspect_ratio(points: Sequence[Sequence[float]]) -> float:
    """6-point EAR/MAR-style ratio: (p2-p6 + p3-p5) / (2 * p1-p4)."""
    if len(points) != 6:
        raise ValueError("aspect_ratio expects 6 points")
    vertical = euclidean(points[1], points[5]) + euclidean(points[2], points[4])
    horizontal = euclidean(points[0], points[3])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(left: Sequence[float], right: Sequence[float],
                       upper: Sequence[float], lower: Sequence[float]) -> float:
    vertical = euclidean(upper, lower)
    horizontal = euclidean(left, right)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _landmark_xy(landmarks: Sequence[Any], index: int, width: int, height: int) -> list[float]:
    point = landmarks[index]
    return [float(point.x) * width, float(point.y) * height]


def estimate_head_pose(landmarks: Sequence[Any], width: int, height: int) -> tuple[float, float, float] | None:
    image_points = np.array(
        [_landmark_xy(landmarks, idx, width, height) for idx in MEDIA_PIPE_POSE],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    pose = np.hstack((rotation, np.array([[0.0], [0.0], [0.0]])))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose)
    pitch, yaw, roll = [float(euler[i][0]) for i in range(3)]
    return pitch, yaw, roll


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_face_landmarker_model(path: Path | None = None) -> Path:
    path = path or default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return path
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


class FaceAnalyzer:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._landmarker = None
        self._import_error: str | None = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        try:
            from mediapipe.tasks.python.vision import FaceLandmarker

            model_path = ensure_face_landmarker_model()
            self._landmarker = FaceLandmarker.create_from_model_path(str(model_path))
        except Exception as exc:  # pragma: no cover - environment dependent
            self._landmarker = None
            self._import_error = str(exc)

    @property
    def available(self) -> bool:
        return self._landmarker is not None

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def analyze_bgr(self, frame_bgr: np.ndarray) -> FaceMetrics:
        height, width = frame_bgr.shape[:2]
        if self._landmarker is None:
            return FaceMetrics(
                False,
                0.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                image_size=(width, height),
                message=f"MediaPipe unavailable: {self._import_error}",
            )

        from mediapipe import Image, ImageFormat

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return FaceMetrics(
                False,
                0.0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                image_size=(width, height),
                message="Face Not Detected",
            )

        landmarks = result.face_landmarks[0]
        left_pts = [_landmark_xy(landmarks, i, width, height) for i in MEDIA_PIPE_LEFT_EYE]
        right_pts = [_landmark_xy(landmarks, i, width, height) for i in MEDIA_PIPE_RIGHT_EYE]
        mouth_pts = [_landmark_xy(landmarks, i, width, height) for i in MEDIA_PIPE_MOUTH]
        left_ear = aspect_ratio(left_pts)
        right_ear = aspect_ratio(right_pts)
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth_pts[0], mouth_pts[1], mouth_pts[2], mouth_pts[3])
        pose = estimate_head_pose(landmarks, width, height)
        pitch = yaw = roll = None
        if pose:
            pitch, yaw, roll = pose

        all_pts = [[lm.x * width, lm.y * height] for lm in landmarks]
        overlay = [_landmark_xy(landmarks, i, width, height) for i in OVERLAY_LANDMARKS]

        visibility = [
            float(getattr(landmarks[i], "visibility", 0.0) or getattr(landmarks[i], "presence", 1.0) or 1.0)
            for i in MEDIA_PIPE_LEFT_EYE + MEDIA_PIPE_RIGHT_EYE
        ]
        # Face Mesh often leaves visibility at 0; fall back to landmark presence via z-spread and bounds.
        xs = [p[0] for p in left_pts + right_pts]
        ys = [p[1] for p in left_pts + right_pts]
        in_frame = all(0 <= x < width and 0 <= y < height for x, y in zip(xs, ys))
        eye_span = max(xs) - min(xs) if xs else 0
        size_ok = eye_span > width * 0.08
        conf = 0.93 if in_frame and size_ok else 0.45
        if visibility and max(visibility) > 0:
            conf = min(conf, float(sum(visibility) / len(visibility)))

        blur = float(cv2.Laplacian(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if blur < 25:
            conf *= 0.55
            message = "Detection Uncertain"
        elif not in_frame or not size_ok:
            conf *= 0.6
            message = "Detection Uncertain"
        else:
            message = ""

        return FaceMetrics(
            face_detected=True,
            confidence=float(max(0.0, min(conf, 1.0))),
            ear=float(ear),
            left_ear=float(left_ear),
            right_ear=float(right_ear),
            mar=float(mar),
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            landmarks_px=all_pts,
            overlay_landmarks=overlay,
            image_size=(width, height),
            message=message,
        )


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


def decode_data_url(data_url: str) -> np.ndarray | None:
    import base64

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
    except Exception:
        return None
    return decode_image_bytes(raw)
