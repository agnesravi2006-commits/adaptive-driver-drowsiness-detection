"""Configurable thresholds used by detection, calibration, and risk scoring."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AppConfig:
    # Sliding window used for temporal risk (seconds).
    window_seconds: float = 20.0

    # Personalized calibration.
    calibration_seconds: float = 8.0
    min_calibration_samples: int = 40
    ear_close_std_factor: float = 1.15
    mar_yawn_std_factor: float = 2.8
    min_ear_close_offset: float = 0.045
    default_ear_close_threshold: float = 0.21
    default_mar_yawn_threshold: float = 0.65
    default_blink_duration_s: float = 0.12
    long_closure_seconds: float = 1.5

    # Head-pose nodding.
    nod_pitch_threshold_deg: float = 18.0
    nod_reset_pitch_deg: float = 8.0

    # Confidence-aware detection.
    min_face_confidence: float = 0.55
    uncertain_confidence: float = 0.72
    max_missing_face_frames: int = 8

    # Risk score weights (must sum to 1.0 for a 0-100 scale).
    weight_perclos: float = 0.35
    weight_long_closure: float = 0.25
    weight_yawn: float = 0.15
    weight_nod: float = 0.15
    weight_blink: float = 0.10

    # Progressive states.
    low_risk_score: float = 25.0
    drowsy_score: float = 50.0
    high_risk_score: float = 75.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update_from(self, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            if hasattr(self, key) and value is not None:
                current = getattr(self, key)
                setattr(self, key, type(current)(value))


DEFAULT_CONFIG = AppConfig()


MEDIA_PIPE_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MEDIA_PIPE_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mouth: left corner, right corner, upper inner lip, lower inner lip.
MEDIA_PIPE_MOUTH = [78, 308, 13, 14]
# solvePnP reference points: nose, chin, left eye outer, right eye outer, mouth corners.
MEDIA_PIPE_POSE = [1, 152, 33, 263, 61, 291]

# Overlay subset keeps the live drawing light for the browser.
OVERLAY_LANDMARKS = sorted(
    set(
        MEDIA_PIPE_LEFT_EYE
        + MEDIA_PIPE_RIGHT_EYE
        + MEDIA_PIPE_MOUTH
        + MEDIA_PIPE_POSE
        + [10, 234, 454, 168]
    )
)
