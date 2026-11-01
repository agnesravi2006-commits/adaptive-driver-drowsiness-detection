"""Personalized adaptive calibration of a driver's baseline facial behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import AppConfig
from detection import FaceMetrics


@dataclass
class PersonalizedThresholds:
    calibrated: bool = False
    sample_count: int = 0
    baseline_ear: float = 0.28
    baseline_ear_std: float = 0.03
    baseline_mar: float = 0.30
    baseline_mar_std: float = 0.05
    baseline_pitch: float = 0.0
    ear_close_threshold: float = 0.21
    mar_yawn_threshold: float = 0.65
    blink_duration_s: float = 0.12
    progress: float = 0.0

    def to_dict(self) -> dict:
        return {
            "calibrated": self.calibrated,
            "sample_count": self.sample_count,
            "baseline_ear": round(self.baseline_ear, 4),
            "baseline_ear_std": round(self.baseline_ear_std, 4),
            "baseline_mar": round(self.baseline_mar, 4),
            "ear_close_threshold": round(self.ear_close_threshold, 4),
            "mar_yawn_threshold": round(self.mar_yawn_threshold, 4),
            "progress": round(self.progress, 3),
        }


class AdaptiveCalibrator:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._ears: list[float] = []
        self._mars: list[float] = []
        self._pitches: list[float] = []
        self.thresholds = PersonalizedThresholds(
            ear_close_threshold=self.config.default_ear_close_threshold,
            mar_yawn_threshold=self.config.default_mar_yawn_threshold,
            blink_duration_s=self.config.default_blink_duration_s,
        )
        self.active = False

    def start(self) -> None:
        self._ears.clear()
        self._mars.clear()
        self._pitches.clear()
        self.active = True
        self.thresholds.calibrated = False
        self.thresholds.sample_count = 0
        self.thresholds.progress = 0.0

    def stop_without_commit(self) -> None:
        self.active = False

    def observe(self, metrics: FaceMetrics) -> PersonalizedThresholds:
        needed = self.config.min_calibration_samples
        if not self.active:
            return self.thresholds
        if not metrics.reliable or metrics.confidence < self.config.uncertain_confidence:
            self.thresholds.progress = min(len(self._ears) / needed, 0.99)
            return self.thresholds
        if metrics.ear is not None:
            self._ears.append(metrics.ear)
        if metrics.mar is not None:
            self._mars.append(metrics.mar)
        if metrics.pitch_deg is not None:
            self._pitches.append(metrics.pitch_deg)

        self.thresholds.sample_count = len(self._ears)
        self.thresholds.progress = min(len(self._ears) / needed, 1.0)
        if len(self._ears) >= needed:
            self._commit()
            self.active = False
        return self.thresholds

    def _commit(self) -> None:
        ears = np.array(self._ears, dtype=np.float64)
        mars = np.array(self._mars, dtype=np.float64) if self._mars else np.array([0.3])
        pitches = np.array(self._pitches, dtype=np.float64) if self._pitches else np.array([0.0])

        ear_mean = float(np.mean(ears))
        ear_std = float(np.std(ears)) if len(ears) > 1 else 0.03
        mar_mean = float(np.mean(mars))
        mar_std = float(np.std(mars)) if len(mars) > 1 else 0.05

        close = ear_mean - max(self.config.ear_close_std_factor * ear_std, self.config.min_ear_close_offset)
        close = max(0.12, min(close, ear_mean - 0.02))
        yawn = max(
            self.config.default_mar_yawn_threshold,
            mar_mean + self.config.mar_yawn_std_factor * mar_std,
        )

        self.thresholds = PersonalizedThresholds(
            calibrated=True,
            sample_count=len(self._ears),
            baseline_ear=ear_mean,
            baseline_ear_std=ear_std,
            baseline_mar=mar_mean,
            baseline_mar_std=mar_std,
            baseline_pitch=float(np.mean(pitches)),
            ear_close_threshold=float(close),
            mar_yawn_threshold=float(yawn),
            blink_duration_s=self.config.default_blink_duration_s,
            progress=1.0,
        )
