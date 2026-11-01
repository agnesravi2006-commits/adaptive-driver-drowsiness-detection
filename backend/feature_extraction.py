"""Window-level feature extraction shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration import AdaptiveCalibrator
from config import AppConfig
from detection import FaceAnalyzer
from risk_assessment import TemporalRiskAssessor

FEATURE_NAMES = [
    "mean_ear",
    "std_ear",
    "min_ear",
    "perclos",
    "longest_closure_s",
    "mean_mar",
    "max_mar",
    "yawn_rate",
    "mean_pitch",
    "std_pitch",
    "nod_count",
    "blinks_per_min",
    "mean_confidence",
]


def window_feature_vector(ears, mars, pitches, confs, perclos, longest, yawns, nods, blinks_per_min) -> list[float]:
    ears = np.asarray(ears, dtype=np.float64) if len(ears) else np.array([0.0])
    mars = np.asarray(mars, dtype=np.float64) if len(mars) else np.array([0.0])
    pitches = np.asarray(pitches, dtype=np.float64) if len(pitches) else np.array([0.0])
    confs = np.asarray(confs, dtype=np.float64) if len(confs) else np.array([0.0])
    return [
        float(np.mean(ears)),
        float(np.std(ears)),
        float(np.min(ears)),
        float(perclos),
        float(longest),
        float(np.mean(mars)),
        float(np.max(mars)),
        float(yawns),
        float(np.mean(pitches)),
        float(np.std(pitches)),
        float(nods),
        float(blinks_per_min),
        float(np.mean(confs)),
    ]


def extract_window_features_from_video(
    video_path: str | Path,
    window_seconds: float = 20.0,
    frame_stride: int = 5,
    max_windows: int | None = None,
) -> list[list[float]]:
    import cv2

    analyzer = FaceAnalyzer()
    if not analyzer.available:
        analyzer.close()
        raise RuntimeError("MediaPipe is required to extract features from videos.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        analyzer.close()
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    config = AppConfig(window_seconds=window_seconds)
    calibrator = AdaptiveCalibrator(config)
    calibrator.start()
    assessor = TemporalRiskAssessor(config)

    windows: list[list[float]] = []
    frame_idx = 0
    t = 0.0
    last_emit = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            t = frame_idx / fps
            metrics = analyzer.analyze_bgr(frame)
            thresholds = calibrator.observe(metrics)
            result = assessor.update(t, metrics, thresholds)
            if calibrator.thresholds.calibrated and t - last_emit >= window_seconds:
                usable = [o for o in assessor.window if o.ear is not None]
                ears = [o.ear for o in usable if o.ear is not None]
                mars = [o.mar for o in usable if o.mar is not None]
                pitches = [o.pitch_deg for o in usable if o.pitch_deg is not None]
                confs = [o.confidence for o in usable]
                row = window_feature_vector(
                    ears,
                    mars,
                    pitches,
                    confs,
                    result.perclos,
                    result.details.get("longest_closure_s", 0.0),
                    result.yawn_count,
                    result.nod_count,
                    result.details.get("blinks_per_min", 0.0),
                )
                windows.append(row)
                last_emit = t
                if max_windows and len(windows) >= max_windows:
                    break
            frame_idx += 1
    finally:
        cap.release()
        analyzer.close()
    return windows


def synthetic_window_dataset(n_per_class: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic toy features for pipeline tests. Not UTA-RLDD data."""
    rng = np.random.default_rng(42)
    rows = []
    labels = []
    centers = {
        0: [0.30, 0.02, 0.24, 0.08, 0.20, 0.32, 0.40, 0.1, 2.0, 3.0, 0.2, 16.0, 0.9],
        1: [0.24, 0.04, 0.16, 0.28, 0.90, 0.40, 0.70, 0.8, 8.0, 7.0, 1.0, 9.0, 0.8],
        2: [0.18, 0.05, 0.10, 0.55, 2.20, 0.55, 0.95, 2.0, 16.0, 10.0, 2.4, 4.0, 0.7],
    }
    for label, center in centers.items():
        base = np.array(center, dtype=np.float64)
        noise = rng.normal(0, 0.08, size=(n_per_class, len(center)))
        data = np.clip(base + noise, 0, None)
        rows.append(data)
        labels.append(np.full(n_per_class, label, dtype=np.int64))
    X = np.vstack(rows)
    y = np.concatenate(labels)
    return X, y
