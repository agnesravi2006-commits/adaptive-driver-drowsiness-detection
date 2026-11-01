import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from calibration import AdaptiveCalibrator
from config import AppConfig
from detection import FaceMetrics


def test_calibration_builds_personalized_threshold():
    cfg = AppConfig(min_calibration_samples=10)
    cal = AdaptiveCalibrator(cfg)
    cal.start()
    for _ in range(12):
        metrics = FaceMetrics(True, 0.95, 0.30, 0.30, 0.30, 0.28, 2.0, 0.0, 0.0)
        cal.observe(metrics)
    assert cal.thresholds.calibrated
    assert cal.thresholds.ear_close_threshold < cal.thresholds.baseline_ear
    assert cal.thresholds.mar_yawn_threshold >= cfg.default_mar_yawn_threshold
