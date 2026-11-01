import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from calibration import PersonalizedThresholds
from detection import FaceMetrics
from risk_assessment import ALERT, FACE_MISSING, HIGH_RISK, UNCERTAIN, TemporalRiskAssessor, score_to_state
from config import AppConfig


def _metrics(face=True, conf=0.9, ear=0.30, mar=0.3, pitch=0.0, message=""):
    return FaceMetrics(face, conf, ear, ear, ear, mar, pitch, 0.0, 0.0, message=message)


def test_missing_face_is_not_alert():
    assessor = TemporalRiskAssessor()
    thresholds = PersonalizedThresholds()
    result = assessor.update(1.0, _metrics(face=False, conf=0.0, ear=None, mar=None, pitch=None), thresholds)
    assert result.driver_state == FACE_MISSING
    assert result.driver_state != ALERT
    assert result.confidence_status == FACE_MISSING


def test_low_confidence_is_uncertain_not_alert():
    assessor = TemporalRiskAssessor()
    thresholds = PersonalizedThresholds()
    result = assessor.update(
        1.0,
        _metrics(conf=0.4, message="Detection Uncertain"),
        thresholds,
    )
    assert result.driver_state == UNCERTAIN
    assert result.driver_state != ALERT


def test_long_eye_closure_raises_score():
    assessor = TemporalRiskAssessor()
    thresholds = PersonalizedThresholds(ear_close_threshold=0.21, calibrated=True)
    for i in range(40):
        result = assessor.update(i * 0.2, _metrics(ear=0.10), thresholds)
    assert result.risk_score > 20
    assert result.details["temporal_state"] in {ALERT, "Low Risk", "Drowsy", HIGH_RISK}
    assert result.perclos > 0.5


def test_score_thresholds():
    cfg = AppConfig()
    assert score_to_state(10, cfg) == ALERT
    assert score_to_state(80, cfg) == HIGH_RISK
