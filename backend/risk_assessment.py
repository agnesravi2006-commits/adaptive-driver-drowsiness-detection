"""Temporal drowsiness risk assessment over a sliding time window.

Risk score (0-100)
------------------
The live system never classifies a single frame as drowsy. Each accepted
frame is pushed into a time window of `window_seconds`. From that window:

    PERCLOS     = fraction of frames with eyes closed
    long_close  = longest consecutive eye-closure (seconds) / 3s, clipped to 1
    yawn_term   = yawn events in the window / 3, clipped to 1
    nod_term    = nod events in the window / 3, clipped to 1
    blink_term  = how far blink rate departed from a calm 10-20 blinks/min

    score = 100 * (
        w_perclos * PERCLOS +
        w_long    * long_close +
        w_yawn    * yawn_term +
        w_nod     * nod_term +
        w_blink   * blink_term
    )

Weights and state cut-offs live in `config.AppConfig` and can be changed
through `POST /api/config` without restarting the app.

Uncertain frames (no face, low confidence, blurry image) are recorded but
do not count as "Alert". The returned driver state becomes
"Face Not Detected" or "Detection Uncertain" instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from config import AppConfig
from detection import FaceMetrics
from calibration import PersonalizedThresholds


ALERT = "Alert"
LOW_RISK = "Low Risk"
DROWSY = "Drowsy"
HIGH_RISK = "High Risk"
FACE_MISSING = "Face Not Detected"
UNCERTAIN = "Detection Uncertain"


@dataclass
class FrameObservation:
    timestamp: float
    face_detected: bool
    confidence: float
    ear: float | None
    mar: float | None
    eyes_closed: bool
    yawning: bool
    pitch_deg: float | None


@dataclass
class RiskResult:
    driver_state: str
    risk_score: float
    confidence: float
    confidence_status: str
    eyes_closed: bool
    eye_closure_duration_s: float
    blink_count: int
    yawn_count: int
    nod_count: int
    perclos: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "driver_state": self.driver_state,
            "risk_score": round(self.risk_score, 1),
            "confidence": round(self.confidence, 3),
            "confidence_status": self.confidence_status,
            "eyes_closed": self.eyes_closed,
            "eye_closure_duration_s": round(self.eye_closure_duration_s, 2),
            "blink_count": self.blink_count,
            "yawn_count": self.yawn_count,
            "nod_count": self.nod_count,
            "perclos": round(self.perclos, 3),
        }
        payload.update(self.details)
        return payload


class TemporalRiskAssessor:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.window: deque[FrameObservation] = deque()
        self.session_blinks = 0
        self.session_yawns = 0
        self.session_nods = 0
        self._closed = False
        self._close_started: float | None = None
        self._current_close_s = 0.0
        self._yawn_open = False
        self._nod_down = False
        self._missing_streak = 0
        self.last_result = self._empty_result(0.0)

    def reset(self) -> None:
        self.window.clear()
        self.session_blinks = 0
        self.session_yawns = 0
        self.session_nods = 0
        self._closed = False
        self._close_started = None
        self._current_close_s = 0.0
        self._yawn_open = False
        self._nod_down = False
        self._missing_streak = 0
        self.last_result = self._empty_result(0.0)

    def update(
        self,
        timestamp: float,
        metrics: FaceMetrics,
        thresholds: PersonalizedThresholds,
    ) -> RiskResult:
        cfg = self.config
        reliable = (
            metrics.face_detected
            and metrics.confidence >= cfg.min_face_confidence
            and metrics.ear is not None
        )

        eyes_closed = False
        yawning = False
        if reliable and metrics.ear is not None:
            eyes_closed = metrics.ear < thresholds.ear_close_threshold
            yawning = bool(metrics.mar is not None and metrics.mar >= thresholds.mar_yawn_threshold)

        if reliable:
            self._missing_streak = 0
            self._track_blinks(timestamp, eyes_closed)
            self._track_yawns(yawning)
            self._track_nods(metrics.pitch_deg, thresholds.baseline_pitch)
        else:
            self._missing_streak += 1
            if self._closed and self._close_started is not None:
                self._current_close_s = timestamp - self._close_started
            else:
                self._current_close_s = 0.0

        obs = FrameObservation(
            timestamp=timestamp,
            face_detected=metrics.face_detected,
            confidence=metrics.confidence,
            ear=metrics.ear,
            mar=metrics.mar,
            eyes_closed=eyes_closed,
            yawning=yawning,
            pitch_deg=metrics.pitch_deg,
        )
        self.window.append(obs)
        self._trim(timestamp)

        result = self._score(timestamp, metrics, reliable, eyes_closed)
        self.last_result = result
        return result

    def _track_blinks(self, timestamp: float, eyes_closed: bool) -> None:
        if eyes_closed:
            if not self._closed:
                self._closed = True
                self._close_started = timestamp
            self._current_close_s = timestamp - (self._close_started or timestamp)
            return
        if self._closed:
            duration = timestamp - (self._close_started or timestamp)
            if 0.06 <= duration <= 0.8:
                self.session_blinks += 1
            self._closed = False
            self._close_started = None
            self._current_close_s = 0.0
        else:
            self._current_close_s = 0.0

    def _track_yawns(self, yawning: bool) -> None:
        if yawning and not self._yawn_open:
            self._yawn_open = True
            self.session_yawns += 1
        elif not yawning:
            self._yawn_open = False

    def _track_nods(self, pitch: float | None, baseline_pitch: float) -> None:
        if pitch is None:
            return
        delta = pitch - baseline_pitch
        if (not self._nod_down) and delta >= self.config.nod_pitch_threshold_deg:
            self._nod_down = True
            self.session_nods += 1
        elif self._nod_down and delta <= self.config.nod_reset_pitch_deg:
            self._nod_down = False

    def _trim(self, timestamp: float) -> None:
        cutoff = timestamp - self.config.window_seconds
        while self.window and self.window[0].timestamp < cutoff:
            self.window.popleft()

    def _score(self, timestamp: float, metrics: FaceMetrics, reliable: bool, eyes_closed: bool) -> RiskResult:
        cfg = self.config
        usable = [o for o in self.window if o.face_detected and o.ear is not None]
        if usable:
            closed_frames = sum(1 for o in usable if o.eyes_closed)
            perclos = closed_frames / len(usable)
            longest = _longest_closure_seconds(usable)
            window_yawns = _count_events(usable, "yawning")
            window_nods = self.session_nods  # session nods are already thresholded events
            window_s = max(self.config.window_seconds, 1.0)
            blinks_in_window = _count_blinks(usable)
            blinks_per_min = blinks_in_window * (60.0 / window_s)
            # Calm driving is roughly 10-20 blinks/min. Very low (microsleep) or
            # very high (fatigue fighting) both raise risk slightly.
            if blinks_per_min < 6:
                blink_term = min((6 - blinks_per_min) / 6.0, 1.0)
            elif blinks_per_min > 28:
                blink_term = min((blinks_per_min - 28) / 20.0, 1.0)
            else:
                blink_term = 0.0
            yawn_term = min(window_yawns / 3.0, 1.0)
            nod_term = min(window_nods / 3.0, 1.0)
            long_term = min(longest / 3.0, 1.0)
            score = 100.0 * (
                cfg.weight_perclos * perclos
                + cfg.weight_long_closure * long_term
                + cfg.weight_yawn * yawn_term
                + cfg.weight_nod * nod_term
                + cfg.weight_blink * blink_term
            )
        else:
            perclos = 0.0
            longest = 0.0
            blink_term = yawn_term = nod_term = long_term = 0.0
            score = 0.0
            blinks_per_min = 0.0

        score = float(max(0.0, min(score, 100.0)))
        state = score_to_state(score, cfg)

        if not metrics.face_detected or self._missing_streak >= cfg.max_missing_face_frames:
            confidence_status = FACE_MISSING
            display_state = FACE_MISSING
        elif (not reliable) or metrics.confidence < cfg.uncertain_confidence or metrics.message == UNCERTAIN:
            confidence_status = UNCERTAIN
            display_state = UNCERTAIN
        else:
            confidence_status = "Reliable"
            display_state = state

        return RiskResult(
            driver_state=display_state,
            risk_score=score if display_state not in {FACE_MISSING, UNCERTAIN} else score,
            confidence=metrics.confidence,
            confidence_status=confidence_status,
            eyes_closed=eyes_closed if reliable else False,
            eye_closure_duration_s=self._current_close_s if reliable or self._closed else 0.0,
            blink_count=self.session_blinks,
            yawn_count=self.session_yawns,
            nod_count=self.session_nods,
            perclos=perclos,
            details={
                "temporal_state": state,
                "blinks_per_min": round(blinks_per_min, 2),
                "longest_closure_s": round(longest, 2),
                "components": {
                    "perclos": round(perclos, 3),
                    "long_closure": round(long_term, 3),
                    "yawn": round(yawn_term, 3),
                    "nod": round(nod_term, 3),
                    "blink": round(blink_term, 3),
                },
            },
        )

    def _empty_result(self, timestamp: float) -> RiskResult:
        return RiskResult(
            driver_state=FACE_MISSING,
            risk_score=0.0,
            confidence=0.0,
            confidence_status=FACE_MISSING,
            eyes_closed=False,
            eye_closure_duration_s=0.0,
            blink_count=0,
            yawn_count=0,
            nod_count=0,
            perclos=0.0,
        )


def score_to_state(score: float, config: AppConfig) -> str:
    if score >= config.high_risk_score:
        return HIGH_RISK
    if score >= config.drowsy_score:
        return DROWSY
    if score >= config.low_risk_score:
        return LOW_RISK
    return ALERT


def _longest_closure_seconds(observations: list[FrameObservation]) -> float:
    longest = 0.0
    run_start = None
    prev = None
    for obs in observations:
        if obs.eyes_closed:
            if run_start is None:
                run_start = obs.timestamp
        else:
            if run_start is not None and prev is not None:
                longest = max(longest, prev.timestamp - run_start)
            run_start = None
        prev = obs
    if run_start is not None and prev is not None:
        longest = max(longest, prev.timestamp - run_start)
    return longest


def _count_events(observations: list[FrameObservation], attr: str) -> int:
    count = 0
    open_event = False
    for obs in observations:
        flag = bool(getattr(obs, attr))
        if flag and not open_event:
            count += 1
            open_event = True
        elif not flag:
            open_event = False
    return count


def _count_blinks(observations: list[FrameObservation]) -> int:
    count = 0
    closed = False
    start = None
    for obs in observations:
        if obs.eyes_closed:
            if not closed:
                closed = True
                start = obs.timestamp
        elif closed:
            duration = obs.timestamp - (start or obs.timestamp)
            if 0.06 <= duration <= 0.8:
                count += 1
            closed = False
            start = None
    return count
