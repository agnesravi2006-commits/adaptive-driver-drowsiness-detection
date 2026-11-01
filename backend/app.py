"""Flask backend for the adaptive multimodal drowsiness detection system."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from alerts import alert_policy
from calibration import AdaptiveCalibrator
from config import AppConfig
from dataset_loader import inspect_dataset
from detection import FaceAnalyzer, decode_data_url, decode_image_bytes
from risk_assessment import FACE_MISSING, TemporalRiskAssessor

FRONTEND = PROJECT / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
CORS(app)


class MonitorSession:
    def __init__(self) -> None:
        self.config = AppConfig()
        self.analyzer = FaceAnalyzer(self.config)
        self.calibrator = AdaptiveCalibrator(self.config)
        self.assessor = TemporalRiskAssessor(self.config)
        self.running = False
        self.muted = False
        self.started_at: float | None = None
        self.frames = 0
        self.last_fps = 0.0
        self._fps_mark = time.time()
        self._fps_count = 0
        self.last_error: str | None = None

    def reset_runtime(self) -> None:
        self.assessor = TemporalRiskAssessor(self.config)
        self.calibrator = AdaptiveCalibrator(self.config)
        self.frames = 0
        self.last_error = None


SESSION = MonitorSession()


def current_payload(extra: dict | None = None) -> dict:
    result = SESSION.assessor.last_result
    thresholds = SESSION.calibrator.thresholds
    policy = alert_policy(result.driver_state, SESSION.muted)
    payload = {
        "ok": True,
        "running": SESSION.running,
        "analyzer_available": SESSION.analyzer.available,
        "calibrating": SESSION.calibrator.active,
        "calibration": thresholds.to_dict(),
        "fps": round(SESSION.last_fps, 1),
        "frames": SESSION.frames,
        "error": SESSION.last_error,
        **result.to_dict(),
        **policy,
    }
    if extra:
        payload.update(extra)
    return payload


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "analyzer_available": SESSION.analyzer.available,
            "running": SESSION.running,
        }
    )


@app.post("/api/start")
def start():
    SESSION.running = True
    SESSION.started_at = time.time()
    SESSION.last_error = None
    return jsonify(current_payload())


@app.post("/api/stop")
def stop():
    SESSION.running = False
    return jsonify(current_payload())


@app.post("/api/reset")
def reset():
    SESSION.reset_runtime()
    SESSION.running = False
    SESSION.muted = False
    return jsonify(current_payload())


@app.post("/api/mute")
def mute():
    body = request.get_json(silent=True) or {}
    if "muted" in body:
        SESSION.muted = bool(body["muted"])
    else:
        SESSION.muted = not SESSION.muted
    return jsonify(current_payload())


@app.post("/api/calibrate/start")
def calibrate_start():
    SESSION.assessor.reset()
    SESSION.calibrator.start()
    SESSION.running = True
    return jsonify(current_payload({"message": "Look at the camera naturally. Calibration in progress."}))


@app.get("/api/status")
def status():
    return jsonify(current_payload())


@app.get("/api/config")
def get_config():
    return jsonify(SESSION.config.to_dict())


@app.post("/api/config")
def set_config():
    body = request.get_json(silent=True) or {}
    SESSION.config.update_from(body)
    return jsonify(SESSION.config.to_dict())


@app.get("/api/dataset")
def dataset_status():
    return jsonify(inspect_dataset())


@app.post("/api/analyze")
def analyze():
    if not SESSION.running:
        SESSION.running = True

    frame = None
    if request.files.get("frame"):
        frame = decode_image_bytes(request.files["frame"].read())
    else:
        body = request.get_json(silent=True) or {}
        data_url = body.get("image") or body.get("frame")
        if data_url:
            frame = decode_data_url(data_url)

    if frame is None:
        SESSION.last_error = "No frame received"
        return jsonify({**current_payload(), "ok": False, "error": "No frame received"}), 400

    if not SESSION.analyzer.available:
        SESSION.last_error = "MediaPipe is not available in this environment"
        return jsonify({**current_payload(), "ok": False, "face_detected": False,
                        "driver_state": FACE_MISSING,
                        "confidence_status": FACE_MISSING}), 503

    timestamp = time.time()
    metrics = SESSION.analyzer.analyze_bgr(frame)
    thresholds = SESSION.calibrator.observe(metrics)
    result = SESSION.assessor.update(timestamp, metrics, thresholds)

    SESSION.frames += 1
    SESSION._fps_count += 1
    elapsed = timestamp - SESSION._fps_mark
    if elapsed >= 1.0:
        SESSION.last_fps = SESSION._fps_count / elapsed
        SESSION._fps_count = 0
        SESSION._fps_mark = timestamp
    SESSION.last_error = metrics.message or None

    overlay = [{"x": p[0], "y": p[1]} for p in metrics.overlay_landmarks]
    width, height = metrics.image_size
    return jsonify(
        current_payload(
            {
                "face_detected": metrics.face_detected,
                "ear": None if metrics.ear is None else round(metrics.ear, 4),
                "left_ear": None if metrics.left_ear is None else round(metrics.left_ear, 4),
                "right_ear": None if metrics.right_ear is None else round(metrics.right_ear, 4),
                "mar": None if metrics.mar is None else round(metrics.mar, 4),
                "head_pitch": None if metrics.pitch_deg is None else round(metrics.pitch_deg, 2),
                "head_yaw": None if metrics.yaw_deg is None else round(metrics.yaw_deg, 2),
                "head_roll": None if metrics.roll_deg is None else round(metrics.roll_deg, 2),
                "landmarks": overlay,
                "image_width": width,
                "image_height": height,
                "message": metrics.message,
            }
        )
    )


@app.errorhandler(Exception)
def on_error(exc):
    SESSION.last_error = str(exc)
    return jsonify({"ok": False, "error": str(exc)}), 500


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    print("Driver Drowsiness System")
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
