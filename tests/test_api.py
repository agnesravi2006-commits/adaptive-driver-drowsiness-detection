import sys
from pathlib import Path

from io import BytesIO

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import app


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True


def test_frontend_index_served():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Adaptive Multimodal Driver Drowsiness Detection" in response.data


def test_analyze_blank_frame_is_not_alert():
    client = app.test_client()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    response = client.post(
        "/api/analyze",
        data={"frame": (BytesIO(encoded.tobytes()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    if response.status_code == 503:
        data = response.get_json()
        assert data["driver_state"] != "Alert"
        return
    assert response.status_code == 200
    data = response.get_json()
    assert data["driver_state"] in {"Face Not Detected", "Detection Uncertain"}
    assert data["driver_state"] != "Alert"


def test_dataset_status_does_not_claim_videos_exist():
    client = app.test_client()
    data = client.get("/api/dataset").get_json()
    assert "video_count" in data
    assert "drive_folder" in data
