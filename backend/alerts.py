"""Progressive alert policy used by the backend and mirrored in the UI."""

from __future__ import annotations

from risk_assessment import ALERT, DROWSY, FACE_MISSING, HIGH_RISK, LOW_RISK, UNCERTAIN


LEVELS = {
    FACE_MISSING: 0,
    UNCERTAIN: 0,
    ALERT: 0,
    LOW_RISK: 1,
    DROWSY: 2,
    HIGH_RISK: 3,
}


def alert_policy(driver_state: str, muted: bool) -> dict:
    level = LEVELS.get(driver_state, 0)
    visual = level >= 1
    sound = (not muted) and level >= 2
    repeated = (not muted) and level >= 3
    messages = {
        0: "",
        1: "Early fatigue signs detected. Stay focused.",
        2: "Drowsiness detected. Consider taking a break.",
        3: "HIGH RISK: Stop driving when safe. Severe drowsiness.",
    }
    if driver_state == FACE_MISSING:
        banner = "Camera cannot see the driver. Sit in view of the webcam."
    elif driver_state == UNCERTAIN:
        banner = "Detection is uncertain. Face the camera and improve lighting."
    else:
        banner = messages[level]
    return {
        "alert_level": level,
        "visual_warning": visual,
        "sound_alert": sound,
        "repeated_sound": repeated,
        "muted": muted,
        "banner": banner,
    }
