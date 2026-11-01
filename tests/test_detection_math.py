import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from detection import aspect_ratio, mouth_aspect_ratio


def test_open_eye_has_higher_ear_than_closed_eye():
    open_eye = [(0, 0), (1, 2), (2, 2), (3, 0), (2, -2), (1, -2)]
    closed_eye = [(0, 0), (1, 0.2), (2, 0.2), (3, 0), (2, -0.2), (1, -0.2)]
    assert aspect_ratio(open_eye) > aspect_ratio(closed_eye)
    assert aspect_ratio(closed_eye) < 0.2


def test_mar_increases_when_mouth_opens():
    closed = mouth_aspect_ratio((0, 0), (10, 0), (5, 1), (5, -1))
    open_mouth = mouth_aspect_ratio((0, 0), (10, 0), (5, 5), (5, -5))
    assert open_mouth > closed
    assert math.isclose(open_mouth, 1.0)
