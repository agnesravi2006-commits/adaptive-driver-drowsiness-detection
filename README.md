# Adaptive Multimodal Driver Drowsiness Detection

Real-time webcam system for a B.E. Computer Science final-year project.
The live detector personalizes eye-closure thresholds, fuses several facial
signals, scores drowsiness over a time window, and raises progressive alerts.

## Run the application

From this project folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000)

1. Allow camera access.
2. Click **Start Monitoring**.
3. Click **Calibrate Driver** and look at the camera normally for a few seconds.
4. The dashboard shows EAR, MAR, closure time, yawns, nods, risk score, and status.

## Project layout

```
driver-drowsiness-system/
  backend/           Flask API, vision, calibration, risk, training
  frontend/          Dashboard UI
  dataset/           UTA-RLDD instructions (videos not bundled)
  models/            Saved classifiers after training
  tests/             Unit and API tests
```

## Novelty

- **Personalized calibration** estimates your baseline EAR/MAR and derives thresholds instead of one global cutoff.
- **Multimodal analysis** uses Eye Aspect Ratio, closure duration, Mouth Aspect Ratio/yawns, and head-pose nodding.
- **Temporal risk score** combines PERCLOS, long closures, yawns, nods, and blink-rate deviation over a sliding window.
- **Progressive alerts**: Low Risk (visual), Drowsy (visual + sound), High Risk (repeated sound). Mute is supported.
- **Confidence-aware output**: missing or unstable faces become `Face Not Detected` / `Detection Uncertain`, never `Alert`.

## Dataset status

UTA-RLDD is documented in `dataset/README.md`. The Drive folder is publicly
listed but each ZIP is 7–13 GB. Videos are not downloaded with this repo.
`python backend/dataset_loader.py --inspect` reports whatever is in `dataset/raw/`.

## Tests

```
python -m pytest tests -q
```

Pipeline smoke test (synthetic features only, not UTA-RLDD accuracy):

```
python backend/train.py --smoke-test
python backend/evaluate.py --smoke-test
```
