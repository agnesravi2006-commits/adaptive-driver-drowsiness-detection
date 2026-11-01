"""Train a temporal-window classifier from UTA-RLDD videos when they exist.

This script extracts the same multimodal features used by the live system
(EAR statistics, PERCLOS, yawns, head-pose variation) from labeled videos
and fits a scikit-learn classifier.

It will not invent UTA-RLDD accuracy numbers. If videos are missing it
exits with instructions. Pass --smoke-test to verify the sklearn pipeline
on clearly-labeled synthetic window features (not real dataset results).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from dataset_loader import LABEL_MAP, discover_videos, inspect_dataset, print_download_instructions, project_root
from feature_extraction import FEATURE_NAMES, extract_window_features_from_video, synthetic_window_dataset


def models_dir() -> Path:
    path = project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_classifier(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    metrics = {}
    if len(np.unique(y)) > 1 and len(y) >= 6:
        folds = min(5, int(np.min(np.bincount(y))), len(y))
        folds = max(2, folds)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        metrics["cv_accuracy_mean"] = float(scores.mean())
        metrics["cv_accuracy_std"] = float(scores.std())
        metrics["cv_folds"] = folds
    pipeline.fit(X, y)
    return pipeline, metrics


def train_from_videos(max_videos: int | None, frame_stride: int, window_seconds: float) -> dict:
    samples = [s for s in discover_videos() if s.label_id is not None]
    if not samples:
        raise FileNotFoundError(
            "No labeled UTA-RLDD videos found under dataset/raw.\n" + print_download_instructions()
        )
    if max_videos:
        samples = samples[:max_videos]

    X_rows = []
    y_rows = []
    used = []
    label_to_index = {0: 0, 5: 1, 10: 2}

    for sample in samples:
        windows = extract_window_features_from_video(
            sample.path,
            window_seconds=window_seconds,
            frame_stride=frame_stride,
        )
        for row in windows:
            X_rows.append(row)
            y_rows.append(label_to_index[sample.label_id])
        used.append({"path": sample.path, "label": sample.label_name, "windows": len(windows)})
        print(f"Extracted {len(windows)} windows from {sample.path} ({sample.label_name})")

    if not X_rows:
        raise RuntimeError("Videos were found but no feature windows could be extracted.")

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    model, metrics = _train_classifier(X, y)
    out = models_dir() / "drowsiness_window_clf.joblib"
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "label_map": {0: "alert", 1: "low_vigilant", 2: "drowsy"},
        "source": "uta-rldd",
        "videos_used": used,
        "metrics": metrics,
    }
    joblib.dump(payload, out)
    summary = {
        "model_path": str(out),
        "n_windows": int(len(y)),
        "n_videos": len(used),
        "class_counts": {LABEL_MAP[k]: int(np.sum(y == label_to_index[k])) for k in LABEL_MAP},
        "metrics": metrics,
        "dataset_inventory": inspect_dataset(),
    }
    (models_dir() / "last_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_smoke_test() -> dict:
    X, y = synthetic_window_dataset(n_per_class=40)
    model, metrics = _train_classifier(X, y)
    out = models_dir() / "drowsiness_window_clf_smoke.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "label_map": {0: "alert", 1: "low_vigilant", 2: "drowsy"},
            "source": "synthetic_smoke_test",
            "metrics": metrics,
        },
        out,
    )
    return {
        "model_path": str(out),
        "source": "synthetic_smoke_test",
        "n_windows": int(len(y)),
        "metrics": metrics,
        "note": "Synthetic features only. This is not a UTA-RLDD result.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    if args.smoke_test:
        summary = train_smoke_test()
        print(json.dumps(summary, indent=2))
        return 0
    try:
        summary = train_from_videos(args.max_videos, args.frame_stride, args.window_seconds)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
