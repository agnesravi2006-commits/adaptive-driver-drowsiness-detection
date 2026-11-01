"""Evaluate a trained window classifier on discovered UTA-RLDD videos."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from dataset_loader import discover_videos, print_download_instructions, project_root
from feature_extraction import extract_window_features_from_video, synthetic_window_dataset
from train import models_dir


def evaluate_model(model_path: Path, max_videos: int | None, frame_stride: int, window_seconds: float) -> dict:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    samples = [s for s in discover_videos() if s.label_id is not None]
    if not samples:
        raise FileNotFoundError("No labeled videos found.\n" + print_download_instructions())
    if max_videos:
        samples = samples[:max_videos]

    label_to_index = {0: 0, 5: 1, 10: 2}
    y_true = []
    y_pred = []
    per_video = []
    for sample in samples:
        windows = extract_window_features_from_video(
            sample.path, window_seconds=window_seconds, frame_stride=frame_stride
        )
        if not windows:
            continue
        preds = model.predict(np.asarray(windows, dtype=np.float32))
        truth = label_to_index[sample.label_id]
        y_true.extend([truth] * len(preds))
        y_pred.extend(preds.tolist())
        majority = int(Counter(preds.tolist()).most_common(1)[0][0])
        per_video.append(
            {
                "path": sample.path,
                "true_label": sample.label_name,
                "majority_pred": bundle["label_map"][majority],
                "n_windows": len(preds),
            }
        )

    if not y_true:
        raise RuntimeError("No windows were extracted for evaluation.")

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    accuracy = float((y_true_a == y_pred_a).mean())
    report = {
        "model_path": str(model_path),
        "source": bundle.get("source"),
        "window_accuracy": accuracy,
        "n_windows": int(len(y_true)),
        "n_videos": len(per_video),
        "per_video": per_video,
    }
    try:
        from sklearn.metrics import classification_report, confusion_matrix

        report["classification_report"] = classification_report(
            y_true_a, y_pred_a, target_names=["alert", "low_vigilant", "drowsy"], output_dict=True
        )
        report["confusion_matrix"] = confusion_matrix(y_true_a, y_pred_a).tolist()
    except Exception as exc:
        report["sklearn_metrics_error"] = str(exc)
    return report


def evaluate_smoke(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    X, y = synthetic_window_dataset(n_per_class=20)
    pred = bundle["model"].predict(X)
    return {
        "model_path": str(model_path),
        "source": bundle.get("source"),
        "note": "Synthetic holdout-style check. Not a UTA-RLDD result.",
        "accuracy": float((pred == y).mean()),
        "n_samples": int(len(y)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    if args.smoke_test:
        model_path = args.model or (models_dir() / "drowsiness_window_clf_smoke.joblib")
        if not model_path.exists():
            print(f"Smoke model not found at {model_path}. Run: python backend/train.py --smoke-test", file=sys.stderr)
            return 1
        print(json.dumps(evaluate_smoke(model_path), indent=2))
        return 0

    model_path = args.model or (models_dir() / "drowsiness_window_clf.joblib")
    if not model_path.exists():
        print(f"Trained model not found at {model_path}.", file=sys.stderr)
        return 1
    try:
        report = evaluate_model(model_path, args.max_videos, args.frame_stride, args.window_seconds)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    out = project_root() / "models" / "last_eval_summary.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
