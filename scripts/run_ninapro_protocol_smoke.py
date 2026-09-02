"""Run a minimal real-data NinaPro DB5 held-out-subject protocol smoke test.

This utility validates data loading, subject separation, feature construction,
prediction artifact creation, and manifest completeness. It is deliberately not
a benchmark runner: two locally retrieved participants cannot support a
comparative, population, clinical, or foundation-model claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from biosignal_fm.data import NinaProDB5Loader
from biosignal_fm.reproducibility import RunManifest, set_global_seed


def _features(samples: list[Any]) -> np.ndarray:
    """Return a deterministic channel-wise RMS vector for each fixed window."""
    return np.vstack([np.sqrt(np.mean(np.square(sample.signal), axis=1)) for sample in samples])


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1_scores: list[float] = []
    for label in labels:
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(f1_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir", type=Path, help="Nested directory containing extracted NinaPro .mat files"
    )
    parser.add_argument(
        "output_dir", type=Path, help="External directory for sanitized study artifacts"
    )
    parser.add_argument("--train-subject", type=int, default=1)
    parser.add_argument("--test-subject", type=int, default=2)
    args = parser.parse_args()
    if args.train_subject == args.test_subject:
        raise ValueError("train-subject and test-subject must be distinct")

    set_global_seed(42)
    loader = NinaProDB5Loader(root_dir=args.root_dir, n_subjects=10)
    samples = loader.samples
    train_samples = [sample for sample in samples if sample.subject_id == args.train_subject]
    test_samples = [sample for sample in samples if sample.subject_id == args.test_subject]
    if not train_samples or not test_samples:
        raise ValueError(
            "The requested train/test participants were not both found in the real data root"
        )

    train_by_label: dict[int, list[Any]] = defaultdict(list)
    for sample in train_samples:
        if sample.label is not None:
            train_by_label[int(sample.label)].append(sample)
    test_samples = [sample for sample in test_samples if sample.label in train_by_label]
    if not test_samples:
        raise ValueError("No test windows share a label with the training participant")

    train_vectors = _features(train_samples)
    train_labels = np.asarray([int(sample.label) for sample in train_samples], dtype=int)
    test_vectors = _features(test_samples)
    test_labels = np.asarray([int(sample.label) for sample in test_samples], dtype=int)
    location = np.mean(train_vectors, axis=0)
    scale = np.std(train_vectors, axis=0)
    scale[scale == 0] = 1.0
    normalized_train = (train_vectors - location) / scale
    normalized_test = (test_vectors - location) / scale
    labels = np.asarray(sorted(train_by_label), dtype=int)
    centroids = np.vstack(
        [np.mean(normalized_train[train_labels == label], axis=0) for label in labels]
    )
    predicted = labels[
        np.argmin(
            np.sum((normalized_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1
        )
    ]

    accuracy = float(np.mean(predicted == test_labels))
    macro_f1 = _macro_f1(test_labels, predicted)
    source_hashes = {
        sample.metadata["source_file"]: sample.metadata["source_file_sha256"] for sample in samples
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "prediction_summary.json"
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_train_windows": int(len(train_samples)),
        "n_test_windows": int(len(test_samples)),
        "n_train_participants": 1,
        "n_test_participants": 1,
        "n_classes": int(len(labels)),
        "claim_boundary": "real-data adapter/protocol smoke only; not a benchmark or inferential result",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    prediction_summary = {
        "held_out_subject": args.test_subject,
        "class_counts": {str(label): int(np.sum(test_labels == label)) for label in labels},
        "predicted_class_counts": {str(label): int(np.sum(predicted == label)) for label in labels},
        "source_file_hashes": source_hashes,
        "raw_predictions_exported": False,
        "reason": "Prevent unnecessary export of participant-level window predictions in a repository artifact.",
    }
    predictions_path.write_text(
        json.dumps(prediction_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = RunManifest.create(
        name="ninapro-db5-real-emg-held-out-subject-protocol-smoke",
        seed=42,
        model_id="rms-nearest-centroid-v1",
        config={
            "dataset_loader": "NinaProDB5Loader",
            "window_seconds": 2.0,
            "window_overlap_seconds": 0.5,
            "feature": "channelwise_rms",
            "normalization": "training-subject feature z-score only",
            "preprocessing": "No signal filtering or resampling; fixed-window feature extraction only.",
        },
        dataset_provenance={
            "dataset_id": "zenodo.1000116",
            "dataset_version": "v1",
            "license_id": "CC-BY-ND-4.0",
            "origin": "real",
            "source_uri": "https://zenodo.org/records/1000116",
            "retrieved_subject_archives": ["s1.zip", "s2.zip"],
            "source_file_sha256": source_hashes,
            "benchmark_eligible": False,
        },
        protocol={
            "protocol_id": "ninapro-db5-two-subject-held-out-smoke-v1",
            "split": {"train_subject": args.train_subject, "test_subject": args.test_subject},
            "metrics": ["accuracy", "macro_f1"],
            "unit_of_analysis": "participant-held-out protocol smoke; window predictions are not independent inferential units",
            "preprocessing_version": "ninapro-window-rms-v1",
            "calibration": "none",
        },
        notes="A two-participant real-data pipeline smoke test. It validates loader, provenance, and subject separation only; it is not an efficacy, comparative, or population claim.",
    )
    manifest.validate_research_readiness()
    manifest.add_metric("accuracy", accuracy)
    manifest.add_metric("macro_f1", macro_f1)
    manifest.add_output(metrics_path)
    manifest.add_output(predictions_path)
    manifest.save(args.output_dir / "manifest.json")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
