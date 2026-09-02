"""Run a minimal real MIT-BIH record-held-out protocol smoke test.

This validates WFDB parsing, annotation-centered windows, record separation,
provenance, and artifact generation. It is not an arrhythmia benchmark or a
clinical-performance result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from biosignal_fm.data import MITBIHLoader
from biosignal_fm.reproducibility import RunManifest, set_global_seed


def _features(samples: list[Any]) -> np.ndarray:
    """Return deterministic per-lead morphology summary features."""
    vectors: list[np.ndarray] = []
    for sample in samples:
        center = sample.n_samples // 2
        signal = sample.signal
        vectors.append(
            np.concatenate(
                [
                    np.mean(signal, axis=1),
                    np.std(signal, axis=1),
                    np.sqrt(np.mean(np.square(signal), axis=1)),
                    signal[:, center],
                ]
            )
        )
    return np.vstack(vectors)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: list[float] = []
    for label in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = (2 * true_positive) + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-record", type=int, default=100)
    parser.add_argument("--test-record", type=int, default=101)
    args = parser.parse_args()
    if args.train_record == args.test_record:
        raise ValueError("train-record and test-record must be distinct")

    set_global_seed(42)
    loader = MITBIHLoader(root_dir=args.root_dir, n_records=48)
    samples = loader.samples
    train_samples = [sample for sample in samples if sample.subject_id == args.train_record]
    test_samples = [sample for sample in samples if sample.subject_id == args.test_record]
    if not train_samples or not test_samples:
        raise ValueError(
            "The requested train/test records were not both found in the real data root"
        )
    shared_labels = sorted(
        {int(sample.label) for sample in train_samples if sample.label is not None}
        & {int(sample.label) for sample in test_samples if sample.label is not None}
    )
    if len(shared_labels) < 2:
        raise ValueError("The selected records do not share at least two annotation classes")
    train_samples = [sample for sample in train_samples if sample.label in shared_labels]
    test_samples = [sample for sample in test_samples if sample.label in shared_labels]
    train_vectors = _features(train_samples)
    test_vectors = _features(test_samples)
    train_labels = np.asarray([int(sample.label) for sample in train_samples], dtype=int)
    test_labels = np.asarray([int(sample.label) for sample in test_samples], dtype=int)
    labels = np.asarray(shared_labels, dtype=int)
    location = np.mean(train_vectors, axis=0)
    scale = np.std(train_vectors, axis=0)
    scale[scale == 0] = 1.0
    normalized_train = (train_vectors - location) / scale
    normalized_test = (test_vectors - location) / scale
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
        str(sample.metadata["source_record"]): sample.metadata["source_file_sha256"]
        for sample in samples
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    prediction_summary_path = args.output_dir / "prediction_summary.json"
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_train_windows": int(len(train_labels)),
        "n_test_windows": int(len(test_labels)),
        "n_train_records": 1,
        "n_test_records": 1,
        "n_shared_classes": int(len(labels)),
        "shared_label_ids": labels.tolist(),
        "claim_boundary": "real-data adapter/protocol smoke only; not a benchmark or clinical result",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    prediction_summary_path.write_text(
        json.dumps(
            {
                "held_out_record": args.test_record,
                "class_counts": {str(label): int(np.sum(test_labels == label)) for label in labels},
                "predicted_class_counts": {
                    str(label): int(np.sum(predicted == label)) for label in labels
                },
                "source_file_hashes": source_hashes,
                "raw_predictions_exported": False,
                "reason": "Avoid exporting beat-level predictions into a repository artifact.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = RunManifest.create(
        name="mitbih-real-ecg-record-held-out-protocol-smoke",
        seed=42,
        model_id="morphology-summary-nearest-centroid-v1",
        config={
            "dataset_loader": "MITBIHLoader",
            "window_seconds": 2.0,
            "feature": "per_lead_mean_std_rms_center_amplitude",
            "normalization": "training-record feature z-score only",
            "preprocessing": "Annotation-centered 2-second windows; no filtering or resampling.",
            "label_policy": "Evaluate only the source annotation classes shared by both locally retrieved records.",
        },
        dataset_provenance={
            "dataset_id": "physionet.mitdb.1.0.0",
            "dataset_version": "1.0.0",
            "license_id": "ODC-BY-1.0",
            "origin": "real",
            "source_uri": "https://www.physionet.org/content/mitdb/1.0.0/",
            "retrieved_records": [str(args.train_record), str(args.test_record)],
            "source_file_sha256": source_hashes,
            "benchmark_eligible": False,
        },
        protocol={
            "protocol_id": "mitbih-two-record-held-out-smoke-v1",
            "split": {"train_record": args.train_record, "test_record": args.test_record},
            "metrics": ["accuracy", "macro_f1"],
            "unit_of_analysis": "record-held-out protocol smoke; beat windows are not independent inferential units",
            "preprocessing_version": "mitbih-annotation-morphology-v1",
            "calibration": "none",
        },
        notes="A two-record real-data pipeline smoke test. It validates source annotations, provenance, and record separation only; it is not an arrhythmia efficacy, comparative, clinical, or population claim.",
    )
    manifest.validate_research_readiness()
    manifest.add_metric("accuracy", accuracy)
    manifest.add_metric("macro_f1", macro_f1)
    manifest.add_output(metrics_path)
    manifest.add_output(prediction_summary_path)
    manifest.save(args.output_dir / "manifest.json")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
