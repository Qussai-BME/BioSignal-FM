"""Run a minimal real EEGMMID participant-held-out protocol smoke test.

The run verifies event-derived motor-imagery windows, train-only feature
normalization, participant separation, provenance, and artifact generation. It
is not a benchmark or an inferential EEG-decoding result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from biosignal_fm.data import EEGMMIDLoader
from biosignal_fm.reproducibility import RunManifest, set_global_seed


def _features(samples: list[Any]) -> np.ndarray:
    """Return deterministic 8–30 Hz log-power features per EEG channel."""
    vectors: list[np.ndarray] = []
    for sample in samples:
        frequencies = np.fft.rfftfreq(sample.n_samples, d=1.0 / sample.sampling_rate_hz)
        spectrum = np.abs(np.fft.rfft(sample.signal, axis=1)) ** 2
        mask = (frequencies >= 8.0) & (frequencies <= 30.0)
        vectors.append(np.log1p(np.mean(spectrum[:, mask], axis=1)))
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
    parser.add_argument("--train-subject", type=int, default=1)
    parser.add_argument("--test-subject", type=int, default=2)
    args = parser.parse_args()
    if args.train_subject == args.test_subject:
        raise ValueError("train-subject and test-subject must be distinct")

    set_global_seed(42)
    loader = EEGMMIDLoader(root_dir=args.root_dir, n_subjects=109, runs=(4,))
    samples = loader.samples
    train_samples = [sample for sample in samples if sample.subject_id == args.train_subject]
    test_samples = [sample for sample in samples if sample.subject_id == args.test_subject]
    if not train_samples or not test_samples:
        raise ValueError(
            "The requested train/test participants were not both found in the real data root"
        )
    train_vectors = _features(train_samples)
    test_vectors = _features(test_samples)
    train_labels = np.asarray([int(sample.label) for sample in train_samples], dtype=int)
    test_labels = np.asarray([int(sample.label) for sample in test_samples], dtype=int)
    labels = np.asarray(sorted(set(train_labels.tolist()) & set(test_labels.tolist())), dtype=int)
    if labels.size < 2:
        raise ValueError("The held-out participant does not contain both motor-imagery labels")
    train_mask = np.isin(train_labels, labels)
    test_mask = np.isin(test_labels, labels)
    train_vectors, train_labels = train_vectors[train_mask], train_labels[train_mask]
    test_vectors, test_labels = test_vectors[test_mask], test_labels[test_mask]
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
        sample.metadata["source_file"]: sample.metadata["source_file_sha256"] for sample in samples
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    prediction_summary_path = args.output_dir / "prediction_summary.json"
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "n_train_windows": int(len(train_labels)),
        "n_test_windows": int(len(test_labels)),
        "n_train_participants": 1,
        "n_test_participants": 1,
        "n_classes": int(len(labels)),
        "claim_boundary": "real-data adapter/protocol smoke only; not a benchmark or inferential result",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    prediction_summary_path.write_text(
        json.dumps(
            {
                "held_out_subject": args.test_subject,
                "class_counts": {str(label): int(np.sum(test_labels == label)) for label in labels},
                "predicted_class_counts": {
                    str(label): int(np.sum(predicted == label)) for label in labels
                },
                "source_file_hashes": source_hashes,
                "raw_predictions_exported": False,
                "reason": "Avoid exporting participant-level windows into a repository artifact.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = RunManifest.create(
        name="eegmmid-real-eeg-held-out-subject-protocol-smoke",
        seed=42,
        model_id="alpha-beta-logpower-nearest-centroid-v1",
        config={
            "dataset_loader": "EEGMMIDLoader",
            "runs": [4],
            "event_labels": {"T1": "left_fist", "T2": "right_fist"},
            "window_seconds": 2.0,
            "feature": "channelwise_log_power_8_to_30_hz",
            "normalization": "training-subject feature z-score only",
            "preprocessing": "Annotation-aligned 2-second windows; no filter, re-reference, or resampling.",
        },
        dataset_provenance={
            "dataset_id": "physionet.eegmmidb.1.0.0",
            "dataset_version": "1.0.0",
            "license_id": "ODC-BY-1.0",
            "origin": "real",
            "source_uri": "https://www.physionet.org/content/eegmmidb/1.0.0/",
            "retrieved_recordings": ["S001R04.edf", "S002R04.edf"],
            "source_file_sha256": source_hashes,
            "benchmark_eligible": False,
        },
        protocol={
            "protocol_id": "eegmmid-r04-two-subject-held-out-smoke-v1",
            "split": {"train_subject": args.train_subject, "test_subject": args.test_subject},
            "metrics": ["accuracy", "macro_f1"],
            "unit_of_analysis": "participant-held-out protocol smoke; event windows are not independent inferential units",
            "preprocessing_version": "eegmmid-event-logpower-v1",
            "calibration": "none",
        },
        notes="A two-participant real-data pipeline smoke test. It validates source-event semantics, provenance, and participant separation only; it is not an efficacy, comparative, or population claim.",
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
