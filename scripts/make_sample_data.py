"""Generate sample data for BioSignal-FM smoke tests.

Creates a small NPZ file with synthetic EMG data that can be used to verify
the full pipeline (data -> preprocessing -> model -> evaluation) works
without downloading the real NinaPro / PhysioNet / Brain-BIDS datasets.

Usage::

    python scripts/make_sample_data.py --output data/sample/smoke_test.npz --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample/smoke_test.npz"),
        help="Output NPZ file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of windows")
    parser.add_argument("--n-channels", type=int, default=16, help="Number of channels")
    parser.add_argument("--n-samples-per-window", type=int, default=400, help="Samples per window")
    parser.add_argument("--sampling-rate", type=int, default=200, help="Sampling rate (Hz)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Generate synthetic EMG-like signals
    n_subjects = 4
    n_classes = 4
    label_names = ("rest", "thumb_flex", "index_flex", "fist")

    signals = np.zeros(
        (args.n_samples, args.n_channels, args.n_samples_per_window), dtype=np.float32
    )
    subject_ids = np.zeros(args.n_samples, dtype=np.int32)
    labels = np.zeros(args.n_samples, dtype=np.int32)

    t = np.arange(args.n_samples_per_window) / args.sampling_rate
    for i in range(args.n_samples):
        subj = i % n_subjects
        cls = i % n_classes
        # Class-dependent amplitude envelope
        envelope = 0.0 if cls == 0 else 0.3 + 0.7 * (cls / (n_classes - 1))
        # 80 Hz carrier with 5 Hz bursts
        carrier = np.sin(2 * np.pi * 80 * t)
        burst = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
        signal = carrier * envelope * burst + 0.05 * rng.standard_normal(args.n_samples_per_window)
        # Per-channel variation
        channel_gains = rng.uniform(0.8, 1.2, args.n_channels)
        signals[i] = (signal[None, :] * channel_gains[:, None]).astype(np.float32)
        subject_ids[i] = subj
        labels[i] = cls

    np.savez(
        args.output,
        signal=signals,
        modality="emg",
        sampling_rate_hz=args.sampling_rate,
        subject_ids=subject_ids,
        labels=labels,
        label_names=label_names,
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"  signal shape: {signals.shape}")
    print(f"  subjects: {sorted(set(subject_ids.tolist()))}")
    print(f"  labels:   {sorted(set(labels.tolist()))}")


if __name__ == "__main__":
    main()
