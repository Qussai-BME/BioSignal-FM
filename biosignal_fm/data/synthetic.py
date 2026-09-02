"""Synthetic biosignal generator for development, testing, and CI.

BioSignal-FM is designed to work on CPU-only compute without access to
the real NinaPro, PhysioNet, or Brain-BIDS datasets (which require manual
download and licensing agreements). This module provides a deterministic
synthetic generator that produces physiologically-plausible biosignals for:

- Unit and integration tests
- CI pipelines
- Quick start tutorials
- Smoke tests of the full training pipeline

The generator uses a superposition of:

- Modality-specific carrier frequencies (EMG: 50-150 Hz bursts, ECG: 1 Hz
  QRS-like waveform, EEG: 10 Hz alpha + noise, fNIRS: 0.1 Hz hemodynamic)
- Class-dependent amplitude modulation (gestures modulate EMG envelope)
- Subject-dependent offset (per-subject baseline)
- Session-dependent noise floor
- Reproducible per-subject random seed

The synthetic data is NOT a substitute for real data in scientific experiments.
It exists purely to validate the engineering pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..config import Modality
from .base import BiosignalSample, ModalityMetadata

__all__ = [
    "SyntheticBiosignalDataset",
    "make_synthetic_sample",
    "SYNTHETIC_LABEL_NAMES",
]


# Default label taxonomy per modality (8-class gesture set inspired by NinaPro)
SYNTHETIC_LABEL_NAMES: tuple[str, ...] = (
    "rest",
    "thumb_flex",
    "index_flex",
    "fist",
    "pinch",
    "wrist_pronation",
    "wrist_supination",
    "lateral_grasp",
)


@dataclass
class SyntheticBiosignalDataset:
    """Deterministic synthetic biosignal dataset.

    Generates ``n_samples`` samples of ``modality`` data with reproducible
    per-subject seeds. The dataset is fully in-memory (small by design).

    Parameters
    ----------
    modality : Modality or str
        The modality to generate ("emg", "ecg", "eeg", "ecog", "fnirs").
    n_subjects : int, optional
        Number of subjects. Default 10.
    n_sessions_per_subject : int, optional
        Number of sessions per subject. Default 3.
    n_samples_per_class : int, optional
        Number of samples per (subject, session, class). Default 10.
    n_channels : int, optional
        Number of channels. Default per modality.
    sampling_rate_hz : int, optional
        Sampling rate. Default per modality.
    window_length_seconds : float, optional
        Window length in seconds. Default 2.0.
    n_classes : int, optional
        Number of classes (1-8). Default 8.
    seed : int, optional
        Master seed. Per-subject seeds are derived deterministically.
    """

    modality: Modality | str
    n_subjects: int = 10
    n_sessions_per_subject: int = 3
    n_samples_per_class: int = 10
    n_channels: int | None = None
    sampling_rate_hz: int | None = None
    window_length_seconds: float = 2.0
    n_classes: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.modality, str):
            self.modality = Modality.from_str(self.modality)

        # Modality-specific defaults
        defaults = _modality_defaults(self.modality)
        if self.n_channels is None:
            self.n_channels = defaults["n_channels"]
        if self.sampling_rate_hz is None:
            self.sampling_rate_hz = defaults["sampling_rate_hz"]

        if self.n_classes < 1 or self.n_classes > len(SYNTHETIC_LABEL_NAMES):
            raise ValueError(
                f"n_classes must be in [1, {len(SYNTHETIC_LABEL_NAMES)}], got {self.n_classes}"
            )

        # Lazily generate samples
        self._samples: list[BiosignalSample] | None = None
        self._metadata = ModalityMetadata(
            modality=self.modality,
            sampling_rate_hz=self.sampling_rate_hz,
            n_channels=self.n_channels,
            channel_names=tuple(f"ch{i:02d}" for i in range(self.n_channels)),
            label_names=SYNTHETIC_LABEL_NAMES[: self.n_classes],
        )

    @property
    def metadata(self) -> ModalityMetadata:
        """The dataset's modality metadata."""
        return self._metadata

    def _generate(self) -> list[BiosignalSample]:
        # __post_init__ always resolves these from Modality|str / int|None to
        # their concrete Modality/int forms before any other method can run;
        # asserting here narrows the type for mypy without weakening the
        # public Modality|str / int|None API those fields intentionally have.
        assert isinstance(self.modality, Modality)
        assert self.n_channels is not None
        assert self.sampling_rate_hz is not None

        samples: list[BiosignalSample] = []
        n_samples_per_window = int(self.sampling_rate_hz * self.window_length_seconds)
        label_names = SYNTHETIC_LABEL_NAMES[: self.n_classes]

        for subj in range(self.n_subjects):
            # Cross-process deterministic per-subject seed.
            # Python's hash() is randomized per-process via PYTHONHASHSEED,
            # so we MUST NOT use it for reproducibility. Instead we derive the
            # seed from a SHA-256 of (seed, modality, subject). This produces
            # the same uint32 across processes, OSes, and Python invocations.
            mod_str = (
                self.modality.value if isinstance(self.modality, Modality) else str(self.modality)
            )
            seed_bytes = f"{self.seed}|{mod_str}|{subj}".encode()
            subj_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
            rng = np.random.default_rng(subj_seed)

            for sess in range(self.n_sessions_per_subject):
                for cls_idx in range(self.n_classes):
                    for _ in range(self.n_samples_per_class):
                        signal = _generate_signal(
                            modality=self.modality,
                            n_channels=self.n_channels,
                            n_samples=n_samples_per_window,
                            sampling_rate_hz=self.sampling_rate_hz,
                            label=cls_idx,
                            subject_id=subj,
                            session_id=sess,
                            rng=rng,
                        )
                        samples.append(
                            BiosignalSample(
                                signal=signal,
                                modality=self.modality,
                                sampling_rate_hz=self.sampling_rate_hz,
                                subject_id=subj,
                                session_id=sess,
                                label=cls_idx,
                                label_name=label_names[cls_idx],
                                metadata={
                                    "synthetic": True,
                                    "generator": "SyntheticBiosignalDataset",
                                    "source_dataset": "synthetic://biosignal-fm",
                                    "dataset_version": "v4",
                                    "fallback_reason": "intentional synthetic generation for testing or demonstration",
                                    "benchmark_eligible": False,
                                },
                            )
                        )
        return samples

    @property
    def samples(self) -> list[BiosignalSample]:
        """Lazy-loaded list of all samples."""
        if self._samples is None:
            self._samples = self._generate()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BiosignalSample:
        return self.samples[idx]

    def get_subject_ids(self) -> list[int]:
        """Return sorted list of subject IDs."""
        return sorted({s.subject_id for s in self.samples})

    def get_session_ids(self, subject_id: int) -> list[int]:
        """Return sorted list of session IDs for a given subject."""
        return sorted({s.session_id for s in self.samples if s.subject_id == subject_id})

    def iter_by_subject(self) -> Iterator[tuple[int, list[BiosignalSample]]]:
        """Iterate over (subject_id, list_of_samples)."""
        for subj in self.get_subject_ids():
            yield subj, [s for s in self.samples if s.subject_id == subj]


def make_synthetic_sample(
    modality: Modality | str = "emg",
    n_channels: int | None = None,
    n_samples: int | None = None,
    sampling_rate_hz: int | None = None,
    label: int = 0,
    subject_id: int = 0,
    session_id: int = 0,
    seed: int = 42,
) -> BiosignalSample:
    """Generate a single synthetic biosignal sample.

    Convenience wrapper around :class:`SyntheticBiosignalDataset` for
    test fixtures and quick experimentation.

    Parameters
    ----------
    modality : Modality or str
        The modality. Default "emg".
    n_channels : int, optional
        Number of channels. Defaults to modality-specific value.
    n_samples : int, optional
        Number of samples. Default = sampling_rate * 2 (2 seconds).
    sampling_rate_hz : int, optional
        Sampling rate. Defaults to modality-specific value.
    label : int
        Class label (0-7).
    subject_id : int
        Subject ID.
    session_id : int
        Session ID.
    seed : int
        Random seed.

    Returns
    -------
    BiosignalSample
        A synthetic sample.
    """
    if isinstance(modality, str):
        modality = Modality.from_str(modality)

    defaults = _modality_defaults(modality)
    if n_channels is None:
        n_channels = defaults["n_channels"]
    if sampling_rate_hz is None:
        sampling_rate_hz = defaults["sampling_rate_hz"]
    if n_samples is None:
        n_samples = sampling_rate_hz * 2  # 2-second window

    rng = np.random.default_rng(seed)
    signal = _generate_signal(
        modality=modality,
        n_channels=n_channels,
        n_samples=n_samples,
        sampling_rate_hz=sampling_rate_hz,
        label=label,
        subject_id=subject_id,
        session_id=session_id,
        rng=rng,
    )
    return BiosignalSample(
        signal=signal,
        modality=modality,
        sampling_rate_hz=sampling_rate_hz,
        subject_id=subject_id,
        session_id=session_id,
        label=label,
        label_name=SYNTHETIC_LABEL_NAMES[label] if label < len(SYNTHETIC_LABEL_NAMES) else None,
        metadata={
            "synthetic": True,
            "generator": "make_synthetic_sample",
            "source_dataset": "synthetic://biosignal-fm",
            "dataset_version": "v4",
            "fallback_reason": "intentional synthetic generation for testing or demonstration",
            "benchmark_eligible": False,
        },
    )


def _modality_defaults(modality: Modality) -> dict:
    """Return modality-specific defaults for channels and sampling rate."""
    defaults = {
        Modality.EMG: {"n_channels": 16, "sampling_rate_hz": 2000},
        Modality.ECG: {"n_channels": 12, "sampling_rate_hz": 360},
        Modality.EEG: {"n_channels": 64, "sampling_rate_hz": 160},
        Modality.ECOG: {"n_channels": 64, "sampling_rate_hz": 1000},
        Modality.FNIRS: {"n_channels": 32, "sampling_rate_hz": 10},
    }
    return defaults[modality]


def _generate_signal(
    modality: Modality,
    n_channels: int,
    n_samples: int,
    sampling_rate_hz: int,
    label: int,
    subject_id: int,
    session_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic biosignal.

    Each modality has a distinct generative model:

    - **EMG:** 50-150 Hz Gaussian-modulated sinusoid, amplitude modulated
      by class label (rest = 0, max effort = 1).
    - **ECG:** Sum of Gaussian peaks at ~1 Hz (QRS-like) with class-dependent
      heart rate.
    - **EEG:** 10 Hz alpha + class-dependent beta/gamma + pink noise.
    - **ECoG:** experimental high-frequency carrier plus noise; this is only a
      smoke-test generator and does not simulate a validated ECoG dataset.
    - **fNIRS:** 0.1 Hz hemodynamic response + class-dependent amplitude.
    """
    t = np.arange(n_samples) / sampling_rate_hz
    # Class amplitude envelope: 0 for rest, increasing for active classes
    class_envelope = 0.0 if label == 0 else 0.3 + 0.7 * (label / 7.0)
    # Subject-dependent baseline
    subject_offset = 0.05 * ((subject_id % 5) - 2)
    # Session-dependent noise floor
    session_noise = 0.02 + 0.01 * session_id

    if modality == Modality.EMG:
        # 80 Hz carrier, Gaussian bursts
        carrier = np.sin(2 * np.pi * 80 * t)
        # Burst envelope: 5 Hz bursts with class-dependent amplitude
        burst = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
        envelope = class_envelope * burst + session_noise
        signal = carrier[None, :] * envelope[None, :] * np.linspace(0.8, 1.2, n_channels)[:, None]
        signal += rng.normal(0, session_noise, signal.shape)
        return np.asarray(signal, dtype=np.float32)

    elif modality == Modality.ECG:
        # Heart rate: 60 bpm for rest, up to 120 bpm for active
        hr_hz = 1.0 + 1.0 * class_envelope
        ecg = np.zeros_like(t)
        for beat_idx in range(int(t[-1] * hr_hz) + 2):
            beat_t = beat_idx / hr_hz
            # QRS complex: sharp peak at beat_t
            ecg += 1.5 * np.exp(-((t - beat_t) ** 2) / (2 * 0.01**2))
            # T wave: smaller, later
            ecg += 0.3 * np.exp(-((t - beat_t - 0.2) ** 2) / (2 * 0.04**2))
        signal = np.stack([ecg + 0.1 * rng.normal(0, 1, n_samples) for _ in range(n_channels)])
        signal += subject_offset
        return np.asarray(signal, dtype=np.float32)

    elif modality == Modality.EEG:
        # 10 Hz alpha + class-dependent beta (20-30 Hz)
        alpha = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
        beta = class_envelope * np.sin(2 * np.pi * 25 * t + rng.uniform(0, 2 * np.pi))
        # Pink-ish noise (1/f spectrum approximation)
        white = rng.normal(0, 1, n_samples)
        pink = np.cumsum(white)
        pink = pink / (np.max(np.abs(pink)) + 1e-8) * 0.3
        base = alpha + beta + pink + subject_offset
        signal = np.stack([base + 0.05 * rng.normal(0, 1, n_samples) for _ in range(n_channels)])
        return np.asarray(signal, dtype=np.float32)

    elif modality == Modality.ECOG:
        # Experimental smoke-test waveform: high-frequency carrier with an
        # amplitude envelope. It is intentionally not presented as a realistic
        # clinical or benchmark ECoG simulation.
        high_gamma = np.sin(2 * np.pi * 85 * t + rng.uniform(0, 2 * np.pi))
        envelope = 0.2 + class_envelope + 0.1 * np.sin(2 * np.pi * 2 * t)
        base = high_gamma * envelope + subject_offset
        signal = np.stack([base + 0.08 * rng.normal(0, 1, n_samples) for _ in range(n_channels)])
        return np.asarray(signal, dtype=np.float32)

    elif modality == Modality.FNIRS:
        # 0.1 Hz hemodynamic response
        hb = np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi))
        # Class-dependent amplitude (active = larger hemodynamic response)
        hb *= 0.5 + class_envelope
        # Slow drift
        drift = 0.1 * np.sin(2 * np.pi * 0.01 * t)
        base = hb + drift + subject_offset
        signal = np.stack([base + 0.02 * rng.normal(0, 1, n_samples) for _ in range(n_channels)])
        return np.asarray(signal, dtype=np.float32)

    else:  # pragma: no cover
        raise ValueError(f"Unknown modality: {modality}")
