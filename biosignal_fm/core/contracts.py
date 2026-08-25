"""Library-independent canonical contracts for BioSignal-FM V4.

The core package intentionally depends only on the Python standard library and
NumPy. Readers such as MNE and WFDB belong in modality adapters; they must
convert their native objects to :class:`Signal` before entering the research
pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "DataOrigin",
    "SignalProvenance",
    "SignalEvent",
    "SignalMetadata",
    "Signal",
    "SignalBatch",
]


class DataOrigin(str, Enum):
    """Evidence class of the signal data used in a run."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SignalProvenance:
    """Structured, immutable provenance for a signal.

    ``origin`` is deliberately explicit. A synthetic sample may be used for a
    smoke test or demo, but it must never be interpreted as a real benchmark
    observation by downstream reporting code.
    """

    origin: DataOrigin = DataOrigin.UNKNOWN
    source_dataset: str | None = None
    dataset_version: str | None = None
    source_uri: str | None = None
    license_id: str | None = None
    adapter_name: str | None = None
    adapter_version: str | None = None
    fallback_reason: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        if self.origin is DataOrigin.SYNTHETIC and not self.fallback_reason:
            # Synthetic data may be intentionally generated, not just used as a
            # fallback. The explicit reason remains required for auditability.
            object.__setattr__(self, "fallback_reason", "synthetic generation")

    @property
    def is_synthetic(self) -> bool:
        """Return whether this signal comes from synthetic generation."""
        return self.origin is DataOrigin.SYNTHETIC


@dataclass(frozen=True)
class SignalEvent:
    """A time-bounded event or annotation associated with a signal."""

    onset_seconds: float
    duration_seconds: float = 0.0
    label: str | None = None
    value: str | int | float | None = None

    def __post_init__(self) -> None:
        if self.onset_seconds < 0:
            raise ValueError("SignalEvent.onset_seconds must be non-negative")
        if self.duration_seconds < 0:
            raise ValueError("SignalEvent.duration_seconds must be non-negative")


@dataclass(frozen=True)
class SignalMetadata:
    """Metadata carried with a canonical signal.

    The data contract distinguishes a dataset, subject, session, recording,
    task, and signal window rather than flattening them into a dataset-specific
    object. Free-form fields are contained in immutable ``extra`` for
    compatibility without weakening the core fields.
    """

    modality: str
    sampling_rate_hz: float
    channel_names: tuple[str, ...]
    units: tuple[str, ...] | str = "unknown"
    subject_id: str | int | None = None
    session_id: str | int | None = None
    recording_id: str | None = None
    task_id: str | None = None
    acquisition_system: str | None = None
    source_dataset: str | None = None
    window_id: str | None = None
    provenance: SignalProvenance = field(default_factory=SignalProvenance)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_modality = self.modality.strip().lower()
        if not normalized_modality:
            raise ValueError("SignalMetadata.modality must not be empty")
        if not np.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0:
            raise ValueError("SignalMetadata.sampling_rate_hz must be finite and positive")
        if not self.channel_names:
            raise ValueError("SignalMetadata.channel_names must not be empty")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("SignalMetadata.channel_names must be unique")
        if isinstance(self.units, tuple) and len(self.units) not in (1, len(self.channel_names)):
            raise ValueError("SignalMetadata.units must contain one unit or one unit per channel")
        object.__setattr__(self, "modality", normalized_modality)
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    @property
    def channel_count(self) -> int:
        """Return the expected number of channels."""
        return len(self.channel_names)


@dataclass(frozen=True)
class Signal:
    """Canonical two-dimensional biosignal with explicit time basis.

    ``data`` is shaped ``(channels, samples)``. Arrays are copied and marked
    read-only at construction so a frozen contract cannot be modified through
    an aliased NumPy array after validation.
    """

    data: np.ndarray
    metadata: SignalMetadata
    timestamps_seconds: np.ndarray | None = None
    events: tuple[SignalEvent, ...] = ()
    missing_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        data = np.array(self.data, dtype=np.float32, copy=True)
        if data.ndim != 2:
            raise ValueError(f"Signal.data must be 2D (channels, samples), got {data.shape}")
        if data.shape[0] != self.metadata.channel_count:
            raise ValueError(
                "Signal data channel count does not match SignalMetadata.channel_names"
            )
        if data.shape[1] == 0:
            raise ValueError("Signal.data must contain at least one sample")
        data.setflags(write=False)
        object.__setattr__(self, "data", data)

        if self.timestamps_seconds is not None:
            timestamps = np.array(self.timestamps_seconds, dtype=np.float64, copy=True)
            if timestamps.ndim != 1 or timestamps.shape[0] != data.shape[1]:
                raise ValueError("timestamps_seconds must be 1D with one entry per signal sample")
            if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
                raise ValueError("timestamps_seconds must be finite and strictly increasing")
            timestamps.setflags(write=False)
            object.__setattr__(self, "timestamps_seconds", timestamps)

        if self.missing_mask is not None:
            missing_mask = np.array(self.missing_mask, dtype=bool, copy=True)
            if missing_mask.shape != data.shape:
                raise ValueError("missing_mask must have the same shape as Signal.data")
            missing_mask.setflags(write=False)
            object.__setattr__(self, "missing_mask", missing_mask)

        object.__setattr__(self, "events", tuple(self.events))

    @property
    def duration_seconds(self) -> float:
        """Return the duration implied by the sampling rate."""
        return float(self.data.shape[1] / self.metadata.sampling_rate_hz)

    @property
    def is_synthetic(self) -> bool:
        """Return whether the signal is explicitly synthetic."""
        return self.metadata.provenance.is_synthetic

    def with_data(
        self,
        data: np.ndarray,
        *,
        sampling_rate_hz: float | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Signal:
        """Return a new signal after a transformation while retaining provenance."""
        metadata = self.metadata
        if sampling_rate_hz is not None or extra_metadata:
            merged_extra = dict(metadata.extra)
            merged_extra.update(extra_metadata or {})
            metadata = SignalMetadata(
                modality=metadata.modality,
                sampling_rate_hz=sampling_rate_hz or metadata.sampling_rate_hz,
                channel_names=metadata.channel_names,
                units=metadata.units,
                subject_id=metadata.subject_id,
                session_id=metadata.session_id,
                recording_id=metadata.recording_id,
                task_id=metadata.task_id,
                acquisition_system=metadata.acquisition_system,
                source_dataset=metadata.source_dataset,
                window_id=metadata.window_id,
                provenance=metadata.provenance,
                extra=merged_extra,
            )
        return Signal(data=data, metadata=metadata, events=self.events)


@dataclass(frozen=True)
class SignalBatch:
    """An immutable collection of canonical signals for a pipeline operation."""

    signals: tuple[Signal, ...]

    def __post_init__(self) -> None:
        signals = tuple(self.signals)
        if not signals:
            raise ValueError("SignalBatch.signals must not be empty")
        object.__setattr__(self, "signals", signals)

    @property
    def modalities(self) -> tuple[str, ...]:
        """Return unique modalities in first-seen order."""
        return tuple(dict.fromkeys(signal.metadata.modality for signal in self.signals))

    @property
    def contains_synthetic_data(self) -> bool:
        """Return whether any signal in the batch is synthetic."""
        return any(signal.is_synthetic for signal in self.signals)

    def by_modality(self, modality: str) -> tuple[Signal, ...]:
        """Return signals belonging to a modality."""
        normalized = modality.strip().lower()
        return tuple(signal for signal in self.signals if signal.metadata.modality == normalized)
