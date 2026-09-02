"""Library-independent canonical contracts for BioSignal-FM V4.

The core package intentionally depends only on the Python standard library and
NumPy. Readers such as MNE and WFDB belong in modality adapters; they must
convert their native objects to :class:`Signal` before entering the research
pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "DataOrigin",
    "SignalProvenance",
    "SignalProcessingStep",
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
class SignalProcessingStep:
    """One explicit, versioned signal transformation recorded in provenance."""

    name: str
    version: str
    config_hash: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    applied_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "config_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SignalProcessingStep.{field_name} must be a non-empty string")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


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
    processing_history: tuple[SignalProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        history = tuple(self.processing_history)
        if not all(isinstance(step, SignalProcessingStep) for step in history):
            raise TypeError(
                "SignalProvenance.processing_history must contain SignalProcessingStep values"
            )
        object.__setattr__(self, "processing_history", history)
        if self.origin is DataOrigin.SYNTHETIC and not self.fallback_reason:
            # Synthetic data may be intentionally generated, not just used as a
            # fallback. The explicit reason remains required for auditability.
            object.__setattr__(self, "fallback_reason", "synthetic generation")

    @property
    def is_synthetic(self) -> bool:
        """Return whether this signal comes from synthetic generation."""
        return self.origin is DataOrigin.SYNTHETIC

    def with_processing_step(self, step: SignalProcessingStep) -> SignalProvenance:
        """Return provenance extended with one explicit transformation record."""
        if not isinstance(step, SignalProcessingStep):
            raise TypeError("step must be a SignalProcessingStep")
        return replace(self, processing_history=(*self.processing_history, step))


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
    preprocessing_status: str = "raw"
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
        if not isinstance(self.preprocessing_status, str) or not self.preprocessing_status.strip():
            raise ValueError("SignalMetadata.preprocessing_status must be a non-empty string")
        object.__setattr__(self, "modality", normalized_modality)
        object.__setattr__(self, "preprocessing_status", self.preprocessing_status.strip().lower())
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
        timestamps_seconds: np.ndarray | None = None,
        missing_mask: np.ndarray | None = None,
        processing_step: SignalProcessingStep | None = None,
        preprocessing_status: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Signal:
        """Return a transformed signal without silently losing scientific context.

        A shape-changing transformation must provide replacement timestamps or a
        replacement missingness mask when the source carried either. This keeps
        resampling and windowing semantics explicit rather than allowing stale
        metadata to survive a changed sample axis.
        """
        transformed = np.asarray(data)
        same_sample_count = transformed.ndim == 2 and transformed.shape[1] == self.data.shape[1]
        if (
            self.timestamps_seconds is not None
            and timestamps_seconds is None
            and not same_sample_count
        ):
            raise ValueError(
                "Shape-changing transformation requires replacement timestamps_seconds"
            )
        if (
            self.missing_mask is not None
            and missing_mask is None
            and transformed.shape != self.data.shape
        ):
            raise ValueError("Shape-changing transformation requires replacement missing_mask")

        resolved_timestamps = timestamps_seconds
        if resolved_timestamps is None and same_sample_count:
            resolved_timestamps = self.timestamps_seconds
        resolved_missing_mask = missing_mask
        if resolved_missing_mask is None and transformed.shape == self.data.shape:
            resolved_missing_mask = self.missing_mask

        metadata = self.metadata
        if (
            sampling_rate_hz is not None
            or extra_metadata
            or processing_step is not None
            or preprocessing_status is not None
        ):
            merged_extra = dict(metadata.extra)
            merged_extra.update(extra_metadata or {})
            provenance = metadata.provenance
            if processing_step is not None:
                provenance = provenance.with_processing_step(processing_step)
            metadata = SignalMetadata(
                modality=metadata.modality,
                sampling_rate_hz=(
                    sampling_rate_hz if sampling_rate_hz is not None else metadata.sampling_rate_hz
                ),
                channel_names=metadata.channel_names,
                units=metadata.units,
                subject_id=metadata.subject_id,
                session_id=metadata.session_id,
                recording_id=metadata.recording_id,
                task_id=metadata.task_id,
                acquisition_system=metadata.acquisition_system,
                source_dataset=metadata.source_dataset,
                window_id=metadata.window_id,
                preprocessing_status=preprocessing_status or metadata.preprocessing_status,
                provenance=provenance,
                extra=merged_extra,
            )
        return Signal(
            data=data,
            metadata=metadata,
            timestamps_seconds=resolved_timestamps,
            events=self.events,
            missing_mask=resolved_missing_mask,
        )


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
