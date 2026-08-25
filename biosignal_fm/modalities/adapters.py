"""Modality edge adapters for converting legacy and reader-specific data.

This module deliberately uses duck typing for optional readers. It can adapt an
MNE-like object exposing ``get_data`` and ``info`` without importing MNE, and
it adapts legacy :class:`BiosignalSample` values without changing existing
loaders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import DataOrigin, Signal, SignalEvent, SignalMetadata, SignalProvenance
from ..data.base import BiosignalSample

__all__ = [
    "ArraySignalAdapter",
    "EMGAdapter",
    "EEGAdapter",
    "ECGAdapter",
    "ECoGAdapter",
    "FNIRSAdapter",
]


@dataclass(frozen=True)
class ArraySignalAdapter:
    """Adapter for a modality that can accept canonical, legacy, or MNE-like input."""

    modality: str
    adapter_name: str

    def to_signal(self, source: Any) -> Signal:
        """Convert a canonical signal, legacy sample, array mapping, or MNE-like object."""
        if isinstance(source, Signal):
            self._validate_modality(source.metadata.modality)
            return source
        if isinstance(source, BiosignalSample):
            return self.from_legacy_sample(source)
        if isinstance(source, Mapping):
            return self.from_mapping(source)
        if hasattr(source, "get_data") and hasattr(source, "info"):
            return self.from_mne_like(source)
        raise TypeError(
            f"{self.adapter_name} cannot adapt {type(source).__name__}. "
            "Pass a Signal, BiosignalSample, mapping, or MNE-like reader object."
        )

    def from_legacy_sample(self, sample: BiosignalSample) -> Signal:
        """Convert the V3.3 sample type while preserving its provenance metadata."""
        self._validate_modality(sample.modality.value)
        raw_metadata = dict(sample.metadata)
        synthetic = bool(raw_metadata.pop("synthetic", False))
        generator = raw_metadata.pop("generator", None)
        source_dataset = raw_metadata.pop("source_dataset", None) or raw_metadata.pop(
            "dataset", None
        )
        dataset_version = raw_metadata.pop("dataset_version", None)
        fallback_reason = raw_metadata.pop("fallback_reason", None) or generator
        benchmark_eligible = raw_metadata.pop("benchmark_eligible", not synthetic)
        provenance = SignalProvenance(
            origin=DataOrigin.SYNTHETIC if synthetic else DataOrigin.REAL,
            source_dataset=source_dataset,
            dataset_version=dataset_version,
            adapter_name=self.adapter_name,
            fallback_reason=fallback_reason if synthetic else None,
            details={"benchmark_eligible": benchmark_eligible, **raw_metadata},
        )
        channel_names = tuple(
            raw_metadata.pop("channel_names", ())
            or tuple(f"{self.modality.upper()}-{index + 1}" for index in range(sample.n_channels))
        )
        return Signal(
            data=sample.signal,
            metadata=SignalMetadata(
                modality=self.modality,
                sampling_rate_hz=sample.sampling_rate_hz,
                channel_names=channel_names,
                subject_id=sample.subject_id,
                session_id=sample.session_id,
                recording_id=str(raw_metadata.pop("recording_id", "")) or None,
                task_id=sample.label_name,
                source_dataset=source_dataset,
                provenance=provenance,
                extra={"legacy_label": sample.label, **raw_metadata},
            ),
        )

    def from_mapping(self, source: Mapping[str, Any]) -> Signal:
        """Convert a transparent array-based source mapping to a canonical signal."""
        try:
            data = source["data"]
            sampling_rate_hz = source["sampling_rate_hz"]
        except KeyError as error:
            raise ValueError("Array source mapping requires data and sampling_rate_hz") from error
        data_array = np.asarray(data)
        channel_names = tuple(
            source.get("channel_names")
            or tuple(f"{self.modality.upper()}-{index + 1}" for index in range(data_array.shape[0]))
        )
        raw_origin = source.get("origin", DataOrigin.UNKNOWN)
        origin = (
            raw_origin
            if isinstance(raw_origin, DataOrigin)
            else DataOrigin(str(raw_origin).lower())
        )
        provenance = SignalProvenance(
            origin=origin,
            source_dataset=source.get("source_dataset"),
            dataset_version=source.get("dataset_version"),
            source_uri=source.get("source_uri"),
            license_id=source.get("license_id"),
            adapter_name=self.adapter_name,
            fallback_reason=source.get("fallback_reason"),
            details=source.get("provenance_details", {}),
        )
        return Signal(
            data=data_array,
            metadata=SignalMetadata(
                modality=self.modality,
                sampling_rate_hz=float(sampling_rate_hz),
                channel_names=channel_names,
                units=source.get("units", "unknown"),
                subject_id=source.get("subject_id"),
                session_id=source.get("session_id"),
                recording_id=source.get("recording_id"),
                task_id=source.get("task_id"),
                acquisition_system=source.get("acquisition_system"),
                source_dataset=source.get("source_dataset"),
                window_id=source.get("window_id"),
                provenance=provenance,
                extra=source.get("extra", {}),
            ),
            timestamps_seconds=source.get("timestamps_seconds"),
            events=tuple(self._event_from_mapping(event) for event in source.get("events", ())),
            missing_mask=source.get("missing_mask"),
        )

    def from_mne_like(self, raw: Any) -> Signal:
        """Adapt an MNE-like Raw/Epochs object without importing MNE in V4 core."""
        info = raw.info
        data = raw.get_data()
        sampling_rate_hz = float(info["sfreq"])
        channel_names = tuple(getattr(raw, "ch_names", info.get("ch_names", ())))
        if not channel_names:
            channel_names = tuple(
                f"{self.modality.upper()}-{index + 1}" for index in range(data.shape[0])
            )
        annotations = getattr(raw, "annotations", None)
        events: tuple[SignalEvent, ...] = ()
        if annotations is not None:
            events = tuple(
                SignalEvent(
                    onset_seconds=float(onset),
                    duration_seconds=float(duration),
                    label=str(description),
                )
                for onset, duration, description in zip(
                    annotations.onset, annotations.duration, annotations.description, strict=False
                )
            )
        dig = info.get("dig") if hasattr(info, "get") else None
        return Signal(
            data=data,
            metadata=SignalMetadata(
                modality=self.modality,
                sampling_rate_hz=sampling_rate_hz,
                channel_names=channel_names,
                acquisition_system=str(info.get("description", "")) or None,
                provenance=SignalProvenance(
                    origin=DataOrigin.REAL,
                    adapter_name=self.adapter_name,
                    details={"reader": type(raw).__name__, "digitization_available": bool(dig)},
                ),
                extra={"electrode_locations_available": bool(dig)},
            ),
            events=events,
        )

    def _validate_modality(self, modality: str) -> None:
        if modality.strip().lower() != self.modality:
            raise ValueError(
                f"{self.adapter_name} handles {self.modality!r}, received {modality!r}"
            )

    @staticmethod
    def _event_from_mapping(event: Any) -> SignalEvent:
        if isinstance(event, SignalEvent):
            return event
        if not isinstance(event, Mapping):
            raise TypeError("Each event must be a SignalEvent or a mapping")
        return SignalEvent(
            onset_seconds=float(event["onset_seconds"]),
            duration_seconds=float(event.get("duration_seconds", 0.0)),
            label=event.get("label"),
            value=event.get("value"),
        )


class EMGAdapter(ArraySignalAdapter):
    """Canonical EMG edge adapter."""

    def __init__(self) -> None:
        super().__init__(modality="emg", adapter_name="EMGAdapter")


class EEGAdapter(ArraySignalAdapter):
    """Canonical EEG edge adapter with MNE/BIDS-compatible duck typing."""

    def __init__(self) -> None:
        super().__init__(modality="eeg", adapter_name="EEGAdapter")


class ECGAdapter(ArraySignalAdapter):
    """Canonical ECG edge adapter for WFDB-derived or array-backed waveforms."""

    def __init__(self) -> None:
        super().__init__(modality="ecg", adapter_name="ECGAdapter")


class ECoGAdapter(ArraySignalAdapter):
    """Experimental ECoG/iEEG adapter; no benchmark capability is implied."""

    def __init__(self) -> None:
        super().__init__(modality="ecog", adapter_name="ECoGAdapter")


class FNIRSAdapter(ArraySignalAdapter):
    """Legacy-compatible optional fNIRS adapter."""

    def __init__(self) -> None:
        super().__init__(modality="fnirs", adapter_name="FNIRSAdapter")
