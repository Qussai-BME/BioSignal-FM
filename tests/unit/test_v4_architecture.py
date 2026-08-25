"""Architecture and scientific-integrity tests for the V4 migration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from biosignal_fm.config import PreprocessingConfig
from biosignal_fm.core import (
    DataOrigin,
    Signal,
    SignalBatch,
    SignalMetadata,
    SignalProvenance,
)
from biosignal_fm.data import make_synthetic_sample
from biosignal_fm.modalities import default_registry
from biosignal_fm.reproducibility import RunManifest
from biosignal_fm.services import Representation, ResearchPipeline


class _Encoder:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def encode(self, signal: Signal) -> Representation:
        self.trace.append(f"encode:{signal.metadata.modality}")
        return Representation(
            values=np.array([signal.data.mean(), signal.data.std()]),
            modality=signal.metadata.modality,
            source_is_synthetic=signal.is_synthetic,
            metadata={},
        )


class _Fusion:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def fuse(self, representations: tuple[Representation, ...]) -> Representation:
        self.trace.append("fuse")
        return Representation(
            values=np.concatenate([representation.values for representation in representations]),
            modality=None,
            source_is_synthetic=any(
                representation.source_is_synthetic for representation in representations
            ),
            metadata={"strategy": "test"},
        )


class _Head:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def predict(self, representation: Representation) -> dict[str, float]:
        self.trace.append("head")
        return {"score": float(representation.values.mean())}


def _signal(modality: str, *, synthetic: bool = False) -> Signal:
    return Signal(
        data=np.ones((2, 8), dtype=np.float32),
        metadata=SignalMetadata(
            modality=modality,
            sampling_rate_hz=100.0,
            channel_names=("C1", "C2"),
            provenance=SignalProvenance(
                origin=DataOrigin.SYNTHETIC if synthetic else DataOrigin.REAL,
                fallback_reason="test fixture" if synthetic else None,
            ),
        ),
    )


def test_signal_contract_rejects_channel_mismatch_and_protects_data() -> None:
    with pytest.raises(ValueError, match="channel count"):
        Signal(
            data=np.ones((1, 4)),
            metadata=SignalMetadata(
                modality="emg",
                sampling_rate_hz=100.0,
                channel_names=("C1", "C2"),
            ),
        )

    signal = _signal("emg")
    assert signal.data.flags.writeable is False
    with pytest.raises(ValueError):
        signal.data[0, 0] = 2.0


def test_registry_exposes_core_experimental_and_legacy_modalities() -> None:
    registry = default_registry()
    assert registry.identifiers() == ("emg", "eeg", "ecg", "ecog", "fnirs")
    assert registry.get("ecog").status.value == "experimental"
    assert registry.get("fnirs").status.value == "legacy_optional"
    assert registry.supports("ecg", "rhythm_analysis")


def test_registry_loads_preprocessing_only_on_explicit_request() -> None:
    plugin = default_registry().get("emg")
    assert plugin.preprocessing_factory is not None
    pipeline = plugin.preprocessing_factory("emg", PreprocessingConfig())
    assert pipeline.modality.value == "emg"


def test_legacy_synthetic_sample_becomes_explicitly_non_benchmark_signal() -> None:
    sample = make_synthetic_sample(modality="emg")
    signal = default_registry().get("emg").adapter_factory().to_signal(sample)  # type: ignore[union-attr]
    assert signal.is_synthetic
    assert signal.metadata.provenance.source_dataset == "synthetic://biosignal-fm"
    assert signal.metadata.provenance.details["benchmark_eligible"] is False


def test_multimodal_pipeline_fuses_before_task_head() -> None:
    trace: list[str] = []
    pipeline = ResearchPipeline(
        registry=default_registry(),
        encoder=_Encoder(trace),
        fusion=_Fusion(trace),
        task_head=_Head(trace),
    )
    result = pipeline.run(SignalBatch((_signal("emg", synthetic=True), _signal("eeg"))))
    assert trace == ["encode:emg", "encode:eeg", "fuse", "head"]
    assert result.contains_synthetic_data
    assert result.result_label == "synthetic_demo"


def test_multimodal_pipeline_rejects_late_task_head_fusion() -> None:
    pipeline = ResearchPipeline(
        registry=default_registry(),
        encoder=_Encoder([]),
        task_head=_Head([]),
    )
    with pytest.raises(ValueError, match="FusionStrategy"):
        pipeline.run(SignalBatch((_signal("emg"), _signal("ecg"))))


def test_run_manifest_records_protocol_and_dataset_provenance() -> None:
    manifest = RunManifest.create(
        name="architecture-test",
        dataset_provenance={"origin": "synthetic", "benchmark_eligible": False},
        protocol={"name": "smoke", "split": "none"},
    )
    serialized = manifest.to_dict()
    assert serialized["dataset_provenance"]["benchmark_eligible"] is False
    assert serialized["protocol"]["name"] == "smoke"
    assert "cpu_count" in serialized["runtime_context"]


def test_core_has_no_optional_modality_or_ui_dependency_imports() -> None:
    core_root = Path(__file__).resolve().parents[2] / "biosignal_fm" / "core"
    forbidden = ("import mne", "import wfdb", "import streamlit", "import fastapi", "import torch")
    for source in core_root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert not any(item in text for item in forbidden), source
