"""Opt-in real-data smoke tests for the licensed NinaPro DB5 EMG path.

Set ``BIOSIGNAL_REAL_NINAPRO_ROOT`` to an extracted subject/archive directory.
Raw dataset files remain external to the repository and this test is skipped in
ordinary CI environments.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_ninapro_db5_loader_and_adapter() -> None:
    root_value = os.environ.get("BIOSIGNAL_REAL_NINAPRO_ROOT")
    if not root_value:
        pytest.skip("BIOSIGNAL_REAL_NINAPRO_ROOT is not configured")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        pytest.skip(f"Configured NinaPro root does not exist: {root}")

    from biosignal_fm.data import NinaProDB5Loader
    from biosignal_fm.modalities import default_registry

    loader = NinaProDB5Loader(
        root_dir=root,
        n_subjects=1,
        window_length_seconds=2.0,
        window_overlap_seconds=0.5,
    )
    samples = loader.samples
    assert samples
    assert loader.is_synthetic is False
    assert loader.metadata.sampling_rate_hz == 200
    assert loader.metadata.n_channels == 16
    assert loader.get_subject_ids() == [1]
    assert {sample.session_id for sample in samples} == {1, 2, 3}

    sample = samples[0]
    assert sample.signal.shape == (16, 400)
    assert sample.sampling_rate_hz == 200
    assert sample.label is not None and sample.label > 0
    assert sample.metadata["dataset_id"] == "zenodo.1000116"
    assert sample.metadata["dataset_version"] == "v1"
    assert sample.metadata["license_id"] == "CC-BY-ND-4.0"
    assert len(sample.metadata["source_file_sha256"]) == 64

    adapter_factory = default_registry().get("emg").adapter_factory
    assert adapter_factory is not None
    signal = adapter_factory().to_signal(sample)
    assert signal.metadata.modality == "emg"
    assert signal.metadata.provenance.source_dataset == "zenodo.1000116"
    assert signal.metadata.provenance.dataset_version == "v1"
    assert signal.metadata.provenance.license_id == "CC-BY-ND-4.0"
    assert signal.metadata.provenance.details["benchmark_eligible"] is True
