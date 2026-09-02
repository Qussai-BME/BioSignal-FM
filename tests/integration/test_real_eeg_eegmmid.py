"""Opt-in real-data smoke tests for PhysioNet EEGMMID.

Set ``BIOSIGNAL_REAL_EEGMMID_ROOT`` to a directory containing official EDF
files. Raw data is external to the repository and this test is skipped in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_eegmmid_annotation_loader_and_adapter() -> None:
    root_value = os.environ.get("BIOSIGNAL_REAL_EEGMMID_ROOT")
    if not root_value:
        pytest.skip("BIOSIGNAL_REAL_EEGMMID_ROOT is not configured")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        pytest.skip(f"Configured EEGMMID root does not exist: {root}")

    from biosignal_fm.data import EEGMMIDLoader
    from biosignal_fm.modalities import default_registry

    loader = EEGMMIDLoader(root_dir=root, n_subjects=1, runs=(4,))
    samples = loader.samples
    assert samples
    assert loader.is_synthetic is False
    assert loader.metadata.sampling_rate_hz == 160
    assert loader.metadata.n_channels == 64
    assert loader.get_subject_ids() == [1]
    assert {sample.session_id for sample in samples} == {4}
    assert {sample.label for sample in samples} == {1, 2}

    sample = samples[0]
    assert sample.signal.shape == (64, 320)
    assert sample.metadata["event_description"] in {"T1", "T2"}
    assert sample.metadata["dataset_id"] == "physionet.eegmmidb.1.0.0"
    assert sample.metadata["dataset_version"] == "1.0.0"
    assert sample.metadata["license_id"] == "ODC-BY-1.0"
    assert len(sample.metadata["source_file_sha256"]) == 64

    adapter_factory = default_registry().get("eeg").adapter_factory
    assert adapter_factory is not None
    signal = adapter_factory().to_signal(sample)
    assert signal.metadata.modality == "eeg"
    assert signal.metadata.provenance.source_dataset == "physionet.eegmmidb.1.0.0"
    assert signal.metadata.provenance.dataset_version == "1.0.0"
    assert signal.metadata.provenance.license_id == "ODC-BY-1.0"
    assert signal.metadata.provenance.details["benchmark_eligible"] is True
