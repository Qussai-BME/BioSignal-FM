"""Opt-in real-data smoke tests for the PhysioNet MIT-BIH ECG path.

Set ``BIOSIGNAL_REAL_MITBIH_ROOT`` to a directory containing official WFDB
records. Raw waveforms and beat annotations remain external to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_mitbih_loader_and_adapter() -> None:
    root_value = os.environ.get("BIOSIGNAL_REAL_MITBIH_ROOT")
    if not root_value:
        pytest.skip("BIOSIGNAL_REAL_MITBIH_ROOT is not configured")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        pytest.skip(f"Configured MIT-BIH root does not exist: {root}")

    from biosignal_fm.data import MITBIHLoader
    from biosignal_fm.modalities import default_registry

    loader = MITBIHLoader(root_dir=root, n_records=1)
    samples = loader.samples
    assert samples
    assert loader.is_synthetic is False
    assert loader.metadata.sampling_rate_hz == 360
    assert loader.metadata.n_channels == 2
    assert loader.get_subject_ids() == [100]

    sample = samples[0]
    assert sample.signal.shape == (2, 720)
    assert sample.sampling_rate_hz == 360
    assert sample.label is not None
    assert sample.metadata["dataset_id"] == "physionet.mitdb.1.0.0"
    assert sample.metadata["dataset_version"] == "1.0.0"
    assert sample.metadata["license_id"] == "ODC-BY-1.0"
    assert sample.metadata["channel_names"] == ("MLII", "V5")
    assert set(sample.metadata["source_file_sha256"]) == {".hea", ".dat", ".atr"}

    adapter_factory = default_registry().get("ecg").adapter_factory
    assert adapter_factory is not None
    signal = adapter_factory().to_signal(sample)
    assert signal.metadata.modality == "ecg"
    assert signal.metadata.channel_names == ("MLII", "V5")
    assert signal.metadata.provenance.source_dataset == "physionet.mitdb.1.0.0"
    assert signal.metadata.provenance.dataset_version == "1.0.0"
    assert signal.metadata.provenance.license_id == "ODC-BY-1.0"
    assert signal.metadata.provenance.details["benchmark_eligible"] is True
