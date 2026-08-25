"""Unit tests for biosignal_fm.data."""

from __future__ import annotations

import numpy as np
import pytest
from biosignal_fm.config import Modality
from biosignal_fm.data import (
    BiosignalSample,
    EEGMMIDLoader,
    FnirsLoader,
    MITBIHLoader,
    NinaProDB5Loader,
    SyntheticBiosignalDataset,
    make_synthetic_sample,
)


class TestBiosignalSample:
    def test_creation(self) -> None:
        signal = np.random.randn(16, 400).astype(np.float32)
        s = BiosignalSample(
            signal=signal,
            modality=Modality.EMG,
            sampling_rate_hz=200,
            subject_id=0,
            session_id=0,
            label=1,
            label_name="fist",
        )
        assert s.n_channels == 16
        assert s.n_samples == 400
        assert abs(s.duration_seconds - 2.0) < 1e-6

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            BiosignalSample(
                signal=np.random.randn(400),
                modality=Modality.EMG,
                sampling_rate_hz=200,
                subject_id=0,
            )


class TestSyntheticBiosignalDataset:
    def test_emg_default(self) -> None:
        ds = SyntheticBiosignalDataset(
            modality="emg", n_subjects=3, n_sessions_per_subject=2, n_samples_per_class=2
        )
        assert len(ds) > 0
        assert len(ds.get_subject_ids()) == 3
        s = ds[0]
        assert s.signal.ndim == 2
        assert s.signal.shape[0] == 16  # default EMG channels
        assert s.modality == Modality.EMG

    def test_deterministic(self) -> None:
        ds1 = SyntheticBiosignalDataset(modality="ecg", n_subjects=2, n_samples_per_class=1)
        ds2 = SyntheticBiosignalDataset(modality="ecg", n_subjects=2, n_samples_per_class=1)
        np.testing.assert_array_equal(ds1[0].signal, ds2[0].signal)

    def test_all_modalities(self) -> None:
        for mod in ["emg", "ecg", "eeg", "fnirs"]:
            ds = SyntheticBiosignalDataset(modality=mod, n_subjects=1, n_samples_per_class=1)
            assert len(ds) > 0
            assert ds[0].modality.value == mod

    def test_invalid_n_classes(self) -> None:
        with pytest.raises(ValueError):
            SyntheticBiosignalDataset(modality="emg", n_classes=99)

    def test_iter_by_subject(self) -> None:
        ds = SyntheticBiosignalDataset(modality="emg", n_subjects=2, n_samples_per_class=1)
        for subj, samples in ds.iter_by_subject():
            assert all(s.subject_id == subj for s in samples)


class TestMakeSyntheticSample:
    def test_default(self) -> None:
        s = make_synthetic_sample()
        assert s.signal.ndim == 2
        assert s.signal.dtype == np.float32

    def test_custom_modality(self) -> None:
        s = make_synthetic_sample(modality="eeg")
        assert s.modality == Modality.EEG


class TestRealLoaders:
    """All real loaders should gracefully fall back to synthetic data when
    the raw datasets are not present."""

    def test_ninapro(self) -> None:
        loader = NinaProDB5Loader(root_dir=None, n_subjects=3)
        assert len(loader) > 0
        assert loader.is_synthetic
        assert loader.metadata.modality == Modality.EMG

    def test_mitbih(self) -> None:
        loader = MITBIHLoader(root_dir=None, n_records=3)
        assert len(loader) > 0
        assert loader.is_synthetic
        assert loader.metadata.modality == Modality.ECG

    def test_eegmmid(self) -> None:
        loader = EEGMMIDLoader(root_dir=None, n_subjects=3)
        assert len(loader) > 0
        assert loader.is_synthetic
        assert loader.metadata.modality == Modality.EEG

    def test_fnirs(self) -> None:
        loader = FnirsLoader(root_dir=None, n_subjects=3)
        assert len(loader) > 0
        assert loader.is_synthetic
        assert loader.metadata.modality == Modality.FNIRS

    def test_get_session_ids(self) -> None:
        loader = NinaProDB5Loader(root_dir=None, n_subjects=1)
        sessions = loader.get_session_ids(loader.get_subject_ids()[0])
        # Synthetic fallback produces 3 sessions per subject
        assert len(sessions) >= 1
