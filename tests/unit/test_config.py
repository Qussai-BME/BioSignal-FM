"""Unit tests for biosignal_fm.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from biosignal_fm.config import (
    MODALITIES,
    ExperimentConfig,
    Modality,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
    load_config,
)


class TestModality:
    def test_values(self) -> None:
        assert Modality.EMG.value == "emg"
        assert Modality.ECG.value == "ecg"
        assert Modality.EEG.value == "eeg"
        assert Modality.ECOG.value == "ecog"
        assert Modality.FNIRS.value == "fnirs"

    def test_from_str_valid(self) -> None:
        assert Modality.from_str("emg") == Modality.EMG
        assert Modality.from_str("ECG") == Modality.ECG  # case-insensitive
        assert Modality.from_str("Eeg") == Modality.EEG

    def test_from_str_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown modality"):
            Modality.from_str("xyz")

    def test_modalities_tuple(self) -> None:
        assert MODALITIES == ("emg", "ecg", "eeg", "ecog", "fnirs")
        assert len(MODALITIES) == 5


class TestPreprocessingConfig:
    def test_defaults(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.target_sampling_rate_hz == 200
        assert cfg.emg_bandpass == (20.0, 450.0)
        assert cfg.ecg_bandpass == (0.5, 40.0)
        assert cfg.eeg_bandpass == (0.5, 45.0)
        assert cfg.ecog_bandpass == (1.0, 95.0)
        assert cfg.fnirs_bandpass == (0.01, 0.5)
        assert cfg.filter_order == 4

    def test_bandpass_for(self) -> None:
        cfg = PreprocessingConfig()
        assert cfg.bandpass_for(Modality.EMG) == cfg.emg_bandpass
        assert cfg.bandpass_for("ecg") == cfg.ecg_bandpass
        assert cfg.bandpass_for("EEG") == cfg.eeg_bandpass
        assert cfg.bandpass_for("ecog") == cfg.ecog_bandpass

    def test_immutable(self) -> None:
        cfg = PreprocessingConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.target_sampling_rate_hz = 500  # type: ignore[misc]


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.d_model == 512
        assert cfg.n_heads == 8
        assert cfg.n_layers == 12
        assert cfg.patch_length == 32
        assert cfg.patch_stride == 16

    def test_immutable(self) -> None:
        cfg = ModelConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.d_model = 1024  # type: ignore[misc]


class TestExperimentConfig:
    def test_create(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="test", output_dir=tmp_path)
        assert cfg.name == "test"
        assert cfg.output_dir == tmp_path.resolve()
        assert cfg.seed == 42
        assert isinstance(cfg.preprocessing, PreprocessingConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_empty_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            ExperimentConfig(name="", output_dir=tmp_path)

    def test_yaml_roundtrip(self, tmp_path: Path) -> None:
        import dataclasses

        cfg = ExperimentConfig(name="test", output_dir=tmp_path)
        cfg = cfg.replace(
            preprocessing=dataclasses.replace(cfg.preprocessing, target_sampling_rate_hz=500)
        )
        yaml_path = cfg.to_yaml(tmp_path / "config.yaml")
        assert yaml_path.exists()

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.name == cfg.name
        assert loaded.preprocessing.target_sampling_rate_hz == 500
        assert loaded.model.d_model == cfg.model.d_model
        assert loaded.training.batch_size == cfg.training.batch_size

    def test_load_config_alias(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="test", output_dir=tmp_path)
        yaml_path = cfg.to_yaml(tmp_path / "c.yaml")
        loaded = load_config(yaml_path)
        assert loaded.name == "test"

    def test_replace(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="test", output_dir=tmp_path)
        cfg2 = cfg.replace(seed=99)
        assert cfg.seed == 42  # original unchanged
        assert cfg2.seed == 99
