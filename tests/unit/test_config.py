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


class TestConfigurationValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"target_sampling_rate_hz": 0}, "target_sampling_rate_hz"),
            ({"emg_bandpass": (40.0, 20.0)}, "emg_bandpass"),
            ({"window_overlap_seconds": 2.0}, "window_overlap_seconds"),
        ],
    )
    def test_preprocessing_rejects_invalid_scientific_settings(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            PreprocessingConfig(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"d_model": 30, "n_heads": 8}, "divisible"),
            ({"patch_length": 16, "patch_stride": 32}, "patch_stride"),
            ({"dropout": 1.0}, "dropout"),
            ({"contrastive_weight": 0.0, "reconstruction_weight": 0.0}, "SSL loss"),
        ],
    )
    def test_model_rejects_invalid_architecture(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            ModelConfig(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"batch_size": 0}, "batch_size"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"ema_decay": 1.0}, "ema_decay"),
        ],
    )
    def test_training_rejects_invalid_hyperparameters(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            TrainingConfig(**kwargs)  # type: ignore[arg-type]

    def test_experiment_config_rejects_non_mapping_input(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            ExperimentConfig.from_dict([])  # type: ignore[arg-type]

    def test_deployment_rejects_out_of_range_port(self) -> None:
        from biosignal_fm.config import DeploymentConfig

        with pytest.raises(ValueError, match="port"):
            DeploymentConfig(port=0)

    def test_evaluation_rejects_invalid_alpha(self) -> None:
        from biosignal_fm.config import EvaluationConfig

        with pytest.raises(ValueError, match="alpha"):
            EvaluationConfig(alpha=1.0)
