"""Configuration system for BioSignal-FM.

This module is the single source of truth for all tunable parameters.
All configurations are immutable (frozen dataclasses) to prevent silent
mutation during training. Configurations can be loaded from YAML files
and serialized back to disk for reproducibility.

Design Principles
-----------------
1. **Immutability:** All dataclasses are frozen. Mutation requires creating
   a new instance via ``dataclasses.replace``.
2. **YAML-first:** Human-readable configuration files; no Python code edits
   needed to run a new experiment.
3. **No magic numbers:** Every tunable parameter lives here, not scattered
   across modules.
4. **Round-trip safe:** ``from_yaml`` -> ``to_yaml`` produces byte-identical
   files (modulo YAML formatting choices).

Example
-------
>>> from biosignal_fm.config import ExperimentConfig, PreprocessingConfig
>>> from pathlib import Path
>>> cfg = ExperimentConfig(name="exp001", output_dir=Path("/tmp/exp001"))
>>> cfg.preprocessing.target_sampling_rate_hz
200
>>> _ = cfg.to_yaml(Path("/tmp/exp001/config.yaml"))
>>> cfg2 = ExperimentConfig.from_yaml(Path("/tmp/exp001/config.yaml"))
>>> cfg == cfg2
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import yaml

__all__ = [
    "Modality",
    "MODALITIES",
    "PreprocessingConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DeploymentConfig",
    "ExperimentConfig",
    "load_config",
]


class Modality(str, Enum):
    """Supported biosignal modalities.

    The string value is used as the canonical identifier in YAML configs,
    dataset metadata, and ONNX model tags.
    """

    EMG = "emg"
    ECG = "ecg"
    EEG = "eeg"
    ECOG = "ecog"
    FNIRS = "fnirs"

    @classmethod
    def from_str(cls, value: str) -> Modality:
        """Convert string to Modality, raising on unknown values.

        Raises
        ------
        ValueError
            If ``value`` is not one of the supported modalities.
        """
        try:
            return cls(value.lower())
        except ValueError as err:
            raise ValueError(
                f"Unknown modality {value!r}. Supported: {[m.value for m in cls]}"
            ) from err


# Convenience tuple of all modality string values (used by data loaders).
MODALITIES: tuple[str, ...] = tuple(m.value for m in Modality)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Modality-aware preprocessing parameters.

    The bandpass ranges are chosen to preserve physiological content of
    each modality (Biomedical lens):

    - **EMG:** 20–450 Hz (surface EMG surface content; below 20 Hz is motion
      artifact, above 450 Hz is aliased noise at 2 kHz).
    - **ECG:** 0.5–40 Hz (diagnostic bandwidth for QRS and basic ST analysis).
    - **EEG:** 0.5–45 Hz (covers delta/theta/alpha/beta and low gamma).
    - **ECoG/iEEG:** 1–95 Hz (compatible with the 200 Hz default target;
      it must be configured for a real dataset and research protocol).
    - **fNIRS:** 0.01–0.5 Hz (slow hemodynamic response, below 0.5 Hz to
      avoid cardiac oscillation contamination).
    """

    target_sampling_rate_hz: int = 200
    emg_bandpass: tuple[float, float] = (20.0, 450.0)
    ecg_bandpass: tuple[float, float] = (0.5, 40.0)
    eeg_bandpass: tuple[float, float] = (0.5, 45.0)
    ecog_bandpass: tuple[float, float] = (1.0, 95.0)
    fnirs_bandpass: tuple[float, float] = (0.01, 0.5)
    notch_freq_hz: float | None = 50.0
    notch_quality_factor: float = 30.0
    filter_order: int = 4
    window_length_seconds: float = 2.0
    window_overlap_seconds: float = 0.5

    def bandpass_for(self, modality: Modality | str) -> tuple[float, float]:
        """Return the bandpass range for a given modality."""
        mod = modality if isinstance(modality, Modality) else Modality.from_str(modality)
        return cast("tuple[float, float]", getattr(self, f"{mod.value}_bandpass"))


@dataclass(frozen=True)
class ModelConfig:
    """Foundation model architecture parameters.

    Defaults target a ``bsfm-base`` configuration: 12 transformer layers,
    d_model=512, 8 attention heads. A ``bsfm-small`` variant (6 layers,
    d_model=256, 4 heads) is supported by overriding these fields.
    """

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    patch_length: int = 32
    patch_stride: int = 16
    dropout: float = 0.1
    n_modalities: int = 5
    max_sequence_length: int = 1024
    layer_norm_eps: float = 1e-5
    activation: Literal["gelu", "relu", "swiglu"] = "gelu"
    use_flash_attention: bool = False  # off by default for CPU portability
    # Span masking (SSL)
    mask_ratio: float = 0.5
    mean_mask_span_length: int = 8
    # Contrastive (SSL)
    contrastive_temperature: float = 0.1
    contrastive_weight: float = 0.5
    reconstruction_weight: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    """SSL pretraining / fine-tuning hyperparameters."""

    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_steps: int = 100_000
    eval_every_steps: int = 1000
    save_every_steps: int = 5000
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    ema_decay: float = 0.999
    ema_use: bool = True
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    lr_scheduler_min_lr: float = 1e-6
    num_workers: int = 4
    pin_memory: bool = False  # CPU-only default
    seed: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation and statistical testing parameters."""

    protocol: Literal["loso", "lodo", "nested"] = "loso"
    alpha: float = 0.05
    n_bootstrap: int = 10_000
    bootstrap_method: Literal["bca", "percentile"] = "bca"
    correction_method: Literal["holm_sidak", "bonferroni", "benjamini_hochberg"] = "holm_sidak"
    effect_size: Literal["cohens_d", "hedges_g"] = "hedges_g"
    power_target: float = 0.8
    random_state: int = 42


@dataclass(frozen=True)
class DeploymentConfig:
    """Deployment / serving parameters."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:8501",)
    quantize: bool = True
    onnx_opset: int = 17
    onnx_numerical_atol: float = 1e-5
    model_registry_dir: str = "~/.cache/biosignal_fm/registry"
    max_request_size_mb: int = 50  # Enforce at the reverse proxy / ASGI server boundary.


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration.

    This is the single object passed around at runtime. It bundles all
    sub-configurations and includes the experiment name and output directory.
    """

    name: str
    output_dir: Path
    seed: int = 42
    description: str = ""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ExperimentConfig.name must not be empty")
        # Normalize output_dir to absolute Path
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (YAML/JSON compatible).

        ``Path`` objects are converted to strings so the dict is
        YAML/JSON-serializable.
        """

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            return obj

        return cast("dict[str, Any]", _serialize(asdict(self)))

    def to_yaml(self, path: Path | str) -> Path:
        """Write the configuration to a YAML file.

        Parameters
        ----------
        path : Path or str
            Destination file path. Parent directories are created.

        Returns
        -------
        Path
            The absolute path of the written file.
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Construct an ExperimentConfig from a plain dict."""
        data = dict(data)  # shallow copy
        # Reconstruct nested dataclasses
        if "preprocessing" in data and isinstance(data["preprocessing"], dict):
            pre_data = dict(data["preprocessing"])
            # YAML/JSON round-trip converts tuples to lists; restore them.
            for band_key in (
                "emg_bandpass",
                "ecg_bandpass",
                "eeg_bandpass",
                "ecog_bandpass",
                "fnirs_bandpass",
            ):
                if band_key in pre_data and isinstance(pre_data[band_key], list):
                    pre_data[band_key] = tuple(pre_data[band_key])
            data["preprocessing"] = PreprocessingConfig(**pre_data)
        if "model" in data and isinstance(data["model"], dict):
            # Convert list back to tuple for bandpass fields
            model_data = dict(data["model"])
            data["model"] = ModelConfig(**model_data)
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvaluationConfig(**data["evaluation"])
        if "deployment" in data and isinstance(data["deployment"], dict):
            dep_data = dict(data["deployment"])
            if "cors_origins" in dep_data and isinstance(dep_data["cors_origins"], list):
                dep_data["cors_origins"] = tuple(dep_data["cors_origins"])
            data["deployment"] = DeploymentConfig(**dep_data)
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load an ExperimentConfig from a YAML file.

        Parameters
        ----------
        path : Path or str
            Path to the YAML file.

        Returns
        -------
        ExperimentConfig
            The loaded configuration.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def replace(self, **kwargs: Any) -> ExperimentConfig:
        """Return a new ExperimentConfig with the given fields replaced.

        Thin wrapper around :func:`dataclasses.replace` for fluent usage.
        """
        return replace(self, **kwargs)


def load_config(path: Path | str) -> ExperimentConfig:
    """Convenience function: alias for ``ExperimentConfig.from_yaml``."""
    return ExperimentConfig.from_yaml(path)
