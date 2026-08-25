"""BioSignal-FM: a modular multimodal biosignal research platform.

BioSignal-FM provides contracts, adapters, processing components, model
integrations, and evaluation utilities for reproducible biosignal research.
EMG, EEG, and ECG are V4 core modalities; ECoG/iEEG is experimental and fNIRS
is a legacy-compatible optional extension. The platform supports future
foundation-model research but does not itself claim a scientifically validated
foundation model without supporting evidence.

Modules
-------
- :mod:`biosignal_fm.config` — Frozen dataclass configuration.
- :mod:`biosignal_fm.reproducibility` — Seeds and RunManifest.
- :mod:`biosignal_fm.core` — Library-independent signal contracts.
- :mod:`biosignal_fm.modalities` — Explicit modality registry and edge adapters.
- :mod:`biosignal_fm.data` — Modality-specific dataset loaders.
- :mod:`biosignal_fm.preprocessing` — Filtering, normalization, patching.
- :mod:`biosignal_fm.models` — Foundation model, SSL heads, task heads.
- :mod:`biosignal_fm.training` — SSL pretraining and fine-tuning.
- :mod:`biosignal_fm.evaluation` — Cross-validation and statistics.
- :mod:`biosignal_fm.deployment` — ONNX export, FastAPI, realtime.
- :mod:`biosignal_fm.tracking` — MLflow + local JSON logging.
- :mod:`biosignal_fm.ui` — Streamlit dashboard.
- :mod:`biosignal_fm.services` — Core-composing application services.
- :mod:`biosignal_fm.cli` — Typer CLI.

Example
-------
>>> import biosignal_fm as bsfm
>>> bsfm.__version__
'4.0.1'
>>> from biosignal_fm.reproducibility import set_global_seed
>>> set_global_seed(42)
"""

from __future__ import annotations

__version__ = "4.0.1"
__author__ = "Qussai Adlbi"
__email__ = "qussai.adlbi@proton.me"
__license__ = "Apache-2.0"

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core contracts
    "DataOrigin",
    "Signal",
    "SignalBatch",
    "SignalEvent",
    "SignalMetadata",
    "SignalProvenance",
    # Modality registry
    "ModalityPlugin",
    "ModalityRegistry",
    "ModalityStatus",
    "default_registry",
    # Config
    "ExperimentConfig",
    "ModelConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DeploymentConfig",
    "MODALITIES",
    "Modality",
    "load_config",
    # Reproducibility
    "set_global_seed",
    "RunManifest",
    # Application services
    "Representation",
    "ResearchPipeline",
    # Data
    "BiosignalSample",
    "ModalityMetadata",
    "SyntheticBiosignalDataset",
    "make_synthetic_sample",
    "NinaProDB5Loader",
    "MITBIHLoader",
    "EEGMMIDLoader",
    "FnirsLoader",
    # Preprocessing
    "PreprocessingPipeline",
    "ModalityFilterBank",
    "Resampler",
    "SubjectAwareNormalizer",
    "ChannelWiseNormalizer",
    "Patcher",
    "Windower",
    # Models
    "FoundationModel",
    "DistilledFoundationModel",
    "distillation_loss",
    "PatchEmbedding",
    "ModalityToken",
    "SwiGLU",
    "SpanMaskedReconstructionHead",
    "ContrastiveHead",
    "JEPAHead",
    "jepa_loss",
    "sample_target_spans",
    "span_mask",
    "LinearProbe",
    "ClassificationHead",
    "SequenceLabelingHead",
    # Training
    "SSLPretrainer",
    "FineTuner",
    # Evaluation
    "LeaveOneSubjectOutCV",
    "LeaveOneDatasetOutCV",
    "cohens_d",
    "hedges_g",
    "bca_bootstrap_ci",
    "friedman_nemenyi_test",
    "wilcoxon_holm_sidak",
    "holm_sidak_correction",
    "confusion_matrix",
    "MixedEffectsAnalyzer",
    # Deployment
    "OnnxExporter",
    "RealtimeInference",
    "ModelRegistry",
    "create_app",
    # Tracking
    "BaseTracker",
    "LocalTracker",
    "MLflowTracker",
]


# Lazy imports — kept lightweight so `import biosignal_fm` is fast.
def __getattr__(name: str):
    # --- Core contracts ---
    if name in (
        "DataOrigin",
        "Signal",
        "SignalBatch",
        "SignalEvent",
        "SignalMetadata",
        "SignalProvenance",
    ):
        from . import core as _core

        return getattr(_core, name)
    # --- Modality registry ---
    if name in ("ModalityPlugin", "ModalityRegistry", "ModalityStatus", "default_registry"):
        from . import modalities as _modalities

        return getattr(_modalities, name)
    # --- Application services ---
    if name in ("Representation", "ResearchPipeline"):
        from . import services as _services

        return getattr(_services, name)
    # --- Config ---
    if name in (
        "ExperimentConfig",
        "ModelConfig",
        "PreprocessingConfig",
        "TrainingConfig",
        "EvaluationConfig",
        "DeploymentConfig",
        "MODALITIES",
        "Modality",
        "load_config",
    ):
        from . import config as _cfg

        if name == "load_config":
            return _cfg.load_config
        return getattr(_cfg, name)
    # --- Reproducibility ---
    if name in ("set_global_seed", "RunManifest"):
        from .reproducibility import RunManifest, set_global_seed

        return {"set_global_seed": set_global_seed, "RunManifest": RunManifest}[name]
    # --- Data ---
    if name in ("BiosignalSample", "ModalityMetadata"):
        from .data.base import BiosignalSample, ModalityMetadata

        return {"BiosignalSample": BiosignalSample, "ModalityMetadata": ModalityMetadata}[name]
    if name in ("SyntheticBiosignalDataset", "make_synthetic_sample"):
        from .data.synthetic import SyntheticBiosignalDataset, make_synthetic_sample

        return {
            "SyntheticBiosignalDataset": SyntheticBiosignalDataset,
            "make_synthetic_sample": make_synthetic_sample,
        }[name]
    if name == "NinaProDB5Loader":
        from .data.ninapro import NinaProDB5Loader

        return NinaProDB5Loader
    if name == "MITBIHLoader":
        from .data.mitbih import MITBIHLoader

        return MITBIHLoader
    if name == "EEGMMIDLoader":
        from .data.eegmmid import EEGMMIDLoader

        return EEGMMIDLoader
    if name == "FnirsLoader":
        from .data.fnirs import FnirsLoader

        return FnirsLoader
    # --- Preprocessing ---
    if name in ("PreprocessingPipeline",):
        from .preprocessing.pipeline import PreprocessingPipeline

        return PreprocessingPipeline
    if name in ("ModalityFilterBank",):
        from .preprocessing.filters import ModalityFilterBank

        return ModalityFilterBank
    if name in ("Resampler",):
        from .preprocessing.resampler import Resampler

        return Resampler
    if name in ("SubjectAwareNormalizer", "ChannelWiseNormalizer"):
        from .preprocessing.normalizer import SubjectAwareNormalizer

        # Both names point to the same class; ChannelWiseNormalizer is the
        # new, honest name. SubjectAwareNormalizer remains as a back-compat alias.
        return SubjectAwareNormalizer
    if name in ("Patcher", "Windower"):
        from .preprocessing.patcher import Patcher, Windower

        return {"Patcher": Patcher, "Windower": Windower}[name]
    # --- Models ---
    if name in ("FoundationModel", "PatchEmbedding", "ModalityToken", "SwiGLU"):
        from .models.foundation import FoundationModel, ModalityToken, PatchEmbedding, SwiGLU

        return {
            "FoundationModel": FoundationModel,
            "PatchEmbedding": PatchEmbedding,
            "ModalityToken": ModalityToken,
            "SwiGLU": SwiGLU,
        }[name]
    if name in ("DistilledFoundationModel", "distillation_loss"):
        from .models.distilled import DistilledFoundationModel, distillation_loss

        return {
            "DistilledFoundationModel": DistilledFoundationModel,
            "distillation_loss": distillation_loss,
        }[name]
    if name in ("JEPAHead", "jepa_loss", "sample_target_spans"):
        from .models.jepa_head import JEPAHead, jepa_loss, sample_target_spans

        return {
            "JEPAHead": JEPAHead,
            "jepa_loss": jepa_loss,
            "sample_target_spans": sample_target_spans,
        }[name]
    if name in ("SpanMaskedReconstructionHead", "ContrastiveHead", "span_mask"):
        from .models.ssl_heads import (
            ContrastiveHead,
            SpanMaskedReconstructionHead,
            span_mask,
        )

        return {
            "SpanMaskedReconstructionHead": SpanMaskedReconstructionHead,
            "ContrastiveHead": ContrastiveHead,
            "span_mask": span_mask,
        }[name]
    if name in ("LinearProbe", "ClassificationHead", "SequenceLabelingHead"):
        from .models.task_heads import (
            ClassificationHead,
            LinearProbe,
            SequenceLabelingHead,
        )

        return {
            "LinearProbe": LinearProbe,
            "ClassificationHead": ClassificationHead,
            "SequenceLabelingHead": SequenceLabelingHead,
        }[name]
    # --- Training ---
    if name == "SSLPretrainer":
        from .training.pretrainer import SSLPretrainer

        return SSLPretrainer
    if name == "FineTuner":
        from .training.finetuner import FineTuner

        return FineTuner
    # --- Evaluation ---
    if name in ("LeaveOneSubjectOutCV", "LeaveOneDatasetOutCV"):
        from .evaluation.cross_validation import (
            LeaveOneDatasetOutCV,
            LeaveOneSubjectOutCV,
        )

        return {
            "LeaveOneSubjectOutCV": LeaveOneSubjectOutCV,
            "LeaveOneDatasetOutCV": LeaveOneDatasetOutCV,
        }[name]
    if name in (
        "cohens_d",
        "hedges_g",
        "bca_bootstrap_ci",
        "friedman_nemenyi_test",
        "wilcoxon_holm_sidak",
        "holm_sidak_correction",
    ):
        from .evaluation import statistics as _stats

        return getattr(_stats, name)
    if name == "confusion_matrix":
        from .evaluation.metrics import confusion_matrix

        return confusion_matrix
    if name == "MixedEffectsAnalyzer":
        from .evaluation.mixed_effects import MixedEffectsAnalyzer

        return MixedEffectsAnalyzer
    # --- Deployment ---
    if name == "OnnxExporter":
        from .deployment.onnx_export import OnnxExporter

        return OnnxExporter
    if name == "RealtimeInference":
        from .deployment.realtime import RealtimeInference

        return RealtimeInference
    if name in ("ModelRegistry", "create_app"):
        from .deployment.serving import ModelRegistry, create_app

        return {"ModelRegistry": ModelRegistry, "create_app": create_app}[name]
    # --- Tracking ---
    if name == "BaseTracker":
        from .tracking.base import BaseTracker

        return BaseTracker
    if name == "LocalTracker":
        from .tracking.local_tracker import LocalTracker

        return LocalTracker
    if name == "MLflowTracker":
        from .tracking.mlflow_tracker import MLflowTracker

        return MLflowTracker
    raise AttributeError(f"module 'biosignal_fm' has no attribute {name!r}")
