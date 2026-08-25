# API Reference

Auto-generated from docstrings via [mkdocstrings](https://mkdocstrings.github.io/).
Every example in every docstring below is verified by `pytest --doctest-modules`
in CI — if an example here doesn't match reality, the build fails before it
ships.

## `biosignal_fm.config`

Frozen dataclasses for every stage of the pipeline. `ExperimentConfig` is the
top-level container (matches `configs/exp.yaml`); the rest nest inside it.

::: biosignal_fm.config
    options:
      members:
        - Modality
        - ModelConfig
        - TrainingConfig
        - PreprocessingConfig
        - EvaluationConfig
        - DeploymentConfig
        - ExperimentConfig
        - load_config

## `biosignal_fm.models`

The transformer backbone, SSL heads (masked reconstruction, contrastive,
JEPA), and downstream task heads.

::: biosignal_fm.models
    options:
      members:
        - FoundationModel
        - DistilledFoundationModel
        - distillation_loss
        - SpanMaskedReconstructionHead
        - ContrastiveHead
        - JEPAHead
        - jepa_loss
        - LinearProbe
        - ClassificationHead
        - SequenceLabelingHead

## `biosignal_fm.training`

::: biosignal_fm.training
    options:
      members:
        - SSLPretrainer
        - FineTuner

## `biosignal_fm.evaluation`

Cross-validation splitters, the full statistical rigor suite, and
classification metrics.

::: biosignal_fm.evaluation
    options:
      members:
        - LeaveOneSubjectOutCV
        - LeaveOneDatasetOutCV
        - friedman_nemenyi_test
        - wilcoxon_holm_sidak
        - holm_sidak_correction
        - hedges_g
        - cohens_d
        - bca_bootstrap_ci
        - power_analysis_ttest
        - MixedEffectsAnalyzer
        - confusion_matrix
        - classification_report

## `biosignal_fm.preprocessing`

::: biosignal_fm.preprocessing
    options:
      members:
        - ModalityFilterBank
        - ChannelWiseNormalizer
        - Resampler
        - Patcher
        - Windower
        - PreprocessingPipeline

## `biosignal_fm.data`

::: biosignal_fm.data
    options:
      members:
        - BiosignalDataset
        - SyntheticBiosignalDataset
        - NinaProDB5Loader
        - MITBIHLoader
        - EEGMMIDLoader
        - FnirsLoader

## `biosignal_fm.deployment`

ONNX export, quantized real-time inference, and the FastAPI REST +
WebSocket server. See [Deployment](deployment.md) for a task-oriented guide
with tested request/response examples.

::: biosignal_fm.deployment
    options:
      members:
        - OnnxExporter
        - RealtimeInference
        - ModelRegistry
        - create_app

## `biosignal_fm.baselines`

Non-transformer comparison methods used throughout the statistical rigor
suite: three classical time-domain-feature classifiers and three deep
learning baselines.

::: biosignal_fm.baselines
    options:
      members:
        - LDATDBaseline
        - SVMTDBaseline
        - RandomForestTDBaseline
        - CNN1DBaseline
        - EEGNetBaseline
        - ResNet1DBaseline
        - run_baseline_loso

## `biosignal_fm.reproducibility`

::: biosignal_fm.reproducibility
    options:
      members:
        - set_global_seed
        - compute_sha256
        - env_fingerprint
        - RunManifest

## `biosignal_fm.tracking`

::: biosignal_fm.tracking
    options:
      members:
        - LocalTracker
        - MLflowTracker
