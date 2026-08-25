"""Application services that compose the BioSignal-FM V4 research core."""

from .research import (
    FusionStrategy,
    PipelineResult,
    Representation,
    ResearchPipeline,
    SignalEncoder,
    SignalPreprocessor,
    TaskHead,
)

__all__ = [
    "FusionStrategy",
    "PipelineResult",
    "Representation",
    "ResearchPipeline",
    "SignalEncoder",
    "SignalPreprocessor",
    "TaskHead",
]
