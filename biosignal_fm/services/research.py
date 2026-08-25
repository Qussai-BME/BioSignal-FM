"""Application service for the canonical V4 research pipeline.

The service establishes one explicit multimodal order:

``signal -> modality preprocessing -> encoder -> optional fusion -> task head``

It contains no UI, HTTP, MNE, WFDB, or model-framework imports. Concrete
models and preprocessors plug in through small protocols at the application
boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..core import Signal, SignalBatch
from ..modalities import ModalityRegistry

__all__ = [
    "Representation",
    "SignalPreprocessor",
    "SignalEncoder",
    "FusionStrategy",
    "TaskHead",
    "PipelineResult",
    "ResearchPipeline",
]


@dataclass(frozen=True)
class Representation:
    """Model-independent representation emitted by an encoder."""

    values: np.ndarray
    modality: str | None
    source_is_synthetic: bool
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, copy=True)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Representation.values must be a non-empty 1D vector")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


class SignalPreprocessor(Protocol):
    """Transforms one canonical signal while preserving its metadata."""

    def __call__(self, signal: Signal) -> Signal:
        """Return a transformed canonical signal."""


class SignalEncoder(Protocol):
    """Produces an encoder representation from a preprocessed signal."""

    def encode(self, signal: Signal) -> Representation:
        """Encode a single canonical signal."""


class FusionStrategy(Protocol):
    """Combines multiple representations before a task head is evaluated."""

    def fuse(self, representations: Sequence[Representation]) -> Representation:
        """Fuse one or more modality representations."""


class TaskHead(Protocol):
    """Performs a task on an already encoded or fused representation."""

    def predict(self, representation: Representation) -> Any:
        """Return a task output from one representation."""


@dataclass(frozen=True)
class PipelineResult:
    """Auditable result from the canonical research pipeline."""

    processed_signals: tuple[Signal, ...]
    representations: tuple[Representation, ...]
    task_input: Representation
    prediction: Any

    @property
    def contains_synthetic_data(self) -> bool:
        """Return whether any input signal was synthetic."""
        return any(signal.is_synthetic for signal in self.processed_signals)

    @property
    def result_label(self) -> str:
        """Return a reporting-safe evidence label for downstream clients."""
        return "synthetic_demo" if self.contains_synthetic_data else "real_data"


@dataclass
class ResearchPipeline:
    """Reusable orchestration service for uni- and multimodal task execution."""

    registry: ModalityRegistry
    encoder: SignalEncoder
    task_head: TaskHead
    preprocessors: dict[str, SignalPreprocessor] | None = None
    fusion: FusionStrategy | None = None

    def run(self, batch: SignalBatch) -> PipelineResult:
        """Execute the V4 pipeline in its canonical architectural order."""
        self.registry.validate_signals(batch.signals)
        processed_signals = tuple(self._preprocess(signal) for signal in batch.signals)
        representations = tuple(self.encoder.encode(signal) for signal in processed_signals)
        if len(representations) == 1:
            task_input = representations[0]
        elif self.fusion is not None:
            task_input = self.fusion.fuse(representations)
        else:
            raise ValueError(
                "A multimodal batch requires an explicit FusionStrategy before a TaskHead. "
                "Late task-head fusion is not the V4 default path."
            )
        return PipelineResult(
            processed_signals=processed_signals,
            representations=representations,
            task_input=task_input,
            prediction=self.task_head.predict(task_input),
        )

    def _preprocess(self, signal: Signal) -> Signal:
        preprocessor = (self.preprocessors or {}).get(signal.metadata.modality)
        return preprocessor(signal) if preprocessor is not None else signal
