"""Base classes and protocols for biosignal datasets.

Defines the canonical data structures used throughout BioSignal-FM:

- :class:`BiosignalSample` — a single windowed biosignal sample with metadata
- :class:`ModalityMetadata` — describes a dataset's modality properties
- :class:`BiosignalDataset` — Protocol every loader implements
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Modality

__all__ = [
    "BiosignalSample",
    "ModalityMetadata",
    "BiosignalDataset",
]


@dataclass(frozen=True)
class BiosignalSample:
    """A single windowed biosignal sample.

    Attributes
    ----------
    signal : np.ndarray
        The biosignal array of shape ``(n_channels, n_samples)``.
    modality : Modality
        The modality of this sample.
    sampling_rate_hz : int
        The sampling rate of the signal.
    subject_id : int
        The subject identifier (used for LOSO/LODO).
    session_id : int
        The session identifier within the subject.
    label : int | None
        The class label (for classification tasks). None if unlabeled.
    label_name : str | None
        Human-readable label name (e.g. "rest", "fist", "pinch").
    metadata : dict
        Free-form metadata (recording date, equipment, etc.).
    """

    signal: np.ndarray
    modality: Modality
    sampling_rate_hz: int
    subject_id: int
    session_id: int = 0
    label: int | None = None
    label_name: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal.ndim != 2:
            raise ValueError(
                f"signal must be 2D (n_channels, n_samples), got shape {self.signal.shape}"
            )

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.signal.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples (time dimension)."""
        return int(self.signal.shape[1])

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.sampling_rate_hz


@dataclass(frozen=True)
class ModalityMetadata:
    """Static metadata describing a dataset's modality.

    Attributes
    ----------
    modality : Modality
        The modality.
    sampling_rate_hz : int
        Native sampling rate of the raw dataset.
    n_channels : int
        Number of channels in the raw dataset.
    channel_names : tuple[str, ...]
        Names of the channels (e.g. lead names, electrode positions).
    label_names : tuple[str, ...]
        Names of the class labels.
    """

    modality: Modality
    sampling_rate_hz: int
    n_channels: int
    channel_names: tuple[str, ...]
    label_names: tuple[str, ...] = ()


@runtime_checkable
class BiosignalDataset(Protocol):
    """Protocol every biosignal dataset loader implements.

    Implementations must provide:

    - ``metadata`` : :class:`ModalityMetadata`
    - ``__len__`` : number of samples
    - ``__getitem__`` : return a :class:`BiosignalSample` by index
    - ``get_subject_ids`` : list of subject IDs (for LOSO)
    - ``get_session_ids`` : list of session IDs per subject
    - ``iter_by_subject`` : iterator yielding (subject_id, list[samples])
    """

    metadata: ModalityMetadata

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> BiosignalSample: ...

    def get_subject_ids(self) -> list[int]: ...

    def get_session_ids(self, subject_id: int) -> list[int]: ...

    def iter_by_subject(self) -> Iterator[tuple[int, list[BiosignalSample]]]: ...


def _hash_file(path: Path) -> str:
    """Compute SHA-256 of a file (used by loaders for cache keys)."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
