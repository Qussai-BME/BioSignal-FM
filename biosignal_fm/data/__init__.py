"""Modality-specific dataset loaders for BioSignal-FM.

This package provides unified loaders for the four supported biosignal
modalities. Each loader implements the :class:`BiosignalDataset` protocol
and produces :class:`BiosignalSample` records.

Loaders are designed to:

- **Never mutate raw files.** All transformations are cached in NPZ format
  keyed by SHA-256 of the raw file.
- **Be subject-aware.** Each loader exposes :meth:`get_subject_ids` and
  :meth:`get_session_ids` for LOSO/LODO cross-validation.
- **Lazy-load.** Data is only loaded into memory when accessed.
- **Work without the raw datasets installed.** A synthetic generator is
  provided for development, testing, and CI.

Datasets
--------
- :class:`NinaProDB5Loader` — NinaPro DB5 sEMG (10 subjects, 16 channels, 2 kHz)
- :class:`MITBIHLoader` — PhysioNet MIT-BIH Arrhythmia (48 records, 2 leads, 360 Hz)
- :class:`EEGMMIDLoader` — PhysioNet EEG Motor Movement/Imagery (109 subjects, 64 ch, 160 Hz)
- :class:`FnirsLoader` — Brain-BIDS fNIRS (8-32 channels, 10 Hz)
- :class:`SyntheticBiosignalDataset` — Procedural generator for development & CI
"""

from __future__ import annotations

from .base import BiosignalDataset, BiosignalSample, ModalityMetadata
from .eegmmid import EEGMMIDLoader
from .fnirs import FnirsLoader
from .mitbih import MITBIHLoader
from .ninapro import NinaProDB5Loader
from .synthetic import SyntheticBiosignalDataset, make_synthetic_sample

__all__ = [
    "BiosignalDataset",
    "BiosignalSample",
    "ModalityMetadata",
    "SyntheticBiosignalDataset",
    "make_synthetic_sample",
    "NinaProDB5Loader",
    "MITBIHLoader",
    "EEGMMIDLoader",
    "FnirsLoader",
]
