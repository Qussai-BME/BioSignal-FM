"""Modality registry and boundary adapters for BioSignal-FM V4."""

from .adapters import ECGAdapter, ECoGAdapter, EEGAdapter, EMGAdapter, FNIRSAdapter
from .registry import ModalityPlugin, ModalityRegistry, ModalityStatus, default_registry

__all__ = [
    "ECGAdapter",
    "EEGAdapter",
    "ECoGAdapter",
    "EMGAdapter",
    "FNIRSAdapter",
    "ModalityPlugin",
    "ModalityRegistry",
    "ModalityStatus",
    "default_registry",
]
