"""Modality-aware preprocessing pipeline.

This package provides:

- :class:`ModalityFilterBank` — Per-modality Butterworth bandpass + notch filters
- :class:`Resampler` — Polyphase resampling with anti-aliasing
- :class:`ChannelWiseNormalizer` — Per-modality z-score normalization
  (the new, honest name; ``SubjectAwareNormalizer`` is kept as a
  backward-compat alias)
- :class:`Patcher` — Windowing and patch extraction
- :class:`PreprocessingPipeline` — End-to-end pipeline orchestrator
  (with ``to_dict`` / ``from_dict`` for deployment round-trips)

Design Principles
-----------------
1. **No leakage.** Normalization statistics are computed on the signals the
   caller passes to ``fit()``; it is the caller's responsibility to pass
   training-fold signals only (Biomedical lens).
2. **Numerical stability.** All IIR filters use second-order sections (SOS)
   to avoid numerical instability at high orders (Systems lens).
3. **Reproducibility.** Filter coefficients are deterministic given the same
   config; normalization statistics are stored as part of the model checkpoint
   via :meth:`PreprocessingPipeline.to_dict` / :meth:`from_dict`.
4. **No silent failures.** If a filter fails to converge, an exception is
   raised (Systems lens anti-pattern avoidance).
"""

from __future__ import annotations

from .filters import ModalityFilterBank
from .normalizer import ChannelWiseNormalizer, SubjectAwareNormalizer
from .patcher import Patcher, Windower
from .pipeline import PreprocessingPipeline
from .resampler import Resampler

__all__ = [
    "ModalityFilterBank",
    "ChannelWiseNormalizer",
    "SubjectAwareNormalizer",  # backward-compat alias
    "Resampler",
    "Patcher",
    "Windower",
    "PreprocessingPipeline",
]
