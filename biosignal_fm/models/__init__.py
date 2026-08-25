"""Foundation model, SSL heads, and task heads.

This package contains the core neural network components of BioSignal-FM:

- :class:`PatchEmbedding` — Conv1d patch embedding
- :class:`ModalityToken` — Learnable per-modality embedding
- :class:`SwiGLU` — SwiGLU activation (LLaMA-style) used when ``activation="swiglu"``
- :class:`FoundationModel` — The core transformer encoder
- :class:`DistilledFoundationModel` — Small CPU-pretrainable variant (d_model=256, n_layers=6)
- :class:`SpanMaskedReconstructionHead` — SSL head for masked reconstruction
- :class:`ContrastiveHead` — SimCLR-style NT-Xent loss
- :class:`JEPAHead` — I-JEPA-style predictive latent head (novel contribution)
- :class:`LinearProbe` — Linear classifier on frozen CLS token
- :class:`SequenceLabelingHead` — Per-patch token classification

All modules are pure PyTorch and CPU-compatible. No HuggingFace dependency
at runtime (avoids version-pinning fragility).
"""

from __future__ import annotations

from .distilled import DistilledFoundationModel, distillation_loss
from .foundation import FoundationModel, ModalityToken, PatchEmbedding, SwiGLU
from .jepa_head import JEPAHead, jepa_loss, sample_target_spans
from .ssl_heads import ContrastiveHead, SpanMaskedReconstructionHead, span_mask
from .task_heads import ClassificationHead, LinearProbe, SequenceLabelingHead

__all__ = [
    "PatchEmbedding",
    "ModalityToken",
    "SwiGLU",
    "FoundationModel",
    "DistilledFoundationModel",
    "distillation_loss",
    "SpanMaskedReconstructionHead",
    "ContrastiveHead",
    "span_mask",
    "JEPAHead",
    "jepa_loss",
    "sample_target_spans",
    "LinearProbe",
    "SequenceLabelingHead",
    "ClassificationHead",
]
