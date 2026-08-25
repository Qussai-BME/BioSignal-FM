"""Task heads for downstream fine-tuning.

- :class:`LinearProbe` — Linear classifier on frozen CLS token.
- :class:`SequenceLabelingHead` — Per-patch token classification.
- :class:`ClassificationHead` — Full MLP classification head with dropout.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

__all__ = ["LinearProbe", "SequenceLabelingHead", "ClassificationHead"]


class LinearProbe(nn.Module):
    """Single linear layer on frozen CLS token.

    Used for zero-shot and few-shot evaluation. The encoder is frozen;
    only this linear layer is trained.

    Parameters
    ----------
    d_model : int
        CLS token dimension.
    n_classes : int
        Number of output classes.

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.models import LinearProbe
    >>> probe = LinearProbe(d_model=64, n_classes=8)
    >>> cls_token = torch.randn(4, 64)
    >>> logits = probe(cls_token)
    >>> logits.shape
    torch.Size([4, 8])
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Classify CLS tokens.

        Parameters
        ----------
        cls_token : torch.Tensor
            CLS token representations of shape ``(B, d_model)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, n_classes)``.
        """
        return cast(torch.Tensor, self.linear(cls_token))


class ClassificationHead(nn.Module):
    """Full MLP classification head with dropout.

    Used for fine-tuning. Two layers with GELU activation and dropout.

    Parameters
    ----------
    d_model : int
        CLS token dimension.
    n_classes : int
        Number of output classes.
    hidden_dim : int, optional
        Hidden layer dimension. Default ``d_model``.
    dropout : float, optional
        Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Classify CLS tokens."""
        return cast(torch.Tensor, self.net(cls_token))


class SequenceLabelingHead(nn.Module):
    """Per-patch token classification head.

    Used for event detection, beat segmentation, and other sequence-labeling
    tasks where each patch (rather than the whole window) gets a label.

    Parameters
    ----------
    d_model : int
        Patch token dimension.
    n_classes : int
        Number of output classes per patch.
    dropout : float, optional
        Dropout rate. Default 0.1.

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.models import SequenceLabelingHead
    >>> head = SequenceLabelingHead(d_model=64, n_classes=3)
    >>> patch_tokens = torch.randn(4, 23, 64)
    >>> logits = head(patch_tokens)
    >>> logits.shape
    torch.Size([4, 23, 3])
    """

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Classify each patch token.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            Patch token representations of shape ``(B, n_patches, d_model)``.

        Returns
        -------
        torch.Tensor
            Per-patch logits of shape ``(B, n_patches, n_classes)``.
        """
        return cast(torch.Tensor, self.classifier(patch_tokens))
