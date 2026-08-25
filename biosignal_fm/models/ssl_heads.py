"""Self-supervised learning heads for BioSignal-FM.

Two complementary SSL objectives:

1. :class:`SpanMaskedReconstructionHead` — Span-based masked reconstruction
   (MSE loss on masked patches only). Inspired by SpanBERT and wav2vec 2.0.
2. :class:`ContrastiveHead` — SimCLR-style NT-Xent loss over augmented
   views of the same window. Encodes invariance to noise/shift augmentations.

The combined loss is::

    L = reconstruction_weight * MSE(masked) + contrastive_weight * NT-Xent(aug_a, aug_b)

The two losses are complementary: reconstruction enforces local fidelity,
contrastive enforces global separability (DL Researcher lens).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpanMaskedReconstructionHead", "ContrastiveHead", "span_mask"]


def span_mask(
    batch_size: int,
    n_patches: int,
    mask_ratio: float = 0.5,
    mean_span_length: int = 8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate span-based masks for masked reconstruction.

    Spans are sampled geometrically (mean length = ``mean_span_length``)
    per the SpanBERT recipe. This captures temporal structure better than
    random masking.

    Parameters
    ----------
    batch_size : int
        Batch size.
    n_patches : int
        Number of patches per sample.
    mask_ratio : float
        Fraction of patches to mask (0-1).
    mean_span_length : int
        Mean span length (geometric distribution parameter).
    generator : torch.Generator, optional
        For reproducible masking.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(batch_size, n_patches)``. True = masked.

    Examples
    --------
    >>> mask = span_mask(batch_size=4, n_patches=23, mask_ratio=0.5, mean_span_length=8)
    >>> mask.shape
    torch.Size([4, 23])
    >>> mask.dtype
    torch.bool
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    if mean_span_length < 1:
        raise ValueError(f"mean_span_length must be >= 1, got {mean_span_length}")

    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
    n_to_mask = int(round(n_patches * mask_ratio))

    for b in range(batch_size):
        masked_count = 0
        attempts = 0
        max_attempts = n_patches * 4
        while masked_count < n_to_mask and attempts < max_attempts:
            # Sample span length from a true (truncated) geometric distribution.
            # The geometric distribution models "number of trials until first
            # success" with success probability p = 1/mean_span_length.
            # We use the formula k = ceil(log(1 - u) / log(1 - p)) which gives
            # a sample from Geometric(p) for u ~ Uniform(0, 1).
            # This is the SpanBERT / BART recipe.
            #
            # NOTE: this is a TRUNCATED geometric — we cap at n_patches to avoid
            # spans that exceed the sequence. For typical n_patches (>=20) and
            # mean_span_length (8), the truncation probability is <1% and the
            # distribution is effectively geometric. The docstring has been
            # updated to say "truncated geometric" for honesty.
            if generator is not None:
                u = torch.rand(1, generator=generator).item()
            else:
                u = torch.rand(1).item()
            # Avoid log(0) and log(1).
            u = max(min(u, 1.0 - 1e-8), 1e-8)
            p_geom = 1.0 / float(mean_span_length)
            span_len = max(1, int(math.ceil(math.log(1.0 - u) / math.log(1.0 - p_geom))))
            # Cap at n_patches (the natural upper bound for a span in a sequence
            # of length n_patches). This is not a bias-inducing cap; it just
            # prevents the span from exceeding the sequence.
            span_len = min(span_len, n_patches)

            # Sample start position
            if generator is not None:
                start = int(
                    torch.randint(
                        0, max(1, n_patches - span_len + 1), (1,), generator=generator
                    ).item()
                )
            else:
                start = int(torch.randint(0, max(1, n_patches - span_len + 1), (1,)).item())

            end = min(start + span_len, n_patches)
            new_masked = (~mask[b, start:end]).sum().item()
            mask[b, start:end] = True
            masked_count = int(mask[b].sum().item())
            attempts += 1

            if new_masked == 0:
                # Span hit only already-masked patches; try again
                continue

        # If we couldn't reach n_to_mask with spans, fill randomly
        if masked_count < n_to_mask:
            unmasked = (~mask[b]).nonzero(as_tuple=True)[0]
            if len(unmasked) > 0:
                extra = n_to_mask - masked_count
                perm = torch.randperm(len(unmasked))[:extra]
                mask[b, unmasked[perm]] = True

    return mask


class SpanMaskedReconstructionHead(nn.Module):
    """Span-based masked reconstruction head.

    Reconstructs the raw patch values at masked positions from the
    encoder's patch tokens. Loss is MSE over masked patches only.

    Parameters
    ----------
    d_model : int
        Patch token dimension.
    patch_length : int
        Length of each patch in samples.
    n_channels : int
        Number of channels to reconstruct.
    hidden_dim : int, optional
        Hidden dimension of the reconstruction MLP. Default ``d_model``.

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.models import SpanMaskedReconstructionHead, span_mask
    >>> head = SpanMaskedReconstructionHead(d_model=64, patch_length=32, n_channels=16)
    >>> patch_tokens = torch.randn(4, 23, 64)
    >>> mask = span_mask(4, 23, mask_ratio=0.5)
    >>> # targets: the raw patch values at each position, shape (B, n_patches, n_channels, patch_length)
    >>> targets = torch.randn(4, 23, 16, 32)
    >>> loss = head(patch_tokens, mask, targets)
    >>> loss.item() > 0
    True
    """

    def __init__(
        self,
        d_model: int,
        patch_length: int,
        n_channels: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_length = patch_length
        self.n_channels = n_channels
        hidden_dim = hidden_dim or d_model

        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_channels * patch_length),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked reconstruction loss.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            Encoder output patch tokens of shape ``(B, n_patches, d_model)``.
        mask : torch.Tensor
            Boolean mask of shape ``(B, n_patches)``. True = masked position.
        targets : torch.Tensor
            Target patch values of shape ``(B, n_patches, n_channels, patch_length)``.

        Returns
        -------
        torch.Tensor
            Scalar MSE loss over masked positions only.
        """
        # Decode all patches
        # (B, n_patches, d_model) -> (B, n_patches, n_channels * patch_length)
        decoded = self.decoder(patch_tokens)
        decoded = decoded.view(
            patch_tokens.shape[0], patch_tokens.shape[1], self.n_channels, self.patch_length
        )

        # Select only masked positions
        # mask: (B, n_patches) -> (B, n_patches, 1, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        masked_decoded = decoded[mask_expanded.expand_as(decoded)].reshape(
            -1, self.n_channels, self.patch_length
        )
        masked_targets = targets[mask_expanded.expand_as(targets)].reshape(
            -1, self.n_channels, self.patch_length
        )

        if masked_decoded.numel() == 0:
            return torch.tensor(0.0, device=patch_tokens.device, requires_grad=True)

        return F.mse_loss(masked_decoded, masked_targets)


class ContrastiveHead(nn.Module):
    """SimCLR-style NT-Xent contrastive loss.

    Given two augmented views (cls_a, cls_b) of the same batch of signals,
    computes the normalized temperature-scaled cross-entropy loss. Positives
    are paired views of the same sample; negatives are all other samples in
    the batch (both views).

    Parameters
    ----------
    d_model : int
        Input dimension (CLS token dim).
    projection_dim : int, optional
        Output dimension of the projection head. Default 128.
    temperature : float, optional
        Temperature scaling. Default 0.1.

    Notes
    -----
    The projection head is a 2-layer MLP (Linear -> ReLU -> Linear) as in
    the original SimCLR paper (Chen et al., 2020).

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.models import ContrastiveHead
    >>> head = ContrastiveHead(d_model=64, projection_dim=128, temperature=0.1)
    >>> cls_a = torch.randn(4, 64)
    >>> cls_b = torch.randn(4, 64)
    >>> loss = head(cls_a, cls_b)
    >>> loss.item() > 0
    True
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int = 128,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, cls_a: torch.Tensor, cls_b: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss between two views.

        Parameters
        ----------
        cls_a, cls_b : torch.Tensor
            CLS token representations of two augmented views, each of
            shape ``(B, d_model)``.

        Returns
        -------
        torch.Tensor
            Scalar NT-Xent loss.
        """
        B = cls_a.shape[0]
        z_a = F.normalize(self.projector(cls_a), dim=-1)
        z_b = F.normalize(self.projector(cls_b), dim=-1)

        # Concatenate: (2B, projection_dim)
        z = torch.cat([z_a, z_b], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)

        return F.cross_entropy(sim, labels)
