"""Unit tests for biosignal_fm.models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from biosignal_fm.config import Modality
from biosignal_fm.models import (
    ClassificationHead,
    ContrastiveHead,
    FoundationModel,
    LinearProbe,
    ModalityToken,
    PatchEmbedding,
    SequenceLabelingHead,
    SpanMaskedReconstructionHead,
    span_mask,
)


def _n_channels_per_modality(n: int = 4) -> dict[str, int]:
    return {m.value: n for m in Modality}


class TestPatchEmbedding:
    def test_output_shape(self) -> None:
        pe = PatchEmbedding(n_channels=16, patch_length=32, stride=16, d_model=64)
        x = torch.randn(4, 16, 400)
        out = pe(x)
        # (400 - 32) / 16 + 1 = 24 patches... actually 23. Let me recompute.
        # (400 - 32) / 16 + 1 = 368/16 + 1 = 23 + 1 = 24? Actually 23.
        # Conv1d formula: floor((T - kernel + 2*padding) / stride) + 1
        # = floor((400 - 32 + 0) / 16) + 1 = floor(368/16) + 1 = 23 + 1 = 24
        # Hmm let me check: 368 / 16 = 23.0 exactly, +1 = 24
        assert out.shape == (4, 24, 64)


class TestModalityToken:
    def test_adds_embedding(self) -> None:
        mt = ModalityToken(n_modalities=4, d_model=32)
        x = torch.randn(4, 23, 32)
        mod_id = torch.tensor([0, 1, 2, 3])
        out = mt(x, mod_id)
        assert out.shape == x.shape


class TestFoundationModel:
    def test_forward_emg(self, small_model_config) -> None:
        n_ch = _n_channels_per_modality(4)
        model = FoundationModel(small_model_config, n_ch)
        x = torch.randn(2, 4, 64)
        cls, patches = model(x, "emg")
        assert cls.shape == (2, 32)
        # (64 - 16) / 8 + 1 = 48/8 + 1 = 6 + 1 = 7 patches
        assert patches.shape == (2, 7, 32)

    def test_forward_with_modality_enum(self, small_model_config) -> None:
        n_ch = _n_channels_per_modality(4)
        model = FoundationModel(small_model_config, n_ch)
        x = torch.randn(2, 4, 64)
        cls, patches = model(x, Modality.EEG)
        assert cls.shape == (2, 32)

    def test_forward_with_tensor_modality(self, small_model_config) -> None:
        n_ch = _n_channels_per_modality(4)
        model = FoundationModel(small_model_config, n_ch)
        x = torch.randn(3, 4, 64)
        mod_id = torch.tensor([0, 1, 2])
        cls, patches = model(x, mod_id)
        assert cls.shape == (3, 32)

    def test_save_load_roundtrip(self, small_model_config, tmp_path: Path) -> None:
        n_ch = _n_channels_per_modality(4)
        model = FoundationModel(small_model_config, n_ch)
        model.eval()  # disable dropout for deterministic comparison
        path = tmp_path / "model.pt"
        model.save(path)
        assert path.exists()

        loaded = FoundationModel.load(path)
        x = torch.randn(2, 4, 64)
        with torch.no_grad():
            cls1, _ = model(x, "emg")
            cls2, _ = loaded(x, "emg")
        torch.testing.assert_close(cls1, cls2, atol=1e-6, rtol=1e-5)


class TestSpanMask:
    def test_shape(self) -> None:
        mask = span_mask(batch_size=4, n_patches=23, mask_ratio=0.5, mean_span_length=8)
        assert mask.shape == (4, 23)
        assert mask.dtype == torch.bool

    def test_mask_ratio_approximate(self) -> None:
        mask = span_mask(batch_size=10, n_patches=100, mask_ratio=0.5, mean_span_length=8)
        ratio = mask.float().mean().item()
        # Should be approximately 0.5 (within +/- 0.2)
        assert abs(ratio - 0.5) < 0.2

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError):
            span_mask(batch_size=2, n_patches=10, mask_ratio=0.0)


class TestSSLHeads:
    def test_reconstruction_head(self, small_model_config) -> None:
        head = SpanMaskedReconstructionHead(
            d_model=small_model_config.d_model,
            patch_length=small_model_config.patch_length,
            n_channels=4,
        )
        patch_tokens = torch.randn(4, 7, 32)
        mask = span_mask(4, 7, mask_ratio=0.5)
        targets = torch.randn(4, 7, 4, 16)
        loss = head(patch_tokens, mask, targets)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_contrastive_head(self) -> None:
        head = ContrastiveHead(d_model=32, projection_dim=64, temperature=0.1)
        cls_a = torch.randn(4, 32)
        cls_b = torch.randn(4, 32)
        loss = head(cls_a, cls_b)
        assert loss.item() > 0
        assert loss.requires_grad


class TestTaskHeads:
    def test_linear_probe(self) -> None:
        probe = LinearProbe(d_model=32, n_classes=8)
        cls = torch.randn(4, 32)
        logits = probe(cls)
        assert logits.shape == (4, 8)

    def test_classification_head(self) -> None:
        head = ClassificationHead(d_model=32, n_classes=5)
        cls = torch.randn(4, 32)
        logits = head(cls)
        assert logits.shape == (4, 5)

    def test_sequence_labeling_head(self) -> None:
        head = SequenceLabelingHead(d_model=32, n_classes=3)
        patch_tokens = torch.randn(4, 7, 32)
        logits = head(patch_tokens)
        assert logits.shape == (4, 7, 3)
