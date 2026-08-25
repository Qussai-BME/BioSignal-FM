"""Tests for v2.1 features: JEPA, DistilledFoundationModel, batch path."""

from __future__ import annotations

from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# JEPA head
# --------------------------------------------------------------------------- #


class TestJEPAHead:
    """Verify the JEPA predictive latent head."""

    def test_sample_target_spans_shape(self) -> None:
        import torch
        from biosignal_fm.models import sample_target_spans

        mask = sample_target_spans(
            batch_size=4, n_patches=20, target_ratio=0.3, n_spans=2, min_span_len=2
        )
        assert mask.shape == (4, 20)
        assert mask.dtype == torch.bool
        # Each row should have ~6 target patches (30% of 20)
        for b in range(4):
            n_target = mask[b].sum().item()
            assert 4 <= n_target <= 10, f"row {b} has {n_target} targets, expected ~6"

    def test_sample_target_spans_validation(self) -> None:
        from biosignal_fm.models import sample_target_spans

        with pytest.raises(ValueError, match="target_ratio"):
            sample_target_spans(2, 20, target_ratio=0.0)
        with pytest.raises(ValueError, match="target_ratio"):
            sample_target_spans(2, 20, target_ratio=1.0)
        with pytest.raises(ValueError, match="n_spans"):
            sample_target_spans(2, 20, n_spans=0)
        with pytest.raises(ValueError, match="min_span_len"):
            sample_target_spans(2, 20, min_span_len=0)

    def test_jepa_head_forward(self) -> None:
        import torch
        from biosignal_fm.models import JEPAHead

        head = JEPAHead(d_model=32, predictor_depth=2, predictor_n_heads=4)
        context = torch.randn(2, 10, 32)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 2:5] = True
        out = head(context, mask)
        # 2 samples × 3 target patches × d_model
        assert out.shape == (2, 3, 32)

    def test_jepa_loss_smooth_l1(self) -> None:
        import torch
        from biosignal_fm.models import jepa_loss

        pred = torch.randn(2, 5, 32)
        target = torch.randn(2, 5, 32)
        loss = jepa_loss(pred, target, loss_type="smooth_l1")
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_jepa_loss_mse(self) -> None:
        import torch
        from biosignal_fm.models import jepa_loss

        pred = torch.randn(2, 5, 32)
        target = torch.randn(2, 5, 32)
        loss = jepa_loss(pred, target, loss_type="mse")
        assert loss.item() >= 0

    def test_jepa_loss_shape_mismatch_raises(self) -> None:
        import torch
        from biosignal_fm.models import jepa_loss

        pred = torch.randn(2, 5, 32)
        target = torch.randn(2, 6, 32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            jepa_loss(pred, target)

    def test_jepa_loss_invalid_type_raises(self) -> None:
        import torch
        from biosignal_fm.models import jepa_loss

        pred = torch.randn(2, 5, 32)
        target = torch.randn(2, 5, 32)
        with pytest.raises(ValueError, match="Unknown loss_type"):
            jepa_loss(pred, target, loss_type="invalid")

    def test_jepa_end_to_end_with_foundation_model(self) -> None:
        """Verify JEPA works end-to-end with the FoundationModel."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel, JEPAHead, jepa_loss, sample_target_spans

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        head = JEPAHead(d_model=32, predictor_depth=1, predictor_n_heads=4)

        x = torch.randn(2, 4, 64)
        mod_id = torch.tensor([0, 1])

        # Encode full signal (target latents)
        with torch.no_grad():
            _, target_patches = model(x, mod_id)

        # Sample target mask
        n_patches = target_patches.shape[1]
        mask = sample_target_spans(2, n_patches, target_ratio=0.3, n_spans=1, min_span_len=2)

        # Encode context (in a real JEPA we'd mask at encoder input; here we
        # just re-encode the full signal as a smoke test)
        _, context_patches = model(x, mod_id)

        # Predict
        pred = head(context_patches, mask)

        # Gather target latents in the same order as pred
        target_list = []
        for b in range(2):
            target_list.append(target_patches[b][mask[b]])
        target_gathered = torch.stack(target_list, dim=0)

        loss = jepa_loss(pred, target_gathered.detach())
        assert loss.item() >= 0


# --------------------------------------------------------------------------- #
# DistilledFoundationModel
# --------------------------------------------------------------------------- #


class TestDistilledFoundationModel:
    """Verify the distilled (small) variant."""

    def test_default_config_smaller_than_full(self) -> None:
        from biosignal_fm.config import ModelConfig
        from biosignal_fm.models import DistilledFoundationModel

        full_cfg = ModelConfig()  # d_model=512, n_layers=12
        dist_cfg = DistilledFoundationModel._default_distilled_config()

        assert dist_cfg.d_model < full_cfg.d_model
        assert dist_cfg.n_layers < full_cfg.n_layers
        # d_ff should be smaller too
        assert dist_cfg.d_ff < full_cfg.d_ff

    def test_from_default_config(self) -> None:
        import torch
        from biosignal_fm.config import Modality
        from biosignal_fm.models import DistilledFoundationModel

        n_ch = {m.value: 4 for m in Modality}
        model = DistilledFoundationModel.from_default_config(n_channels_per_modality=n_ch)
        # Default distilled: d_model=256, n_layers=6
        assert model.config.d_model == 256
        assert model.config.n_layers == 6

        # Forward pass
        x = torch.randn(2, 4, 64)
        mod_id = torch.tensor([0, 1])
        cls, patches = model(x, mod_id)
        assert cls.shape == (2, 256)
        assert patches.shape[0] == 2

    def test_n_parameters(self) -> None:
        from biosignal_fm.config import Modality
        from biosignal_fm.models import DistilledFoundationModel

        n_ch = {m.value: 4 for m in Modality}
        model = DistilledFoundationModel.from_default_config(n_channels_per_modality=n_ch)
        n_params = model.n_parameters
        # Should be in the 5M-20M range
        assert 5_000_000 < n_params < 20_000_000, f"got {n_params} params"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        import torch
        from biosignal_fm.config import Modality
        from biosignal_fm.models import DistilledFoundationModel

        n_ch = {m.value: 4 for m in Modality}
        model = DistilledFoundationModel.from_default_config(n_channels_per_modality=n_ch)
        model.eval()
        ckpt = tmp_path / "distilled.pt"
        model.save(ckpt)
        loaded = DistilledFoundationModel.load(ckpt)
        x = torch.randn(2, 4, 64)
        mod_id = torch.tensor([0, 1])
        with torch.no_grad():
            cls1, _ = model(x, mod_id)
            cls2, _ = loaded(x, mod_id)
        torch.testing.assert_close(cls1, cls2, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# Distillation loss
# --------------------------------------------------------------------------- #


class TestDistillationLoss:
    """Verify Hinton-style distillation loss."""

    def test_soft_only(self) -> None:
        import torch
        from biosignal_fm.models import distillation_loss

        student = torch.randn(4, 8)
        teacher = torch.randn(4, 8)
        loss = distillation_loss(student, teacher, temperature=2.0)
        assert loss.item() >= 0

    def test_soft_plus_hard(self) -> None:
        import torch
        from biosignal_fm.models import distillation_loss

        student = torch.randn(4, 8)
        teacher = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])
        loss = distillation_loss(student, teacher, temperature=2.0, alpha=0.5, hard_labels=labels)
        assert loss.item() >= 0

    def test_alpha_zero_is_pure_hard(self) -> None:
        import torch
        from biosignal_fm.models import distillation_loss

        student = torch.randn(4, 8)
        teacher = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])
        loss = distillation_loss(student, teacher, temperature=2.0, alpha=0.0, hard_labels=labels)
        assert loss.item() >= 0

    def test_shape_mismatch_raises(self) -> None:
        import torch
        from biosignal_fm.models import distillation_loss

        with pytest.raises(ValueError, match="Shape mismatch"):
            distillation_loss(torch.randn(4, 8), torch.randn(4, 7))

    def test_invalid_temperature_raises(self) -> None:
        import torch
        from biosignal_fm.models import distillation_loss

        with pytest.raises(ValueError, match="temperature"):
            distillation_loss(torch.randn(4, 8), torch.randn(4, 8), temperature=0.0)

    def test_invalid_alpha_raises(self) -> None:
        import torch
        from biosignal_fm.models import distillation_loss

        with pytest.raises(ValueError, match="alpha"):
            distillation_loss(torch.randn(4, 8), torch.randn(4, 8), alpha=1.5)


# --------------------------------------------------------------------------- #
# Multi-modality batch path optimization (F4)
# --------------------------------------------------------------------------- #


class TestMultiModalityBatchPath:
    """Verify the group-by-modality batch path produces correct outputs."""

    def test_multi_modality_matches_single_modality(self) -> None:
        """The group-by-modality path must produce identical outputs to
        running each sample individually through its modality's patch embedding.
        """
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        model.eval()

        x = torch.randn(4, 4, 64)  # 4 samples, 4 channels, 64 samples
        # Preserve the legacy EMG/ECG/EEG/fNIRS check while deriving indices
        # from the V4 enum, where ECoG is an inserted experimental modality.
        modality_names = ["emg", "ecg", "eeg", "fnirs"]
        mod_id = torch.tensor(
            [list(Modality).index(Modality.from_str(name)) for name in modality_names]
        )

        with torch.no_grad():
            # Group-by-modality path
            cls_multi, patches_multi = model(x, mod_id)

            # Per-sample path: encode each sample individually with its modality
            cls_list = []
            patches_list = []
            for b in range(4):
                mod_str = modality_names[b]
                cls_s, patches_s = model(x[b : b + 1], mod_str)
                cls_list.append(cls_s)
                patches_list.append(patches_s)
            cls_single = torch.cat(cls_list, dim=0)
            patches_single = torch.cat(patches_list, dim=0)

        torch.testing.assert_close(cls_multi, cls_single, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(patches_multi, patches_single, atol=1e-6, rtol=1e-5)

    def test_multi_modality_preserves_order(self) -> None:
        """The output must be in the original batch order, not modality-grouped order."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        model.eval()

        # Mix modalities in non-sorted order: emg, eeg, emg, eeg
        x = torch.randn(4, 4, 64)
        mod_id = torch.tensor([0, 2, 0, 2])  # emg=0, eeg=2

        with torch.no_grad():
            cls_multi, _ = model(x, mod_id)

            # Per-sample in order
            cls_list = []
            for b in range(4):
                mod_str = ["emg", "eeg", "emg", "eeg"][b]
                cls_s, _ = model(x[b : b + 1], mod_str)
                cls_list.append(cls_s)
            cls_single = torch.cat(cls_list, dim=0)

        torch.testing.assert_close(cls_multi, cls_single, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# New public symbols are exported
# --------------------------------------------------------------------------- #


class TestNewSymbolsExported:
    """Verify v2.1 symbols are importable from biosignal_fm top-level."""

    def test_imports(self) -> None:
        import biosignal_fm as bsfm

        new_symbols = [
            "DistilledFoundationModel",
            "distillation_loss",
            "JEPAHead",
            "jepa_loss",
            "sample_target_spans",
        ]
        missing = [s for s in new_symbols if not hasattr(bsfm, s)]
        assert not missing, f"Missing new public symbols: {missing}"
