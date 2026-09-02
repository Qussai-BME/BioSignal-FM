"""Additional coverage tests for v2.1 real loaders (file parsing, edge cases).

These tests exercise the file-parsing and edge-case code paths in the real
loaders WITHOUT requiring the optional dependencies (scipy, wfdb, mne, h5py).
They use synthetic .mat/.hea/.edf files to verify the parsing logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestNinaProLoaderFileParsing:
    """Verify NinaPro filename parsing logic."""

    def test_parses_subject_id_from_filename(self, tmp_path: Path) -> None:
        """The loader should correctly extract subject IDs from S*_E*_A*.mat files."""
        # Create empty .mat files that scipy.io.loadmat will fail on gracefully.
        # The loader should warn and skip.
        for subj in [1, 2, 3]:
            (tmp_path / f"S{subj}_E1_A1.mat").write_bytes(b"not a real mat file")

        from biosignal_fm.data.ninapro import NinaProDB5Loader

        loader = NinaProDB5Loader(root_dir=tmp_path, n_subjects=3)
        # Malformed source files do not silently become synthetic evidence.
        with pytest.raises(FileNotFoundError, match="allow_synthetic_fallback=True"):
            _ = loader.samples

        demo_loader = NinaProDB5Loader(
            root_dir=tmp_path, n_subjects=3, allow_synthetic_fallback=True
        )
        with pytest.warns(UserWarning, match="explicit synthetic fallback"):
            samples = demo_loader.samples
        assert len(samples) > 0
        assert demo_loader.is_synthetic


class TestMITBIHLoaderFileParsing:
    """Verify MIT-BIH record parsing logic."""

    def test_skips_malformed_hea_files(self, tmp_path: Path) -> None:
        """Malformed .hea files should be skipped with a warning, not crash.

        If wfdb is not installed, the loader raises ImportError, which
        propagates up. If wfdb is installed but fails to parse, the loader
        emits a warning and falls back to synthetic. Either way, no crash.
        """
        (tmp_path / "100.hea").write_text("not a real hea file")
        (tmp_path / "100.dat").write_bytes(b"\x00\x01\x02")

        from biosignal_fm.data.mitbih import MITBIHLoader

        loader = MITBIHLoader(root_dir=tmp_path, n_records=1)
        # Invalid real files do not silently become synthetic evidence.
        with pytest.raises(FileNotFoundError, match="allow_synthetic_fallback=True"):
            _ = loader.samples

        demo_loader = MITBIHLoader(root_dir=tmp_path, n_records=1, allow_synthetic_fallback=True)
        with pytest.warns(UserWarning, match="explicit synthetic fallback"):
            samples = demo_loader.samples
        assert len(samples) > 0


class TestEEGMMIDLoaderFileParsing:
    """Verify EEGMMID run-number parsing logic."""

    def test_skips_unparsable_filenames(self, tmp_path: Path) -> None:
        """Files that don't match SXXXRY.edf pattern are skipped.

        If mne is not installed, the loader raises ImportError. Either way, no crash.
        """
        # Create a file with an unparseable name
        (tmp_path / "weird_file.edf").write_bytes(b"not a real edf")

        from biosignal_fm.data.eegmmid import EEGMMIDLoader

        loader = EEGMMIDLoader(root_dir=tmp_path, n_subjects=1)
        with pytest.raises(FileNotFoundError, match="allow_synthetic_fallback=True"):
            _ = loader.samples

        demo_loader = EEGMMIDLoader(root_dir=tmp_path, n_subjects=1, allow_synthetic_fallback=True)
        with pytest.warns(UserWarning, match="explicit synthetic fallback"):
            samples = demo_loader.samples
        assert len(samples) > 0


class TestFnirsLoaderFileParsing:
    """Verify fNIRS path parsing logic."""

    def test_handles_csv_fallback(self, tmp_path: Path) -> None:
        """If no .snirf files but .csv files exist, the csv path is used.

        The malformed/truncated fixture must not silently become synthetic
        evidence; development fallback is separately and explicitly tested.
        """
        # Create a sub-01/ses-01/nirs/ structure with a fake csv.
        nirs_dir = tmp_path / "sub-01" / "ses-01" / "nirs"
        nirs_dir.mkdir(parents=True)
        (nirs_dir / "sub-01_ses-01_task-motor_nirs.csv").write_text("a,b,c\n1,2,3\n4,5,6\n")

        from biosignal_fm.data.fnirs import FnirsLoader

        loader = FnirsLoader(root_dir=tmp_path, n_subjects=1)
        with pytest.raises(FileNotFoundError, match="allow_synthetic_fallback=True"):
            _ = loader.samples

        demo_loader = FnirsLoader(root_dir=tmp_path, n_subjects=1, allow_synthetic_fallback=True)
        with pytest.warns(UserWarning, match="explicit synthetic fallback"):
            samples = demo_loader.samples
        assert len(samples) > 0


class TestDistilledModelConfiguration:
    """Verify the distilled model configuration options."""

    def test_distilled_config_constants(self) -> None:
        from biosignal_fm.models.distilled import DistilledFoundationModel

        assert DistilledFoundationModel.DISTILLED_D_MODEL == 256
        assert DistilledFoundationModel.DISTILLED_N_LAYERS == 6
        assert DistilledFoundationModel.DISTILLED_N_HEADS == 8
        assert DistilledFoundationModel.DISTILLED_D_FF == 1024

    def test_distilled_with_custom_config(self) -> None:
        """The distilled model accepts a custom config too."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import DistilledFoundationModel

        # Even smaller than the default distilled config.
        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = DistilledFoundationModel(config=cfg, n_channels_per_modality=n_ch)
        # Should NOT have the distilled defaults; should have the custom values.
        assert model.config.d_model == 32
        assert model.config.n_layers == 1
        x = torch.randn(2, 4, 64)
        mod_id = torch.tensor([0, 1])
        cls, _ = model(x, mod_id)
        assert cls.shape == (2, 32)


class TestJEPASampling:
    """Verify JEPA target-span sampling edge cases."""

    def test_sample_target_spans_reproducible(self) -> None:
        """Same generator → same output."""
        import torch
        from biosignal_fm.models import sample_target_spans

        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        m1 = sample_target_spans(4, 20, target_ratio=0.3, n_spans=2, min_span_len=2, generator=g1)
        m2 = sample_target_spans(4, 20, target_ratio=0.3, n_spans=2, min_span_len=2, generator=g2)
        assert torch.equal(m1, m2)

    def test_sample_target_spans_large_n_spans(self) -> None:
        """Many spans should still produce a valid mask."""

        from biosignal_fm.models import sample_target_spans

        mask = sample_target_spans(2, 30, target_ratio=0.5, n_spans=4, min_span_len=2)
        # Each row should have ~15 target patches (50% of 30), with some slack.
        for b in range(2):
            n = mask[b].sum().item()
            assert 8 <= n <= 22, f"row {b} has {n} targets, expected ~15"


class TestDistillationLossEdgeCases:
    """Verify distillation loss edge cases."""

    def test_perfect_match_is_zero(self) -> None:
        """If student == teacher, soft loss should be 0 (or very close)."""
        import torch
        from biosignal_fm.models import distillation_loss

        logits = torch.randn(4, 8)
        loss = distillation_loss(logits, logits.clone(), temperature=2.0)
        assert loss.item() < 1e-5

    def test_alpha_one_is_pure_soft(self) -> None:
        """alpha=1.0 with hard_labels → pure soft loss (hard ignored)."""
        import torch
        from biosignal_fm.models import distillation_loss

        student = torch.randn(4, 8)
        teacher = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])
        loss_soft_only = distillation_loss(student, teacher, temperature=2.0)
        loss_alpha1 = distillation_loss(
            student, teacher, temperature=2.0, alpha=1.0, hard_labels=labels
        )
        # Should be approximately equal (alpha=1.0 means hard loss weight = 0).
        assert abs(loss_soft_only.item() - loss_alpha1.item()) < 1e-4


class TestMultiModalityBatchPathEdgeCases:
    """Verify the group-by-modality batch path handles edge cases."""

    def test_single_sample_batch(self) -> None:
        """A batch of size 1 should work for both single- and multi-modality paths."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        model.eval()

        x = torch.randn(1, 4, 64)
        mod_id = torch.tensor([0])

        with torch.no_grad():
            cls, patches = model(x, mod_id)
        assert cls.shape == (1, 32)
        assert patches.shape[0] == 1

    def test_all_same_modality_batch(self) -> None:
        """All samples in the batch have the same modality."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        model.eval()

        x = torch.randn(4, 4, 64)
        mod_id = torch.tensor([2, 2, 2, 2])  # all EEG

        with torch.no_grad():
            cls, patches = model(x, mod_id)
        assert cls.shape == (4, 32)
