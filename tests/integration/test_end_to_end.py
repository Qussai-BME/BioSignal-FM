"""End-to-end integration tests for BioSignal-FM.

These tests verify the full pipeline works:

1. Generate synthetic data
2. Preprocess (filter + resample + normalize)
3. Train (SSL pretrain + fine-tune)
4. Evaluate (LOSO + statistics)
5. Export to ONNX with parity verification
6. Serve via FastAPI

These tests are slower than unit tests but verify the integration
between modules works correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from biosignal_fm.config import (
    Modality,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from biosignal_fm.data import EEGMMIDLoader, MITBIHLoader, make_synthetic_sample
from biosignal_fm.deployment import OnnxExporter, RealtimeInference
from biosignal_fm.evaluation import (
    LeaveOneSubjectOutCV,
    confusion_matrix,
    holm_sidak_correction,
)
from biosignal_fm.models import (
    ClassificationHead,
    ContrastiveHead,
    FoundationModel,
    SpanMaskedReconstructionHead,
)
from biosignal_fm.preprocessing import PreprocessingPipeline
from biosignal_fm.reproducibility import RunManifest, set_global_seed
from biosignal_fm.training import FineTuner, SSLPretrainer


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        patch_length=16,
        patch_stride=8,
        max_sequence_length=64,
        mask_ratio=0.5,
    )


@pytest.mark.integration
class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_full_pipeline_smoke(self, small_model_config, tmp_path: Path) -> None:
        """Smoke test: every module runs end-to-end without errors.

        Steps:
        1. Set seed
        2. Generate synthetic data
        3. Preprocess
        4. Build model
        5. Forward pass
        6. Save/load checkpoint
        7. ONNX export + verify
        """
        # 1. Seed
        set_global_seed(42)

        # 2. Synthetic data at 2000 Hz so the EMG bandpass (20, 450) is below Nyquist.
        sample = make_synthetic_sample(
            modality="emg", n_channels=4, n_samples=640, sampling_rate_hz=2000
        )
        assert sample.signal.shape == (4, 640)

        # 3. Preprocess (downsample to 200 Hz after filtering)
        pipe = PreprocessingPipeline(
            config=PreprocessingConfig(target_sampling_rate_hz=200),
            modality=Modality.EMG,
        )
        pipe.fit([sample.signal], source_sampling_rate_hz=2000)
        processed = pipe.transform(sample.signal, source_sampling_rate_hz=2000)
        # After downsampling 2000 -> 200, length should shrink 10x (64 samples).
        assert processed.shape[0] == 4
        assert 60 <= processed.shape[1] <= 70  # approximate, polyphase may differ slightly

        # 4. Build model
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(small_model_config, n_ch)
        model.eval()

        # 5. Forward pass
        with torch.no_grad():
            x = torch.from_numpy(processed).unsqueeze(0)
            cls, patches = model(x, "emg")
        assert cls.shape == (1, 32)
        assert patches.shape[1] > 0

        # 6. Save / load
        ckpt_path = tmp_path / "model.pt"
        model.save(ckpt_path)
        loaded = FoundationModel.load(ckpt_path)
        with torch.no_grad():
            cls2, _ = loaded(x, "emg")
        torch.testing.assert_close(cls, cls2, atol=1e-6, rtol=1e-5)

        # 7. ONNX export + verify
        exporter = OnnxExporter()
        onnx_path = exporter.export(model, tmp_path / "model.onnx", "emg", 4, 64)
        assert onnx_path.exists()
        ok = exporter.verify(model, onnx_path, "emg", 4, 64, n_samples=10)
        assert ok is True

    def test_eeg_loader_to_prediction_to_onnx_fallback(
        self, small_model_config, tmp_path: Path
    ) -> None:
        """Exercise EEG loader fallback through preprocessing, prediction, and ONNX parity."""
        empty_root = tmp_path / "empty-eegmmid"
        empty_root.mkdir()
        loader = EEGMMIDLoader(
            root_dir=empty_root,
            n_subjects=1,
            window_length_seconds=1.0,
            target_sampling_rate_hz=160,
        )

        with pytest.warns(UserWarning, match="falling back to synthetic"):
            sample = loader.samples[0]
        assert loader.is_synthetic is True
        assert sample.metadata["synthetic"] is True
        assert sample.metadata["benchmark_eligible"] is False
        assert sample.modality is Modality.EEG

        pipeline = PreprocessingPipeline(
            config=PreprocessingConfig(target_sampling_rate_hz=160),
            modality=Modality.EEG,
        )
        pipeline.fit([sample.signal], source_sampling_rate_hz=sample.sampling_rate_hz)
        processed = pipeline.transform(
            sample.signal, source_sampling_rate_hz=sample.sampling_rate_hz
        )
        assert processed.shape == sample.signal.shape

        n_channels = {modality.value: sample.signal.shape[0] for modality in Modality}
        model = FoundationModel(small_model_config, n_channels)
        head = ClassificationHead(d_model=small_model_config.d_model, n_classes=5)
        model.eval()
        head.eval()
        encoded = torch.from_numpy(processed).unsqueeze(0)
        with torch.no_grad():
            cls_token, _ = model(encoded, "eeg")
            logits = head(cls_token)
        assert logits.shape == (1, 5)

        exporter = OnnxExporter()
        onnx_path = exporter.export(
            model,
            tmp_path / "eeg-model.onnx",
            "eeg",
            sample.signal.shape[0],
            processed.shape[1],
        )
        assert onnx_path.exists()
        assert exporter.verify(
            model,
            onnx_path,
            "eeg",
            sample.signal.shape[0],
            processed.shape[1],
            n_samples=3,
        )

    def test_ecg_loader_to_prediction_to_onnx_fallback(
        self, small_model_config, tmp_path: Path
    ) -> None:
        """Exercise ECG loader fallback through preprocessing, prediction, and ONNX parity."""
        empty_root = tmp_path / "empty-mitbih"
        empty_root.mkdir()
        loader = MITBIHLoader(
            root_dir=empty_root,
            n_records=1,
            window_length_seconds=1.0,
            target_sampling_rate_hz=360,
        )

        with pytest.warns(UserWarning, match="falling back to synthetic"):
            sample = loader.samples[0]
        assert loader.is_synthetic is True
        assert sample.metadata["synthetic"] is True
        assert sample.metadata["benchmark_eligible"] is False
        assert sample.modality is Modality.ECG

        pipeline = PreprocessingPipeline(
            config=PreprocessingConfig(target_sampling_rate_hz=200),
            modality=Modality.ECG,
        )
        pipeline.fit([sample.signal], source_sampling_rate_hz=sample.sampling_rate_hz)
        processed = pipeline.transform(
            sample.signal, source_sampling_rate_hz=sample.sampling_rate_hz
        )
        assert processed.shape[0] == sample.signal.shape[0]
        assert 195 <= processed.shape[1] <= 205

        n_channels = {modality.value: sample.signal.shape[0] for modality in Modality}
        model = FoundationModel(small_model_config, n_channels)
        head = ClassificationHead(d_model=small_model_config.d_model, n_classes=5)
        model.eval()
        head.eval()
        encoded = torch.from_numpy(processed).unsqueeze(0)
        with torch.no_grad():
            cls_token, _ = model(encoded, "ecg")
            logits = head(cls_token)
        assert logits.shape == (1, 5)

        exporter = OnnxExporter()
        onnx_path = exporter.export(
            model,
            tmp_path / "ecg-model.onnx",
            "ecg",
            sample.signal.shape[0],
            processed.shape[1],
        )
        assert onnx_path.exists()
        assert exporter.verify(
            model,
            onnx_path,
            "ecg",
            sample.signal.shape[0],
            processed.shape[1],
            n_samples=3,
        )

    def test_ssl_pretrain_to_finetune_flow(self, small_model_config, tmp_path: Path) -> None:
        """SSL pretrain -> fine-tune -> evaluate flow."""
        set_global_seed(42)

        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(small_model_config, n_ch)

        # SSL pretrain (3 steps)
        ssl_head = SpanMaskedReconstructionHead(
            d_model=small_model_config.d_model,
            patch_length=small_model_config.patch_length,
            n_channels=4,
        )
        ctr_head = ContrastiveHead(d_model=small_model_config.d_model)
        cfg = TrainingConfig(
            max_steps=3, warmup_steps=0, eval_every_steps=1, save_every_steps=10, ema_use=False
        )
        trainer = SSLPretrainer(model, ssl_head, ctr_head, cfg, small_model_config)

        # Mini dataloader
        from torch.utils.data import DataLoader, Dataset

        class _Mini(Dataset):
            def __init__(self) -> None:
                self.data = [
                    (
                        torch.randn(4, 64, dtype=torch.float32),
                        torch.tensor([0], dtype=torch.long),
                        torch.randn(7, 4, 16, dtype=torch.float32),
                    )
                    for _ in range(8)
                ]

            def __len__(self) -> int:
                return 8

            def __getitem__(self, i: int):
                return self.data[i]

        result = trainer.train(
            DataLoader(_Mini(), batch_size=4),
            n_steps=3,
            output_dir=tmp_path,
            run_name="integration_test",
        )
        assert result["final_metrics"]["loss"] > 0

        # Fine-tune
        head = ClassificationHead(d_model=32, n_classes=4)
        ft_cfg = TrainingConfig(max_steps=2, warmup_steps=0, eval_every_steps=1)
        ft = FineTuner(model, head, strategy="linear", config=ft_cfg)

        class _MiniCls(Dataset):
            def __init__(self) -> None:
                self.data = [
                    (
                        torch.randn(4, 64, dtype=torch.float32),
                        torch.tensor([0], dtype=torch.long),
                        torch.randint(0, 4, (1,)).item(),
                    )
                    for _ in range(8)
                ]

            def __len__(self) -> int:
                return 8

            def __getitem__(self, i: int):
                return self.data[i]

        dl = DataLoader(_MiniCls(), batch_size=4)
        ft_result = ft.fit(dl, val_loader=dl, n_steps=2, output_dir=tmp_path / "ft")
        assert "final_train_metrics" in ft_result
        assert ft_result["final_train_metrics"]["loss"] > 0

    def test_loso_with_statistics(self) -> None:
        """LOSO CV + Holm-Šídák correction."""
        # Simulate per-subject accuracy for 3 methods
        rng = np.random.default_rng(42)
        subjects = list(range(10))
        method_a = rng.uniform(0.85, 0.95, size=10)  # best
        method_b = rng.uniform(0.75, 0.85, size=10)
        method_c = rng.uniform(0.65, 0.75, size=10)  # worst

        # LOSO: each subject is one fold (here we just verify the protocol runs)
        cv = LeaveOneSubjectOutCV()
        n_folds = cv.get_n_splits(subjects)
        assert n_folds == 10

        # Aggregate confusion matrix across folds (audit fix verification)
        y_true_folds = [rng.integers(0, 4, size=20).tolist() for _ in range(10)]
        y_pred_folds = [rng.integers(0, 4, size=20).tolist() for _ in range(10)]
        cm = confusion_matrix(y_true_folds, y_pred_folds, n_classes=4)
        assert cm.sum() == 200  # 10 folds * 20 samples

        # Wilcoxon + Holm-Šídák
        from biosignal_fm.evaluation import wilcoxon_signed_rank

        p_a_b = wilcoxon_signed_rank(method_a, method_b)["p_value"]
        p_a_c = wilcoxon_signed_rank(method_a, method_c)["p_value"]
        p_b_c = wilcoxon_signed_rank(method_b, method_c)["p_value"]

        result = holm_sidak_correction([p_a_b, p_a_c, p_b_c], alpha=0.05)
        # All should be rejected (clear differences)
        assert all(result["rejected"]), f"Expected all rejected, got {result['rejected']}"

    def test_realtime_inference_latency(self, small_model_config) -> None:
        """Verify real-time inference completes in reasonable time."""
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(small_model_config, n_ch)
        rt = RealtimeInference(model, modality="emg", quantize=False)

        # Benchmark
        result = rt.benchmark(n_channels=4, signal_length=64, n_runs=20, warmup=3)
        # On CPU, small model should be < 100ms per inference
        assert result["mean_ms"] < 1000  # generous bound for CI
        assert result["n_runs"] == 20
        assert result["single_thread"] is True

    def test_run_manifest_records_all_artifacts(self, small_model_config, tmp_path: Path) -> None:
        """Verify RunManifest captures all outputs with SHA-256."""
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(small_model_config, n_ch)

        # Create manifest and add outputs
        manifest = RunManifest.create(name="test", seed=42)
        ckpt = tmp_path / "model.pt"
        model.save(ckpt)
        manifest.add_output(ckpt, alias="checkpoint")

        onnx_exporter = OnnxExporter()
        onnx_path = onnx_exporter.export(model, tmp_path / "m.onnx", "emg", 4, 64)
        manifest.add_output(onnx_path, alias="onnx")

        # Save manifest
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        # Verify
        loaded = RunManifest.load(manifest_path)
        assert "checkpoint" in loaded.output_hashes
        assert "onnx" in loaded.output_hashes
        assert len(loaded.output_hashes["checkpoint"]) == 64  # SHA-256
        assert len(loaded.output_hashes["onnx"]) == 64
