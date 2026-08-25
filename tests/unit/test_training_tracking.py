"""Unit tests for biosignal_fm.training and biosignal_fm.tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from biosignal_fm.config import Modality, ModelConfig, TrainingConfig
from biosignal_fm.models import (
    ClassificationHead,
    ContrastiveHead,
    FoundationModel,
    LinearProbe,
    SpanMaskedReconstructionHead,
)
from biosignal_fm.tracking import LocalTracker, MLflowTracker
from biosignal_fm.training import FineTuner, SSLPretrainer
from torch.utils.data import DataLoader, Dataset


def _make_model(n_channels: int = 4) -> FoundationModel:
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        patch_length=16,
        patch_stride=8,
        max_sequence_length=64,
        mask_ratio=0.5,
    )
    n_ch = {m.value: n_channels for m in Modality}
    return FoundationModel(cfg, n_ch)


class _DummySSLDataset(Dataset):
    def __init__(self, n: int = 20, n_channels: int = 4, signal_length: int = 64) -> None:
        self.n = n
        self.data = [
            (
                torch.randn(n_channels, signal_length, dtype=torch.float32),
                torch.tensor([0], dtype=torch.long),
                torch.randn(7, n_channels, 16, dtype=torch.float32),  # targets
            )
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.data[idx]


class _DummyClassifDataset(Dataset):
    def __init__(
        self, n: int = 20, n_channels: int = 4, signal_length: int = 64, n_classes: int = 4
    ) -> None:
        self.n = n
        self.data = [
            (
                torch.randn(n_channels, signal_length, dtype=torch.float32),
                torch.tensor([0], dtype=torch.long),
                torch.randint(0, n_classes, (1,), dtype=torch.long).item(),
            )
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.data[idx]


class TestSSLPretrainer:
    def test_train_step(self) -> None:
        model = _make_model()
        ssl_head = SpanMaskedReconstructionHead(d_model=32, patch_length=16, n_channels=4)
        ctr_head = ContrastiveHead(d_model=32, projection_dim=32)
        cfg = TrainingConfig(max_steps=1, warmup_steps=0, eval_every_steps=1, save_every_steps=1)
        model_cfg = model.config
        trainer = SSLPretrainer(model, ssl_head, ctr_head, cfg, model_cfg)

        signals = torch.randn(4, 4, 64)
        mod_ids = torch.tensor([0, 0, 0, 0])
        targets = torch.randn(4, 7, 4, 16)
        metrics = trainer.train_step((signals, mod_ids, targets))
        assert "loss" in metrics
        assert "mse" in metrics
        assert "contrastive" in metrics
        assert metrics["step"] == 1

    def test_train_short_run(self, tmp_path: Path) -> None:
        model = _make_model()
        ssl_head = SpanMaskedReconstructionHead(d_model=32, patch_length=16, n_channels=4)
        ctr_head = ContrastiveHead(d_model=32, projection_dim=32)
        cfg = TrainingConfig(max_steps=3, warmup_steps=0, eval_every_steps=1, save_every_steps=2)
        trainer = SSLPretrainer(model, ssl_head, ctr_head, cfg, model.config)

        dl = DataLoader(_DummySSLDataset(n=10), batch_size=2, shuffle=False)
        result = trainer.train(dl, n_steps=3, output_dir=tmp_path, run_name="test")
        assert "run_id" in result
        assert "final_metrics" in result
        # Checkpoint saved
        assert (tmp_path / "final_model.pt").exists()
        assert (tmp_path / "manifest.json").exists()


class TestFineTuner:
    def test_linear_probe_step(self) -> None:
        model = _make_model()
        head = LinearProbe(d_model=32, n_classes=4)
        cfg = TrainingConfig(max_steps=1, warmup_steps=0, eval_every_steps=1)
        ft = FineTuner(model, head, strategy="linear", config=cfg)

        # Encoder should be frozen
        for p in model.parameters():
            assert not p.requires_grad

        signals = torch.randn(4, 4, 64)
        mod_ids = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 1, 2, 3])
        metrics = ft.train_step((signals, mod_ids, labels))
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_full_finetune_step(self) -> None:
        model = _make_model()
        head = ClassificationHead(d_model=32, n_classes=4)
        cfg = TrainingConfig(max_steps=1, warmup_steps=0, eval_every_steps=1)
        ft = FineTuner(model, head, strategy="full", config=cfg)

        # Encoder should be trainable
        assert any(p.requires_grad for p in model.parameters())

        signals = torch.randn(4, 4, 64)
        mod_ids = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 1, 2, 3])
        metrics = ft.train_step((signals, mod_ids, labels))
        assert "loss" in metrics

    def test_evaluate(self) -> None:
        model = _make_model()
        head = LinearProbe(d_model=32, n_classes=4)
        cfg = TrainingConfig(max_steps=1, eval_every_steps=1)
        ft = FineTuner(model, head, strategy="linear", config=cfg)

        dl = DataLoader(_DummyClassifDataset(n=10, n_classes=4), batch_size=5)
        result = ft.evaluate(dl)
        assert "loss" in result
        assert "accuracy" in result


class TestLocalTracker:
    def test_log_params(self, tmp_path: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_path, run_name="t")
        tracker.log_params({"lr": 1e-4, "batch_size": 64})
        assert (tmp_path / "params.json").exists()

    def test_log_metrics_append(self, tmp_path: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_path)
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_metrics({"loss": 0.4}, step=2)
        # Two lines in metrics.jsonl
        content = (tmp_path / "metrics.jsonl").read_text()
        assert content.count("\n") == 2

    def test_log_artifact(self, tmp_path: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_path)
        # Create an artifact file
        artifact = tmp_path / "artifact.txt"
        artifact.write_text("hello")
        tracker.log_artifact(artifact)
        content = (tmp_path / "artifacts.jsonl").read_text()
        assert "sha256" in content

    def test_finish_writes_summary(self, tmp_path: Path) -> None:
        tracker = LocalTracker(output_dir=tmp_path, run_name="t")
        tracker.finish()
        assert (tmp_path / "summary.json").exists()

    def test_numpy_values_serialized_properly(self, tmp_path: Path) -> None:
        """CRITICAL: numpy values must be serialized as JSON numbers, not strings.

        This is the fix for the MyoControl v2.0 audit defect where
        LocalTracker used ``default=str`` which stringified numpy types.
        """
        tracker = LocalTracker(output_dir=tmp_path)
        tracker.log_metrics({"loss": np.float64(0.5), "acc": np.float32(0.9)}, step=1)
        import json

        with (tmp_path / "metrics.jsonl").open() as fh:
            entry = json.loads(fh.readline())
        assert isinstance(entry["loss"], (int, float))  # NOT a string
        assert isinstance(entry["acc"], (int, float))  # NOT a string


class TestMLflowTracker:
    def test_works_with_or_without_mlflow(self, tmp_path: Path) -> None:
        """MLflowTracker works whether or not mlflow is installed.

        If mlflow is installed: uses MLflow backend (writes to ./mlruns).
        If mlflow is not installed: falls back to LocalTracker (writes to output_dir).
        Either way, the tracker should not raise and should accept log calls.
        """
        tracker = MLflowTracker(experiment_name="test_exp", output_dir=tmp_path)
        tracker.log_params({"lr": 1e-4})
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.finish()
        # Either MLflow run directory or LocalTracker files should exist
        # We only verify no exceptions were raised.
        assert tracker is not None
