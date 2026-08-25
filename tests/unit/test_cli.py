"""Unit tests for biosignal_fm.cli."""

from __future__ import annotations

from pathlib import Path

from biosignal_fm.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


class TestCLI:
    def test_info(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "BioSignal-FM" in result.stdout

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "info" in result.stdout
        assert "pretrain" in result.stdout
        assert "export-onnx" in result.stdout
        assert "benchmark" in result.stdout
        assert "serve" in result.stdout

    def test_benchmark_no_checkpoint(self) -> None:
        """Benchmark with a random model (no checkpoint needed)."""
        result = runner.invoke(
            app,
            [
                "benchmark",
                "--n-channels",
                "4",
                "--signal-length",
                "64",
                "--n-runs",
                "5",
                "--no-quantize",
            ],
        )
        assert result.exit_code == 0
        assert "mean_ms" in result.stdout or "Benchmark" in result.stdout

    def test_export_onnx_missing_checkpoint(self, tmp_path: Path) -> None:
        """export-onnx should fail with a clear error for missing checkpoint."""
        result = runner.invoke(
            app,
            [
                "export-onnx",
                "--checkpoint",
                str(tmp_path / "nonexistent.pt"),
                "--output",
                str(tmp_path / "out.onnx"),
                "--no-verify",
            ],
        )
        assert result.exit_code != 0
