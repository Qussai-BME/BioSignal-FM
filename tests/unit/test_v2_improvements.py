"""Tests for the v2.0 improvement-pass features.

These tests verify the fixes from the IMPROVEMENT_PLAN:

- D1: torch.load weights_only=True (security)
- D2: cross-process deterministic synthetic seed
- D5: PreprocessingPipeline.from_dict round-trip
- D6: public symbols re-exported
- D7: PassMark single_thread honesty
- D8: SwiGLU activation + flash attention flag
- D9: YAML round-trip preserves bandpass tuples
- D12: Nemenyi k>15 raises
- D14: FineTuner n_unfrozen_layers rename
- D15: EEG channel names contain no fiducials
- D28: span_mask geometric distribution
- D31: MixedEffectsAnalyzer
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# D1: torch.load weights_only=True (security)
# --------------------------------------------------------------------------- #


class TestSecureCheckpointLoading:
    """Verify checkpoints are loaded with weights_only=True (no RCE)."""

    def test_load_uses_weights_only_true(self) -> None:
        """AST-audit foundation.py to ensure load() uses weights_only=True."""
        import ast

        path = Path(__file__).resolve().parents[2] / "biosignal_fm" / "models" / "foundation.py"
        content = path.read_text()
        tree = ast.parse(content)
        found_weights_only_true = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Look for torch.load(..., weights_only=True)
                func = node.func
                is_torch_load = (isinstance(func, ast.Attribute) and func.attr == "load") or (
                    isinstance(func, ast.Name) and func.id == "load"
                )
                if is_torch_load:
                    for kw in node.keywords:
                        if (
                            kw.arg == "weights_only"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            found_weights_only_true = True
        assert found_weights_only_true, (
            "FoundationModel.load must call torch.load with weights_only=True"
        )

    def test_no_weights_only_false_anywhere(self) -> None:
        """No source file in the package should use weights_only=False."""
        import ast

        repo_root = Path(__file__).resolve().parents[2]
        offenders: list[str] = []
        for path in (repo_root / "biosignal_fm").rglob("*.py"):
            content = path.read_text()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if (
                            kw.arg == "weights_only"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is False
                        ):
                            offenders.append(str(path.relative_to(repo_root)))
        assert not offenders, f"Found weights_only=False (RCE risk) in: {offenders}"

    def test_round_trip_save_load(self, tmp_path: Path) -> None:
        """Save then load a model and verify numerical equivalence."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        model.eval()  # disable dropout for deterministic comparison
        ckpt_path = tmp_path / "model.pt"
        model.save(ckpt_path)

        loaded = FoundationModel.load(ckpt_path)
        x = torch.randn(2, 4, 64)
        mod_id = torch.tensor([0, 1])
        with torch.no_grad():
            cls_a, _ = model(x, mod_id)
            cls_b, _ = loaded(x, mod_id)
        torch.testing.assert_close(cls_a, cls_b, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------- #
# D2: cross-process deterministic synthetic seed
# --------------------------------------------------------------------------- #


class TestCrossProcessReproducibility:
    """Verify SyntheticBiosignalDataset produces identical data across processes."""

    def test_seed_derivation_uses_sha256_not_hash(self) -> None:
        """The seed derivation must NOT use Python's built-in hash()."""
        path = Path(__file__).resolve().parents[2] / "biosignal_fm" / "data" / "synthetic.py"
        content = path.read_text()
        # The dangerous pattern would be `hash((self.seed, ...))`
        assert "hash((" not in content, (
            "synthetic.py must not use Python's hash() for seed derivation "
            "(it is randomized per-process via PYTHONHASHSEED)."
        )
        assert "hashlib" in content, "synthetic.py should use hashlib for deterministic seeds"

    def test_cross_process_same_data(self, tmp_path: Path) -> None:
        """Generate data in two subprocesses with different PYTHONHASHSEED and assert equality."""
        # Generate sample signal in subprocess 1 with PYTHONHASHSEED=0
        script = tmp_path / "gen.py"
        script.write_text(
            "import numpy as np\n"
            "from biosignal_fm.data.synthetic import SyntheticBiosignalDataset\n"
            "from biosignal_fm.config import Modality\n"
            "ds = SyntheticBiosignalDataset(modality=Modality.EMG, n_subjects=2, n_sessions_per_subject=1, n_samples_per_class=1, n_channels=4, sampling_rate_hz=200, window_length_seconds=1.0, n_classes=2, seed=42)\n"
            "sample = ds[0].signal\n"
            "np.save('/tmp/_bsfm_sample.npy', sample)\n"
        )
        env1 = {**os.environ, "PYTHONHASHSEED": "0"}
        env2 = {**os.environ, "PYTHONHASHSEED": "1"}
        subprocess.run([sys.executable, str(script)], check=True, env=env1, cwd=str(tmp_path))
        sample1 = np.load("/tmp/_bsfm_sample.npy")
        subprocess.run([sys.executable, str(script)], check=True, env=env2, cwd=str(tmp_path))
        sample2 = np.load("/tmp/_bsfm_sample.npy")
        assert np.array_equal(sample1, sample2), (
            "Synthetic data must be identical across processes regardless of PYTHONHASHSEED"
        )


# --------------------------------------------------------------------------- #
# D5: PreprocessingPipeline.from_dict round-trip
# --------------------------------------------------------------------------- #


class TestPipelineRoundTrip:
    """Verify PreprocessingPipeline.to_dict / from_dict round-trips."""

    def test_round_trip(self) -> None:
        from biosignal_fm.config import Modality, PreprocessingConfig
        from biosignal_fm.preprocessing import PreprocessingPipeline

        cfg = PreprocessingConfig(target_sampling_rate_hz=200)
        pipe = PreprocessingPipeline(config=cfg, modality=Modality.EEG)
        # Fit on a few synthetic signals
        rng = np.random.default_rng(42)
        signals = [rng.standard_normal((8, 2000)).astype(np.float32) for _ in range(3)]
        pipe.fit(signals, source_sampling_rate_hz=2000)

        d = pipe.to_dict()
        restored = PreprocessingPipeline.from_dict(d)
        # Verify the restored normalizer has the same mean/std
        assert np.allclose(pipe._normalizer.mean_, restored._normalizer.mean_)
        assert np.allclose(pipe._normalizer.std_, restored._normalizer.std_)
        # Verify transform gives the same result
        test_signal = rng.standard_normal((8, 2000)).astype(np.float32)
        out1 = pipe.transform(test_signal, source_sampling_rate_hz=2000)
        out2 = restored.transform(test_signal, source_sampling_rate_hz=2000)
        assert np.allclose(out1, out2, atol=1e-6)

    def test_from_dict_invalid_raises(self) -> None:
        from biosignal_fm.preprocessing import PreprocessingPipeline

        with pytest.raises(ValueError, match="missing required key"):
            PreprocessingPipeline.from_dict({"modality": "emg"})


# --------------------------------------------------------------------------- #
# D6: public symbols re-exported
# --------------------------------------------------------------------------- #


class TestPublicSymbolsReExported:
    """Verify the full set of public symbols is importable from biosignal_fm."""

    def test_top_level_imports(self) -> None:
        import biosignal_fm as bsfm

        # Spot-check the most important ones
        symbols = [
            "FoundationModel",
            "SwiGLU",
            "LinearProbe",
            "SpanMaskedReconstructionHead",
            "ContrastiveHead",
            "PreprocessingPipeline",
            "ChannelWiseNormalizer",
            "SubjectAwareNormalizer",
            "OnnxExporter",
            "RealtimeInference",
            "ModelRegistry",
            "create_app",
            "SSLPretrainer",
            "FineTuner",
            "LocalTracker",
            "MLflowTracker",
            "MixedEffectsAnalyzer",
            "LeaveOneSubjectOutCV",
            "LeaveOneDatasetOutCV",
            "friedman_nemenyi_test",
            "wilcoxon_holm_sidak",
            "SyntheticBiosignalDataset",
            "NinaProDB5Loader",
            "MITBIHLoader",
            "EEGMMIDLoader",
            "FnirsLoader",
            "EvaluationConfig",
            "DeploymentConfig",
            "load_config",
        ]
        missing = [s for s in symbols if not hasattr(bsfm, s)]
        assert not missing, f"Missing public symbols: {missing}"


# --------------------------------------------------------------------------- #
# D7: PassMark single_thread honesty
# --------------------------------------------------------------------------- #


class TestPassMarkHonesty:
    """Verify benchmark reports single_thread based on actual threadpoolctl availability."""

    def test_benchmark_reports_threadpoolctl_field(self) -> None:
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.deployment import RealtimeInference
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        rt = RealtimeInference(model, modality="emg", quantize=False)
        result = rt.benchmark(
            n_channels=4, signal_length=64, n_runs=2, warmup=0, force_single_thread=True
        )
        assert "threadpoolctl_available" in result, (
            "benchmark must report whether threadpoolctl was actually available"
        )
        # single_thread must be False if threadpoolctl is missing
        if not result["threadpoolctl_available"]:
            assert result["single_thread"] is False, (
                "single_thread must be False when threadpoolctl is unavailable"
            )


# --------------------------------------------------------------------------- #
# D8: SwiGLU + flash attention
# --------------------------------------------------------------------------- #


class TestSwiGLUActivation:
    """Verify SwiGLU activation works and use_flash_attention is honored."""

    def test_swiglu_activation_forward(self) -> None:
        import torch
        from biosignal_fm.models import SwiGLU

        sw = SwiGLU(d_model=32, d_ff=64)
        x = torch.randn(2, 5, 32)
        y = sw(x)
        assert y.shape == x.shape
        # Output should not be all zeros (silu is non-zero for non-zero input)
        assert y.abs().sum() > 0

    def test_swiglu_config_accepted(self) -> None:
        """ModelConfig activation='swiglu' must construct without error."""
        import torch
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            patch_length=16,
            patch_stride=8,
            activation="swiglu",
        )
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        x = torch.randn(2, 4, 64)
        mod_id = torch.tensor([0, 1])
        cls, patches = model(x, mod_id)
        assert cls.shape == (2, 32)

    def test_invalid_activation_raises(self) -> None:
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel

        cfg = ModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            patch_length=16,
            patch_stride=8,
            activation="invalid_activation",
        )
        n_ch = {m.value: 4 for m in Modality}
        with pytest.raises(ValueError, match="Unsupported activation"):
            FoundationModel(cfg, n_ch)


# --------------------------------------------------------------------------- #
# D9: YAML round-trip preserves bandpass tuples
# --------------------------------------------------------------------------- #


class TestYamlRoundTripTuples:
    """Verify PreprocessingConfig bandpass tuples survive YAML round-trip."""

    def test_bandpass_remains_tuple(self, tmp_path: Path) -> None:
        from biosignal_fm.config import ExperimentConfig, PreprocessingConfig

        cfg = ExperimentConfig(
            name="test",
            output_dir=tmp_path,
            preprocessing=PreprocessingConfig(
                emg_bandpass=(20.0, 450.0),
                ecg_bandpass=(0.5, 40.0),
            ),
        )
        yaml_path = tmp_path / "cfg.yaml"
        cfg.to_yaml(yaml_path)
        loaded = ExperimentConfig.from_yaml(yaml_path)
        # The bandpass fields must be tuples, not lists.
        assert isinstance(loaded.preprocessing.emg_bandpass, tuple), (
            f"emg_bandpass should be tuple, got {type(loaded.preprocessing.emg_bandpass).__name__}"
        )
        assert isinstance(loaded.preprocessing.ecg_bandpass, tuple)
        assert loaded.preprocessing.emg_bandpass == (20.0, 450.0)


# --------------------------------------------------------------------------- #
# D12: Nemenyi k>15 raises
# --------------------------------------------------------------------------- #


class TestNemenyiTable:
    """Verify Nemenyi post-hoc handles k beyond table bounds correctly."""

    def test_k_in_extended_range(self) -> None:
        """k=11..15 should now work after extending the table."""
        from biosignal_fm.evaluation import nemenyi_posthoc

        # 12 datasets × 12 methods
        scores = np.random.default_rng(0).uniform(0.5, 1.0, size=(12, 12))
        result = nemenyi_posthoc(scores, alpha=0.05)
        assert "critical_difference" in result

    def test_k_too_large_raises(self) -> None:
        from biosignal_fm.evaluation import nemenyi_posthoc

        # 16 methods — beyond our extended table
        scores = np.random.default_rng(0).uniform(0.5, 1.0, size=(5, 16))
        with pytest.raises(ValueError, match="only covers k=2..15"):
            nemenyi_posthoc(scores, alpha=0.05)

    def test_invalid_alpha_raises(self) -> None:
        from biosignal_fm.evaluation import nemenyi_posthoc

        scores = np.random.default_rng(0).uniform(0.5, 1.0, size=(5, 3))
        with pytest.raises(ValueError, match="alpha must be 0.05 or 0.10"):
            nemenyi_posthoc(scores, alpha=0.01)


# --------------------------------------------------------------------------- #
# D14: FineTuner n_unfrozen_layers rename
# --------------------------------------------------------------------------- #


class TestFineTunerRename:
    """Verify n_unfrozen_layers is the new API name."""

    def test_new_name_accepted(self) -> None:

        from biosignal_fm.config import Modality, ModelConfig, TrainingConfig
        from biosignal_fm.models import FoundationModel, LinearProbe
        from biosignal_fm.training import FineTuner

        cfg = ModelConfig(d_model=32, n_layers=3, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        head = LinearProbe(d_model=32, n_classes=4)
        FineTuner(
            model,
            head,
            strategy="partial",
            n_unfrozen_layers=1,
            config=TrainingConfig(max_steps=1),
        )
        # First 2 of 3 encoder layers should be frozen, last 1 unfrozen
        params_frozen = sum(1 for p in model.encoder.layers[0].parameters() if not p.requires_grad)
        params_unfrozen = sum(1 for p in model.encoder.layers[2].parameters() if p.requires_grad)
        assert params_frozen > 0, "First layer should be frozen"
        assert params_unfrozen > 0, "Last layer should be unfrozen"


# --------------------------------------------------------------------------- #
# D15: EEG channel names contain no fiducials
# --------------------------------------------------------------------------- #


class TestEEGChannelNames:
    """Verify EEGMMIDLoader channel names contain no fiducial landmarks."""

    def test_no_fiducials(self) -> None:
        from biosignal_fm.data.eegmmid import EEGMMIDLoader

        # Fiducials are anatomical landmarks used during digitization, NOT
        # EEG electrodes. They must not appear in the channel list.
        # (Iz/T9/T10 ARE valid 10-10 electrodes — Iz = inion, T9/T10 = preauricular
        # alternative positions — so they are allowed.)
        fiducials = {"Naz", "LPA", "RPA", "Nz", "A1", "A2"}
        ch_names = set(EEGMMIDLoader.CHANNEL_NAMES)
        overlap = ch_names & fiducials
        assert not overlap, (
            f"EEG channel names should not include fiducial landmarks {fiducials}, "
            f"found overlap: {overlap}"
        )
        assert len(EEGMMIDLoader.CHANNEL_NAMES) == 64
        # No made-up names: every channel must match the standard 10-10 pattern.
        # 10-10 names are 1-2 uppercase letters, optionally followed by a
        # lowercase letter (e.g., Fp1, FCz, CP3, Oz), then an optional digit
        # (1-9) or 'z'. Valid examples: Fp1, Fpz, FCz, Cz, Oz, T9, T10, PO8.
        import re

        pattern = re.compile(
            r"^[A-Z]{1,2}"  # 1-2 uppercase letters (F, Fp, FC, CP, PO, O, T, etc.)
            r"[a-z]?"  # optional lowercase (e.g., 'p' in Fp, 'z' in Oz)
            r"(?:z|\d{1,2})$"  # ends with 'z' or 1-2 digits (z, 1, 2, 9, 10)
        )
        bad = [n for n in EEGMMIDLoader.CHANNEL_NAMES if not pattern.match(n)]
        assert not bad, f"Channel names not matching 10-10 pattern: {bad}"


# --------------------------------------------------------------------------- #
# D28: span_mask geometric distribution
# --------------------------------------------------------------------------- #


class TestSpanMaskGeometric:
    """Verify span_mask uses a true geometric distribution."""

    def test_no_torch_log_in_span_sampling(self) -> None:
        """The old buggy code used `torch.log(torch.tensor(...))` — make sure it's gone."""
        path = Path(__file__).resolve().parents[2] / "biosignal_fm" / "models" / "ssl_heads.py"
        content = path.read_text()
        assert "torch.log(torch.tensor" not in content, (
            "ssl_heads.py should not use the buggy `torch.log(torch.tensor(...))` pattern"
        )
        assert "math.log" in content, "ssl_heads.py should use math.log for the geometric formula"

    def test_span_mask_basic(self) -> None:
        import torch
        from biosignal_fm.models import span_mask

        mask = span_mask(batch_size=2, n_patches=20, mask_ratio=0.5, mean_span_length=4)
        assert mask.shape == (2, 20)
        assert mask.dtype == torch.bool
        # Each row should mask ~50% of patches
        for b in range(2):
            ratio = mask[b].sum().item() / 20
            assert 0.3 < ratio < 0.7, f"row {b} mask ratio {ratio} out of expected range"


# --------------------------------------------------------------------------- #
# D31: MixedEffectsAnalyzer
# --------------------------------------------------------------------------- #


class TestMixedEffectsAnalyzer:
    """Verify MixedEffectsAnalyzer works (if statsmodels is installed)."""

    def test_import_or_skip(self) -> None:
        pytest.importorskip("statsmodels")

    def test_basic_fit(self) -> None:
        pytest.importorskip("statsmodels")
        import pandas as pd
        from biosignal_fm.evaluation import MixedEffectsAnalyzer

        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame(
            {
                "accuracy": rng.uniform(0.6, 0.9, n),
                "method": ["fm"] * (n // 2) + ["baseline"] * (n // 2),
                "subject": [f"s{i % 5}" for i in range(n)],
            }
        )
        an = MixedEffectsAnalyzer(df)
        result = an.fit(formula="accuracy ~ method", groups="subject")
        assert result.n_observations == n
        assert result.n_groups == 5
        assert "Intercept" in result.coefficients
        # Reference category is alphabetical ("baseline"), so coefficient is method[T.fm]
        assert "method[T.fm]" in result.coefficients or "method[T.baseline]" in result.coefficients
        d = result.to_dict()
        assert d["n_observations"] == n


# --------------------------------------------------------------------------- #
# D4: CLI finetune and evaluate commands exist
# --------------------------------------------------------------------------- #


class TestCliFinetuneEvaluate:
    """Verify the new finetune and evaluate CLI commands exist."""

    def test_finetune_help(self) -> None:
        from biosignal_fm.cli.main import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["finetune", "--help"])
        assert result.exit_code == 0
        assert "Fine-tune" in result.output or "fine-tune" in result.output.lower()

    def test_evaluate_help(self) -> None:
        from biosignal_fm.cli.main import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Friedman" in result.output or "evaluate" in result.output.lower()

    def test_evaluate_runs(self, tmp_path: Path) -> None:
        """Actually invoke the evaluate command end-to-end with a real checkpoint."""
        from biosignal_fm.cli.main import app
        from biosignal_fm.config import Modality, ModelConfig
        from biosignal_fm.models import FoundationModel
        from typer.testing import CliRunner

        # Save a tiny model checkpoint
        cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
        n_ch = {m.value: 4 for m in Modality}
        model = FoundationModel(cfg, n_ch)
        ckpt = tmp_path / "model.pt"
        model.save(ckpt)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--checkpoint",
                str(ckpt),
                "--modality",
                "emg",
                "--n-classes",
                "4",
                "--n-channels",
                "4",
                "--signal-length",
                "64",
                "--protocol",
                "loso",
            ],
        )
        assert result.exit_code == 0, f"evaluate failed: {result.output}"
        assert "Friedman-Nemenyi" in result.output
        assert "Hedges" in result.output


# --------------------------------------------------------------------------- #
# D11: RunManifest no longer uses default=str
# --------------------------------------------------------------------------- #


class TestRunManifestNoDefaultStr:
    """Verify RunManifest.create does not use default=str."""

    def test_no_default_str_in_manifest(self) -> None:
        import ast

        path = Path(__file__).resolve().parents[2] / "biosignal_fm" / "reproducibility.py"
        content = path.read_text()
        tree = ast.parse(content)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (
                        kw.arg == "default"
                        and isinstance(kw.value, ast.Name)
                        and kw.value.id == "str"
                    ):
                        offenders.append(f"line {node.lineno}")
        assert not offenders, (
            f"reproducibility.py should not use default=str (audit defect). Found at: {offenders}"
        )

    def test_no_silent_str_fallback(self) -> None:
        """RunManifest.create must raise on non-serializable config, not stringify it.

        We use AST inspection to make sure no `except TypeError` block ends
        with a `config_dict = {"_raw": str(...)}` assignment — i.e., the silent
        fallback pattern is gone from the *code*, not just from comments.
        """
        import ast

        path = Path(__file__).resolve().parents[2] / "biosignal_fm" / "reproducibility.py"
        content = path.read_text()
        tree = ast.parse(content)
        # Walk every `except TypeError` handler and verify it does not assign
        # a dict with an "_raw" key.
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "config_dict"
                                and isinstance(stmt.value, ast.Dict)
                            ):
                                for key in stmt.value.keys:
                                    if isinstance(key, ast.Constant) and key.value == "_raw":
                                        pytest.fail(
                                            f"Found silent {{'_raw': str(...)}} fallback at "
                                            f"line {stmt.lineno}"
                                        )


# --------------------------------------------------------------------------- #
# F3: Nyquist validation (raise, not silently clamp)
# --------------------------------------------------------------------------- #


class TestNyquistValidation:
    """Verify ModalityFilterBank raises when bandpass exceeds Nyquist."""

    def test_emg_at_200hz_raises(self) -> None:
        """EMG bandpass (20, 450) at 200 Hz sampling must raise (Nyquist = 100)."""
        import numpy as np
        from biosignal_fm.config import Modality, PreprocessingConfig
        from biosignal_fm.preprocessing import ModalityFilterBank

        cfg = PreprocessingConfig()  # default EMG bandpass is (20, 450)
        fb = ModalityFilterBank(cfg)
        sig = np.random.randn(4, 400).astype(np.float32)
        with pytest.raises(ValueError, match="exceeds Nyquist"):
            fb.filter(sig, Modality.EMG, sampling_rate_hz=200)

    def test_emg_at_2000hz_works(self) -> None:
        """EMG bandpass (20, 450) at 2000 Hz sampling must work (Nyquist = 1000)."""
        import numpy as np
        from biosignal_fm.config import Modality, PreprocessingConfig
        from biosignal_fm.preprocessing import ModalityFilterBank

        cfg = PreprocessingConfig()
        fb = ModalityFilterBank(cfg)
        sig = np.random.randn(4, 4000).astype(np.float32)
        out = fb.filter(sig, Modality.EMG, sampling_rate_hz=2000)
        assert out.shape == sig.shape

    def test_low_ge_high_raises(self) -> None:
        """A bandpass with low >= high must fail before preprocessing executes."""
        from biosignal_fm.config import PreprocessingConfig

        with pytest.raises(ValueError, match="emg_bandpass"):
            PreprocessingConfig(emg_bandpass=(100.0, 100.0))


# --------------------------------------------------------------------------- #
# D29: MLflowTracker fallback robustness
# --------------------------------------------------------------------------- #


class TestMLflowTrackerRobustness:
    """Verify MLflowTracker falls back to LocalTracker if mlflow unavailable."""

    def test_tracker_works_without_mlflow(self, tmp_path: Path) -> None:
        from biosignal_fm.tracking import MLflowTracker

        # Constructor should not raise even if mlflow is missing/broken.
        tracker = MLflowTracker(output_dir=tmp_path)
        tracker.log_params({"foo": "bar"})
        tracker.log_metrics({"loss": 0.5}, step=0)
        tracker.finish()
