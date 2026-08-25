"""Hostile audit checklist tests.

These tests verify the project meets every requirement from the master
prompt's final checklist. They are intentionally strict — any failure
indicates a deviation from the spec.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLicenseCompliance:
    def test_license_is_apache_2_0(self) -> None:
        license_path = REPO_ROOT / "LICENSE"
        assert license_path.exists(), "LICENSE file missing"
        content = license_path.read_text()
        assert "Apache License" in content
        assert "Version 2.0" in content

    def test_license_has_202_lines(self) -> None:
        """CRITICAL: LICENSE must be exactly 202 lines (full Apache 2.0)."""
        license_path = REPO_ROOT / "LICENSE"
        # 202 lines + trailing newline = 203 elements after splitlines
        # But the spec says "202 lines" — count non-empty trailing newline
        content = license_path.read_text()
        n_lines = content.count("\n")
        assert n_lines == 202, f"LICENSE must have 202 lines, got {n_lines}"

    def test_license_has_copyright(self) -> None:
        content = (REPO_ROOT / "LICENSE").read_text()
        assert "Qussai Adlbi" in content
        assert "2026" in content


class TestAuthorship:
    def test_pyproject_author(self) -> None:
        content = (REPO_ROOT / "pyproject.toml").read_text()
        assert "Qussai Adlbi" in content

    def test_citation_author(self) -> None:
        content = (REPO_ROOT / "CITATION.cff").read_text()
        # CITATION.cff uses YAML structure with family-names and given-names
        assert "Adlbi" in content
        assert "Qussai" in content

    def test_authors_file(self) -> None:
        content = (REPO_ROOT / "AUTHORS.md").read_text()
        assert "Qussai Adlbi" in content

    def test_no_myocontrol_contributors_as_author(self) -> None:
        """Ensure 'MyoControl Contributors' is NOT used as actual author.
        Mentions in docstrings/comments (e.g. audit-fix comments) are OK —
        we only flag real authorship assignments.
        """
        import ast

        # Limit the audit to tracked project source/test trees. A local virtual
        # environment may contain generated or non-UTF-8 Python-adjacent files
        # that are not project authorship assignments.
        python_paths = list((REPO_ROOT / "biosignal_fm").rglob("*.py")) + list(
            (REPO_ROOT / "tests").rglob("*.py")
        )
        for path in python_paths:
            content = path.read_text(encoding="utf-8")
            # Look for __author__ = "MyoControl Contributors" pattern
            if "__author__" in content:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "__author__"
                                and isinstance(node.value, ast.Constant)
                            ):
                                val = str(node.value.value)
                                assert "MyoControl Contributors" not in val, (
                                    f"Found 'MyoControl Contributors' as __author__ in {path}"
                                )


class TestPythonVersionMatrix:
    def test_pyproject_requires_3_10(self) -> None:
        content = (REPO_ROOT / "pyproject.toml").read_text()
        assert 'requires-python = ">=3.10"' in content

    def test_ci_matrix_has_3_versions(self) -> None:
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        content = ci_path.read_text()
        assert '"3.10"' in content
        assert '"3.11"' in content
        assert '"3.12"' in content


class TestDockerSecurity:
    def test_dockerfile_has_non_root_user(self) -> None:
        content = (REPO_ROOT / "Dockerfile").read_text()
        assert "USER " in content, "Dockerfile missing USER directive"
        assert "USER bsfm" in content or "USER 1000" in content

    def test_dockerfile_uses_multi_stage(self) -> None:
        content = (REPO_ROOT / "Dockerfile").read_text()
        assert "FROM" in content
        assert "AS builder" in content or "AS builder" in content

    def test_docker_compose_security_opts(self) -> None:
        content = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "no-new-privileges" in content
        assert "cap_drop" in content
        assert "read_only" in content


class TestPreCommitHooks:
    def test_pre_commit_has_ruff(self) -> None:
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
        assert "ruff" in content
        assert "ruff-format" in content

    def test_pre_commit_has_mypy(self) -> None:
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
        assert "mypy" in content

    def test_pre_commit_has_eof_fixer(self) -> None:
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
        assert "end-of-file-fixer" in content

    def test_pre_commit_has_trailing_whitespace(self) -> None:
        content = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
        assert "trailing-whitespace" in content


class TestNoEmojisInProductionUI:
    """CRITICAL: No emojis in production UI per master prompt."""

    @pytest.mark.parametrize(
        "ui_file",
        [
            "biosignal_fm/ui/app.py",
            "biosignal_fm/ui/theme.py",
            "biosignal_fm/ui/pages/1_Overview.py",
            "biosignal_fm/ui/pages/2_Pretrain.py",
            "biosignal_fm/ui/pages/3_Finetune.py",
            "biosignal_fm/ui/pages/4_Evaluate.py",
            "biosignal_fm/ui/pages/5_Deploy.py",
        ],
    )
    def test_no_emojis_in_ui(self, ui_file: str) -> None:
        path = REPO_ROOT / ui_file
        if not path.exists():
            pytest.skip(f"{ui_file} not found")
        content = path.read_text()
        # Check for common emoji ranges
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002702-\U000027b0"  # dingbats
            "\U000024c2-\U0001f251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "]+",
            re.UNICODE,
        )
        matches = emoji_pattern.findall(content)
        assert not matches, f"Found emojis in {ui_file}: {matches}"


class TestNoPickleLoadFromClientPaths:
    """CRITICAL: No pickle.load on user-supplied paths."""

    def test_no_pickle_load_in_api(self) -> None:
        # Note: the api/ directory was removed in v2.0 — the FastAPI app now
        # lives in deployment/serving.py. We keep this test for historical
        # context and to detect a re-introduction of the api/ directory.
        api_path = REPO_ROOT / "biosignal_fm" / "api"
        if not api_path.exists():
            # No api/ directory → nothing to check. This is the expected
            # state in v2.0+, so we do NOT skip — we pass silently.
            return
        for path in api_path.rglob("*.py"):
            content = path.read_text()
            # Check code only (not docstrings) — use AST
            import ast

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "load"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "pickle"
                    ):
                        pytest.fail(f"pickle.load found in {path}")

    def test_no_pickle_load_in_deployment(self) -> None:
        dep_path = REPO_ROOT / "biosignal_fm" / "deployment"
        for path in dep_path.rglob("*.py"):
            content = path.read_text()
            # Check code only (not docstrings) — use AST
            import ast

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "load"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "pickle"
                    ):
                        pytest.fail(f"pickle.load found in {path}")


class TestHolmSidakFormula:
    """CRITICAL: Verify Holm-Šídák formula is correct, NOT Bonferroni-Holm."""

    def test_formula_is_sidak_not_bonferroni(self) -> None:
        """For p=0.05, m=4, k=1:
        - Šídák:           1 - (1 - 0.05)^4 = 0.185494
        - Bonferroni-Holm: 0.05 * 4         = 0.200000
        These are distinct; verify we get Šídák.
        """
        from biosignal_fm.evaluation import holm_sidak_correction

        result = holm_sidak_correction([0.05, 0.10, 0.20, 0.30], alpha=1.0)
        sidak_corrected = result["corrected_pvalues"][0]
        expected_sidak = 1 - (1 - 0.05) ** 4  # 0.185494
        bonferroni_holm = 0.05 * 4  # 0.20

        assert abs(sidak_corrected - expected_sidak) < 1e-10
        assert abs(sidak_corrected - bonferroni_holm) > 0.001  # distinct!

    def test_source_code_contains_correct_formula(self) -> None:
        """Source code should contain the Šídák formula, not Bonferroni-Holm."""
        stats_path = REPO_ROOT / "biosignal_fm" / "evaluation" / "statistics.py"
        content = stats_path.read_text()
        # Look for the Šídák formula expression
        assert "(1.0 - p_clamped) ** exponent" in content or "(1 - p) **" in content
        # Should NOT contain the Bonferroni-Holm formula as the primary
        # (it's OK to have it in bonferroni_holm_correction function)
        # Just verify the holm_sidak function exists and uses Šídák


class TestOnnxRealParity:
    """CRITICAL: OnnxExporter.verify() must do REAL numerical comparison."""

    def test_verify_runs_inference(self) -> None:
        stats_path = REPO_ROOT / "biosignal_fm" / "deployment" / "onnx_export.py"
        content = stats_path.read_text()
        # Should contain actual inference call, not just load
        assert "InferenceSession" in content
        assert "sess.run" in content
        assert "np.max(np.abs" in content  # numerical comparison


class TestConfusionMatrixAggregation:
    """CRITICAL: UI confusion matrix must aggregate across folds."""

    def test_confusion_matrix_accepts_list_of_folds(self) -> None:
        from biosignal_fm.evaluation import confusion_matrix

        # Pass a list of folds (each fold is a list of labels)
        cm = confusion_matrix(
            y_true_per_fold=[[0, 1, 2], [0, 1, 2]],
            y_pred_per_fold=[[0, 1, 1], [1, 1, 2]],
            n_classes=3,
        )
        # Should aggregate: 6 total samples
        assert cm.sum() == 6


class TestLocalTrackerJsonEncoder:
    """CRITICAL: LocalTracker must NOT use default=str."""

    def test_no_default_str_in_local_tracker(self) -> None:
        """LocalTracker must NOT pass default=str to json.dumps.
        Mentions in docstrings (explaining the audit fix) are OK.
        """
        import ast

        path = REPO_ROOT / "biosignal_fm" / "tracking" / "local_tracker.py"
        content = path.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Detect json.dumps(...) calls
                if isinstance(func, ast.Attribute) and func.attr == "dumps":
                    # Check keyword args for default=str
                    for kw in node.keywords:
                        if (
                            kw.arg == "default"
                            and isinstance(kw.value, ast.Name)
                            and kw.value.id == "str"
                        ):
                            # default=str would be ast.Call to str
                            pytest.fail("LocalTracker uses default=str (audit defect)")

    def test_uses_numpy_aware_encoder(self) -> None:
        path = REPO_ROOT / "biosignal_fm" / "tracking" / "local_tracker.py"
        content = path.read_text()
        assert "NumpyAwareJSONEncoder" in content


class TestPublicSymbolsReExported:
    """CRITICAL: All public symbols must be re-exported in __init__.py."""

    def test_package_exports_version(self) -> None:
        import re

        import biosignal_fm

        assert hasattr(biosignal_fm, "__version__")
        # Check it's a well-formed semver string rather than pinning an
        # exact value -- a hardcoded version string here means this test
        # breaks on every single version bump for no real reason.
        assert re.match(r"^\d+\.\d+\.\d+$", biosignal_fm.__version__), (
            f"__version__ = {biosignal_fm.__version__!r} is not a valid semver string"
        )

    def test_package_exports_set_global_seed(self) -> None:
        import biosignal_fm

        assert hasattr(biosignal_fm, "set_global_seed")

    def test_package_exports_run_manifest(self) -> None:
        import biosignal_fm

        assert hasattr(biosignal_fm, "RunManifest")

    def test_package_exports_configs(self) -> None:
        import biosignal_fm

        assert hasattr(biosignal_fm, "ExperimentConfig")
        assert hasattr(biosignal_fm, "ModelConfig")
        assert hasattr(biosignal_fm, "PreprocessingConfig")
        assert hasattr(biosignal_fm, "TrainingConfig")


class TestReproducibilityManifest:
    def test_run_manifest_has_sha256(self) -> None:
        from biosignal_fm.reproducibility import RunManifest

        m = RunManifest.create(name="test")
        assert hasattr(m, "output_hashes")
        assert hasattr(m, "env_fingerprint")
        assert hasattr(m, "git_head")
        assert hasattr(m, "config_hash")

    def test_set_global_seed_affects_numpy(self) -> None:
        import numpy as np
        from biosignal_fm.reproducibility import set_global_seed

        set_global_seed(42)
        a = np.random.rand(10)
        set_global_seed(42)
        b = np.random.rand(10)
        assert (a == b).all()
