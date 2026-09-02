"""Unit tests for biosignal_fm.reproducibility."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from biosignal_fm.reproducibility import (
    NumpyAwareJSONEncoder,
    RunManifest,
    compute_sha256,
    env_fingerprint,
    set_global_seed,
)


class TestSetGlobalSeed:
    def test_deterministic_numpy(self) -> None:
        set_global_seed(42)
        a = np.random.rand(10)
        set_global_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_python_random(self) -> None:
        import random

        set_global_seed(42)
        a = [random.random() for _ in range(10)]
        set_global_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b


class TestComputeSha256:
    def test_known_content(self, tmp_path: Path) -> None:
        path = tmp_path / "test.txt"
        path.write_text("hello world")
        sha = compute_sha256(path)
        assert isinstance(sha, str)
        assert len(sha) == 64  # SHA-256 hex length

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "nonexistent.txt")


class TestEnvFingerprint:
    def test_has_required_keys(self) -> None:
        fp = env_fingerprint()
        assert "python_version" in fp
        assert "platform" in fp
        assert "packages" in fp
        assert "biosignal_fm_version" in fp


class TestRunManifest:
    def test_create(self) -> None:
        m = RunManifest.create(name="test_run", seed=42)
        assert m.name == "test_run"
        assert m.seed == 42
        assert len(m.run_id) == 36  # UUID4 string length
        assert m.timestamp  # not empty
        assert m.config_hash  # not empty

    def test_records_real_git_head_and_dirty_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture clean and deliberately dirty state from the real project repository."""
        repository_root = Path(__file__).resolve().parents[2]
        if not (repository_root / ".git").is_dir():
            pytest.skip("This test requires the project root to be a Git repository.")

        monkeypatch.chdir(repository_root)
        clean_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
        )
        assert clean_status.stdout == ""

        clean_manifest = RunManifest.create(name="real-git-clean")
        assert len(clean_manifest.git_head) == 40
        assert clean_manifest.git_dirty is False

        probe_path = repository_root / ".biosignal_fm_git_dirty_probe"
        probe_path.write_text("intentional dirty-state probe\n", encoding="utf-8")
        try:
            dirty_manifest = RunManifest.create(name="real-git-dirty")
            assert dirty_manifest.git_head == clean_manifest.git_head
            assert dirty_manifest.git_dirty is True
        finally:
            probe_path.unlink(missing_ok=True)

        restored_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
        )
        assert restored_status.stdout == ""

    def test_add_metric(self) -> None:
        m = RunManifest.create(name="t")
        m.add_metric("accuracy", 0.92)
        assert m.metrics["accuracy"] == 0.92

    def test_add_output(self, tmp_path: Path) -> None:
        path = tmp_path / "output.txt"
        path.write_text("result")
        m = RunManifest.create(name="t")
        sha = m.add_output(path, alias="result")
        assert sha == compute_sha256(path)
        assert "result" in m.output_hashes

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        m = RunManifest.create(name="test", seed=42, notes="hello")
        m.add_metric("loss", 0.5)
        path = tmp_path / "manifest.json"
        m.save(path)

        loaded = RunManifest.load(path)
        assert loaded.name == "test"
        assert loaded.seed == 42
        assert loaded.notes == "hello"
        assert loaded.metrics["loss"] == 0.5

    def test_to_json_serializable(self) -> None:
        m = RunManifest.create(name="test")
        s = m.to_json()
        # Should be valid JSON
        d = json.loads(s)
        assert d["name"] == "test"


class TestNumpyAwareJSONEncoder:
    def test_numpy_int(self) -> None:
        s = json.dumps(np.int64(42), cls=NumpyAwareJSONEncoder)
        assert json.loads(s) == 42

    def test_numpy_float(self) -> None:
        s = json.dumps(np.float64(3.14), cls=NumpyAwareJSONEncoder)
        assert abs(json.loads(s) - 3.14) < 1e-10

    def test_numpy_array(self) -> None:
        arr = np.array([1, 2, 3])
        s = json.dumps(arr, cls=NumpyAwareJSONEncoder)
        assert json.loads(s) == [1, 2, 3]

    def test_path(self, tmp_path: Path) -> None:
        s = json.dumps(tmp_path, cls=NumpyAwareJSONEncoder)
        assert tmp_path.as_posix() in s or str(tmp_path) in s

    def test_unknown_type_raises(self) -> None:
        class Custom:
            pass

        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(Custom(), cls=NumpyAwareJSONEncoder)


class TestResearchRecordCompleteness:
    def test_experiment_id_is_deterministic_for_equivalent_definitions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        environment = {"python_version": "3.12", "packages": []}
        monkeypatch.setattr("biosignal_fm.reproducibility.env_fingerprint", lambda: environment)
        monkeypatch.setattr("biosignal_fm.reproducibility._get_git_head", lambda: "a" * 40)
        monkeypatch.setattr("biosignal_fm.reproducibility._get_git_dirty", lambda: False)
        kwargs = {
            "name": "same-definition",
            "config": {"preprocessing": {"version": "v1"}},
            "seed": 7,
            "model_id": "encoder-v1",
            "dataset_provenance": {
                "dataset_id": "demo-dataset",
                "dataset_version": "1.0",
                "license_id": "CC-BY-4.0",
                "origin": "real",
            },
            "protocol": {
                "protocol_id": "loso-v1",
                "split": "leave-one-subject-out",
                "metrics": ["macro_f1"],
                "unit_of_analysis": "subject",
            },
        }
        first = RunManifest.create(**kwargs)
        second = RunManifest.create(**kwargs)
        assert first.run_id != second.run_id
        assert first.experiment_id == second.experiment_id
        assert first.research_readiness_issues == ()
        first.validate_research_readiness()

    def test_incomplete_manifest_cannot_support_a_real_data_claim(self) -> None:
        manifest = RunManifest.create(name="incomplete")
        assert "dataset_provenance" in manifest.research_readiness_issues
        assert "model_id" in manifest.research_readiness_issues
        with pytest.raises(ValueError, match="not complete"):
            manifest.validate_research_readiness()

    def test_serialized_manifest_recomputes_experiment_identity(self, tmp_path: Path) -> None:
        manifest = RunManifest.create(name="serialized", model_id="encoder-v1")
        path = manifest.save(tmp_path / "manifest.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["experiment_id"] == manifest.experiment_id
        loaded = RunManifest.load(path)
        assert loaded.experiment_id == manifest.experiment_id
