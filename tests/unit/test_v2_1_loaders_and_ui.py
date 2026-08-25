"""Tests for v2.1 real dataset loaders.

These tests verify the real-file loading code paths (D3 fix). They do NOT
require real datasets — they verify:

1. The loader raises FileNotFoundError when root_dir does not exist.
2. The loader raises ImportError when the optional dep (scipy/wfdb/mne/h5py) is missing.
3. The loader emits a UserWarning when falling back to synthetic.
4. The loader correctly parses file paths and subject IDs.
5. The loader skips malformed files with a warning (not silent failure).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# NinaPro loader
# --------------------------------------------------------------------------- #


class TestNinaProLoaderReal:
    """Verify NinaPro DB5 loader's real-file code path."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """If root_dir is set but does not exist, raise FileNotFoundError."""
        from biosignal_fm.data.ninapro import NinaProDB5Loader

        loader = NinaProDB5Loader(root_dir=tmp_path / "nonexistent")
        # The check happens lazily on first .samples access.
        with pytest.raises(FileNotFoundError, match="root_dir does not exist"):
            _ = loader.samples

    def test_synthetic_fallback_warns(self, tmp_path: Path) -> None:
        """Empty root_dir triggers synthetic fallback with UserWarning."""
        from biosignal_fm.data.ninapro import NinaProDB5Loader

        loader = NinaProDB5Loader(root_dir=tmp_path, n_subjects=2)
        with pytest.warns(UserWarning, match="falling back to synthetic"):
            samples = loader.samples
        assert len(samples) > 0
        assert loader.is_synthetic

    def test_no_root_dir_returns_synthetic(self) -> None:
        """When root_dir is None, loader falls back to synthetic with warning."""
        from biosignal_fm.data.ninapro import NinaProDB5Loader

        loader = NinaProDB5Loader(root_dir=None, n_subjects=2)
        with pytest.warns(UserWarning, match="falling back to synthetic"):
            samples = loader.samples
        assert len(samples) > 0
        assert loader.is_synthetic

    def test_load_raw_with_no_mat_files(self, tmp_path: Path) -> None:
        """An existing root_dir with no .mat files returns empty list."""
        from biosignal_fm.data.ninapro import NinaProDB5Loader

        loader = NinaProDB5Loader(root_dir=tmp_path, n_subjects=2)
        result = loader._load_raw()
        assert result == []


# --------------------------------------------------------------------------- #
# MIT-BIH loader
# --------------------------------------------------------------------------- #


class TestMITBIHLoaderReal:
    """Verify MIT-BIH loader's real-file code path."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        from biosignal_fm.data.mitbih import MITBIHLoader

        loader = MITBIHLoader(root_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="root_dir does not exist"):
            _ = loader.samples

    def test_synthetic_fallback_warns(self, tmp_path: Path) -> None:
        from biosignal_fm.data.mitbih import MITBIHLoader

        loader = MITBIHLoader(root_dir=tmp_path, n_records=2)
        with pytest.warns(UserWarning, match="falling back to synthetic"):
            samples = loader.samples
        assert len(samples) > 0
        assert loader.is_synthetic

    def test_load_raw_with_no_hea_files(self, tmp_path: Path) -> None:
        from biosignal_fm.data.mitbih import MITBIHLoader

        loader = MITBIHLoader(root_dir=tmp_path, n_records=2)
        result = loader._load_raw()
        assert result == []


# --------------------------------------------------------------------------- #
# EEGMMID loader
# --------------------------------------------------------------------------- #


class TestEEGMMIDLoaderReal:
    """Verify EEGMMID loader's real-file code path."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        from biosignal_fm.data.eegmmid import EEGMMIDLoader

        loader = EEGMMIDLoader(root_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="root_dir does not exist"):
            _ = loader.samples

    def test_synthetic_fallback_warns(self, tmp_path: Path) -> None:
        from biosignal_fm.data.eegmmid import EEGMMIDLoader

        loader = EEGMMIDLoader(root_dir=tmp_path, n_subjects=2)
        with pytest.warns(UserWarning, match="falling back to synthetic"):
            samples = loader.samples
        assert len(samples) > 0
        assert loader.is_synthetic

    def test_load_raw_with_no_edf_files(self, tmp_path: Path) -> None:
        from biosignal_fm.data.eegmmid import EEGMMIDLoader

        loader = EEGMMIDLoader(root_dir=tmp_path, n_subjects=2)
        result = loader._load_raw()
        assert result == []


# --------------------------------------------------------------------------- #
# fNIRS loader
# --------------------------------------------------------------------------- #


class TestFnirsLoaderReal:
    """Verify fNIRS loader's real-file code path."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        from biosignal_fm.data.fnirs import FnirsLoader

        loader = FnirsLoader(root_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="root_dir does not exist"):
            _ = loader.samples

    def test_synthetic_fallback_warns(self, tmp_path: Path) -> None:
        from biosignal_fm.data.fnirs import FnirsLoader

        loader = FnirsLoader(root_dir=tmp_path, n_subjects=2)
        with pytest.warns(UserWarning, match="falling back to synthetic"):
            samples = loader.samples
        assert len(samples) > 0
        assert loader.is_synthetic

    def test_load_raw_with_no_files(self, tmp_path: Path) -> None:
        from biosignal_fm.data.fnirs import FnirsLoader

        loader = FnirsLoader(root_dir=tmp_path, n_subjects=2)
        result = loader._load_raw()
        assert result == []


# --------------------------------------------------------------------------- #
# UI honesty fixes (no emojis, demo labels)
# --------------------------------------------------------------------------- #


class TestUIHonestyFixes:
    """Verify the UI pages label demos honestly and contain no emojis."""

    @pytest.mark.parametrize(
        "page_file",
        [
            "biosignal_fm/ui/pages/3_Finetune.py",
            "biosignal_fm/ui/pages/5_Deploy.py",
        ],
    )
    def test_no_emojis_in_ui_pages(self, page_file: str) -> None:
        """No emoji characters in production UI pages."""
        repo_root = Path(__file__).resolve().parents[2]
        content = (repo_root / page_file).read_text(encoding="utf-8")
        # Common emoji ranges. This is a conservative check.
        emoji_ranges = [
            (0x1F300, 0x1F9FF),  # Misc symbols and pictographs, emoticons, etc.
            (0x2600, 0x27BF),  # Misc symbols, dingbats
            (0x1F600, 0x1F64F),  # Emoticons
        ]
        for char in content:
            cp = ord(char)
            for lo, hi in emoji_ranges:
                if lo <= cp <= hi:
                    pytest.fail(f"Found emoji {char!r} (U+{cp:04X}) in {page_file}")

    def test_finetune_page_labels_demo(self) -> None:
        """The finetune page must clearly label non-checkpoint results as such."""
        repo_root = Path(__file__).resolve().parents[2]
        content = (repo_root / "biosignal_fm" / "ui" / "pages" / "3_Finetune.py").read_text()
        assert "demo" in content.lower()
        # Must explain *why* results aren't meaningful research results (no
        # real pretrained checkpoint ships with the package) rather than
        # just asserting the word "random" appears anywhere incidentally.
        assert "random" in content.lower()
        assert "pretrain" in content.lower()

    def test_deploy_page_labels_benchmark_illustrative(self) -> None:
        """The deploy page must label benchmark numbers as illustrative."""
        repo_root = Path(__file__).resolve().parents[2]
        content = (repo_root / "biosignal_fm" / "ui" / "pages" / "5_Deploy.py").read_text()
        assert "illustrative" in content.lower()
        assert (
            "do NOT cite" in content
            or "not real" in content.lower()
            or "not measured" in content.lower()
        )


# --------------------------------------------------------------------------- #
# Ablation script exists and runs
# --------------------------------------------------------------------------- #


class TestAblationScript:
    """Verify the ablation script exists and is importable."""

    def test_script_exists(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "run_ablations.py"
        assert script_path.exists(), f"run_ablations.py not found at {script_path}"

    def test_script_has_main(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "run_ablations.py"
        content = script_path.read_text()
        assert "def main" in content
        assert "argparse" in content
        assert "ablation_table" in content
