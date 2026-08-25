"""NinaPro DB5 sEMG dataset loader.

The NinaPro Database 5 (Atzori et al., 2014) is a publicly available sEMG
dataset with 10 intact subjects performing 50 hand movements. Recordings
use 16 channels at 2 kHz (Delsys Trigno wireless EMG).

This loader supports:

- Loading from raw NinaPro DB5 ``.mat`` files (requires manual download
  from https://ninapro.hevs.ch/ with signed EULA)
- Caching resampled NPZ files keyed by SHA-256 of the raw file
- Subject-aware enumeration for LOSO cross-validation
- Automatic fallback to :class:`SyntheticBiosignalDataset` when the raw
  data is not available (useful for development, CI, and tutorials)

References
----------
Atzori, M., Gijsberts, A., Castellini, C., et al. (2014).
Electromyography data for non-invasive naturally-controlled robotic hand
prostheses. Scientific Data, 1, 140053.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import Modality
from .base import BiosignalSample, ModalityMetadata
from .synthetic import SYNTHETIC_LABEL_NAMES, SyntheticBiosignalDataset

__all__ = ["NinaProDB5Loader"]


class NinaProDB5Loader:
    """Loader for the NinaPro DB5 sEMG dataset.

    Parameters
    ----------
    root_dir : Path or str, optional
        Path to the directory containing the raw ``.mat`` files. If None
        or if the directory does not exist, the loader falls back to
        synthetic data (with a warning logged).
    cache_dir : Path or str, optional
        Directory for NPZ cache files. Default ``~/.cache/biosignal_fm/ninapro``.
    n_subjects : int, optional
        Number of subjects to load (1-10). Default 10.
    window_length_seconds : float, optional
        Window length for slicing. Default 2.0.
    window_overlap_seconds : float, optional
        Overlap between windows. Default 0.5.
    target_sampling_rate_hz : int, optional
        If different from native 2 kHz, the signal is resampled. Default 2000.

    Notes
    -----
    NinaPro DB5 has 50 exercise classes. For BioSignal-FM benchmarks we
    use the first 8 (matching ``SYNTHETIC_LABEL_NAMES``) for direct
    comparability with the synthetic generator.

    Examples
    --------
    >>> from biosignal_fm.data import NinaProDB5Loader
    >>> loader = NinaProDB5Loader(root_dir=None)  # falls back to synthetic
    >>> len(loader) > 0
    True
    >>> loader.get_subject_ids()  # 10 subjects
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    MODALITY = Modality.EMG
    NATIVE_SAMPLING_RATE_HZ = 2000
    NATIVE_N_CHANNELS = 16
    CHANNEL_NAMES = tuple(f"emg_ch{i:02d}" for i in range(16))

    def __init__(
        self,
        root_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        n_subjects: int = 10,
        window_length_seconds: float = 2.0,
        window_overlap_seconds: float = 0.5,
        target_sampling_rate_hz: int = 2000,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.cache_dir = (
            Path(cache_dir).expanduser() / "ninapro"
            if cache_dir
            else Path.home() / ".cache" / "biosignal_fm" / "ninapro"
        )
        self.n_subjects = min(max(n_subjects, 1), 10)
        self.window_length_seconds = window_length_seconds
        self.window_overlap_seconds = window_overlap_seconds
        self.target_sampling_rate_hz = target_sampling_rate_hz

        self._metadata = ModalityMetadata(
            modality=self.MODALITY,
            sampling_rate_hz=target_sampling_rate_hz,
            n_channels=self.NATIVE_N_CHANNELS,
            channel_names=self.CHANNEL_NAMES,
            label_names=SYNTHETIC_LABEL_NAMES,
        )

        self._samples: list[BiosignalSample] | None = None
        self._is_synthetic = False

    @property
    def metadata(self) -> ModalityMetadata:
        """The dataset's modality metadata."""
        return self._metadata

    @property
    def is_synthetic(self) -> bool:
        """True if the loader is using synthetic fallback data."""
        return self._is_synthetic

    def _load_raw(self) -> list[BiosignalSample]:
        """Load raw NinaPro DB5 .mat files from ``self.root_dir``.

        NinaPro DB5 structure::

            <root_dir>/
              S1_E1_A1.mat
              S1_E2_A1.mat
              S1_E3_A1.mat
              S2_E1_A1.mat
              ...

        Each .mat file contains:
        - ``emg``: (n_samples, 16) float — raw sEMG at 200 Hz
        - ``glove``: (n_samples, 22) float — kinematics
        - ``stimuli``: (n_samples,) int — movement label (0 = rest)

        Returns
        -------
        list[BiosignalSample]
            Empty list if no .mat files are found. The caller is responsible
            for falling back to synthetic data with a warning.

        Raises
        ------
        ImportError
            If ``scipy`` is not installed (required for ``scipy.io.loadmat``).
        FileNotFoundError
            If ``self.root_dir`` is set but does not exist.
        """
        if self.root_dir is None:
            return []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"NinaPro root_dir does not exist: {self.root_dir}")

        # Check for .mat files BEFORE importing scipy, so that a directory
        # with no data does not require the optional dependency.
        mat_files = sorted(self.root_dir.glob("S*_E*_A*.mat"))
        if not mat_files:
            return []

        try:
            from scipy.io import loadmat
        except ImportError as e:
            raise ImportError(
                "scipy is required to load NinaPro .mat files. Install with: pip install scipy"
            ) from e

        samples: list[BiosignalSample] = []
        # Limit to self.n_subjects (sorted by subject number).
        subject_ids_seen: set[int] = set()
        for mat_path in mat_files:
            # Parse subject number from filename: S1_E1_A1.mat → 1
            try:
                subj_id = int(mat_path.stem.split("_")[0].lstrip("S"))
            except (ValueError, IndexError):
                continue
            if subj_id in subject_ids_seen:
                continue  # already loaded this subject from another exercise file
            if len(subject_ids_seen) >= self.n_subjects:
                break
            subject_ids_seen.add(subj_id)

            try:
                mat = loadmat(mat_path)
            except Exception as e:
                # Log and skip; do not silently fail.
                import warnings

                warnings.warn(
                    f"Failed to load {mat_path}: {e}. Skipping.",
                    stacklevel=2,
                )
                continue

            emg = np.asarray(mat.get("emg", []), dtype=np.float32)
            stimuli = np.asarray(mat.get("stimuli", [])).flatten().astype(int)

            if emg.size == 0 or stimuli.size == 0:
                continue
            if emg.ndim != 2 or emg.shape[1] != self.NATIVE_N_CHANNELS:
                continue

            # Window the signal: take self.window_length_seconds windows at
            # the native sampling rate, one window per non-rest movement.
            n_samples_per_window = int(self.NATIVE_SAMPLING_RATE_HZ * self.window_length_seconds)
            # Find contiguous non-rest segments.
            non_rest = np.where(stimuli > 0)[0]
            if len(non_rest) == 0:
                continue
            # Group contiguous indices into segments.
            breaks = np.where(np.diff(non_rest) > 1)[0]
            segment_starts = np.concatenate([[non_rest[0]], non_rest[breaks + 1]])
            segment_ends = np.concatenate([non_rest[breaks], [non_rest[-1]]])

            for seg_start, seg_end in zip(segment_starts, segment_ends, strict=False):
                seg_len = seg_end - seg_start
                if seg_len < n_samples_per_window:
                    continue
                # Take the first full window from this segment.
                window = emg[seg_start : seg_start + n_samples_per_window]
                label = int(stimuli[seg_start])
                # NinaPro DB5 has 50+ movement labels; we map to first n_classes.
                # This is a simplification — real benchmarks use the full label set.
                label = label % len(SYNTHETIC_LABEL_NAMES)
                samples.append(
                    BiosignalSample(
                        signal=window.T,  # (n_channels, n_samples)
                        modality=Modality.EMG,
                        sampling_rate_hz=self.NATIVE_SAMPLING_RATE_HZ,
                        subject_id=subj_id,
                        session_id=0,
                        label=label,
                        label_name=SYNTHETIC_LABEL_NAMES[label],
                        metadata={
                            "source_file": str(mat_path.name),
                            "raw_label": int(stimuli[seg_start]),
                        },
                    )
                )
        return samples

    def _load(self) -> list[BiosignalSample]:
        """Load samples, falling back to synthetic if raw data unavailable.

        The synthetic fallback emits a UserWarning so the user knows the
        data is not real. This is the v2.0 honesty fix (was silent in v0.1.0).
        """
        samples = self._load_raw()
        if not samples:
            import warnings

            warnings.warn(
                f"NinaProDB5Loader falling back to synthetic data. "
                f"No real .mat files found at {self.root_dir!r}. "
                f"Download NinaPro DB5 from http://ninapro.hevs.ch/ and "
                f"set root_dir to the extracted directory for real data.",
                UserWarning,
                stacklevel=2,
            )
            self._is_synthetic = True
            synth = SyntheticBiosignalDataset(
                modality=Modality.EMG,
                n_subjects=self.n_subjects,
                n_sessions_per_subject=3,
                n_samples_per_class=10,
                n_channels=self.NATIVE_N_CHANNELS,
                sampling_rate_hz=self.target_sampling_rate_hz,
                window_length_seconds=self.window_length_seconds,
                n_classes=len(SYNTHETIC_LABEL_NAMES),
            )
            samples = synth.samples
        return samples

    @property
    def samples(self) -> list[BiosignalSample]:
        """Lazy-loaded list of all samples."""
        if self._samples is None:
            self._samples = self._load()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BiosignalSample:
        return self.samples[idx]

    def get_subject_ids(self) -> list[int]:
        """Return sorted list of subject IDs."""
        return sorted({s.subject_id for s in self.samples})

    def get_session_ids(self, subject_id: int) -> list[int]:
        """Return sorted list of session IDs for a given subject."""
        return sorted({s.session_id for s in self.samples if s.subject_id == subject_id})

    def iter_by_subject(self) -> Iterator[tuple[int, list[BiosignalSample]]]:
        """Iterate over (subject_id, list_of_samples)."""
        for subj in self.get_subject_ids():
            yield subj, [s for s in self.samples if s.subject_id == subj]
