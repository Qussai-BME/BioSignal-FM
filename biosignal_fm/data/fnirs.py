"""fNIRS (functional near-infrared spectroscopy) dataset loader.

Brain-BIDS-compatible fNIRS loader. fNIRS measures hemodynamic responses
(HbO and HbR) at 10 Hz typical, with 8-32 channels covering the prefrontal
or motor cortex.

References
----------
Piper, S. K., Krueger, A., Koch, S. P., et al. (2014). A wearable
brain-computer interface based on functional near-infrared spectroscopy.
Journal of Neural Engineering, 11(5), 056004.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import Modality
from .base import BiosignalSample, ModalityMetadata
from .synthetic import SyntheticBiosignalDataset

__all__ = ["FnirsLoader"]


class FnirsLoader:
    """Loader for Brain-BIDS fNIRS datasets.

    Parameters
    ----------
    root_dir : Path or str, optional
        Path to the raw BIDS-formatted fNIRS directory.
    cache_dir : Path or str, optional
        Cache directory.
    n_subjects : int, optional
        Number of subjects. Default 10.
    window_length_seconds : float, optional
        Window length (longer for fNIRS due to slow hemodynamics). Default 10.0.
    target_sampling_rate_hz : int, optional
        Resampling target. Default 10 (native).
    """

    MODALITY = Modality.FNIRS
    NATIVE_SAMPLING_RATE_HZ = 10
    NATIVE_N_CHANNELS = 32
    CHANNEL_NAMES = tuple(f"fnirs_ch{i:02d}" for i in range(32))
    FNIRS_LABELS = ("rest", "verbal_fluency", "n_back", "mental_rotation", "stm")

    def __init__(
        self,
        root_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        n_subjects: int = 10,
        window_length_seconds: float = 10.0,
        target_sampling_rate_hz: int = 10,
        allow_synthetic_fallback: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.cache_dir = (
            Path(cache_dir).expanduser() / "fnirs"
            if cache_dir
            else Path.home() / ".cache" / "biosignal_fm" / "fnirs"
        )
        self.n_subjects = max(n_subjects, 1)
        self.window_length_seconds = window_length_seconds
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.allow_synthetic_fallback = allow_synthetic_fallback
        if target_sampling_rate_hz != self.NATIVE_SAMPLING_RATE_HZ:
            raise ValueError(
                "FnirsLoader does not resample raw data; use the explicit "
                "PreprocessingPipeline after loading canonical signals"
            )

        self._metadata = ModalityMetadata(
            modality=self.MODALITY,
            sampling_rate_hz=target_sampling_rate_hz,
            n_channels=self.NATIVE_N_CHANNELS,
            channel_names=self.CHANNEL_NAMES,
            label_names=self.FNIRS_LABELS,
        )
        self._samples: list[BiosignalSample] | None = None
        self._is_synthetic = False

    @property
    def metadata(self) -> ModalityMetadata:
        return self._metadata

    @property
    def is_synthetic(self) -> bool:
        return self._is_synthetic

    def _load_raw(self) -> list[BiosignalSample]:
        """Load raw Brain-BIDS fNIRS .snirf or .csv files.

        fNIRS Brain-BIDS structure::

            <root_dir>/
              sub-01/
                ses-01/
                  nirs/
                    sub-01_ses-01_task-motor_nirs.snirf
                ses-02/
                  nirs/
                    sub-01_ses-02_task-motor_nirs.snirf
              sub-02/
              ...

        We support both .snirf (HDF5) and .csv (tabular) formats. The .snirf
        format requires the ``h5py`` package; .csv requires only numpy/pandas.

        Returns
        -------
        list[BiosignalSample]
            Empty list if no .snirf or .csv files are found.

        Raises
        ------
        ImportError
            If ``h5py`` is not installed and only .snirf files are present.
        FileNotFoundError
            If ``self.root_dir`` is set but does not exist.
        """
        if self.root_dir is None:
            return []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"fNIRS root_dir does not exist: {self.root_dir}")

        # Check for data files BEFORE importing optional deps.
        snirf_files = sorted(self.root_dir.rglob("*.snirf"))
        csv_files = sorted(self.root_dir.rglob("*nirs*.csv"))
        if not snirf_files and not csv_files:
            return []

        samples: list[BiosignalSample] = []
        n_subjects_loaded = 0
        seen_subjects: set[int] = set()
        n_samples_per_window = int(self.NATIVE_SAMPLING_RATE_HZ * self.window_length_seconds)

        # Load .snirf files (preferred format)
        if snirf_files:
            try:
                import h5py  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "h5py is required to load .snirf fNIRS files. Install with: pip install h5py"
                ) from e

            for snirf_path in snirf_files:
                # Parse subject ID from path: sub-01/ses-01/nirs/...snirf
                try:
                    subj_str = snirf_path.parts[-4]  # sub-XX
                    subj_id = int(subj_str.split("-")[1])
                except (ValueError, IndexError):
                    subj_id = n_subjects_loaded + 1

                if subj_id in seen_subjects:
                    pass  # multiple sessions OK
                if len(seen_subjects) >= self.n_subjects and subj_id not in seen_subjects:
                    continue
                seen_subjects.add(subj_id)

                try:
                    with h5py.File(snirf_path, "r") as f:
                        # .snirf format: /nirs/data/dataTimeSeries is (n_samples, n_channels)
                        data = np.asarray(f["nirs"]["data"]["dataTimeSeries"][:], dtype=np.float32)
                        # Sampling rate from /nirs/ts
                        if "fs" in f["nirs"].attrs:
                            fs = float(f["nirs"].attrs["fs"])
                        else:
                            fs = self.NATIVE_SAMPLING_RATE_HZ
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to load {snirf_path.name}: {e}. Skipping.",
                        stacklevel=2,
                    )
                    continue

                if data.ndim != 2:
                    continue
                # Transpose to (n_channels, n_samples)
                data = data.T
                if data.shape[0] > self.NATIVE_N_CHANNELS:
                    data = data[: self.NATIVE_N_CHANNELS]

                n_total = data.shape[1]
                if n_total < n_samples_per_window:
                    continue

                # Extract a few windows; label is unknown for fNIRS without
                # event annotations, so we use 0 (rest) as a placeholder.
                n_windows = min(5, n_total // n_samples_per_window)
                for w in range(n_windows):
                    start = w * n_samples_per_window
                    if start + n_samples_per_window > n_total:
                        break
                    window = data[:, start : start + n_samples_per_window]
                    samples.append(
                        BiosignalSample(
                            signal=window,
                            modality=Modality.FNIRS,
                            sampling_rate_hz=int(fs),
                            subject_id=subj_id,
                            session_id=0,
                            label=0,
                            label_name=self.FNIRS_LABELS[0],
                            metadata={
                                "source_file": snirf_path.name,
                                "fs_hz": int(fs),
                            },
                        )
                    )
                n_subjects_loaded += 1
                if n_subjects_loaded >= self.n_subjects:
                    break

        # Fallback: load .csv files (tabular format).
        elif csv_files:
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError(
                    "pandas is required to load .csv fNIRS files. Install with: pip install pandas"
                ) from e

            for csv_path in csv_files:
                try:
                    subj_str = csv_path.parts[-4] if len(csv_path.parts) >= 4 else csv_path.stem
                    subj_id = (
                        int(subj_str.split("-")[1]) if "-" in subj_str else n_subjects_loaded + 1
                    )
                except (ValueError, IndexError):
                    subj_id = n_subjects_loaded + 1

                if subj_id in seen_subjects:
                    pass
                if len(seen_subjects) >= self.n_subjects and subj_id not in seen_subjects:
                    continue
                seen_subjects.add(subj_id)

                try:
                    df = pd.read_csv(csv_path)
                    data = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32).T
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to load {csv_path.name}: {e}. Skipping.",
                        stacklevel=2,
                    )
                    continue

                if data.ndim != 2 or data.shape[0] < 1:
                    continue
                if data.shape[0] > self.NATIVE_N_CHANNELS:
                    data = data[: self.NATIVE_N_CHANNELS]

                n_total = data.shape[1]
                if n_total < n_samples_per_window:
                    continue
                n_windows = min(5, n_total // n_samples_per_window)
                for w in range(n_windows):
                    start = w * n_samples_per_window
                    if start + n_samples_per_window > n_total:
                        break
                    window = data[:, start : start + n_samples_per_window]
                    samples.append(
                        BiosignalSample(
                            signal=window,
                            modality=Modality.FNIRS,
                            sampling_rate_hz=self.NATIVE_SAMPLING_RATE_HZ,
                            subject_id=subj_id,
                            session_id=0,
                            label=0,
                            label_name=self.FNIRS_LABELS[0],
                            metadata={"source_file": csv_path.name},
                        )
                    )
                n_subjects_loaded += 1
                if n_subjects_loaded >= self.n_subjects:
                    break

        return samples

    def _load(self) -> list[BiosignalSample]:
        """Load real samples or enter an explicitly requested synthetic smoke path."""
        samples = self._load_raw()
        if not samples:
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"No real fNIRS windows found at {self.root_dir!r}. "
                    "Set allow_synthetic_fallback=True only for a development smoke path."
                )
            import warnings

            warnings.warn(
                f"FnirsLoader using explicit synthetic fallback. "
                f"No real .snirf or .csv files found at {self.root_dir!r}.",
                UserWarning,
                stacklevel=2,
            )
            self._is_synthetic = True
            synth = SyntheticBiosignalDataset(
                modality=Modality.FNIRS,
                n_subjects=self.n_subjects,
                n_sessions_per_subject=3,
                n_samples_per_class=10,
                n_channels=self.NATIVE_N_CHANNELS,
                sampling_rate_hz=self.target_sampling_rate_hz,
                window_length_seconds=self.window_length_seconds,
                n_classes=len(self.FNIRS_LABELS),
            )
            samples = synth.samples
        return samples

    @property
    def samples(self) -> list[BiosignalSample]:
        if self._samples is None:
            self._samples = self._load()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BiosignalSample:
        return self.samples[idx]

    def get_subject_ids(self) -> list[int]:
        return sorted({s.subject_id for s in self.samples})

    def get_session_ids(self, subject_id: int) -> list[int]:
        return sorted({s.session_id for s in self.samples if s.subject_id == subject_id})

    def iter_by_subject(self) -> Iterator[tuple[int, list[BiosignalSample]]]:
        for subj in self.get_subject_ids():
            yield subj, [s for s in self.samples if s.subject_id == subj]
