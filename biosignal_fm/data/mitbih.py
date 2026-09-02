"""PhysioNet MIT-BIH Arrhythmia Database ECG loader.

The MIT-BIH Arrhythmia Database (Goldberger et al., 2000; Moody & Mark, 2001)
contains 48 half-hour excerpts of two-channel ambulatory ECG recordings,
sampled at 360 Hz with 11-bit resolution.

This loader supports:

- Loading from raw WFDB-format records (requires ``wfdb`` package and manual
  download from https://physionet.org/content/mitdb/)
- Subject-aware (record-aware) enumeration for LOSO
- Explicit opt-in synthetic fallback for development-only smoke paths

References
----------
Goldberger, A. L., Amaral, L. A., Glass, L., et al. (2000). PhysioBank,
PhysioToolkit, and PhysioNet: components of a new research resource for
complex physiologic signals. Circulation, 101(23), e215-e220.

Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import Modality
from .base import BiosignalSample, ModalityMetadata
from .synthetic import SyntheticBiosignalDataset

__all__ = ["MITBIHLoader"]


class MITBIHLoader:
    """Loader for the PhysioNet MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    root_dir : Path or str, optional
        Path to the directory containing the raw WFDB records.
    cache_dir : Path or str, optional
        Directory for NPZ cache files.
    n_records : int, optional
        Number of records (subjects) to use (1-48). Default 10.
    window_length_seconds : float, optional
        Window length. Default 2.0.
    target_sampling_rate_hz : int, optional
        Resampling target. Default 360 (native).

    Notes
    -----
    MIT-BIH has 5 AAMI beat classes: N (normal), SVEB, VEB, F, Q. For
    BioSignal-FM benchmarks, we map these to the first 5 labels of
    ``SYNTHETIC_LABEL_NAMES`` (renamed) to keep the unified label space.
    """

    MODALITY = Modality.ECG
    NATIVE_SAMPLING_RATE_HZ = 360
    NATIVE_N_CHANNELS = 2
    CHANNEL_NAMES = ("lead_1", "lead_2")
    AAMI_LABELS = ("normal", "sveb", "veb", "fusion", "paced")

    def __init__(
        self,
        root_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        n_records: int = 10,
        window_length_seconds: float = 2.0,
        target_sampling_rate_hz: int = 360,
        allow_synthetic_fallback: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.cache_dir = (
            Path(cache_dir).expanduser() / "mitbih"
            if cache_dir
            else Path.home() / ".cache" / "biosignal_fm" / "mitbih"
        )
        self.n_records = min(max(n_records, 1), 48)
        self.window_length_seconds = window_length_seconds
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.allow_synthetic_fallback = allow_synthetic_fallback
        if target_sampling_rate_hz != self.NATIVE_SAMPLING_RATE_HZ:
            raise ValueError(
                "MITBIHLoader does not resample raw data; use the explicit "
                "PreprocessingPipeline after loading canonical signals"
            )

        self._metadata = ModalityMetadata(
            modality=self.MODALITY,
            sampling_rate_hz=target_sampling_rate_hz,
            n_channels=self.NATIVE_N_CHANNELS,
            channel_names=self.CHANNEL_NAMES,
            label_names=self.AAMI_LABELS,
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
        """Load raw MIT-BIH .dat + .hea records via wfdb.

        MIT-BIH structure::

            <root_dir>/
              100.dat  100.hea
              101.dat  101.hea
              ...
              234.dat  234.hea

        Returns
        -------
        list[BiosignalSample]
            Empty list if no records are found.

        Raises
        ------
        ImportError
            If ``wfdb`` is not installed.
        FileNotFoundError
            If ``self.root_dir`` is set but does not exist.
        """
        if self.root_dir is None:
            return []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"MIT-BIH root_dir does not exist: {self.root_dir}")

        # Check for .hea files BEFORE importing wfdb, so that a directory
        # with no data does not require the optional dependency.
        hea_files = sorted(self.root_dir.rglob("*.hea"))
        if not hea_files:
            return []

        try:
            import wfdb  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "wfdb is required to load MIT-BIH records. Install with: pip install wfdb"
            ) from e

        samples: list[BiosignalSample] = []
        n_records_loaded = 0
        for hea_path in hea_files:
            if n_records_loaded >= self.n_records:
                break
            record_name = hea_path.stem
            try:
                record = wfdb.rdrecord(str(hea_path.with_suffix("")))
                annotation = wfdb.rdann(str(hea_path.with_suffix("")), "atr")
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load MIT-BIH record {record_name}: {e}. Skipping.",
                    stacklevel=2,
                )
                continue

            # record.p_signal: (n_samples, n_leads)
            signal = np.asarray(record.p_signal, dtype=np.float32)
            if (
                signal.ndim != 2
                or signal.shape[1] != self.NATIVE_N_CHANNELS
                or int(round(float(record.fs))) != self.NATIVE_SAMPLING_RATE_HZ
            ):
                continue
            channel_names = tuple(str(name) for name in record.sig_name)
            units = tuple(str(unit) for unit in record.units)
            source_file_sha256 = {
                suffix: hashlib.sha256(hea_path.with_suffix(suffix).read_bytes()).hexdigest()
                for suffix in (".hea", ".dat", ".atr")
                if hea_path.with_suffix(suffix).is_file()
            }

            # Map annotation symbols to AAMI classes.
            # AAMI: N=0, S=1, V=2, F=3, Q=4
            symbol_to_aami = {
                "N": 0,
                "L": 0,
                "R": 0,
                "e": 0,
                "j": 0,  # Normal
                "A": 1,
                "a": 1,
                "J": 1,
                "S": 1,  # Supraventricular
                "V": 2,
                "E": 2,  # Ventricular
                "F": 3,  # Fusion
                "Q": 4,
                "?": 4,  # Unknown
            }

            n_samples_per_window = int(self.NATIVE_SAMPLING_RATE_HZ * self.window_length_seconds)
            # For each R-peak annotation, extract a window centered on it.
            for i, sym in enumerate(annotation.symbol):
                if sym not in symbol_to_aami:
                    continue
                label = symbol_to_aami[sym]
                r_peak = annotation.sample[i]
                start = r_peak - n_samples_per_window // 2
                end = start + n_samples_per_window
                if start < 0 or end > signal.shape[0]:
                    continue
                window = signal[start:end]
                samples.append(
                    BiosignalSample(
                        signal=window.T,  # (n_channels, n_samples)
                        modality=Modality.ECG,
                        sampling_rate_hz=self.NATIVE_SAMPLING_RATE_HZ,
                        subject_id=int(record_name),
                        session_id=0,
                        label=label,
                        label_name=self.AAMI_LABELS[label],
                        metadata={
                            "dataset_id": "physionet.mitdb.1.0.0",
                            "dataset_version": "1.0.0",
                            "source_uri": "https://www.physionet.org/content/mitdb/1.0.0/",
                            "license_id": "ODC-BY-1.0",
                            "source_record": record_name,
                            "source_file_sha256": source_file_sha256,
                            "channel_names": channel_names,
                            "units": units,
                            "r_peak_sample": int(r_peak),
                            "raw_symbol": sym,
                        },
                    )
                )
            n_records_loaded += 1
        return samples

    def _load(self) -> list[BiosignalSample]:
        """Load real samples or enter an explicitly requested synthetic smoke path."""
        samples = self._load_raw()
        if not samples:
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"No real MIT-BIH windows found at {self.root_dir!r}. "
                    "Set allow_synthetic_fallback=True only for a development smoke path."
                )
            import warnings

            warnings.warn(
                f"MITBIHLoader using explicit synthetic fallback. "
                f"No real .dat/.hea files found at {self.root_dir!r}.",
                UserWarning,
                stacklevel=2,
            )
            self._is_synthetic = True
            synth = SyntheticBiosignalDataset(
                modality=Modality.ECG,
                n_subjects=self.n_records,
                n_sessions_per_subject=3,
                n_samples_per_class=10,
                n_channels=self.NATIVE_N_CHANNELS,
                sampling_rate_hz=self.target_sampling_rate_hz,
                window_length_seconds=self.window_length_seconds,
                n_classes=len(self.AAMI_LABELS),
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
