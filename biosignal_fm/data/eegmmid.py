"""PhysioNet EEG Motor Movement/Imagery Dataset loader.

The EEGMMID database (Schalk et al., 2004) contains 109 subjects performing
motor/imagery tasks with 64-channel EEG at 160 Hz.

References
----------
Schalk, G., McFarland, D. J., Hinterberger, T., et al. (2004). BCI2000:
a general-purpose brain-computer interface (BCI) system. IEEE Transactions
on Biomedical Engineering, 51(6), 1034-1043.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..config import Modality
from .base import BiosignalSample, ModalityMetadata
from .synthetic import SyntheticBiosignalDataset

__all__ = ["EEGMMIDLoader"]


class EEGMMIDLoader:
    """Loader for the PhysioNet EEG Motor Movement/Imagery Dataset.

    Parameters
    ----------
    root_dir : Path or str, optional
        Path to the raw EEGMMID files (EdfFormat).
    cache_dir : Path or str, optional
        Cache directory.
    n_subjects : int, optional
        Number of subjects (1-109). Default 10.
    window_length_seconds : float, optional
        Window length. Default 2.0.
    target_sampling_rate_hz : int, optional
        Resampling target. Default 160 (native).
    """

    MODALITY = Modality.EEG
    NATIVE_SAMPLING_RATE_HZ = 160
    NATIVE_N_CHANNELS = 64
    EEGMMID_LABELS = ("rest", "left_fist", "right_fist", "both_fists", "both_feet")

    # Standard 64-channel 10-10 montage. The previous list mixed in fiducial
    # landmarks (Naz, LPA, RPA) which are NOT EEG electrodes, and made-up
    # names (Fpz2, FFC1, FFC2, P1a, P2a) that do not exist in the standard.
    # We now use only valid 10-10 system electrode names per Oostenveld &
    # Praamstra (2001), Clin Neurophysiol 132(7): 1539-1545. The 64 channels
    # below are a complete, anatomically-valid subset of the international
    # 10-10 system, consistent with the BrainVision 64-channel cap.
    CHANNEL_NAMES = (
        "Fp1",
        "Fpz",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "AF4",
        "AF8",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
        "Iz",
        "T9",
        "T10",
    )

    def __init__(
        self,
        root_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        n_subjects: int = 10,
        window_length_seconds: float = 2.0,
        target_sampling_rate_hz: int = 160,
        runs: tuple[int, ...] = (4, 8, 12),
        allow_synthetic_fallback: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.cache_dir = (
            Path(cache_dir).expanduser() / "eegmmid"
            if cache_dir
            else Path.home() / ".cache" / "biosignal_fm" / "eegmmid"
        )
        self.n_subjects = min(max(n_subjects, 1), 109)
        self.window_length_seconds = window_length_seconds
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.runs = tuple(sorted(set(runs)))
        self.allow_synthetic_fallback = allow_synthetic_fallback
        if target_sampling_rate_hz != self.NATIVE_SAMPLING_RATE_HZ:
            raise ValueError(
                "EEGMMIDLoader does not resample raw data; use the explicit "
                "PreprocessingPipeline after loading canonical signals"
            )
        if not self.runs or any(run not in {4, 8, 12} for run in self.runs):
            raise ValueError(
                "EEGMMIDLoader supports annotation-defined left/right motor-imagery "
                "runs 4, 8, and 12 only"
            )

        self._metadata = ModalityMetadata(
            modality=self.MODALITY,
            sampling_rate_hz=target_sampling_rate_hz,
            n_channels=self.NATIVE_N_CHANNELS,
            channel_names=self.CHANNEL_NAMES,
            label_names=self.EEGMMID_LABELS,
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
        """Load raw PhysioNet EEGMMID .edf files via MNE.

        EEGMMID structure::

            <root_dir>/
              S001/
                S001R01.edf  S001R01.edf.event
                S001R02.edf
                ...
              S002/
              ...

        Returns
        -------
        list[BiosignalSample]
            Empty list if no .edf files are found.

        Raises
        ------
        ImportError
            If ``mne`` is not installed.
        FileNotFoundError
            If ``self.root_dir`` is set but does not exist.
        """
        if self.root_dir is None:
            return []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"EEGMMID root_dir does not exist: {self.root_dir}")

        # Check for .edf files BEFORE importing mne, so that a directory
        # with no data does not require the optional dependency.
        edf_files = sorted(self.root_dir.rglob("*.edf"))
        if not edf_files:
            return []

        try:
            import mne  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "mne is required to load EEGMMID .edf files. Install with: pip install mne"
            ) from e

        samples: list[BiosignalSample] = []
        # Official EEGMMID semantics depend on the run. Restrict this loader to
        # runs 4/8/12, where T1 and T2 denote imagined left and right fist.
        annotation_to_label = {"T1": 1, "T2": 2}
        seen_subjects: set[int] = set()
        n_samples_per_window = int(self.NATIVE_SAMPLING_RATE_HZ * self.window_length_seconds)
        if n_samples_per_window <= 0:
            raise ValueError("window_length_seconds must be positive")

        for edf_path in edf_files:
            # Parse subject + run number from filename like "S001R03.edf".
            try:
                subj_id = int(edf_path.stem[1:4])
                run_id = int(edf_path.stem[4:].lstrip("R"))
            except (ValueError, IndexError):
                continue
            if len(seen_subjects) >= self.n_subjects and subj_id not in seen_subjects:
                continue
            seen_subjects.add(subj_id)
            if run_id not in self.runs:
                continue

            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load EEGMMID file {edf_path.name}: {e}. Skipping.",
                    stacklevel=2,
                )
                continue

            if len(raw.ch_names) != self.NATIVE_N_CHANNELS:
                continue
            source_file_sha256 = hashlib.sha256(edf_path.read_bytes()).hexdigest()
            channel_names = tuple(name.rstrip(".") for name in raw.ch_names)
            for annotation in raw.annotations:
                description = str(annotation["description"])
                if description not in annotation_to_label:
                    continue
                start = int(round(float(annotation["onset"]) * self.NATIVE_SAMPLING_RATE_HZ))
                duration_samples = int(
                    round(float(annotation["duration"]) * self.NATIVE_SAMPLING_RATE_HZ)
                )
                if duration_samples < n_samples_per_window:
                    continue
                window = raw.get_data(start=start, stop=start + n_samples_per_window).astype(
                    np.float32
                )
                if window.shape != (self.NATIVE_N_CHANNELS, n_samples_per_window):
                    continue
                label = annotation_to_label[description]
                samples.append(
                    BiosignalSample(
                        signal=window,
                        modality=Modality.EEG,
                        sampling_rate_hz=self.NATIVE_SAMPLING_RATE_HZ,
                        subject_id=subj_id,
                        session_id=run_id,
                        label=label,
                        label_name=self.EEGMMID_LABELS[label],
                        metadata={
                            "dataset_id": "physionet.eegmmidb.1.0.0",
                            "dataset_version": "1.0.0",
                            "source_uri": "https://www.physionet.org/content/eegmmidb/1.0.0/",
                            "license_id": "ODC-BY-1.0",
                            "source_file": edf_path.name,
                            "source_file_sha256": source_file_sha256,
                            "run_id": run_id,
                            "event_description": description,
                            "event_onset_seconds": float(annotation["onset"]),
                            "channel_names": channel_names,
                        },
                    )
                )
        return samples

    def _load(self) -> list[BiosignalSample]:
        """Load real samples or enter an explicitly requested synthetic smoke path."""
        samples = self._load_raw()
        if not samples:
            if not self.allow_synthetic_fallback:
                raise FileNotFoundError(
                    f"No real EEGMMID windows found at {self.root_dir!r}. "
                    "Set allow_synthetic_fallback=True only for a development smoke path."
                )
            import warnings

            warnings.warn(
                f"EEGMMIDLoader using explicit synthetic fallback. "
                f"No real .edf files found at {self.root_dir!r}.",
                UserWarning,
                stacklevel=2,
            )
            self._is_synthetic = True
            synth = SyntheticBiosignalDataset(
                modality=Modality.EEG,
                n_subjects=self.n_subjects,
                n_sessions_per_subject=3,
                n_samples_per_class=10,
                n_channels=self.NATIVE_N_CHANNELS,
                sampling_rate_hz=self.target_sampling_rate_hz,
                window_length_seconds=self.window_length_seconds,
                n_classes=len(self.EEGMMID_LABELS),
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
