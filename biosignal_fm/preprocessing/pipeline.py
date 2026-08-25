"""End-to-end preprocessing pipeline orchestrator.

Combines :class:`ModalityFilterBank`, :class:`Resampler`, and
:class:`SubjectAwareNormalizer` into a single callable pipeline. The
pipeline is stateful (it holds normalization statistics) and must be
fit on training data before transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Modality, PreprocessingConfig
from .filters import ModalityFilterBank
from .normalizer import ChannelWiseNormalizer, SubjectAwareNormalizer
from .resampler import Resampler

__all__ = ["PreprocessingPipeline"]


@dataclass
class PreprocessingPipeline:
    """End-to-end modality-aware preprocessing pipeline.

    Steps (in order):

    1. Bandpass filter (modality-specific)
    2. Notch filter (line noise, 50/60 Hz)
    3. Resample to target sampling rate
    4. Z-score normalize (subject-aware)

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration.
    modality : Modality
        The modality this pipeline is for.

    Examples
    --------
    >>> import numpy as np
    >>> from biosignal_fm.config import PreprocessingConfig, Modality
    >>> from biosignal_fm.preprocessing import PreprocessingPipeline
    >>> cfg = PreprocessingConfig(target_sampling_rate_hz=200)
    >>> pipe = PreprocessingPipeline(config=cfg, modality=Modality.EMG)
    >>> train = [np.random.randn(16, 4000) for _ in range(5)]
    >>> _ = pipe.fit(train, source_sampling_rate_hz=2000)
    >>> out = pipe.transform(train[0], source_sampling_rate_hz=2000)
    >>> out.shape
    (16, 400)
    """

    config: PreprocessingConfig
    modality: Modality
    # Fitted state
    _filter_bank: ModalityFilterBank = field(init=False, repr=False)
    _resampler: Resampler = field(init=False, repr=False)
    _normalizer: SubjectAwareNormalizer | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._filter_bank = ModalityFilterBank(self.config)
        self._resampler = Resampler(self.config.target_sampling_rate_hz)
        if not hasattr(self, "_normalizer") or self._normalizer is None:
            self._normalizer = SubjectAwareNormalizer(modality=self.modality)

    def fit(
        self,
        signals: list[np.ndarray],
        source_sampling_rate_hz: int,
    ) -> PreprocessingPipeline:
        """Fit normalization statistics on filtered + resampled training signals.

        Parameters
        ----------
        signals : list of np.ndarray
            Raw training signals of shape ``(n_channels, n_samples)``.
        source_sampling_rate_hz : int
            Native sampling rate of the raw signals.

        Returns
        -------
        self
        """
        if not signals:
            raise ValueError("Cannot fit on empty signal list")

        processed: list[np.ndarray] = []
        for sig in signals:
            sig = np.asarray(sig, dtype=np.float32)
            # 1. Bandpass + notch
            filtered = self._filter_bank.filter(sig, self.modality, source_sampling_rate_hz)
            # 2. Resample
            resampled = self._resampler.resample(filtered, source_sampling_rate_hz)
            processed.append(resampled)

        # 3. Fit normalizer on processed signals
        self._normalizer = SubjectAwareNormalizer(modality=self.modality)
        self._normalizer.fit(processed)
        return self

    def transform(
        self,
        signal: np.ndarray,
        source_sampling_rate_hz: int,
    ) -> np.ndarray:
        """Apply the full pipeline to a single signal.

        Parameters
        ----------
        signal : np.ndarray
            Raw signal of shape ``(n_channels, n_samples)``.
        source_sampling_rate_hz : int
            Native sampling rate.

        Returns
        -------
        np.ndarray
            Processed signal of shape ``(n_channels, n_samples_resampled)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._normalizer is None or self._normalizer.mean_ is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        signal = np.asarray(signal, dtype=np.float32)
        filtered = self._filter_bank.filter(signal, self.modality, source_sampling_rate_hz)
        resampled = self._resampler.resample(filtered, source_sampling_rate_hz)
        normalized = self._normalizer.transform(resampled)
        return normalized

    def fit_transform(
        self,
        signals: list[np.ndarray],
        source_sampling_rate_hz: int,
    ) -> list[np.ndarray]:
        """Convenience: fit then transform a list of signals."""
        self.fit(signals, source_sampling_rate_hz)
        return [self.transform(s, source_sampling_rate_hz) for s in signals]

    def to_dict(self) -> dict:
        """Serialize the fitted pipeline state to a dict.

        The dict contains the full :class:`PreprocessingConfig` (so the same
        filters can be rebuilt) plus the fitted normalizer state. The dict is
        JSON-serializable and can be round-tripped through :meth:`from_dict`.
        """
        if self._normalizer is None or self._normalizer.mean_ is None:
            raise RuntimeError("Cannot serialize unfitted pipeline.")
        return {
            "modality": self.modality.value,
            "config": {
                "target_sampling_rate_hz": self.config.target_sampling_rate_hz,
                "emg_bandpass": list(self.config.emg_bandpass),
                "ecg_bandpass": list(self.config.ecg_bandpass),
                "eeg_bandpass": list(self.config.eeg_bandpass),
                "ecog_bandpass": list(self.config.ecog_bandpass),
                "fnirs_bandpass": list(self.config.fnirs_bandpass),
                "notch_freq_hz": self.config.notch_freq_hz,
                "notch_quality_factor": self.config.notch_quality_factor,
                "filter_order": self.config.filter_order,
                "window_length_seconds": self.config.window_length_seconds,
                "window_overlap_seconds": self.config.window_overlap_seconds,
            },
            "normalizer": self._normalizer.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PreprocessingPipeline:
        """Reconstruct a fitted pipeline from a dict produced by :meth:`to_dict`.

        This is the inverse of :meth:`to_dict`. The returned pipeline is
        already fitted (the normalizer state is restored) and can be used
        immediately for :meth:`transform`.

        Parameters
        ----------
        data : dict
            Output of :meth:`to_dict`.

        Returns
        -------
        PreprocessingPipeline
            A fitted pipeline.

        Raises
        ------
        ValueError
            If the dict schema is invalid.
        """
        for required_key in ("modality", "config", "normalizer"):
            if required_key not in data:
                raise ValueError(f"Invalid pipeline dict: missing required key {required_key!r}.")
        cfg_data = dict(data["config"])
        # Restore tuples from lists (YAML/JSON round-trip converts tuples to lists)
        for band_key in (
            "emg_bandpass",
            "ecg_bandpass",
            "eeg_bandpass",
            "ecog_bandpass",
            "fnirs_bandpass",
        ):
            if band_key in cfg_data and isinstance(cfg_data[band_key], list):
                cfg_data[band_key] = tuple(cfg_data[band_key])
        config = PreprocessingConfig(**cfg_data)
        modality = Modality.from_str(data["modality"])
        # Build the pipeline (this also creates a fresh normalizer that we will overwrite)
        pipe = cls(config=config, modality=modality)
        # Restore the fitted normalizer
        pipe._normalizer = ChannelWiseNormalizer.from_dict(data["normalizer"])
        return pipe
