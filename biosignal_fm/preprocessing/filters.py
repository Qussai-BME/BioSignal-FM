"""Modality-aware bandpass and notch filtering.

Uses ``scipy.signal`` second-order sections (SOS) for numerical stability
with high-order IIR filters. All filters are designed once at construction
time and applied identically to every signal — this guarantees that the
same filter is applied across training and inference (Systems lens).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt

from ..config import Modality, PreprocessingConfig

__all__ = ["ModalityFilterBank"]


@dataclass
class ModalityFilterBank:
    """Per-modality Butterworth bandpass + optional notch filter.

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object specifying bandpass ranges and notch frequency.

    Notes
    -----
    The filter is applied using ``scipy.signal.sosfiltfilt`` for zero-phase
    filtering (no group delay). This is critical for time-aligned downstream
    tasks like event detection.

    Examples
    --------
    >>> from biosignal_fm.config import PreprocessingConfig, Modality
    >>> from biosignal_fm.preprocessing import ModalityFilterBank
    >>> import numpy as np
    >>> cfg = PreprocessingConfig()
    >>> fb = ModalityFilterBank(cfg)
    >>> signal = np.random.randn(16, 4000).astype(np.float32)  # 16ch, 2s @ 2kHz
    >>> filtered = fb.filter(signal, Modality.EMG, sampling_rate_hz=2000)
    >>> filtered.shape
    (16, 4000)
    """

    config: PreprocessingConfig

    def __post_init__(self) -> None:
        # Pre-design filters for each modality at common sampling rates
        self._sos_cache: dict[tuple[Modality, int], np.ndarray] = {}
        self._notch_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _design_bandpass(self, modality: Modality, sampling_rate_hz: int) -> np.ndarray:
        """Design Butterworth bandpass filter (cached).

        Raises
        ------
        ValueError
            If ``high`` exceeds the Nyquist frequency. Silently clamping would
            produce a filter that destroys the signal (e.g., EMG bandpass
            (20, 450) at 200 Hz sampling would silently become (20, 95),
            eliminating the EMG band entirely).
        """
        key = (modality, sampling_rate_hz)
        if key in self._sos_cache:
            return self._sos_cache[key]

        low, high = self.config.bandpass_for(modality)
        nyq = sampling_rate_hz / 2.0

        # Reject bandpass specs that exceed Nyquist. A silent clamp here would
        # be a serious scientific bug — it would silently filter the signal
        # into the wrong band and the user would never know.
        if high >= nyq:
            raise ValueError(
                f"Bandpass high cutoff {high} Hz for modality {modality.value} "
                f"exceeds Nyquist frequency {nyq} Hz (sampling rate "
                f"{sampling_rate_hz} Hz). Either increase the sampling rate "
                f"or reduce the bandpass high cutoff in PreprocessingConfig."
            )
        if low <= 0:
            low = 0.001
        if low >= high:
            raise ValueError(
                f"Bandpass low cutoff {low} Hz must be < high cutoff {high} Hz "
                f"for modality {modality.value}."
            )

        wn = [low / nyq, high / nyq]
        sos = np.asarray(butter(self.config.filter_order, wn, btype="bandpass", output="sos"))
        self._sos_cache[key] = sos
        return sos

    def _design_notch(self, sampling_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
        """Design IIR notch filter at line frequency (50 or 60 Hz)."""
        if sampling_rate_hz in self._notch_cache:
            return self._notch_cache[sampling_rate_hz]

        if self.config.notch_freq_hz is None:
            # No notch filter; return identity (b=[1], a=[1])
            self._notch_cache[sampling_rate_hz] = (np.array([1.0]), np.array([1.0]))
            return self._notch_cache[sampling_rate_hz]

        b, a = iirnotch(
            self.config.notch_freq_hz,
            self.config.notch_quality_factor,
            fs=sampling_rate_hz,
        )
        self._notch_cache[sampling_rate_hz] = (b, a)
        return b, a

    def filter(
        self,
        signal: np.ndarray,
        modality: Modality | str,
        sampling_rate_hz: int,
    ) -> np.ndarray:
        """Apply the modality-specific filter to a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape ``(n_channels, n_samples)`` or ``(n_samples,)``.
        modality : Modality or str
            The modality, used to select the bandpass range.
        sampling_rate_hz : int
            The sampling rate of the input signal in Hz.

        Returns
        -------
        np.ndarray
            Filtered signal, same shape as input.

        Raises
        ------
        ValueError
            If the signal is not 1D or 2D, or if the bandpass range is
            invalid for the given sampling rate.
        """
        if isinstance(modality, str):
            modality = Modality.from_str(modality)

        if signal.ndim not in (1, 2):
            raise ValueError(f"signal must be 1D or 2D, got shape {signal.shape}")

        signal = np.asarray(signal, dtype=np.float64)

        # Bandpass via SOS zero-phase filtfilt
        sos = self._design_bandpass(modality, sampling_rate_hz)

        # Notch (b, a) — applied via filtfilt (zero-phase)
        b, a = self._design_notch(sampling_rate_hz)

        if signal.ndim == 1:
            # 1D: apply bandpass then notch
            filtered = sosfiltfilt(sos, signal)
            if self.config.notch_freq_hz is not None:
                from scipy.signal import filtfilt

                filtered = filtfilt(b, a, filtered)
            return np.asarray(filtered, dtype=np.float32)

        # 2D: apply per-channel
        filtered = np.empty_like(signal, dtype=np.float64)
        for ch in range(signal.shape[0]):
            x = sosfiltfilt(sos, signal[ch])
            if self.config.notch_freq_hz is not None:
                from scipy.signal import filtfilt

                x = filtfilt(b, a, x)
            filtered[ch] = x

        return filtered.astype(np.float32)
