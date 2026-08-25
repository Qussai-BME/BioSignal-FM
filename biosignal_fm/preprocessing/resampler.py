"""Polyphase resampling with anti-aliasing.

Resamples biosignals to a target sampling rate using
``scipy.signal.resample_poly`` (polyphase FIR) which is more numerically
stable than FFT-based resampling for high-rate ratios.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

__all__ = ["Resampler"]


class Resampler:
    """Resample biosignals to a target sampling rate.

    Parameters
    ----------
    target_sampling_rate_hz : int
        Target sampling rate in Hz.

    Notes
    -----
    Uses polyphase filtering with default Kaiser-window FIR anti-aliasing
    filter. For typical biosignal ratios (e.g. 2000 -> 200, ratio 10:1)
    the default filter is more than adequate.

    Examples
    --------
    >>> import numpy as np
    >>> from biosignal_fm.preprocessing import Resampler
    >>> r = Resampler(target_sampling_rate_hz=200)
    >>> signal = np.random.randn(16, 4000).astype(np.float32)  # 2s @ 2kHz
    >>> out = r.resample(signal, source_sampling_rate_hz=2000)
    >>> out.shape
    (16, 400)
    """

    def __init__(self, target_sampling_rate_hz: int) -> None:
        if target_sampling_rate_hz <= 0:
            raise ValueError(f"target_sampling_rate_hz must be > 0, got {target_sampling_rate_hz}")
        self.target_sampling_rate_hz = int(target_sampling_rate_hz)

    def resample(
        self,
        signal: np.ndarray,
        source_sampling_rate_hz: int,
    ) -> np.ndarray:
        """Resample a signal to the target sampling rate.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape ``(n_channels, n_samples)`` or ``(n_samples,)``.
        source_sampling_rate_hz : int
            Source sampling rate in Hz.

        Returns
        -------
        np.ndarray
            Resampled signal with shape matching input but with the time
            dimension resampled to ``target_sampling_rate_hz``.

        Raises
        ------
        ValueError
            If source and target rates are not positive.
        """
        if source_sampling_rate_hz <= 0:
            raise ValueError(f"source_sampling_rate_hz must be > 0, got {source_sampling_rate_hz}")

        if source_sampling_rate_hz == self.target_sampling_rate_hz:
            return np.asarray(signal, dtype=np.float32)

        # Compute integer up/down factors
        from math import gcd

        g = gcd(int(source_sampling_rate_hz), int(self.target_sampling_rate_hz))
        up = int(self.target_sampling_rate_hz) // g
        down = int(source_sampling_rate_hz) // g

        signal = np.asarray(signal, dtype=np.float64)

        if signal.ndim == 1:
            out = resample_poly(signal, up, down)
            return np.asarray(out, dtype=np.float32)

        # 2D: resample per channel
        out = np.empty(
            (signal.shape[0], int(np.ceil(signal.shape[1] * up / down))), dtype=np.float64
        )
        for ch in range(signal.shape[0]):
            out[ch] = resample_poly(signal[ch], up, down)
        return out.astype(np.float32)
