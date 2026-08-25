"""Windowing and patch extraction for biosignal transformers.

Two utilities:

1. :class:`Windower` — Slices long signals into overlapping windows of
   ``(n_channels, window_samples)`` suitable for transformer input.
2. :class:`Patcher` — Slices a window into non-overlapping or overlapping
   patches of ``(d_model,)`` embeddings via Conv1d-style slicing (here
   implemented as a NumPy reshape for inspection/debugging; the actual
   Conv1d happens inside the PyTorch model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Windower", "Patcher"]


@dataclass
class Windower:
    """Slice signals into overlapping windows.

    Parameters
    ----------
    window_length_samples : int
        Window length in samples.
    overlap_samples : int
        Overlap between consecutive windows in samples.

    Examples
    --------
    >>> import numpy as np
    >>> from biosignal_fm.preprocessing import Windower
    >>> w = Windower(window_length_samples=400, overlap_samples=100)
    >>> signal = np.random.randn(16, 2000)
    >>> windows = w.window(signal)
    >>> len(windows)
    6
    >>> windows[0].shape
    (16, 400)
    """

    window_length_samples: int
    overlap_samples: int = 0

    def __post_init__(self) -> None:
        if self.window_length_samples <= 0:
            raise ValueError("window_length_samples must be > 0")
        if self.overlap_samples < 0:
            raise ValueError("overlap_samples must be >= 0")
        if self.overlap_samples >= self.window_length_samples:
            raise ValueError(
                "overlap_samples must be < window_length_samples "
                f"(got {self.overlap_samples} >= {self.window_length_samples})"
            )

    @property
    def step_samples(self) -> int:
        """Step between consecutive window starts."""
        return self.window_length_samples - self.overlap_samples

    def window(self, signal: np.ndarray) -> list[np.ndarray]:
        """Slice a 2D signal into overlapping windows.

        Parameters
        ----------
        signal : np.ndarray
            2D array of shape ``(n_channels, n_samples)``.

        Returns
        -------
        list of np.ndarray
            List of windows, each of shape ``(n_channels, window_length_samples)``.
            If the signal is shorter than one window, returns an empty list.

        Raises
        ------
        ValueError
            If signal is not 2D.
        """
        if signal.ndim != 2:
            raise ValueError(f"signal must be 2D, got shape {signal.shape}")

        n_samples = signal.shape[1]
        if n_samples < self.window_length_samples:
            return []

        windows: list[np.ndarray] = []
        start = 0
        while start + self.window_length_samples <= n_samples:
            end = start + self.window_length_samples
            windows.append(signal[:, start:end].copy())
            start += self.step_samples
        return windows


@dataclass
class Patcher:
    """Slice a window into patches (NumPy implementation for inspection).

    The actual learned patch embedding (Conv1d) happens in the PyTorch model.
    This utility exists for:

    - Debugging (inspect what patches will be fed to the transformer)
    - Computing the expected number of patches for a given config
    - Test fixtures

    Parameters
    ----------
    patch_length : int
        Length of each patch in samples.
    stride : int
        Stride between patches. Default equals ``patch_length`` (no overlap).
    """

    patch_length: int
    stride: int | None = None

    def __post_init__(self) -> None:
        if self.patch_length <= 0:
            raise ValueError("patch_length must be > 0")
        if self.stride is None:
            self.stride = self.patch_length
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.stride > self.patch_length:
            # Allowed but unusual; warn via docstring only to avoid print spam.
            pass

    def n_patches(self, n_samples: int) -> int:
        """Compute the number of patches for a given signal length."""
        assert self.stride is not None  # resolved in __post_init__
        if n_samples < self.patch_length:
            return 0
        return 1 + (n_samples - self.patch_length) // self.stride

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extract patches from a 1D or 2D signal.

        Parameters
        ----------
        signal : np.ndarray
            1D array ``(n_samples,)`` or 2D array ``(n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            If 1D input: shape ``(n_patches, patch_length)``.
            If 2D input: shape ``(n_patches, n_channels, patch_length)``.
        """
        assert self.stride is not None  # resolved in __post_init__
        signal = np.asarray(signal)
        if signal.ndim == 1:
            n_samples = signal.shape[0]
            n_p = self.n_patches(n_samples)
            if n_p == 0:
                return np.empty((0, self.patch_length), dtype=signal.dtype)
            patches = np.empty((n_p, self.patch_length), dtype=signal.dtype)
            for i in range(n_p):
                start = i * self.stride
                patches[i] = signal[start : start + self.patch_length]
            return patches

        if signal.ndim == 2:
            n_channels, n_samples = signal.shape
            n_p = self.n_patches(n_samples)
            if n_p == 0:
                return np.empty((0, n_channels, self.patch_length), dtype=signal.dtype)
            patches = np.empty((n_p, n_channels, self.patch_length), dtype=signal.dtype)
            for i in range(n_p):
                start = i * self.stride
                patches[i] = signal[:, start : start + self.patch_length]
            return patches

        raise ValueError(f"signal must be 1D or 2D, got shape {signal.shape}")
