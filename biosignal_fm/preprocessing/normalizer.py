"""Channel-wise z-score normalization.

Per-modality z-score normalization with statistics computed on the
**training fold only**. This is critical for preventing domain leakage
in LOSO and LODO cross-validation.

.. note::
    This class was originally called ``SubjectAwareNormalizer`` but the
    name was misleading: the normalizer itself is NOT subject-aware —
    the user is responsible for passing only training-fold signals to
    :meth:`fit`. The new, honest name is :class:`ChannelWiseNormalizer`.
    ``SubjectAwareNormalizer`` remains as a backward-compat alias.

The normalizer stores:

- ``mean_`` : per-channel mean (shape ``(n_channels, 1)``)
- ``std_`` : per-channel standard deviation (shape ``(n_channels, 1)``)
- ``modality_`` : the modality it was fit on
- ``n_samples_seen_`` : total samples used for fitting

These statistics are stored as part of the model checkpoint to guarantee
identical inference-time normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Modality

__all__ = ["ChannelWiseNormalizer", "SubjectAwareNormalizer"]


@dataclass
class ChannelWiseNormalizer:
    """Per-modality z-score normalizer.

    Statistics are computed across all signals passed to :meth:`fit`. The
    caller is responsible for ensuring those signals come from the training
    fold only (no test-fold leakage).

    Examples
    --------
    >>> import numpy as np
    >>> from biosignal_fm.preprocessing import ChannelWiseNormalizer
    >>> from biosignal_fm.config import Modality
    >>> norm = ChannelWiseNormalizer(modality=Modality.EMG)
    >>> train_signals = [np.random.randn(16, 4000) for _ in range(10)]
    >>> _ = norm.fit(train_signals)
    >>> test_signal = np.random.randn(16, 4000)
    >>> normalized = norm.transform(test_signal)
    >>> normalized.shape
    (16, 4000)
    >>> bool(abs(normalized.mean()) < 0.1)  # roughly zero-mean
    True
    """

    modality: Modality
    epsilon: float = 1e-8
    # Fitted attributes
    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)
    n_samples_seen_: int = field(default=0, repr=False)

    def fit(self, signals: list[np.ndarray]) -> ChannelWiseNormalizer:
        """Fit normalization statistics on a list of training signals.

        Parameters
        ----------
        signals : list of np.ndarray
            List of 2D signals of shape ``(n_channels, n_samples)``. All
            signals must have the same number of channels.

        Returns
        -------
        self
            The fitted normalizer.

        Raises
        ------
        ValueError
            If ``signals`` is empty or if signals have inconsistent
            channel counts.
        """
        if not signals:
            raise ValueError("Cannot fit on empty signal list")

        n_channels = signals[0].shape[0]
        for i, s in enumerate(signals):
            if s.shape[0] != n_channels:
                raise ValueError(f"Signal {i} has {s.shape[0]} channels, expected {n_channels}")

        # Concatenate along time dimension for efficient mean/std computation
        concatenated = np.concatenate([np.asarray(s, dtype=np.float64) for s in signals], axis=1)
        self.mean_ = concatenated.mean(axis=1, keepdims=True)
        self.std_ = concatenated.std(axis=1, keepdims=True)
        # Avoid division by zero
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        self.n_samples_seen_ = int(concatenated.shape[1])
        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape ``(n_channels, n_samples)`` or ``(n_samples,)``.

        Returns
        -------
        np.ndarray
            Normalized signal, same shape as input.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            # Treat as single channel
            return np.asarray(
                ((signal - self.mean_.flatten()) / self.std_.flatten()), dtype=np.float32
            )

        if signal.shape[0] != self.mean_.shape[0]:
            raise ValueError(
                f"Signal has {signal.shape[0]} channels, but normalizer was fit on "
                f"{self.mean_.shape[0]} channels"
            )

        return np.asarray((signal - self.mean_) / self.std_, dtype=np.float32)

    def fit_transform(self, signals: list[np.ndarray]) -> list[np.ndarray]:
        """Convenience: fit then transform a list of signals."""
        self.fit(signals)
        return [self.transform(s) for s in signals]

    def inverse_transform(self, signal: np.ndarray) -> np.ndarray:
        """Reverse the normalization (e.g. for visualization)."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
        signal = np.asarray(signal, dtype=np.float32)
        return np.asarray(signal * self.std_ + self.mean_, dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize fitted state to a dict (for checkpointing)."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot serialize unfitted normalizer.")
        return {
            "modality": self.modality.value,
            "epsilon": self.epsilon,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "n_samples_seen": self.n_samples_seen_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChannelWiseNormalizer:
        """Reconstruct a fitted normalizer from a dict."""
        norm = cls(
            modality=Modality.from_str(data["modality"]),
            epsilon=data["epsilon"],
        )
        norm.mean_ = np.array(data["mean"], dtype=np.float64)
        norm.std_ = np.array(data["std"], dtype=np.float64)
        norm.n_samples_seen_ = int(data["n_samples_seen"])
        return norm


# Backward-compat alias. New code should use ChannelWiseNormalizer.
SubjectAwareNormalizer = ChannelWiseNormalizer
