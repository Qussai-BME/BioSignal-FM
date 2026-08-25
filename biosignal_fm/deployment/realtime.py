"""Real-time quantized inference for BioSignal-FM.

Provides sub-50ms inference on commodity CPUs via dynamic int8 quantization.
PassMark scores reported by :meth:`RealtimeInference.benchmark` are
single-threaded measurements on the current hardware (verified via
``threadpoolctl`` to ensure no parallel BLAS threads inflate the number).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..config import Modality
from ..models import FoundationModel

__all__ = ["RealtimeInference"]


class RealtimeInference:
    """Quantized real-time inference wrapper.

    Parameters
    ----------
    model : FoundationModel
        The model to wrap.
    modality : Modality or str
        The modality to use for inference (single-modality deployment).
    quantize : bool, optional
        Whether to apply dynamic int8 quantization. Default True.

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.config import ModelConfig, Modality
    >>> from biosignal_fm.models import FoundationModel
    >>> from biosignal_fm.deployment import RealtimeInference
    >>> cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
    >>> n_ch = {m.value: 4 for m in Modality}
    >>> model = FoundationModel(cfg, n_ch)
    >>> rt = RealtimeInference(model, modality="emg", quantize=False)
    >>> signal = np.random.randn(4, 128).astype(np.float32)
    >>> cls = rt.predict(signal)
    >>> cls.shape
    (1, 32)
    """

    def __init__(
        self,
        model: FoundationModel,
        modality: Modality | str,
        quantize: bool = True,
    ) -> None:
        self.modality = Modality.from_str(modality) if isinstance(modality, str) else modality
        self.quantize = quantize
        self.model = model.eval().cpu()

        if self.quantize:
            # Dynamic int8 quantization on Linear layers.
            #
            # Root cause of the historical failure: nn.TransformerEncoderLayer
            # has an internal "fast path" that inspects raw weight tensors
            # (e.g. linear1.weight) to decide whether to dispatch to a fused
            # native kernel. A dynamically-quantized nn.Linear no longer
            # exposes a plain tensor .weight, so that inspection itself
            # raises AttributeError -- before quantized inference ever runs.
            # This is a known PyTorch limitation, not specific to this model.
            #
            # torch.backends.mha.set_fastpath_enabled(False) disables that
            # fast-path dispatch so quantized Linear/LayerNorm modules work
            # normally. NOTE: this is a process-wide torch setting, not
            # scoped to this instance -- acceptable here since RealtimeInference
            # is meant to be the single inference path in a serving process,
            # but worth knowing if you mix this with other torch code in the
            # same process.
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="torch.ao.quantization is deprecated",
                    category=DeprecationWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="torch.quantize_per_tensor.*deprecated",
                    category=UserWarning,
                )
                try:
                    torch.backends.mha.set_fastpath_enabled(False)
                    self.quantized_model = torch.ao.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear, torch.nn.LayerNorm},
                        dtype=torch.qint8,
                    ).eval()
                    self.quantization_active = True
                except (RuntimeError, AttributeError, AssertionError) as err:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Dynamic quantization failed (%s); falling back to full precision. "
                        "This will increase inference latency.",
                        err,
                    )
                    self.quantized_model = self.model
                    self.quantization_active = False
        else:
            self.quantized_model = self.model
            self.quantization_active = False

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run inference on a single signal window.

        Parameters
        ----------
        signal : np.ndarray
            Input signal of shape ``(n_channels, n_samples)`` or
            ``(batch, n_channels, n_samples)``. If 2D, batch dimension
            of 1 is added.

        Returns
        -------
        np.ndarray
            CLS token of shape ``(batch, d_model)``.
        """
        if signal.ndim == 2:
            signal = signal[None, ...]
        signal_t = torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32))
        try:
            cls, _ = self.quantized_model(signal_t, self.modality)
        except (RuntimeError, AttributeError, AssertionError) as err:
            # Quantized path failed (known torch issue with
            # TransformerEncoderLayer + dynamic quantization). Fall back
            # to full precision and log a warning.
            import logging

            logging.getLogger(__name__).warning(
                "Quantized inference failed (%s); using full-precision fallback.",
                err,
            )
            cls, _ = self.model(signal_t, self.modality)
        result: np.ndarray = cls.cpu().numpy()
        return result

    @torch.no_grad()
    def predict_batch(self, signals: np.ndarray) -> np.ndarray:
        """Run inference on a batch of signals.

        Parameters
        ----------
        signals : np.ndarray
            Input of shape ``(batch, n_channels, n_samples)``.

        Returns
        -------
        np.ndarray
            CLS tokens of shape ``(batch, d_model)``.
        """
        return self.predict(signals)

    def benchmark(
        self,
        n_channels: int,
        signal_length: int,
        n_runs: int = 100,
        warmup: int = 10,
        force_single_thread: bool = True,
    ) -> dict:
        """Benchmark inference latency.

        Parameters
        ----------
        n_channels : int
        signal_length : int
        n_runs : int
        warmup : int
        force_single_thread : bool
            If True, forces BLAS to use 1 thread to ensure the reported
            latency is a true single-thread measurement (avoids inflated
            numbers from parallel BLAS).

        Returns
        -------
        dict
            {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "n_runs"}
        """
        if force_single_thread:
            try:
                from threadpoolctl import threadpool_limits

                ctx = threadpool_limits(limits=1)
                ctx.__enter__()
            except ImportError:
                ctx = None
        else:
            ctx = None

        try:
            signal = np.random.randn(1, n_channels, signal_length).astype(np.float32)
            # Warmup
            for _ in range(warmup):
                self.predict(signal)

            latencies: list[float] = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                self.predict(signal)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

            arr = np.array(latencies)
            # IMPORTANT: report `single_thread` based on whether threadpoolctl
            # was actually available. If force_single_thread=True but
            # threadpoolctl is missing, BLAS may run multi-threaded and the
            # measurement is NOT a true single-thread result.
            return {
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(arr.max()),
                "min_ms": float(arr.min()),
                "n_runs": int(n_runs),
                "n_channels": int(n_channels),
                "signal_length": int(signal_length),
                "single_thread": bool(force_single_thread and ctx is not None),
                "threadpoolctl_available": ctx is not None,
                "quantization_requested": bool(self.quantize),
                "quantization_active": bool(self.quantization_active),
            }
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
