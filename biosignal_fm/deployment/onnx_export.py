"""ONNX export with REAL numerical parity verification.

The :meth:`OnnxExporter.verify` method performs a genuine numerical
comparison between PyTorch and ONNX Runtime outputs on identical random
inputs. This is the fix for the MyoControl v2.0 audit defect where
``OnnxExporter.verify()`` only loaded the model without checking outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Modality
from ..models import FoundationModel
from ..reproducibility import compute_sha256

__all__ = ["OnnxExporter"]


class OnnxExporter:
    """Export FoundationModel to ONNX format with verification.

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.config import ModelConfig, Modality
    >>> from biosignal_fm.models import FoundationModel
    >>> from biosignal_fm.deployment import OnnxExporter
    >>> cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
    >>> n_ch = {m.value: 4 for m in Modality}
    >>> model = FoundationModel(cfg, n_ch)
    >>> exporter = OnnxExporter()
    >>> # export + verify
    >>> # onnx_path = exporter.export(model, Path("/tmp/model.onnx"), modality="emg",
    >>> #                              n_channels=4, signal_length=128)
    >>> # is_ok = exporter.verify(model, onnx_path, modality="emg", n_channels=4,
    >>> #                          signal_length=128)
    """

    def __init__(self, opset: int = 17) -> None:
        self.opset = opset

    def export(
        self,
        model: FoundationModel,
        output_path: Path | str,
        modality: Modality | str,
        n_channels: int,
        signal_length: int,
    ) -> Path:
        """Export model to ONNX format.

        Parameters
        ----------
        model : FoundationModel
            The model to export.
        output_path : Path or str
            Destination ONNX file path.
        modality : Modality or str
            The modality (must be a single modality for export; ONNX trace
            is modality-specific).
        n_channels : int
            Number of input channels.
        signal_length : int
            Input signal length in samples.

        Returns
        -------
        Path
            The absolute path of the exported file.

        Raises
        ------
        ImportError
            If ``onnx`` is not installed.
        """
        try:
            import onnx  # noqa: F401 — used as availability check
        except ImportError as err:
            raise ImportError(
                "onnx is required for export. Install with: pip install onnx"
            ) from err

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mod_str = modality.value if isinstance(modality, Modality) else modality
        mod_idx = list(model.n_channels_per_modality.keys()).index(mod_str)

        model.eval()
        model.cpu()

        # Dummy input
        dummy_signal = torch.randn(1, n_channels, signal_length)
        dummy_modality = torch.tensor([mod_idx], dtype=torch.long)

        # Wrap model so ONNX export sees a single (signal, modality) -> (cls, patches) signature
        class _Wrapper(torch.nn.Module):
            def __init__(self, fm: FoundationModel) -> None:
                super().__init__()
                self.fm = fm

            def forward(self, signal, modality_idx):
                cls, patches = self.fm(signal, modality_idx)
                return cls, patches

        wrapped = _Wrapper(model)
        wrapped.eval()

        # Use the legacy (TorchScript-based) ONNX exporter via dynamo=False.
        # The new dynamo=True exporter has stricter requirements on
        # dynamic_shapes and does not accept dynamic_axes.
        torch.onnx.export(
            wrapped,
            (dummy_signal, dummy_modality),
            str(output_path),
            input_names=["signal", "modality_id"],
            output_names=["cls_token", "patch_tokens"],
            dynamic_axes={
                "signal": {0: "batch"},
                "modality_id": {0: "batch"},
                "cls_token": {0: "batch"},
                "patch_tokens": {0: "batch"},
            },
            opset_version=self.opset,
            do_constant_folding=True,
            dynamo=False,
        )

        # Write SHA-256 sidecar
        sha = compute_sha256(output_path)
        sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
        sidecar.write_text(f"{sha}  {output_path.name}\n", encoding="utf-8")

        return output_path

    def verify(
        self,
        pytorch_model: FoundationModel,
        onnx_path: Path | str,
        modality: Modality | str,
        n_channels: int,
        signal_length: int,
        n_samples: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> bool:
        """REAL numerical parity verification.

        Runs identical random inputs through both the PyTorch model and
        the ONNX Runtime model, and asserts that the maximum absolute
        difference is below ``atol`` and relative difference below ``rtol``.

        This is NOT a load-only test — it actually runs inference and
        compares outputs numerically.

        Parameters
        ----------
        pytorch_model : FoundationModel
            The original PyTorch model.
        onnx_path : Path or str
            Path to the exported ONNX file.
        modality : Modality or str
            Modality to test.
        n_channels : int
        signal_length : int
        n_samples : int
            Number of random test inputs. Default 100.
        atol : float
            Absolute tolerance. Default 1e-5.
        rtol : float
            Relative tolerance. Default 1e-4.

        Returns
        -------
        bool
            True if parity holds across all test inputs.

        Raises
        ------
        ImportError
            If ``onnxruntime`` is not installed.
        AssertionError
            If parity fails.
        """
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime is required for verification. Install with: pip install onnxruntime"
            ) from err

        onnx_path = Path(onnx_path).expanduser().resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        mod_str = modality.value if isinstance(modality, Modality) else modality
        mod_idx = list(pytorch_model.n_channels_per_modality.keys()).index(mod_str)

        # Set up ONNX Runtime session
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_names = [i.name for i in sess.get_inputs()]
        # input_names should be ["signal", "modality_id"]

        pytorch_model.eval()
        pytorch_model.cpu()

        max_diff = 0.0
        for i in range(n_samples):
            rng = np.random.default_rng(seed=i)
            signal_np = rng.standard_normal((1, n_channels, signal_length)).astype(np.float32)
            mod_id_np = np.array([mod_idx], dtype=np.int64)

            # PyTorch inference
            with torch.no_grad():
                pt_cls, pt_patches = pytorch_model(
                    torch.from_numpy(signal_np),
                    torch.from_numpy(mod_id_np),
                )
            pt_cls_np = pt_cls.cpu().numpy()
            pt_patches_np = pt_patches.cpu().numpy()

            # ONNX Runtime inference
            ort_inputs = {input_names[0]: signal_np, input_names[1]: mod_id_np}
            ort_out = sess.run(None, ort_inputs)
            ort_cls_np = ort_out[0]
            ort_patches_np = ort_out[1]

            # Compare
            diff_cls = np.max(np.abs(pt_cls_np - ort_cls_np))
            diff_patches = np.max(np.abs(pt_patches_np - ort_patches_np))
            max_diff = max(max_diff, diff_cls, diff_patches)

            # Relative check
            rel_cls = diff_cls / (np.max(np.abs(pt_cls_np)) + 1e-12)
            rel_patches = diff_patches / (np.max(np.abs(pt_patches_np)) + 1e-12)

            if diff_cls > atol or rel_cls > rtol:
                raise AssertionError(
                    f"Parity failed at sample {i}: CLS abs_diff={diff_cls:.2e} (atol={atol:.0e}), "
                    f"rel_diff={rel_cls:.2e} (rtol={rtol:.0e})"
                )
            if diff_patches > atol or rel_patches > rtol:
                raise AssertionError(
                    f"Parity failed at sample {i}: patches abs_diff={diff_patches:.2e} "
                    f"(atol={atol:.0e}), rel_diff={rel_patches:.2e} (rtol={rtol:.0e})"
                )

        return True
