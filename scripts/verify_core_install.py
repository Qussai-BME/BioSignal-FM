"""Verify that the V4 core installs and operates without optional heavy extras."""

from __future__ import annotations

import biosignal_fm
import numpy as np
from biosignal_fm.core import DataOrigin, Signal, SignalMetadata, SignalProvenance
from biosignal_fm.modalities import default_registry


def main() -> None:
    registry = default_registry()
    signal = Signal(
        data=np.array([[0.0, 1.0, 0.5], [0.2, 0.3, 0.4]], dtype=np.float32),
        metadata=SignalMetadata(
            modality="emg",
            sampling_rate_hz=100.0,
            channel_names=("EMG-1", "EMG-2"),
            provenance=SignalProvenance(
                origin=DataOrigin.SYNTHETIC,
                fallback_reason="clean-install verification",
            ),
        ),
    )
    assert biosignal_fm.__version__ == "4.0.0"
    assert registry.identifiers() == ("emg", "eeg", "ecg", "ecog", "fnirs")
    assert signal.is_synthetic
    print("core-install-ok")
    print(f"version={biosignal_fm.__version__}")
    print(f"modalities={','.join(registry.identifiers())}")


if __name__ == "__main__":
    main()
