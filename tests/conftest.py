"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible NumPy generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_emg_signal() -> np.ndarray:
    """Small EMG-like signal: 4 channels, 400 samples (2s @ 200 Hz)."""
    rng = np.random.default_rng(0)
    t = np.arange(400) / 200.0
    carrier = np.sin(2 * np.pi * 80 * t)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    signal = np.stack([carrier * envelope + 0.1 * rng.standard_normal(400) for _ in range(4)])
    return signal.astype(np.float32)


@pytest.fixture
def small_model_config():
    from biosignal_fm.config import ModelConfig

    return ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        patch_length=16,
        patch_stride=8,
        max_sequence_length=128,
    )
