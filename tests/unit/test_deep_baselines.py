"""Tests for the deep baselines: EEGNet and ResNet1D (v3.1)."""

from __future__ import annotations

import numpy as np
import pytest


class TestEEGNetBaseline:
    """Verify the EEGNet baseline."""

    def test_fit_predict(self) -> None:
        from biosignal_fm.baselines import EEGNetBaseline

        rng = np.random.default_rng(42)
        n = 20
        n_channels = 4
        signal_length = 64
        signals = np.zeros((n, n_channels, signal_length), dtype=np.float32)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels[i] = i % 2
            signals[i] = rng.standard_normal((n_channels, signal_length)).astype(np.float32)
            if labels[i] == 1:
                signals[i] += 1.0

        clf = EEGNetBaseline(
            n_channels=n_channels,
            n_classes=2,
            signal_length=signal_length,
            epochs=5,
            batch_size=8,
        )
        clf.fit(signals, labels)
        preds = clf.predict(signals)
        assert preds.shape == (n,)
        acc = float(np.mean(preds == labels))
        assert acc > 0.6, f"EEGNet accuracy too low: {acc}"

    def test_predict_before_fit_raises(self) -> None:
        from biosignal_fm.baselines import EEGNetBaseline

        clf = EEGNetBaseline(n_channels=4, n_classes=2, signal_length=64)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.random.randn(5, 4, 64).astype(np.float32))

    def test_invalid_input_shape(self) -> None:
        from biosignal_fm.baselines import EEGNetBaseline

        clf = EEGNetBaseline(n_channels=4, n_classes=2, signal_length=64, epochs=1)
        clf.fit(np.random.randn(10, 4, 64).astype(np.float32), np.zeros(10, dtype=int))
        # 2D input should fail (EEGNet expects 3D)
        with pytest.raises((ValueError, RuntimeError)):
            clf.predict(np.random.randn(10, 64).astype(np.float32))


class TestResNet1DBaseline:
    """Verify the ResNet1D baseline."""

    def test_fit_predict(self) -> None:
        from biosignal_fm.baselines import ResNet1DBaseline

        rng = np.random.default_rng(42)
        n = 20
        n_channels = 4
        signal_length = 64
        signals = np.zeros((n, n_channels, signal_length), dtype=np.float32)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels[i] = i % 2
            signals[i] = rng.standard_normal((n_channels, signal_length)).astype(np.float32)
            if labels[i] == 1:
                signals[i] += 1.0

        clf = ResNet1DBaseline(
            n_channels=n_channels,
            n_classes=2,
            signal_length=signal_length,
            epochs=5,
            batch_size=8,
        )
        clf.fit(signals, labels)
        preds = clf.predict(signals)
        assert preds.shape == (n,)
        acc = float(np.mean(preds == labels))
        assert acc > 0.6, f"ResNet1D accuracy too low: {acc}"

    def test_predict_before_fit_raises(self) -> None:
        from biosignal_fm.baselines import ResNet1DBaseline

        clf = ResNet1DBaseline(n_channels=4, n_classes=2, signal_length=64)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.random.randn(5, 4, 64).astype(np.float32))


class TestBaselinesModule:
    """Verify the baselines module exports all 6 baselines."""

    def test_all_baselines_importable(self) -> None:
        from biosignal_fm.baselines import (
            CNN1DBaseline,
            EEGNetBaseline,
            LDATDBaseline,
            RandomForestTDBaseline,
            ResNet1DBaseline,
            SVMTDBaseline,
        )

        # All should be classes
        for cls in [
            LDATDBaseline,
            SVMTDBaseline,
            RandomForestTDBaseline,
            CNN1DBaseline,
            EEGNetBaseline,
            ResNet1DBaseline,
        ]:
            assert isinstance(cls, type)

    def test_all_baselines_have_names(self) -> None:
        from biosignal_fm.baselines import (
            CNN1DBaseline,
            EEGNetBaseline,
            LDATDBaseline,
            RandomForestTDBaseline,
            ResNet1DBaseline,
            SVMTDBaseline,
        )

        names = {
            LDATDBaseline.name,
            SVMTDBaseline.name,
            RandomForestTDBaseline.name,
            CNN1DBaseline.name,
            EEGNetBaseline.name,
            ResNet1DBaseline.name,
        }
        assert len(names) == 6  # all unique
        assert "LDA+TD" in names
        assert "EEGNet" in names
        assert "ResNet1D" in names
