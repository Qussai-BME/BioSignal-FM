"""Tests for the classical baselines module (v3.0)."""

from __future__ import annotations

import numpy as np
import pytest


class TestTDFeatureExtraction:
    """Verify Hudgins time-domain feature extraction."""

    def test_shape_2d_input(self) -> None:
        """2D input (C, T) returns 1D feature vector of shape (C*4,)."""
        from biosignal_fm.baselines import extract_td_features

        signal = np.random.randn(8, 400)
        feats = extract_td_features(signal)
        assert feats.shape == (32,)  # 8 channels * 4 features

    def test_shape_3d_input(self) -> None:
        """3D input (N, C, T) returns 2D feature matrix of shape (N, C*4)."""
        from biosignal_fm.baselines import extract_td_features

        signals = np.random.randn(10, 8, 400)
        feats = extract_td_features(signals)
        assert feats.shape == (10, 32)

    def test_invalid_shape_raises(self) -> None:
        from biosignal_fm.baselines import extract_td_features

        with pytest.raises(ValueError, match="2D or 3D"):
            extract_td_features(np.random.randn(5))  # 1D

    def test_mav_is_nonneg(self) -> None:
        """MAV (mean absolute value) must be non-negative."""
        from biosignal_fm.baselines import extract_td_features

        signal = np.random.randn(4, 100)
        feats = extract_td_features(signal)
        # MAV is at indices 0, 4, 8, 12
        for c in range(4):
            assert feats[c * 4] >= 0, f"MAV for channel {c} is negative"

    def test_wl_is_nonneg(self) -> None:
        """Waveform length must be non-negative."""
        from biosignal_fm.baselines import extract_td_features

        signal = np.random.randn(4, 100)
        feats = extract_td_features(signal)
        for c in range(4):
            assert feats[c * 4 + 3] >= 0, f"WL for channel {c} is negative"

    def test_zero_signal(self) -> None:
        """A zero signal should produce zero MAV, zero WL, zero ZC, zero SSC."""
        from biosignal_fm.baselines import extract_td_features

        signal = np.zeros((4, 100))
        feats = extract_td_features(signal)
        assert np.allclose(feats, 0.0)

    def test_constant_signal(self) -> None:
        """A constant non-zero signal should have zero ZC and zero SSC."""
        from biosignal_fm.baselines import extract_td_features

        signal = np.ones((1, 100)) * 0.5
        feats = extract_td_features(signal)
        assert feats[0] == 0.5  # MAV = 0.5
        assert feats[1] == 0  # ZC = 0 (no sign changes)
        assert feats[2] == 0  # SSC = 0
        assert feats[3] == 0  # WL = 0 (no changes)


class TestLDABaseline:
    """Verify the LDA + TD baseline."""

    def test_fit_predict(self) -> None:
        from biosignal_fm.baselines import LDATDBaseline, extract_td_features

        # Build a simple separable dataset
        rng = np.random.default_rng(42)
        n = 40
        signals = np.zeros((n, 4, 100))
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels[i] = i % 2
            signals[i] = rng.standard_normal((4, 100)) + labels[i] * 2.0

        X = extract_td_features(signals)
        clf = LDATDBaseline()
        clf.fit(X, labels)
        preds = clf.predict(X)
        # Should achieve high accuracy on training data
        acc = float(np.mean(preds == labels))
        assert acc > 0.9, f"LDA training accuracy too low: {acc}"

    def test_predict_before_fit_raises(self) -> None:
        from biosignal_fm.baselines import LDATDBaseline

        clf = LDATDBaseline()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.random.randn(5, 32))


class TestSVMBaseline:
    """Verify the SVM + TD baseline."""

    def test_fit_predict(self) -> None:
        from biosignal_fm.baselines import SVMTDBaseline, extract_td_features

        rng = np.random.default_rng(42)
        n = 40
        signals = np.zeros((n, 4, 100))
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels[i] = i % 2
            signals[i] = rng.standard_normal((4, 100)) + labels[i] * 3.0

        X = extract_td_features(signals)
        clf = SVMTDBaseline()
        clf.fit(X, labels)
        preds = clf.predict(X)
        acc = float(np.mean(preds == labels))
        assert acc > 0.9


class TestRandomForestBaseline:
    """Verify the Random Forest + TD baseline."""

    def test_fit_predict(self) -> None:
        from biosignal_fm.baselines import RandomForestTDBaseline, extract_td_features

        rng = np.random.default_rng(42)
        n = 40
        signals = np.zeros((n, 4, 100))
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            labels[i] = i % 2
            signals[i] = rng.standard_normal((4, 100)) + labels[i] * 2.0

        X = extract_td_features(signals)
        clf = RandomForestTDBaseline(n_estimators=20)
        clf.fit(X, labels)
        preds = clf.predict(X)
        acc = float(np.mean(preds == labels))
        assert acc > 0.85


class TestCNN1DBaseline:
    """Verify the CNN1D baseline."""

    def test_fit_predict(self) -> None:
        from biosignal_fm.baselines import CNN1DBaseline

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

        clf = CNN1DBaseline(
            n_channels=n_channels,
            n_classes=2,
            signal_length=signal_length,
            epochs=5,
            batch_size=8,
        )
        clf.fit(signals, labels)
        preds = clf.predict(signals)
        assert preds.shape == (n,)
        # Should do better than chance
        acc = float(np.mean(preds == labels))
        assert acc > 0.6, f"CNN1D accuracy too low: {acc}"

    def test_invalid_input_shape(self) -> None:
        from biosignal_fm.baselines import CNN1DBaseline

        clf = CNN1DBaseline(n_channels=4, n_classes=2, signal_length=64, epochs=1)
        clf.fit(np.random.randn(10, 4, 64).astype(np.float32), np.zeros(10, dtype=int))
        with pytest.raises(ValueError, match="3D"):
            clf.predict(np.random.randn(10, 64).astype(np.float32))


class TestRunBaselineLOSO:
    """Verify the LOSO evaluation helper."""

    def test_loso_returns_correct_structure(self) -> None:
        from biosignal_fm.baselines import LDATDBaseline, run_baseline_loso

        rng = np.random.default_rng(42)
        n_subjects = 4
        n_per_subj = 10
        signals = np.zeros((n_subjects * n_per_subj, 4, 100))
        labels = np.zeros(n_subjects * n_per_subj, dtype=int)
        subjects = np.zeros(n_subjects * n_per_subj, dtype=int)
        for s in range(n_subjects):
            for i in range(n_per_subj):
                idx = s * n_per_subj + i
                labels[idx] = i % 2
                subjects[idx] = s
                signals[idx] = rng.standard_normal((4, 100)) + labels[idx] * 2.0

        result = run_baseline_loso(LDATDBaseline(), signals, labels, subjects, use_td_features=True)
        assert result["baseline_name"] == "LDA+TD"
        assert len(result["fold_accuracies"]) == n_subjects
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        assert result["std_accuracy"] >= 0
        assert len(result["per_fold_predictions"]) == n_subjects
