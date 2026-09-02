"""Unit tests for biosignal_fm.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from biosignal_fm.config import Modality, PreprocessingConfig
from biosignal_fm.preprocessing import (
    ModalityFilterBank,
    Patcher,
    PreprocessingPipeline,
    Resampler,
    SubjectAwareNormalizer,
    Windower,
)


class TestModalityFilterBank:
    def test_filter_2d(self, small_emg_signal) -> None:
        # Use a sampling rate high enough that EMG's 450 Hz high cutoff is
        # below Nyquist (need fs > 900 Hz). 2000 Hz matches NinaPro DB5 native.
        cfg = PreprocessingConfig()
        fb = ModalityFilterBank(cfg)
        out = fb.filter(small_emg_signal, Modality.EMG, sampling_rate_hz=2000)
        assert out.shape == small_emg_signal.shape
        assert out.dtype == np.float32

    def test_filter_1d(self) -> None:
        # Same Nyquist reasoning: use 2000 Hz.
        cfg = PreprocessingConfig()
        fb = ModalityFilterBank(cfg)
        sig = np.random.randn(1000).astype(np.float32)
        out = fb.filter(sig, Modality.EMG, sampling_rate_hz=2000)
        assert out.shape == (1000,)

    def test_rejects_3d(self) -> None:
        cfg = PreprocessingConfig()
        fb = ModalityFilterBank(cfg)
        sig = np.random.randn(4, 4, 100)
        with pytest.raises(ValueError):
            fb.filter(sig, Modality.EMG, sampling_rate_hz=200)

    def test_highpass_reduces_dc(self) -> None:
        cfg = PreprocessingConfig(emg_bandpass=(20.0, 90.0))
        fb = ModalityFilterBank(cfg)
        sig = np.ones((1, 1000), dtype=np.float32) + 0.1 * np.random.randn(1, 1000)
        out = fb.filter(sig, Modality.EMG, sampling_rate_hz=200)
        # DC component should be largely removed
        assert abs(out.mean()) < abs(sig.mean())


class TestResampler:
    def test_downsample(self) -> None:
        r = Resampler(target_sampling_rate_hz=100)
        sig = np.random.randn(4, 1000).astype(np.float32)
        out = r.resample(sig, source_sampling_rate_hz=200)
        assert out.shape == (4, 500)

    def test_upsample(self) -> None:
        r = Resampler(target_sampling_rate_hz=400)
        sig = np.random.randn(4, 1000).astype(np.float32)
        out = r.resample(sig, source_sampling_rate_hz=200)
        assert out.shape[0] == 4
        assert abs(out.shape[1] - 2000) < 50  # ~2x

    def test_no_resample(self) -> None:
        r = Resampler(target_sampling_rate_hz=200)
        sig = np.random.randn(4, 1000).astype(np.float32)
        out = r.resample(sig, source_sampling_rate_hz=200)
        assert out.shape == (4, 1000)

    def test_invalid_rate(self) -> None:
        r = Resampler(target_sampling_rate_hz=200)
        with pytest.raises(ValueError):
            r.resample(np.array([1.0]), source_sampling_rate_hz=0)


class TestSubjectAwareNormalizer:
    def test_fit_transform(self) -> None:
        norm = SubjectAwareNormalizer(modality=Modality.EMG)
        signals = [np.random.randn(4, 1000) for _ in range(5)]
        norm.fit(signals)
        out = norm.transform(signals[0])
        # After normalization, mean should be near zero
        assert abs(out.mean()) < 0.1

    def test_not_fitted_raises(self) -> None:
        norm = SubjectAwareNormalizer(modality=Modality.EMG)
        with pytest.raises(RuntimeError):
            norm.transform(np.random.randn(4, 100))

    def test_channel_mismatch_raises(self) -> None:
        norm = SubjectAwareNormalizer(modality=Modality.EMG)
        norm.fit([np.random.randn(4, 100)])
        with pytest.raises(ValueError, match="channels"):
            norm.transform(np.random.randn(8, 100))

    def test_serialization(self) -> None:
        norm = SubjectAwareNormalizer(modality=Modality.EMG)
        norm.fit([np.random.randn(4, 100)])
        d = norm.to_dict()
        norm2 = SubjectAwareNormalizer.from_dict(d)
        sig = np.random.randn(4, 100)
        np.testing.assert_array_almost_equal(norm.transform(sig), norm2.transform(sig))


class TestWindower:
    def test_basic(self) -> None:
        w = Windower(window_length_samples=100, overlap_samples=50)
        sig = np.random.randn(4, 1000)
        windows = w.window(sig)
        assert len(windows) == 19  # (1000 - 100) / 50 + 1 = 19
        assert windows[0].shape == (4, 100)

    def test_signal_too_short(self) -> None:
        w = Windower(window_length_samples=100, overlap_samples=0)
        sig = np.random.randn(4, 50)
        assert len(w.window(sig)) == 0

    def test_overlap_too_large(self) -> None:
        with pytest.raises(ValueError):
            Windower(window_length_samples=100, overlap_samples=100)


class TestPatcher:
    def test_n_patches(self) -> None:
        p = Patcher(patch_length=32, stride=16)
        assert p.n_patches(64) == 3  # (64-32)/16 + 1 = 3
        assert p.n_patches(32) == 1
        assert p.n_patches(16) == 0

    def test_extract_1d(self) -> None:
        p = Patcher(patch_length=32, stride=16)
        sig = np.random.randn(64)
        patches = p.extract(sig)
        assert patches.shape == (3, 32)

    def test_extract_2d(self) -> None:
        p = Patcher(patch_length=32, stride=16)
        sig = np.random.randn(4, 64)
        patches = p.extract(sig)
        assert patches.shape == (3, 4, 32)


class TestPreprocessingPipeline:
    def test_fit_transform(self) -> None:
        cfg = PreprocessingConfig(target_sampling_rate_hz=200)
        pipe = PreprocessingPipeline(config=cfg, modality=Modality.EMG)
        signals = [np.random.randn(4, 4000) for _ in range(3)]  # 2s @ 2kHz
        pipe.fit(signals, source_sampling_rate_hz=2000)
        out = pipe.transform(signals[0], source_sampling_rate_hz=2000)
        # Should be resampled from 2kHz to 200 Hz: 4000 -> 400
        assert out.shape == (4, 400)
        assert out.dtype == np.float32

    def test_not_fitted_raises(self) -> None:
        cfg = PreprocessingConfig()
        pipe = PreprocessingPipeline(config=cfg, modality=Modality.EMG)
        with pytest.raises(RuntimeError):
            pipe.transform(np.random.randn(4, 1000), source_sampling_rate_hz=2000)


class TestCanonicalSignalPreprocessing:
    def test_with_data_records_processing_history_and_preserves_time_basis(self) -> None:
        from biosignal_fm.core import (
            DataOrigin,
            Signal,
            SignalMetadata,
            SignalProcessingStep,
            SignalProvenance,
        )

        signal = Signal(
            data=np.ones((2, 8), dtype=np.float32),
            metadata=SignalMetadata(
                modality="emg",
                sampling_rate_hz=100.0,
                channel_names=("E1", "E2"),
                provenance=SignalProvenance(origin=DataOrigin.REAL),
            ),
            timestamps_seconds=np.arange(8, dtype=np.float64) / 100.0,
            missing_mask=np.zeros((2, 8), dtype=bool),
        )
        processed = signal.with_data(
            signal.data * 2,
            processing_step=SignalProcessingStep(
                name="unit-test-filter",
                version="v1",
                config_hash="a" * 64,
            ),
            preprocessing_status="preprocessed",
        )
        assert processed.metadata.preprocessing_status == "preprocessed"
        assert processed.timestamps_seconds is not None
        np.testing.assert_array_equal(processed.timestamps_seconds, signal.timestamps_seconds)
        assert processed.missing_mask is not None
        assert len(processed.metadata.provenance.processing_history) == 1
        assert processed.metadata.provenance.processing_history[0].name == "unit-test-filter"

    def test_shape_change_requires_explicit_time_and_missingness_policy(self) -> None:
        from biosignal_fm.core import Signal, SignalMetadata

        signal = Signal(
            data=np.ones((1, 8), dtype=np.float32),
            metadata=SignalMetadata(modality="ecg", sampling_rate_hz=100.0, channel_names=("I",)),
            timestamps_seconds=np.arange(8, dtype=np.float64) / 100.0,
        )
        with pytest.raises(ValueError, match="replacement timestamps_seconds"):
            signal.with_data(np.ones((1, 4), dtype=np.float32), sampling_rate_hz=50.0)

    def test_transform_signal_records_fitted_pipeline_provenance(self) -> None:
        from biosignal_fm.core import Signal, SignalMetadata

        rng = np.random.default_rng(7)
        raw_signals = [rng.normal(size=(2, 4000)).astype(np.float32) for _ in range(2)]
        pipeline = PreprocessingPipeline(
            PreprocessingConfig(target_sampling_rate_hz=200), Modality.EMG
        )
        pipeline.fit(raw_signals, source_sampling_rate_hz=2000)
        signal = Signal(
            data=raw_signals[0],
            metadata=SignalMetadata(
                modality="emg", sampling_rate_hz=2000.0, channel_names=("E1", "E2")
            ),
            timestamps_seconds=np.arange(4000, dtype=np.float64) / 2000.0,
        )
        processed = pipeline.transform_signal(signal)
        assert processed.metadata.sampling_rate_hz == 200.0
        assert processed.metadata.preprocessing_status == "preprocessed"
        assert processed.data.shape == (2, 400)
        assert processed.timestamps_seconds is not None
        assert len(processed.metadata.provenance.processing_history) == 1
        assert processed.metadata.provenance.processing_history[0].name == "emg_preprocessing"
