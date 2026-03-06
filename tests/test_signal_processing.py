"""Tests for FFT-based signal processing utilities."""

import numpy as np
import pytest

from utils.signal_processing import (
    compute_dominant_frequency,
    compute_tremor_rejection_ratio,
    low_pass_filter,
)


class TestDominantFrequency:
    def test_detects_known_frequency(self) -> None:
        """Should detect a 5 Hz sinusoid."""
        sample_rate = 200.0
        t = np.arange(0, 2.0, 1.0 / sample_rate)
        signal = np.sin(2 * np.pi * 5.0 * t)

        freq = compute_dominant_frequency(signal, sample_rate)
        assert abs(freq - 5.0) < 1.0, f"Detected {freq} Hz, expected ~5 Hz"

    def test_respects_frequency_range(self) -> None:
        """Should ignore frequencies outside the specified range."""
        sample_rate = 200.0
        t = np.arange(0, 2.0, 1.0 / sample_rate)
        # 1 Hz signal (below default range of 3-12 Hz)
        signal = np.sin(2 * np.pi * 1.0 * t)

        freq = compute_dominant_frequency(signal, sample_rate, freq_range=(3.0, 12.0))
        # Should not return 1.0 Hz since it's outside the range
        assert freq != 1.0

    def test_empty_signal(self) -> None:
        assert compute_dominant_frequency(np.array([]), 200.0) == 0.0

    def test_single_sample(self) -> None:
        assert compute_dominant_frequency(np.array([1.0]), 200.0) == 0.0


class TestTremorRejectionRatio:
    def test_perfect_rejection(self) -> None:
        """Full tremor removal should give high dB ratio."""
        sample_rate = 200.0
        t = np.arange(0, 2.0, 1.0 / sample_rate)
        raw = np.sin(2 * np.pi * 5.0 * t)
        compensated = np.zeros_like(raw)

        ratio = compute_tremor_rejection_ratio(raw, compensated, sample_rate)
        assert ratio >= 50.0

    def test_no_rejection(self) -> None:
        """Same signal in and out should give ~0 dB."""
        sample_rate = 200.0
        t = np.arange(0, 2.0, 1.0 / sample_rate)
        signal = np.sin(2 * np.pi * 5.0 * t)

        ratio = compute_tremor_rejection_ratio(signal, signal, sample_rate)
        assert abs(ratio) < 1.0

    def test_empty_signal(self) -> None:
        assert compute_tremor_rejection_ratio(np.array([]), np.array([]), 200.0) == 0.0


class TestLowPassFilter:
    def test_removes_high_frequency(self) -> None:
        """Low-pass at 3 Hz should remove a 10 Hz component."""
        sample_rate = 200.0
        t = np.arange(0, 2.0, 1.0 / sample_rate)
        low = np.sin(2 * np.pi * 1.0 * t)
        high = 0.5 * np.sin(2 * np.pi * 10.0 * t)
        signal = low + high

        filtered = low_pass_filter(signal, cutoff_hz=3.0, sample_rate_hz=sample_rate)

        # Filtered signal should be closer to the low-frequency component
        error_with_filter = np.mean((filtered - low) ** 2)
        error_without_filter = np.mean((signal - low) ** 2)
        assert error_with_filter < error_without_filter

    def test_preserves_shape(self) -> None:
        signal = np.random.randn(100).astype(np.float32)
        filtered = low_pass_filter(signal, cutoff_hz=5.0, sample_rate_hz=200.0)
        assert filtered.shape == signal.shape

    def test_empty_signal(self) -> None:
        result = low_pass_filter(np.array([], dtype=np.float32), 5.0, 200.0)
        assert len(result) == 0
