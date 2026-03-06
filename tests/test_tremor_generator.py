"""Tests for tremor signal generator."""

import numpy as np
import pytest

from env.tremor_generator import TremorGenerator


@pytest.fixture
def essential_tremor() -> TremorGenerator:
    gen = TremorGenerator(tremor_type="essential")
    gen.reset(np.random.default_rng(42))
    return gen


@pytest.fixture
def parkinsons_tremor() -> TremorGenerator:
    gen = TremorGenerator(tremor_type="parkinsons")
    gen.reset(np.random.default_rng(42))
    return gen


@pytest.fixture
def physiological_tremor() -> TremorGenerator:
    gen = TremorGenerator(tremor_type="physiological")
    gen.reset(np.random.default_rng(42))
    return gen


class TestTremorGenerator:
    def test_output_shape(self, essential_tremor: TremorGenerator) -> None:
        signal = essential_tremor.generate(0.0)
        assert signal.shape == (3,)

    def test_output_dtype(self, essential_tremor: TremorGenerator) -> None:
        signal = essential_tremor.generate(0.0)
        assert signal.dtype == np.float32

    def test_essential_dominant_frequency(self, essential_tremor: TremorGenerator) -> None:
        freq = essential_tremor.dominant_frequency
        assert 4.0 <= freq <= 8.0, f"Essential tremor dominant freq {freq} outside 4-8 Hz"

    def test_parkinsons_dominant_frequency(self, parkinsons_tremor: TremorGenerator) -> None:
        freq = parkinsons_tremor.dominant_frequency
        assert 3.0 <= freq <= 6.0, f"Parkinson's dominant freq {freq} outside 3-6 Hz"

    def test_physiological_dominant_frequency(
        self, physiological_tremor: TremorGenerator
    ) -> None:
        freq = physiological_tremor.dominant_frequency
        assert 8.0 <= freq <= 12.0, f"Physiological dominant freq {freq} outside 8-12 Hz"

    def test_amplitude_range(self, essential_tremor: TremorGenerator) -> None:
        """Tremor signal should be within reasonable amplitude range."""
        signals = [essential_tremor.generate(t * 0.005) for t in range(1000)]
        max_amplitude = max(np.max(np.abs(s)) for s in signals)
        assert max_amplitude < 1.0, f"Max amplitude {max_amplitude} exceeds 1mm"

    def test_phase_coherence(self, essential_tremor: TremorGenerator) -> None:
        """Same time should produce same signal (phases fixed per episode)."""
        s1 = essential_tremor.generate(0.5)
        s2 = essential_tremor.generate(0.5)
        np.testing.assert_array_equal(s1, s2)

    def test_reset_changes_phases(self) -> None:
        """Resetting with different seed should produce different signals."""
        gen = TremorGenerator(tremor_type="essential")
        gen.reset(np.random.default_rng(1))
        s1 = gen.generate(0.5)
        gen.reset(np.random.default_rng(2))
        s2 = gen.generate(0.5)
        assert not np.allclose(s1, s2)

    def test_invalid_tremor_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown tremor type"):
            TremorGenerator(tremor_type="nonexistent")

    def test_signal_varies_over_time(self, essential_tremor: TremorGenerator) -> None:
        """Signal should change over time (not constant)."""
        s1 = essential_tremor.generate(0.0)
        s2 = essential_tremor.generate(0.1)
        assert not np.allclose(s1, s2)
