"""Physiologically accurate tremor signal generator.

Generates synthetic hand tremor signals based on medical literature:
- Essential tremor:   4-8 Hz
- Parkinson's tremor: 3-6 Hz
- Physiological:      8-12 Hz
"""

from __future__ import annotations

import numpy as np
import yaml


def load_tremor_profiles(config_path: str = "config.yaml") -> dict:
    """Load tremor profiles from config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["tremor"]["profiles"]


class TremorGenerator:
    """Generates coherent tremor signals with per-episode random phases.

    The phase for each frequency component is sampled once at reset,
    ensuring a smooth, realistic tremor waveform throughout the episode
    rather than random noise.
    """

    def __init__(
        self,
        tremor_type: str = "essential",
        config_path: str = "config.yaml",
    ) -> None:
        profiles = load_tremor_profiles(config_path)
        if tremor_type not in profiles:
            raise ValueError(
                f"Unknown tremor type '{tremor_type}'. "
                f"Available: {list(profiles.keys())}"
            )
        profile = profiles[tremor_type]
        self.tremor_type = tremor_type
        self.frequencies: np.ndarray = np.array(profile["frequencies"], dtype=np.float32)
        self.amplitudes: np.ndarray = np.array(profile["amplitudes"], dtype=np.float32)
        self.phases: np.ndarray = np.zeros_like(self.frequencies)

    def reset(self, rng: np.random.Generator) -> None:
        """Sample random phases once per episode for coherent tremor."""
        self.phases = rng.uniform(0, 2 * np.pi, size=len(self.frequencies)).astype(
            np.float32
        )

    def generate(self, t: float) -> np.ndarray:
        """Generate 3D tremor signal at time t.

        Args:
            t: Current time in seconds.

        Returns:
            Tremor displacement vector of shape (3,) in mm.
            Each axis gets the same base signal with slight phase offsets.
        """
        # Base 1D tremor signal (superposition of sinusoids)
        signal_1d = np.sum(
            self.amplitudes
            * np.sin(2 * np.pi * self.frequencies * t + self.phases)
        )

        # Distribute across 3 axes with slight decorrelation
        tremor_3d = np.array(
            [
                signal_1d,
                signal_1d * 0.8,  # y-axis slightly lower amplitude
                signal_1d * 0.6,  # z-axis even lower (gravity axis)
            ],
            dtype=np.float32,
        )
        return tremor_3d

    @property
    def dominant_frequency(self) -> float:
        """Return the frequency with the highest amplitude."""
        idx = np.argmax(self.amplitudes)
        return float(self.frequencies[idx])
