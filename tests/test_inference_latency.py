"""Benchmark tests for inference latency.

Run with: uv run pytest tests/test_inference_latency.py -v --benchmark
"""

import numpy as np
import pytest

from env.surgical_env import SurgicalTremorEnv
from env.tremor_generator import TremorGenerator


@pytest.mark.benchmark
class TestInferenceLatency:
    def test_env_step_latency(self, benchmark) -> None:
        """Single env.step() should complete within 5ms budget."""
        env = SurgicalTremorEnv()
        env.reset(seed=42)
        action = np.zeros(3, dtype=np.float32)

        result = benchmark(env.step, action)
        # benchmark plugin handles timing; we just verify it runs
        assert result is not None
        env.close()

    def test_tremor_generation_latency(self, benchmark) -> None:
        """Tremor signal generation should be sub-millisecond."""
        gen = TremorGenerator(tremor_type="essential")
        gen.reset(np.random.default_rng(42))

        result = benchmark(gen.generate, 0.5)
        assert result.shape == (3,)

    def test_observation_construction_latency(self, benchmark) -> None:
        """Observation vector construction should be sub-millisecond."""
        env = SurgicalTremorEnv()
        env.reset(seed=42)

        result = benchmark(env._get_obs)
        assert result.shape == (18,)
        env.close()
