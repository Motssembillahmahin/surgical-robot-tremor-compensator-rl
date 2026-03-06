"""Integration test: full training loop for a small number of steps."""

import pytest
import numpy as np

from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv


@pytest.mark.integration
class TestTrainingLoop:
    def test_env_runs_full_episode(self) -> None:
        """Environment should complete a full episode without errors."""
        base_env = SurgicalTremorEnv()
        env = SafetySurgicalEnv(base_env)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        assert obs.shape == (18,)
        assert isinstance(total_reward, float)
        env.close()

    def test_reward_components_are_finite(self) -> None:
        """All reward components should be finite numbers."""
        env = SurgicalTremorEnv()
        env.reset(seed=42)

        for _ in range(50):
            _, _, _, _, info = env.step(env.action_space.sample())
            for key in ["reward_tracking", "reward_smooth", "reward_safety", "reward_latency"]:
                assert np.isfinite(info[key]), f"{key} is not finite: {info[key]}"
        env.close()
