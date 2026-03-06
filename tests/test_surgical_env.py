"""Tests for the surgical tremor compensation environment."""

import gymnasium as gym
import numpy as np
import pytest

from env.surgical_env import SurgicalTremorEnv


@pytest.fixture
def env() -> SurgicalTremorEnv:
    e = SurgicalTremorEnv()
    yield e
    e.close()


class TestSurgicalEnv:
    def test_observation_space_shape(self, env: SurgicalTremorEnv) -> None:
        assert env.observation_space.shape == (18,)

    def test_action_space_shape(self, env: SurgicalTremorEnv) -> None:
        assert env.action_space.shape == (3,)

    def test_reset_returns_valid_obs(self, env: SurgicalTremorEnv) -> None:
        obs, info = env.reset(seed=42)
        assert obs.shape == (18,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self, env: SurgicalTremorEnv) -> None:
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (18,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_info_contains_metrics(self, env: SurgicalTremorEnv) -> None:
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "compensation_error_mm" in info
        assert "tissue_proximity_mm" in info
        assert "reward_tracking" in info
        assert "reward_smooth" in info
        assert "reward_safety" in info

    def test_episode_truncates_at_max_steps(self, env: SurgicalTremorEnv) -> None:
        env.reset(seed=42)
        for _ in range(env.max_steps - 1):
            _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
            assert not truncated
        _, _, _, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        assert truncated

    def test_action_clipping(self, env: SurgicalTremorEnv) -> None:
        """Large actions should be clipped to max_correction."""
        env.reset(seed=42)
        large_action = np.array([100.0, -100.0, 50.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(large_action)
        # Should not crash; action is clipped internally
        assert obs.shape == (18,)

    def test_human_feedback_injection(self, env: SurgicalTremorEnv) -> None:
        env.reset(seed=42)
        env.inject_human_feedback(1.0)
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["reward_human"] == 1.0

    def test_deterministic_with_same_seed(self) -> None:
        env1 = SurgicalTremorEnv()
        env2 = SurgicalTremorEnv()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env1.close()
        env2.close()
