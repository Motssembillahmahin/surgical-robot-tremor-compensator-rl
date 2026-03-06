"""Tests for reward function components in isolation."""

import numpy as np
import pytest

from env.surgical_env import SurgicalTremorEnv


@pytest.fixture
def env() -> SurgicalTremorEnv:
    e = SurgicalTremorEnv()
    e.reset(seed=42)
    yield e
    e.close()


class TestRewardFunction:
    def test_zero_action_gives_nonzero_reward(self, env: SurgicalTremorEnv) -> None:
        """Zero correction should still yield a reward from tracking error."""
        _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        assert reward != 0.0

    def test_tracking_reward_is_negative(self, env: SurgicalTremorEnv) -> None:
        """Tracking reward should be negative (distance-based penalty)."""
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["reward_tracking"] <= 0.0

    def test_smoothness_penalty_zero_for_consistent_action(
        self, env: SurgicalTremorEnv
    ) -> None:
        """First step with zero action should have zero smoothness penalty."""
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["reward_smooth"] == 0.0

    def test_smoothness_penalty_increases_with_jerk(
        self, env: SurgicalTremorEnv
    ) -> None:
        """Changing action rapidly should increase smoothness penalty."""
        env.step(np.zeros(3, dtype=np.float32))
        _, _, _, _, info_small = env.step(np.array([0.1, 0.0, 0.0], dtype=np.float32))
        env.reset(seed=42)
        env.step(np.zeros(3, dtype=np.float32))
        _, _, _, _, info_large = env.step(np.array([2.0, 0.0, 0.0], dtype=np.float32))
        assert info_large["reward_smooth"] < info_small["reward_smooth"]

    def test_safety_penalty_far_from_tissue(self, env: SurgicalTremorEnv) -> None:
        """No safety penalty when far from tissue boundary."""
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["reward_safety"] == 0.0

    def test_human_feedback_default_zero(self, env: SurgicalTremorEnv) -> None:
        """Human feedback should be 0 when not injected."""
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["reward_human"] == 0.0

    def test_reward_is_sum_of_components(self, env: SurgicalTremorEnv) -> None:
        """Total reward should equal sum of all components."""
        _, reward, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        component_sum = (
            info["reward_tracking"]
            + info["reward_smooth"]
            + info["reward_safety"]
            + info["reward_latency"]
            + info["reward_human"]
        )
        assert abs(reward - component_sum) < 1e-6
