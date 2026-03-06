"""Tests for safety constraint wrapper."""

import numpy as np
import pytest

from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv


@pytest.fixture
def safe_env() -> SafetySurgicalEnv:
    base = SurgicalTremorEnv()
    env = SafetySurgicalEnv(base)
    env.reset(seed=42)
    yield env
    env.close()


class TestSafetyConstraints:
    def test_step_returns_valid_output(self, safe_env: SafetySurgicalEnv) -> None:
        obs, reward, terminated, truncated, info = safe_env.step(
            np.zeros(3, dtype=np.float32)
        )
        assert obs.shape == (18,)
        assert isinstance(reward, float)
        assert "safety_mode" in info

    def test_info_contains_safety_mode(self, safe_env: SafetySurgicalEnv) -> None:
        _, _, _, _, info = safe_env.step(np.zeros(3, dtype=np.float32))
        assert info["safety_mode"] in ("hard", "soft", "transitioning")

    def test_adaptive_starts_soft(self, safe_env: SafetySurgicalEnv) -> None:
        assert safe_env.effective_mode == "soft"

    def test_adaptive_transitions_to_hard(self) -> None:
        base = SurgicalTremorEnv()
        env = SafetySurgicalEnv(base)
        env._total_steps = env.hard_threshold_steps + env._transition_window + 1
        assert env.effective_mode == "hard"
        env.close()

    def test_adaptive_transitioning_phase(self) -> None:
        base = SurgicalTremorEnv()
        env = SafetySurgicalEnv(base)
        env._total_steps = env.hard_threshold_steps + 1
        assert env.effective_mode == "transitioning"
        env.close()

    def test_hard_mode_projects_action(self) -> None:
        """In hard mode, dangerous actions should be projected to safe ones."""
        base = SurgicalTremorEnv()
        env = SafetySurgicalEnv(base)
        env.mode = "hard"
        env.reset(seed=42)
        # Step should not crash
        obs, _, _, _, info = env.step(np.array([100.0, 100.0, 100.0], dtype=np.float32))
        assert obs.shape == (18,)
        assert info["safety_mode"] == "hard"
        env.close()

    def test_step_counter_persists_across_episodes(
        self, safe_env: SafetySurgicalEnv
    ) -> None:
        safe_env.step(np.zeros(3, dtype=np.float32))
        steps_before = safe_env._total_steps
        safe_env.reset(seed=43)
        assert safe_env._total_steps == steps_before
