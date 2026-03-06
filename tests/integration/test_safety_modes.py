"""Integration test: safety mode transitions during training."""

import numpy as np
import pytest

from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv


@pytest.mark.integration
class TestSafetyModes:
    def test_adaptive_mode_transition(self) -> None:
        """Verify adaptive mode transitions from soft to hard."""
        base = SurgicalTremorEnv()
        env = SafetySurgicalEnv(base)
        env.reset(seed=42)

        # Initially soft
        assert env.effective_mode == "soft"

        # Simulate reaching threshold
        env._total_steps = env.hard_threshold_steps + 1
        assert env.effective_mode == "transitioning"

        # Past transition window
        env._total_steps = env.hard_threshold_steps + env._transition_window + 1
        assert env.effective_mode == "hard"

        env.close()
