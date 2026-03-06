"""Safety constraint wrapper for the surgical environment.

Supports three modes:
- hard:     Always projects actions onto safe manifold
- soft:     Allows violations but penalises them in reward
- adaptive: Starts soft, transitions to hard after threshold steps
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import yaml


class SafetySurgicalEnv(gym.Wrapper):
    """Intercepts actions before execution to enforce tissue safety.

    In hard mode, any action that would move the robot tip closer to the
    tissue boundary than safety_margin_mm is projected onto the safe manifold.
    This is non-negotiable in production — the agent cannot learn to violate
    tissue safety.
    """

    def __init__(
        self,
        env: gym.Env,
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__(env)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        safety_cfg = config["safety"]
        self.mode = safety_cfg["mode"]
        self.safety_margin_mm = safety_cfg["safety_margin_mm"]
        self.soft_penalty_weight = safety_cfg["soft_penalty_weight"]
        self.hard_threshold_steps = safety_cfg["hard_threshold_steps"]

        self._total_steps = 0
        self._transition_window = 10_000  # Gradual soft→hard over this many steps

    @property
    def effective_mode(self) -> str:
        """Determine current mode based on step count for adaptive mode."""
        if self.mode != "adaptive":
            return self.mode
        if self._total_steps >= self.hard_threshold_steps + self._transition_window:
            return "hard"
        if self._total_steps >= self.hard_threshold_steps:
            return "transitioning"
        return "soft"

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply safety projection then step the wrapped environment."""
        mode = self.effective_mode

        if mode == "hard":
            safe_action = self._project_to_safe_action(action)
        elif mode == "transitioning":
            # Linear interpolation: gradually increase projection strength
            progress = (
                self._total_steps - self.hard_threshold_steps
            ) / self._transition_window
            projected = self._project_to_safe_action(action)
            safe_action = (1 - progress) * action + progress * projected
            safe_action = safe_action.astype(np.float32)
        else:
            # Soft mode: allow the action through
            safe_action = action

        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self._total_steps += 1

        # In soft mode, add penalty to reward instead of clipping action
        if mode == "soft" or mode == "transitioning":
            tissue_prox = info.get("tissue_proximity_mm", float("inf"))
            if tissue_prox < self.safety_margin_mm:
                violation_depth = self.safety_margin_mm - tissue_prox
                soft_penalty = -self.soft_penalty_weight * violation_depth
                reward += soft_penalty
                info["soft_safety_penalty"] = soft_penalty

        info["safety_mode"] = mode
        info["action_was_projected"] = mode == "hard" or (
            mode == "transitioning" and not np.allclose(action, safe_action)
        )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the wrapped environment. Step counter persists across episodes."""
        return self.env.reset(**kwargs)

    def _project_to_safe_action(self, action: np.ndarray) -> np.ndarray:
        """Project action onto safe manifold using tissue proximity gradient.

        If the current robot tip position plus the proposed action would bring
        it within safety_margin_mm of the tissue boundary, scale back the
        component of the action pointing toward the tissue.
        """
        # Get current robot tip position from the unwrapped env
        robot_pos = self.env._robot_tip_pos  # type: ignore[attr-defined]
        tissue_boundary = self.env.tissue_boundary  # type: ignore[attr-defined]

        # Predicted position after action
        predicted_pos = robot_pos + action

        # Vector from predicted position to tissue boundary
        to_tissue = tissue_boundary - predicted_pos
        distance = float(np.linalg.norm(to_tissue))

        if distance >= self.safety_margin_mm:
            return action  # Already safe

        # Project out the component of action pointing toward tissue
        if distance < 1e-8:
            # At tissue boundary: zero out action
            return np.zeros_like(action)

        # Direction toward tissue
        tissue_dir = to_tissue / distance

        # Remove the component of action that moves toward tissue
        action_toward_tissue = np.dot(action, tissue_dir) * tissue_dir
        safe_action = action - action_toward_tissue

        # Scale to maintain safety margin
        scale = max(0.0, (distance - 0.1) / self.safety_margin_mm)
        return (safe_action * scale).astype(np.float32)
