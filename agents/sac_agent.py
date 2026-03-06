"""SAC agent wrapper for surgical tremor compensation.

Phase 2: Uses Stable-Baselines3 SAC.
Phase 3: Will add custom SAC implementation from scratch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym


class SACAgent:
    """Wraps SB3 SAC with project-specific config and logging.

    All hyperparameters are loaded from config.yaml — nothing is hardcoded.
    """

    def __init__(
        self,
        env: gym.Env,
        config_path: str = "config.yaml",
    ) -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        sac_cfg = config["sac"]
        seed = config.get("seed", 42)

        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            ent_coef=sac_cfg["ent_coef"],
            target_entropy=sac_cfg["target_entropy"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            seed=seed + 1,
            verbose=0,
        )

    def train(self, total_timesteps: int, callback: BaseCallback | None = None) -> None:
        """Train the SAC agent."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, Any]:
        """Select an action given an observation."""
        return self.model.predict(obs, deterministic=deterministic)

    def save(self, path: str | Path) -> None:
        """Save the model to disk."""
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: gym.Env) -> SACAgent:
        """Load a saved model."""
        agent = cls.__new__(cls)
        agent.model = SAC.load(str(path), env=env)
        return agent
