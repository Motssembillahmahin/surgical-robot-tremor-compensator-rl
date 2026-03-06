"""Main training entrypoint for the surgical tremor compensator.

Usage:
    uv run train.py
    uv run train.py --config config.yaml
    uv run train.py --resume checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from stable_baselines3.common.callbacks import BaseCallback

from agents.sac_agent import SACAgent
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv
from utils.logger import TrainingLogger


class MetricsCallback(BaseCallback):
    """SB3 callback that logs env metrics to TensorBoard and tracks progress."""

    def __init__(
        self,
        logger_obj: TrainingLogger,
        log_freq: int = 100,
        eval_freq: int = 10_000,
        checkpoint_dir: str = "checkpoints/",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._logger = logger_obj
        self._log_freq = log_freq
        self._eval_freq = eval_freq
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Rolling tracking
        self._episode_rewards: list[float] = []
        self._episode_errors: list[float] = []
        self._episode_lengths: list[int] = []
        self._current_ep_reward = 0.0
        self._current_ep_errors: list[float] = []
        self._current_ep_len = 0
        self._best_error = float("inf")
        self._episodes_done = 0

    def _on_step(self) -> bool:
        # Accumulate per-step info
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])

        for info, reward in zip(infos, rewards):
            self._current_ep_reward += reward
            self._current_ep_len += 1
            if "compensation_error_mm" in info:
                self._current_ep_errors.append(info["compensation_error_mm"])

        # Check for episode done
        dones = self.locals.get("dones", [False])
        for i, done in enumerate(dones):
            if done:
                self._episodes_done += 1
                self._episode_rewards.append(self._current_ep_reward)
                avg_error = (
                    sum(self._current_ep_errors) / len(self._current_ep_errors)
                    if self._current_ep_errors
                    else 0.0
                )
                self._episode_errors.append(avg_error)
                self._episode_lengths.append(self._current_ep_len)

                # Log episode metrics
                self._logger.log_scalars(
                    {
                        "episode/reward": self._current_ep_reward,
                        "episode/compensation_error_mm": avg_error,
                        "episode/length": self._current_ep_len,
                    },
                    self.num_timesteps,
                )

                # Print progress every 10 episodes
                if self._episodes_done % 10 == 0:
                    recent_rewards = self._episode_rewards[-20:]
                    recent_errors = self._episode_errors[-20:]
                    avg_r = sum(recent_rewards) / len(recent_rewards)
                    avg_e = sum(recent_errors) / len(recent_errors)
                    print(
                        f"  Episode {self._episodes_done} | "
                        f"Step {self.num_timesteps:,} | "
                        f"Avg Return(20): {avg_r:.2f} | "
                        f"Avg Error(20): {avg_e:.4f} mm"
                    )

                # Save best model
                if avg_error < self._best_error and self._episodes_done > 5:
                    self._best_error = avg_error
                    best_path = self._checkpoint_dir / "best_model"
                    self.model.save(str(best_path))
                    self._logger.log_audit_event(
                        "best_model_saved",
                        {"error_mm": avg_error, "episode": self._episodes_done},
                    )

                # Reset episode accumulators
                self._current_ep_reward = 0.0
                self._current_ep_errors = []
                self._current_ep_len = 0

        # Log step-level metrics periodically
        if self.num_timesteps % self._log_freq == 0:
            for info in infos:
                step_metrics = {}
                for key in [
                    "reward_tracking",
                    "reward_smooth",
                    "reward_safety",
                    "reward_latency",
                    "tissue_proximity_mm",
                ]:
                    if key in info:
                        step_metrics[f"step/{key}"] = info[key]
                if step_metrics:
                    self._logger.log_scalars(step_metrics, self.num_timesteps)

        # Periodic checkpoint
        if self.num_timesteps % self._eval_freq == 0 and self.num_timesteps > 0:
            ckpt_path = self._checkpoint_dir / f"step_{self.num_timesteps}"
            self.model.save(str(ckpt_path))

        return True

    def _on_training_end(self) -> None:
        if self._episode_errors:
            print(f"\n--- Training Summary ---")
            print(f"Episodes: {self._episodes_done}")
            first_10 = self._episode_errors[:10]
            last_10 = self._episode_errors[-10:]
            avg_first = sum(first_10) / len(first_10) if first_10 else 0
            avg_last = sum(last_10) / len(last_10) if last_10 else 0
            print(f"Avg error (first 10 eps): {avg_first:.4f} mm")
            print(f"Avg error (last 10 eps):  {avg_last:.4f} mm")
            print(f"Best error:               {self._best_error:.4f} mm")
            improvement = ((avg_first - avg_last) / avg_first * 100) if avg_first > 0 else 0
            print(f"Improvement:              {improvement:.1f}%")
            self._logger.log_audit_event(
                "training_summary",
                {
                    "episodes": self._episodes_done,
                    "avg_error_first_10": avg_first,
                    "avg_error_last_10": avg_last,
                    "best_error": self._best_error,
                    "improvement_pct": improvement,
                },
            )


def seed_everything(master_seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    np.random.seed(master_seed)
    torch.manual_seed(master_seed + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surgical tremor compensator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    master_seed = config.get("seed", 42)
    seed_everything(master_seed)

    # Create environment with safety wrapper
    base_env = SurgicalTremorEnv(config_path=args.config)
    env = SafetySurgicalEnv(base_env, config_path=args.config)

    # Logger
    logger = TrainingLogger(
        log_dir=config["logging"]["log_dir"],
        config_path=args.config,
    )
    logger.log_audit_event("training_start", {"config": args.config, "resume": args.resume})

    # Agent
    if args.resume:
        agent = SACAgent.load(args.resume, env=env)
        logger.log_audit_event("checkpoint_loaded", {"path": args.resume})
    else:
        agent = SACAgent(env=env, config_path=args.config)

    # Graceful shutdown handler
    def graceful_shutdown(signum: int, frame: object) -> None:
        checkpoint_path = Path(config["checkpointing"]["checkpoint_dir"]) / "emergency.zip"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint_path))
        logger.log_audit_event("emergency_checkpoint", {"path": str(checkpoint_path)})
        logger.close()
        print(f"\nEmergency checkpoint saved to {checkpoint_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Training callback
    log_freq = config["logging"].get("log_freq_steps", 100)
    checkpoint_dir = config["checkpointing"]["checkpoint_dir"]
    callback = MetricsCallback(
        logger_obj=logger,
        log_freq=log_freq,
        eval_freq=20_000,
        checkpoint_dir=checkpoint_dir,
    )

    # Training
    total_steps = args.steps or config["environment"]["episode_length_steps"] * 500
    print(f"Training for {total_steps:,} steps...")
    print(f"Logging to: {logger.run_dir}")

    agent.train(total_timesteps=total_steps, callback=callback)

    # Save final model
    checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "final_model.zip"
    agent.save(str(final_path))

    logger.log_audit_event("training_stop", {"total_steps": total_steps})
    logger.close()
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
