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

from agents.sac_agent import SACAgent
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv
from utils.logger import TrainingLogger


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

    # Training
    total_steps = args.steps or config["environment"]["episode_length_steps"] * 500
    print(f"Training for {total_steps} steps...")
    print(f"Logging to: {logger.run_dir}")

    agent.train(total_timesteps=total_steps)

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
