"""Side-by-side comparison of SB3 SAC vs Custom SAC (Phase 3).

Trains both agents for the same number of steps and compares
compensation_error_mm to verify the custom implementation matches
SB3 performance within 10%.

Usage:
    uv run compare_sac.py
    uv run compare_sac.py --steps 100000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from agents.sac_agent import SACAgent
from agents.sac_custom import CustomSACAgent
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv
from utils.logger import TrainingLogger


def seed_everything(master_seed: int) -> None:
    np.random.seed(master_seed)
    torch.manual_seed(master_seed + 1)


def evaluate_agent(agent: object, env_cfg_path: str, n_episodes: int = 10) -> dict[str, float]:
    """Run evaluation episodes and return aggregate metrics."""
    env = SafetySurgicalEnv(SurgicalTremorEnv(config_path=env_cfg_path), config_path=env_cfg_path)
    rewards = []
    errors = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        ep_errors = []
        for _ in range(2000):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_errors.append(info["compensation_error_mm"])
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        errors.append(np.mean(ep_errors))

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_error_mm": float(np.mean(errors)),
        "std_error_mm": float(np.std(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SB3 SAC vs Custom SAC")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Train SB3 SAC ──────────────────────────────────────────
    print("=" * 60)
    print("Training SB3 SAC baseline...")
    print("=" * 60)

    seed_everything(config.get("seed", 42))

    sb3_env = SafetySurgicalEnv(
        SurgicalTremorEnv(config_path=args.config),
        config_path=args.config,
    )
    sb3_logger = TrainingLogger(
        log_dir=config["logging"]["log_dir"],
        config_path=args.config,
        run_id="compare_sb3",
    )

    from train import MetricsCallback

    sb3_agent = SACAgent(env=sb3_env, config_path=args.config)
    sb3_callback = MetricsCallback(
        logger_obj=sb3_logger,
        log_freq=100,
        eval_freq=50_000,
        checkpoint_dir="checkpoints/sb3_compare/",
    )
    sb3_agent.train(total_timesteps=args.steps, callback=sb3_callback)
    sb3_logger.close()

    # ── Train Custom SAC ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Custom SAC...")
    print("=" * 60)

    seed_everything(config.get("seed", 42))

    custom_env = SafetySurgicalEnv(
        SurgicalTremorEnv(config_path=args.config),
        config_path=args.config,
    )
    custom_logger = TrainingLogger(
        log_dir=config["logging"]["log_dir"],
        config_path=args.config,
        run_id="compare_custom",
    )

    custom_agent = CustomSACAgent(env=custom_env, config_path=args.config)
    custom_agent.train(
        total_timesteps=args.steps,
        logger=custom_logger,
        log_freq=100,
        checkpoint_dir="checkpoints/custom_compare/",
    )
    custom_logger.close()

    # ── Evaluate both ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Evaluating both agents ({args.eval_episodes} episodes each)...")
    print("=" * 60)

    sb3_results = evaluate_agent(sb3_agent, args.config, args.eval_episodes)
    custom_results = evaluate_agent(custom_agent, args.config, args.eval_episodes)

    print(f"\n{'Metric':<30} {'SB3 SAC':>12} {'Custom SAC':>12}")
    print("-" * 56)
    print(f"{'Avg Reward':<30} {sb3_results['avg_reward']:>12.2f} {custom_results['avg_reward']:>12.2f}")
    print(f"{'Avg Error (mm)':<30} {sb3_results['avg_error_mm']:>12.4f} {custom_results['avg_error_mm']:>12.4f}")
    print(f"{'Std Error (mm)':<30} {sb3_results['std_error_mm']:>12.4f} {custom_results['std_error_mm']:>12.4f}")

    # Check 10% threshold
    sb3_err = sb3_results["avg_error_mm"]
    custom_err = custom_results["avg_error_mm"]

    if sb3_err > 0:
        pct_diff = abs(custom_err - sb3_err) / sb3_err * 100
    else:
        pct_diff = 0.0

    print(f"\nPerformance gap: {pct_diff:.1f}%")
    if pct_diff <= 10.0:
        print("PASS: Custom SAC matches SB3 within 10% threshold.")
    elif custom_err < sb3_err:
        print("PASS: Custom SAC outperforms SB3!")
    else:
        print(f"NOTE: Custom SAC is {pct_diff:.1f}% behind SB3 (threshold: 10%).")
        print("Consider training longer or tuning hyperparameters.")


if __name__ == "__main__":
    main()
