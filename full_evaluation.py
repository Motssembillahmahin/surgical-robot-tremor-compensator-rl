"""Full evaluation suite across all tremor types (Phase 7).

Evaluates the trained agent on all 3 tremor profiles and generates
a final metrics report with all 5 evaluation metrics.

Usage:
    uv run full_evaluation.py
    uv run full_evaluation.py --checkpoint checkpoints/best_model_custom.pt --agent custom
    uv run full_evaluation.py --episodes 20

Metrics reported:
    1. compensation_error_mm — RMS error between compensated and true trajectory
    2. tremor_rejection_ratio_dB — how much tremor power was removed
    3. safety_violations — count of tissue boundary breaches (target: 0)
    4. compensation_latency_ms — must stay below 20ms threshold
    5. human_feedback_score — average reward model prediction
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agents.reward_model import RewardModelTrainer, compute_trajectory_features
from agents.sac_agent import SACAgent
from agents.sac_custom import CustomSACAgent
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv
from utils.signal_processing import compute_tremor_rejection_ratio


def evaluate_tremor_type(
    agent: Any,
    tremor_type: str,
    config_path: str,
    num_episodes: int,
    reward_model_path: str | None,
) -> dict[str, Any]:
    """Evaluate agent on a specific tremor type.

    Returns all 5 evaluation metrics.
    """
    # Modify config to use specified tremor type
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["tremor"]["default_type"] = tremor_type
    eval_seed = config.get("seed", 42) + 2000

    # Write temp config
    tmp_config = Path(f"/tmp/eval_config_{tremor_type}.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(config, f)

    base_env = SurgicalTremorEnv(config_path=str(tmp_config))
    env = SafetySurgicalEnv(base_env, config_path=str(tmp_config))

    # Load reward model if available
    reward_trainer = None
    if reward_model_path and Path(reward_model_path).exists():
        reward_trainer = RewardModelTrainer()
        reward_trainer.load(reward_model_path)

    all_errors: list[float] = []
    all_violations: list[int] = []
    all_latencies: list[float] = []
    episode_returns: list[float] = []
    all_rejection_ratios: list[float] = []
    all_feedback_scores: list[float] = []
    episode_details: list[dict[str, float]] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        ep_return = 0.0
        ep_violations = 0
        ep_errors: list[float] = []
        ep_tissue: list[float] = []
        raw_signals: list[float] = []
        compensated_signals: list[float] = []
        step_rewards: list[float] = []

        done = False
        step_count = 0
        t_start = time.perf_counter()

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            ep_return += reward
            ep_errors.append(info["compensation_error_mm"])
            ep_tissue.append(info["tissue_proximity_mm"])
            step_rewards.append(reward)

            if info["tissue_proximity_mm"] < config["safety"]["safety_margin_mm"]:
                ep_violations += 1

            # Collect signals for rejection ratio
            raw_signals.append(float(obs[6]))  # surgeon_raw x-component
            compensated_signals.append(float(obs[0]))  # robot_tip x-component

        t_elapsed = time.perf_counter() - t_start

        # Compensation latency (time per step in ms)
        latency_ms = (t_elapsed / max(step_count, 1)) * 1000

        # Tremor rejection ratio
        if len(raw_signals) > 50:
            sample_rate = 1.0 / config["environment"]["simulation_timestep_ms"] * 1000
            rejection_ratio = compute_tremor_rejection_ratio(
                np.array(raw_signals),
                np.array(compensated_signals),
                sample_rate_hz=sample_rate,
            )
        else:
            rejection_ratio = 0.0

        # Human feedback score from reward model
        if reward_trainer:
            traj_data = {
                "compensation_error_mm": ep_errors,
                "reward_smooth": [0.0] * len(ep_errors),
                "tissue_proximity_mm": ep_tissue,
                "reward_total": step_rewards,
                "max_steps": config["environment"]["episode_length_steps"],
            }
            features = compute_trajectory_features(traj_data)
            feedback_score = reward_trainer.predict(features)
        else:
            feedback_score = 0.0

        rms_error = float(np.sqrt(np.mean(np.array(ep_errors) ** 2)))

        all_errors.extend(ep_errors)
        all_violations.append(ep_violations)
        all_latencies.append(latency_ms)
        all_rejection_ratios.append(rejection_ratio)
        all_feedback_scores.append(feedback_score)
        episode_returns.append(ep_return)

        episode_details.append({
            "episode": ep,
            "rms_error_mm": rms_error,
            "safety_violations": ep_violations,
            "latency_ms": latency_ms,
            "rejection_ratio_dB": rejection_ratio,
            "feedback_score": feedback_score,
            "return": ep_return,
            "steps": step_count,
        })

    return {
        "tremor_type": tremor_type,
        "num_episodes": num_episodes,
        "compensation_error_mm": float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
        "tremor_rejection_ratio_dB": float(np.mean(all_rejection_ratios)),
        "safety_violations": int(np.sum(all_violations)),
        "compensation_latency_ms": float(np.mean(all_latencies)),
        "human_feedback_score": float(np.mean(all_feedback_scores)),
        "mean_episode_return": float(np.mean(episode_returns)),
        "episodes": episode_details,
    }


def generate_report(
    results: list[dict[str, Any]],
    agent_type: str,
    checkpoint: str,
) -> str:
    """Generate a formatted evaluation report."""
    lines = [
        "=" * 70,
        "SURGICAL TREMOR COMPENSATOR — FINAL EVALUATION REPORT",
        "=" * 70,
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Agent:      {agent_type}",
        f"Checkpoint: {checkpoint}",
        "",
    ]

    # Summary table
    header = f"{'Tremor Type':<16} {'Error(mm)':>10} {'Reject(dB)':>11} {'Violations':>11} {'Latency(ms)':>12} {'Feedback':>9}"
    lines.append(header)
    lines.append("-" * 70)

    all_violations = 0
    for r in results:
        lines.append(
            f"{r['tremor_type']:<16} "
            f"{r['compensation_error_mm']:>10.4f} "
            f"{r['tremor_rejection_ratio_dB']:>11.2f} "
            f"{r['safety_violations']:>11d} "
            f"{r['compensation_latency_ms']:>12.4f} "
            f"{r['human_feedback_score']:>9.4f}"
        )
        all_violations += r["safety_violations"]

    lines.append("-" * 70)

    # Averages
    avg_error = np.mean([r["compensation_error_mm"] for r in results])
    avg_reject = np.mean([r["tremor_rejection_ratio_dB"] for r in results])
    avg_latency = np.mean([r["compensation_latency_ms"] for r in results])
    avg_feedback = np.mean([r["human_feedback_score"] for r in results])

    lines.append(
        f"{'AVERAGE':<16} "
        f"{avg_error:>10.4f} "
        f"{avg_reject:>11.2f} "
        f"{all_violations:>11d} "
        f"{avg_latency:>12.4f} "
        f"{avg_feedback:>9.4f}"
    )

    # Pass/fail criteria
    lines.append("")
    lines.append("── Exit Criteria ──")
    lines.append(f"  Compensation error decreasing:  {'PASS' if avg_error < 2.0 else 'CHECK'} ({avg_error:.4f} mm)")
    lines.append(f"  Safety violations = 0:          {'PASS' if all_violations == 0 else 'FAIL'} ({all_violations})")
    lines.append(f"  Latency < 20ms:                 {'PASS' if avg_latency < 20 else 'FAIL'} ({avg_latency:.4f} ms)")
    lines.append(f"  Tremor rejection positive:      {'PASS' if avg_reject > 0 else 'CHECK'} ({avg_reject:.2f} dB)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation across all tremor types")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model_custom.pt",
        help="Model checkpoint path",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--agent", type=str, default="custom", choices=["sb3", "custom"],
    )
    parser.add_argument("--reward-model", type=str, default="checkpoints/reward_model.pt")
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    args = parser.parse_args()

    # Load agent
    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_env = SurgicalTremorEnv(config_path=args.config)
    env = SafetySurgicalEnv(base_env, config_path=args.config)

    if args.agent == "custom":
        agent = CustomSACAgent(env=env, config_path=args.config)
        agent.load(args.checkpoint)
    else:
        agent = SACAgent.load(args.checkpoint, env=env)

    # Evaluate across all tremor types
    tremor_types = list(config["tremor"]["profiles"].keys())
    print(f"Evaluating across {len(tremor_types)} tremor types: {tremor_types}")
    print(f"Episodes per type: {args.episodes}")
    print()

    results = []
    for tremor_type in tremor_types:
        print(f"Evaluating {tremor_type}...")
        result = evaluate_tremor_type(
            agent, tremor_type, args.config, args.episodes, args.reward_model,
        )
        results.append(result)
        print(f"  Error: {result['compensation_error_mm']:.4f} mm | "
              f"Violations: {result['safety_violations']} | "
              f"Rejection: {result['tremor_rejection_ratio_dB']:.2f} dB")

    # Generate report
    report = generate_report(results, args.agent, args.checkpoint)
    print()
    print(report)

    # Save detailed results as JSON
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": args.agent,
            "checkpoint": args.checkpoint,
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
