"""Evaluation script and FastAPI feedback server.

Usage:
    # Run evaluation
    uv run evaluate.py --checkpoint checkpoints/best_model.zip

    # Start feedback server
    uv run uvicorn evaluate:app --reload --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.reward_model import (
    RewardModelTrainer,
    compute_trajectory_features,
)
from agents.sac_agent import SACAgent
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv
from utils.signal_processing import compute_tremor_rejection_ratio

# ── FastAPI App ──────────────────────────────────────────────

app = FastAPI(title="Surgical Tremor Compensator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "human_labels.jsonl"
TRAJECTORY_DIR = FEEDBACK_DIR / "trajectories"
TRAJECTORY_DIR.mkdir(exist_ok=True)

# Active WebSocket connections for live metrics streaming
_ws_connections: list[WebSocket] = []


# ── Request/Response Schemas ─────────────────────────────────


class FeedbackRequest(BaseModel):
    episode_id: int
    score: int = Field(ge=1, le=5)
    evaluator_id: str


class FeedbackResponse(BaseModel):
    status: str
    total_labels: int
    features: list[float]


class StatsResponse(BaseModel):
    total_labels: int
    score_distribution: dict[str, int]
    average_score: float
    evaluator_count: int


class TrainingStatusResponse(BaseModel):
    status: str
    message: str


class TrajectoryResponse(BaseModel):
    episode_id: int
    steps: int
    compensation_error_mm: list[float]
    tissue_proximity_mm: list[float]
    reward_tracking: list[float]
    reward_smooth: list[float]
    reward_total: list[float]
    features: list[float]


class MetricsSummaryResponse(BaseModel):
    total_episodes: int
    avg_error_mm: float
    avg_reward: float
    total_safety_violations: int


# ── Trajectory Storage ───────────────────────────────────────


def save_trajectory(episode_id: int, trajectory: dict[str, list[float]]) -> Path:
    """Save episode trajectory data for later human review."""
    path = TRAJECTORY_DIR / f"episode_{episode_id}.json"
    with open(path, "w") as f:
        json.dump({"episode_id": episode_id, **trajectory}, f)
    return path


def load_trajectory(episode_id: int) -> dict[str, Any] | None:
    """Load saved trajectory data for an episode."""
    path = TRAJECTORY_DIR / f"episode_{episode_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_episode_trajectory(
    agent: object,
    env: SafetySurgicalEnv,
    episode_id: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one episode and collect full trajectory data."""
    obs, _ = env.reset(seed=seed)
    trajectory: dict[str, list[float]] = {
        "compensation_error_mm": [],
        "tissue_proximity_mm": [],
        "reward_tracking": [],
        "reward_smooth": [],
        "reward_safety": [],
        "reward_latency": [],
        "reward_total": [],
    }

    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory["compensation_error_mm"].append(info["compensation_error_mm"])
        trajectory["tissue_proximity_mm"].append(info["tissue_proximity_mm"])
        trajectory["reward_tracking"].append(info["reward_tracking"])
        trajectory["reward_smooth"].append(info["reward_smooth"])
        trajectory["reward_safety"].append(info["reward_safety"])
        trajectory["reward_latency"].append(info["reward_latency"])
        trajectory["reward_total"].append(reward)

    trajectory["max_steps"] = env.env.max_steps
    trajectory["tremor_rejection_ratio"] = 0.0  # placeholder

    save_trajectory(episode_id, trajectory)
    return trajectory


# ── Human Feedback Endpoints ─────────────────────────────────


@app.post("/feedback/evaluate", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Store a human evaluation score for an episode with trajectory features."""
    # Load trajectory to compute features
    traj = load_trajectory(req.episode_id)
    if traj is not None:
        features = compute_trajectory_features(traj)
    else:
        features = [0.0] * 10

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "episode_id": req.episode_id,
        "score": req.score,
        "evaluator_id": req.evaluator_id,
        "features": features,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    total = sum(1 for _ in open(FEEDBACK_FILE))
    return FeedbackResponse(status="ok", total_labels=total, features=features)


@app.get("/feedback/trajectory/{episode_id}", response_model=TrajectoryResponse)
async def get_trajectory(episode_id: int) -> TrajectoryResponse:
    """Return trajectory data for human review."""
    traj = load_trajectory(episode_id)
    if traj is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    features = compute_trajectory_features(traj)
    return TrajectoryResponse(
        episode_id=episode_id,
        steps=len(traj.get("compensation_error_mm", [])),
        compensation_error_mm=traj.get("compensation_error_mm", []),
        tissue_proximity_mm=traj.get("tissue_proximity_mm", []),
        reward_tracking=traj.get("reward_tracking", []),
        reward_smooth=traj.get("reward_smooth", []),
        reward_total=traj.get("reward_total", []),
        features=features,
    )


@app.get("/feedback/stats", response_model=StatsResponse)
async def feedback_stats() -> StatsResponse:
    """Return aggregate feedback statistics with inter-rater info."""
    if not FEEDBACK_FILE.exists():
        return StatsResponse(
            total_labels=0, score_distribution={}, average_score=0.0, evaluator_count=0
        )

    scores: list[int] = []
    evaluators: set[str] = set()
    with open(FEEDBACK_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                scores.append(entry["score"])
                evaluators.add(entry.get("evaluator_id", "unknown"))

    if not scores:
        return StatsResponse(
            total_labels=0, score_distribution={}, average_score=0.0, evaluator_count=0
        )

    distribution = {str(i): scores.count(i) for i in range(1, 6)}
    return StatsResponse(
        total_labels=len(scores),
        score_distribution=distribution,
        average_score=sum(scores) / len(scores),
        evaluator_count=len(evaluators),
    )


@app.post("/feedback/retrain-reward-model")
async def retrain_reward_model() -> dict[str, Any]:
    """Trigger reward model retraining on collected labels."""
    trainer = RewardModelTrainer()
    loss = trainer.train()
    if loss == float("inf"):
        raise HTTPException(status_code=400, detail="Not enough labels to train (minimum 5)")
    trainer.save("checkpoints/reward_model.pt")

    labels = trainer.load_labels()
    return {
        "status": "ok",
        "final_loss": loss,
        "labels_used": len(labels),
    }


# ── Training Control Endpoints ──────────────────────────────


@app.get("/training/status", response_model=TrainingStatusResponse)
async def training_status() -> TrainingStatusResponse:
    """Return current training status."""
    return TrainingStatusResponse(
        status="idle",
        message="Use 'uv run train.py' to start training",
    )


# ── Dashboard Endpoints ─────────────────────────────────────


@app.get("/api/episodes/{episode_id}")
async def get_episode(episode_id: int) -> dict[str, Any]:
    """Return full episode data for dashboard replay."""
    traj = load_trajectory(episode_id)
    if traj is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    return traj


@app.get("/api/metrics/summary", response_model=MetricsSummaryResponse)
async def metrics_summary() -> MetricsSummaryResponse:
    """Return aggregated metrics across all saved trajectories."""
    traj_files = list(TRAJECTORY_DIR.glob("episode_*.json"))
    if not traj_files:
        return MetricsSummaryResponse(
            total_episodes=0, avg_error_mm=0.0, avg_reward=0.0, total_safety_violations=0
        )

    all_errors = []
    all_rewards = []
    total_violations = 0

    for path in traj_files:
        with open(path) as f:
            traj = json.load(f)
        errors = traj.get("compensation_error_mm", [])
        rewards = traj.get("reward_total", [])
        tissue = np.array(traj.get("tissue_proximity_mm", [50.0]))

        if errors:
            all_errors.append(np.mean(errors))
        if rewards:
            all_rewards.append(np.sum(rewards))
        total_violations += int(np.sum(tissue < 2.0))

    return MetricsSummaryResponse(
        total_episodes=len(traj_files),
        avg_error_mm=float(np.mean(all_errors)) if all_errors else 0.0,
        avg_reward=float(np.mean(all_rewards)) if all_rewards else 0.0,
        total_safety_violations=total_violations,
    )


# ── WebSocket for Live Metrics ──────────────────────────────


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket) -> None:
    """Stream live training metrics to the React dashboard."""
    await websocket.accept()
    _ws_connections.append(websocket)
    try:
        while True:
            # Keep connection alive; actual metrics are pushed via broadcast_metrics()
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_connections.remove(websocket)


async def broadcast_metrics(metrics: dict[str, Any]) -> None:
    """Push metrics to all connected WebSocket clients."""
    dead = []
    for ws in _ws_connections:
        try:
            await ws.send_json(metrics)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections.remove(ws)


# ── Evaluation Logic ─────────────────────────────────────────


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    num_episodes: int = 10,
    save_trajectories: bool = True,
) -> dict[str, float]:
    """Evaluate a saved model checkpoint and optionally save trajectories.

    Returns:
        Dictionary of evaluation metrics.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_env = SurgicalTremorEnv(config_path=config_path)
    env = SafetySurgicalEnv(base_env, config_path=config_path)
    agent = SACAgent.load(checkpoint_path, env=env)

    eval_seed = config.get("seed", 42) + 1000

    all_errors: list[float] = []
    all_violations: list[int] = []
    episode_returns: list[float] = []

    for ep in range(num_episodes):
        episode_id = eval_seed + ep

        if save_trajectories:
            traj = collect_episode_trajectory(agent, env, episode_id, seed=episode_id)
            errors = traj["compensation_error_mm"]
            tissue = np.array(traj["tissue_proximity_mm"])
            ep_return = sum(traj["reward_total"])
        else:
            obs, _ = env.reset(seed=episode_id)
            ep_return = 0.0
            violations = 0
            errors = []
            tissue = []

            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += reward
                errors.append(info["compensation_error_mm"])
                tissue.append(info["tissue_proximity_mm"])

            tissue = np.array(tissue)

        all_errors.extend(errors)
        all_violations.append(int(np.sum(tissue < config["safety"]["safety_margin_mm"])))
        episode_returns.append(ep_return)

    metrics = {
        "compensation_error_mm": float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
        "safety_violations": float(np.sum(all_violations)),
        "mean_episode_return": float(np.mean(episode_returns)),
        "num_episodes": num_episodes,
    }
    return metrics


# ── Feedback Injection Helper ────────────────────────────────


def inject_feedback_into_env(
    env: SurgicalTremorEnv,
    reward_model_path: str = "checkpoints/reward_model.pt",
    trajectory: dict[str, list[float]] | None = None,
) -> float:
    """Compute r_human from the reward model and inject into env.

    Args:
        env: The surgical environment to inject feedback into.
        reward_model_path: Path to trained reward model weights.
        trajectory: Episode trajectory data for feature extraction.

    Returns:
        The predicted human feedback signal.
    """
    model_path = Path(reward_model_path)
    if not model_path.exists() or trajectory is None:
        return 0.0

    trainer = RewardModelTrainer()
    trainer.load(model_path)
    features = compute_trajectory_features(trajectory)
    r_human = trainer.predict(features)
    env.inject_human_feedback(r_human)
    return r_human


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tremor compensator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="Save trajectory data for human review",
    )
    args = parser.parse_args()

    print(f"Evaluating {args.checkpoint}...")
    metrics = evaluate_checkpoint(
        args.checkpoint, args.config, args.episodes,
        save_trajectories=args.save_trajectories,
    )

    print("\n── Evaluation Results ──")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    if args.save_trajectories:
        traj_count = len(list(TRAJECTORY_DIR.glob("episode_*.json")))
        print(f"\n  Trajectories saved: {traj_count} (in {TRAJECTORY_DIR}/)")
        print("  Submit feedback via: POST /feedback/evaluate")


if __name__ == "__main__":
    main()
