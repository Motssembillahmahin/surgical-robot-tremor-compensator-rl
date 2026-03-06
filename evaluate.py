"""Evaluation script and FastAPI feedback server.

Usage:
    # Run evaluation
    uv run evaluate.py --checkpoint checkpoints/best_model.zip

    # Start feedback server
    uv run uvicorn evaluate:app --reload --port 8000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.reward_model import RewardModelTrainer
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


# ── Request/Response Schemas ─────────────────────────────────


class FeedbackRequest(BaseModel):
    episode_id: int
    score: int = Field(ge=1, le=5)
    evaluator_id: str


class FeedbackResponse(BaseModel):
    status: str
    total_labels: int


class TrainingStatusResponse(BaseModel):
    status: str
    message: str


class StatsResponse(BaseModel):
    total_labels: int
    score_distribution: dict[str, int]
    average_score: float


# ── Human Feedback Endpoints ─────────────────────────────────


@app.post("/feedback/evaluate", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Store a human evaluation score for an episode."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "episode_id": req.episode_id,
        "score": req.score,
        "evaluator_id": req.evaluator_id,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    total = sum(1 for _ in open(FEEDBACK_FILE))
    return FeedbackResponse(status="ok", total_labels=total)


@app.get("/feedback/stats", response_model=StatsResponse)
async def feedback_stats() -> StatsResponse:
    """Return aggregate feedback statistics."""
    if not FEEDBACK_FILE.exists():
        return StatsResponse(
            total_labels=0, score_distribution={}, average_score=0.0
        )

    scores: list[int] = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                scores.append(entry["score"])

    if not scores:
        return StatsResponse(
            total_labels=0, score_distribution={}, average_score=0.0
        )

    distribution = {str(i): scores.count(i) for i in range(1, 6)}
    return StatsResponse(
        total_labels=len(scores),
        score_distribution=distribution,
        average_score=sum(scores) / len(scores),
    )


@app.post("/feedback/retrain-reward-model")
async def retrain_reward_model() -> dict[str, Any]:
    """Trigger reward model retraining on collected labels."""
    trainer = RewardModelTrainer()
    loss = trainer.train()
    if loss == float("inf"):
        raise HTTPException(status_code=400, detail="Not enough labels to train (minimum 5)")
    trainer.save("checkpoints/reward_model.pt")
    return {"status": "ok", "final_loss": loss}


@app.get("/training/status", response_model=TrainingStatusResponse)
async def training_status() -> TrainingStatusResponse:
    """Return current training status."""
    return TrainingStatusResponse(
        status="idle",
        message="Use 'uv run train.py' to start training",
    )


# ── Evaluation Logic ─────────────────────────────────────────


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str = "config.yaml",
    num_episodes: int = 10,
) -> dict[str, float]:
    """Evaluate a saved model checkpoint.

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
        obs, _ = env.reset(seed=eval_seed + ep)
        episode_return = 0.0
        violations = 0
        errors: list[float] = []

        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            errors.append(info["compensation_error_mm"])
            if info["tissue_proximity_mm"] < config["safety"]["safety_margin_mm"]:
                violations += 1

        all_errors.extend(errors)
        all_violations.append(violations)
        episode_returns.append(episode_return)

    metrics = {
        "compensation_error_mm": float(np.sqrt(np.mean(np.array(all_errors) ** 2))),
        "safety_violations": float(np.sum(all_violations)),
        "mean_episode_return": float(np.mean(episode_returns)),
        "num_episodes": num_episodes,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tremor compensator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    print(f"Evaluating {args.checkpoint}...")
    metrics = evaluate_checkpoint(args.checkpoint, args.config, args.episodes)

    print("\n── Evaluation Results ──")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
