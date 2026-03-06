"""Human-in-the-loop reward model.

A small neural network trained on human feedback scores to predict
compensation quality. Outputs a sparse reward signal injected into
the SAC training loop.

The features vector (10-dim) summarises an episode's trajectory:
  [mean_error, std_error, max_error, mean_smoothness, max_jerk,
   safety_violations, min_tissue_proximity, mean_reward,
   tremor_rejection_ratio, episode_length_ratio]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


FEATURE_DIM = 10
FEATURE_NAMES = [
    "mean_error",
    "std_error",
    "max_error",
    "mean_smoothness",
    "max_jerk",
    "safety_violations",
    "min_tissue_proximity",
    "mean_reward",
    "tremor_rejection_ratio",
    "episode_length_ratio",
]


def compute_trajectory_features(trajectory: dict[str, list[float]]) -> list[float]:
    """Extract 10-dim feature vector from episode trajectory data.

    Args:
        trajectory: Dict with keys matching info dict fields, each a list
                    of per-step values.

    Returns:
        10-dim feature list.
    """
    errors = np.array(trajectory.get("compensation_error_mm", [0.0]))
    smoothness = np.array(trajectory.get("reward_smooth", [0.0]))
    tissue_prox = np.array(trajectory.get("tissue_proximity_mm", [50.0]))
    rewards = np.array(trajectory.get("reward_total", [0.0]))
    max_steps = trajectory.get("max_steps", 2000)
    actual_steps = len(errors)

    return [
        float(np.mean(errors)),
        float(np.std(errors)),
        float(np.max(errors)) if len(errors) > 0 else 0.0,
        float(np.mean(np.abs(smoothness))),
        float(np.max(np.abs(np.diff(errors)))) if len(errors) > 1 else 0.0,
        float(np.sum(tissue_prox < 2.0)),
        float(np.min(tissue_prox)) if len(tissue_prox) > 0 else 50.0,
        float(np.mean(rewards)),
        trajectory.get("tremor_rejection_ratio", 0.0),
        actual_steps / max(max_steps, 1),
    ]


class RewardModel(nn.Module):
    """Small MLP that predicts human feedback score from trajectory features.

    Input: trajectory summary features (10-dim)
    Output: predicted score normalised to [-1, 1]
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RewardModelTrainer:
    """Manages training the reward model on human feedback labels."""

    def __init__(
        self,
        model: RewardModel | None = None,
        feedback_path: str = "feedback/human_labels.jsonl",
        learning_rate: float = 1e-3,
    ) -> None:
        self.model = model or RewardModel()
        self.feedback_path = Path(feedback_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def load_labels(self) -> list[dict[str, Any]]:
        """Load human feedback labels from JSONL file."""
        if not self.feedback_path.exists():
            return []
        labels = []
        with open(self.feedback_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(json.loads(line))
        return labels

    def train(self, epochs: int = 50) -> float:
        """Train reward model on collected human labels.

        Returns:
            Final training loss.
        """
        labels = self.load_labels()
        if len(labels) < 5:
            return float("inf")  # Not enough data to train

        # Extract features and scores
        features = []
        scores = []
        for label in labels:
            feat = label.get("features", [0.0] * FEATURE_DIM)
            if len(feat) < FEATURE_DIM:
                feat = feat + [0.0] * (FEATURE_DIM - len(feat))
            score = label.get("score", 3)
            features.append(feat[:FEATURE_DIM])
            # Normalise score from [1, 5] to [-1, 1]
            scores.append((score - 3.0) / 2.0)

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

        self.model.train()
        final_loss = float("inf")

        for _ in range(epochs):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            final_loss = loss.item()

        return final_loss

    def predict(self, features: np.ndarray | list[float]) -> float:
        """Predict reward signal from trajectory features."""
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                features if isinstance(features, list) else features.tolist(),
                dtype=torch.float32,
            ).unsqueeze(0)
            pred = self.model(x)
        return float(pred.item())

    def save(self, path: str | Path) -> None:
        """Save reward model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        """Load reward model weights."""
        self.model.load_state_dict(torch.load(str(path), weights_only=True))
