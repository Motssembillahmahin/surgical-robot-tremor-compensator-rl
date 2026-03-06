"""Tests for the human feedback reward model."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.reward_model import RewardModel, RewardModelTrainer


class TestRewardModel:
    def test_forward_pass_shape(self) -> None:
        model = RewardModel(input_dim=10)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_range(self) -> None:
        """Output should be in [-1, 1] due to Tanh."""
        model = RewardModel(input_dim=10)
        x = torch.randn(100, 10)
        out = model(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestRewardModelTrainer:
    def test_train_with_enough_labels(self, tmp_path: Path) -> None:
        feedback_file = tmp_path / "labels.jsonl"
        with open(feedback_file, "w") as f:
            for i in range(10):
                entry = {
                    "episode_id": i,
                    "score": (i % 5) + 1,
                    "features": list(np.random.randn(10)),
                }
                f.write(json.dumps(entry) + "\n")

        trainer = RewardModelTrainer(feedback_path=str(feedback_file))
        loss = trainer.train(epochs=10)
        assert loss < float("inf")
        assert loss >= 0

    def test_train_with_insufficient_labels(self, tmp_path: Path) -> None:
        feedback_file = tmp_path / "labels.jsonl"
        with open(feedback_file, "w") as f:
            entry = {"episode_id": 0, "score": 3, "features": [0.0] * 10}
            f.write(json.dumps(entry) + "\n")

        trainer = RewardModelTrainer(feedback_path=str(feedback_file))
        loss = trainer.train()
        assert loss == float("inf")

    def test_predict_returns_float(self) -> None:
        trainer = RewardModelTrainer()
        features = np.random.randn(10).astype(np.float32)
        result = trainer.predict(features)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_save_and_load(self, tmp_path: Path) -> None:
        model = RewardModel(input_dim=10)
        save_path = tmp_path / "model.pt"

        trainer = RewardModelTrainer(model=model)
        trainer.save(save_path)

        new_model = RewardModel(input_dim=10)
        new_trainer = RewardModelTrainer(model=new_model)
        new_trainer.load(save_path)

        # Weights should match
        x = torch.randn(1, 10)
        model.eval()
        new_model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), new_model(x))
