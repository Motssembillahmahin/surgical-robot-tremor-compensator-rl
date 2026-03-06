"""End-to-end tests for the human-in-the-loop feedback pipeline (Phase 5).

Tests the full cycle: trajectory collection → human scoring → reward model
training → r_human injection → agent behavior change.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from agents.reward_model import (
    RewardModel,
    RewardModelTrainer,
    compute_trajectory_features,
)
from env.surgical_env import SurgicalTremorEnv
from evaluate import (
    collect_episode_trajectory,
    inject_feedback_into_env,
    load_trajectory,
    save_trajectory,
)
from safety.constraints import SafetySurgicalEnv


@pytest.fixture
def env():
    return SafetySurgicalEnv(SurgicalTremorEnv(), config_path="config.yaml")


@pytest.fixture
def feedback_dir(tmp_path):
    """Provide a temp feedback directory and clean up after."""
    return tmp_path / "feedback"


class TestTrajectoryStorage:
    def test_save_and_load(self, tmp_path):
        traj = {
            "compensation_error_mm": [1.0, 0.8, 0.6],
            "tissue_proximity_mm": [45.0, 44.0, 43.0],
            "reward_total": [-1.0, -0.8, -0.6],
        }
        import evaluate

        old_dir = evaluate.TRAJECTORY_DIR
        evaluate.TRAJECTORY_DIR = tmp_path
        try:
            save_trajectory(42, traj)
            loaded = load_trajectory(42)
            assert loaded is not None
            assert loaded["episode_id"] == 42
            assert loaded["compensation_error_mm"] == [1.0, 0.8, 0.6]
        finally:
            evaluate.TRAJECTORY_DIR = old_dir

    def test_load_missing_returns_none(self):
        assert load_trajectory(999999) is None


class TestTrajectoryFeatures:
    def test_feature_dimension(self):
        traj = {
            "compensation_error_mm": [1.0, 0.8, 0.6, 0.4],
            "reward_smooth": [-0.1, -0.05, -0.02, -0.01],
            "tissue_proximity_mm": [45.0, 44.0, 43.0, 42.0],
            "reward_total": [-1.0, -0.8, -0.6, -0.4],
            "max_steps": 2000,
        }
        features = compute_trajectory_features(traj)
        assert len(features) == 10

    def test_features_values_reasonable(self):
        traj = {
            "compensation_error_mm": [2.0, 1.5, 1.0, 0.5],
            "reward_smooth": [-0.1, -0.05, -0.02, -0.01],
            "tissue_proximity_mm": [45.0, 44.0, 43.0, 42.0],
            "reward_total": [-2.0, -1.5, -1.0, -0.5],
            "max_steps": 2000,
        }
        features = compute_trajectory_features(traj)
        assert features[0] == pytest.approx(1.25)  # mean_error
        assert features[5] == 0.0  # safety_violations (all > 2.0)
        assert features[6] == pytest.approx(42.0)  # min_tissue_proximity


class TestCollectTrajectory:
    def test_collect_episode(self, env):
        # Use a simple agent that returns zero actions
        class ZeroAgent:
            def predict(self, obs, deterministic=True):
                return np.zeros(3, dtype=np.float32), None

        import evaluate

        old_dir = evaluate.TRAJECTORY_DIR
        evaluate.TRAJECTORY_DIR = Path("/tmp/test_traj_collect")
        evaluate.TRAJECTORY_DIR.mkdir(exist_ok=True)
        try:
            agent = ZeroAgent()
            traj = collect_episode_trajectory(agent, env, episode_id=1, seed=42)
            assert len(traj["compensation_error_mm"]) > 0
            assert len(traj["tissue_proximity_mm"]) > 0
            assert len(traj["reward_total"]) > 0
        finally:
            shutil.rmtree(evaluate.TRAJECTORY_DIR, ignore_errors=True)
            evaluate.TRAJECTORY_DIR = old_dir


class TestRewardModelIntegration:
    def test_train_with_50_labels(self, tmp_path):
        """Generate 50+ labels and train the reward model (exit criteria)."""
        feedback_file = tmp_path / "human_labels.jsonl"
        rng = np.random.default_rng(42)

        # Generate 60 synthetic labels with varied features and scores
        for i in range(60):
            error = rng.uniform(0.2, 3.0)
            # Score correlates with error: low error → high score
            score = int(np.clip(5 - error, 1, 5))
            features = [
                error,                     # mean_error
                rng.uniform(0.1, 0.5),     # std_error
                error * 1.5,               # max_error
                rng.uniform(0.01, 0.1),    # mean_smoothness
                rng.uniform(0.01, 0.5),    # max_jerk
                rng.integers(0, 3),        # safety_violations
                rng.uniform(10.0, 50.0),   # min_tissue_proximity
                -error * 2,                # mean_reward
                rng.uniform(5.0, 20.0),    # tremor_rejection_ratio
                rng.uniform(0.8, 1.0),     # episode_length_ratio
            ]
            entry = {
                "episode_id": i,
                "score": score,
                "evaluator_id": f"eval_{i % 3}",
                "features": [float(f) for f in features],
            }
            with open(feedback_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        # Train reward model
        trainer = RewardModelTrainer(feedback_path=str(feedback_file))
        loss = trainer.train(epochs=100)
        assert loss < float("inf"), "Training should succeed with 60 labels"
        assert loss < 0.5, f"Loss should be reasonable, got {loss}"

        # Model should predict higher reward for low-error trajectories
        good_features = [0.3, 0.1, 0.5, 0.02, 0.05, 0, 45.0, -0.6, 15.0, 1.0]
        bad_features = [2.5, 0.4, 4.0, 0.08, 0.4, 2, 15.0, -5.0, 5.0, 0.8]

        good_pred = trainer.predict(good_features)
        bad_pred = trainer.predict(bad_features)
        assert good_pred > bad_pred, (
            f"Good trajectory ({good_pred:.3f}) should score higher "
            f"than bad ({bad_pred:.3f})"
        )

    def test_save_load_reward_model(self, tmp_path):
        model = RewardModel()
        trainer = RewardModelTrainer(model=model)
        path = tmp_path / "reward_model.pt"
        trainer.save(path)

        trainer2 = RewardModelTrainer()
        trainer2.load(path)

        import torch
        x = torch.randn(1, 10)
        with torch.no_grad():
            p1 = model(x)
            p2 = trainer2.model(x)
        assert torch.allclose(p1, p2)


class TestFeedbackInjection:
    def test_inject_changes_reward(self, tmp_path):
        """Verify that r_human actually affects the env reward."""
        env = SurgicalTremorEnv()
        obs, _ = env.reset(seed=42)

        # Step without feedback
        action = np.zeros(3, dtype=np.float32)
        _, reward_no_fb, _, _, info_no_fb = env.step(action)
        assert info_no_fb["reward_human"] == 0.0

        # Inject positive feedback
        env.inject_human_feedback(0.5)
        obs, _ = env.reset(seed=42)
        env.step(np.zeros(3, dtype=np.float32))  # skip first step
        env.inject_human_feedback(0.5)
        _, reward_with_fb, _, _, info_with_fb = env.step(action)
        assert info_with_fb["reward_human"] == pytest.approx(0.5)

    def test_inject_from_reward_model(self, tmp_path):
        """Test the inject_feedback_into_env helper."""
        # Create and save a trained reward model
        feedback_file = tmp_path / "labels.jsonl"
        for i in range(10):
            entry = {
                "score": 4,
                "features": [0.5, 0.1, 0.8, 0.02, 0.05, 0, 45.0, -1.0, 15.0, 1.0],
            }
            with open(feedback_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        trainer = RewardModelTrainer(feedback_path=str(feedback_file))
        trainer.train(epochs=50)
        model_path = tmp_path / "reward_model.pt"
        trainer.save(model_path)

        # Inject into env
        env = SurgicalTremorEnv()
        env.reset(seed=42)
        traj = {
            "compensation_error_mm": [0.5, 0.4, 0.3],
            "reward_smooth": [-0.01, -0.01, -0.01],
            "tissue_proximity_mm": [45.0, 44.0, 43.0],
            "reward_total": [-0.5, -0.4, -0.3],
            "max_steps": 2000,
        }

        r_human = inject_feedback_into_env(env, str(model_path), traj)
        # Should return a non-zero value
        assert r_human != 0.0
        assert -1.0 <= r_human <= 1.0
