"""Unit tests for the custom SAC implementation (Phase 3)."""

import numpy as np
import pytest
import torch

from agents.sac_custom import (
    CustomSACAgent,
    ReplayBuffer,
    SquashedGaussianActor,
    TwinQCritic,
)
from env.surgical_env import SurgicalTremorEnv
from safety.constraints import SafetySurgicalEnv


@pytest.fixture
def env():
    return SafetySurgicalEnv(SurgicalTremorEnv(), config_path="config.yaml")


class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(capacity=100, obs_dim=18, act_dim=3)
        assert buf.size == 0
        buf.add(np.zeros(18), np.zeros(3), 1.0, np.zeros(18), False)
        assert buf.size == 1

    def test_circular_overwrite(self):
        buf = ReplayBuffer(capacity=5, obs_dim=2, act_dim=1)
        for i in range(10):
            buf.add(np.array([i, i]), np.array([i]), float(i), np.array([i, i]), False)
        assert buf.size == 5
        assert buf.ptr == 0  # wrapped around

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, obs_dim=18, act_dim=3)
        for _ in range(50):
            buf.add(np.random.randn(18), np.random.randn(3), 0.5, np.random.randn(18), False)
        batch = buf.sample(32, torch.device("cpu"))
        assert batch["obs"].shape == (32, 18)
        assert batch["actions"].shape == (32, 3)
        assert batch["rewards"].shape == (32,)
        assert batch["dones"].shape == (32,)


class TestSquashedGaussianActor:
    def test_output_shape(self):
        actor = SquashedGaussianActor(obs_dim=18, act_dim=3)
        obs = torch.randn(4, 18)
        mu, log_std = actor(obs)
        assert mu.shape == (4, 3)
        assert log_std.shape == (4, 3)

    def test_sample_bounded(self):
        actor = SquashedGaussianActor(obs_dim=18, act_dim=3)
        obs = torch.randn(100, 18)
        action, log_prob = actor.sample(obs)
        assert action.shape == (100, 3)
        assert log_prob.shape == (100, 1)
        # tanh output should be in (-1, 1)
        assert action.abs().max() <= 1.0

    def test_deterministic_no_noise(self):
        actor = SquashedGaussianActor(obs_dim=18, act_dim=3)
        obs = torch.randn(1, 18)
        a1 = actor.deterministic(obs)
        a2 = actor.deterministic(obs)
        assert torch.allclose(a1, a2)


class TestTwinQCritic:
    def test_output_shape(self):
        critic = TwinQCritic(obs_dim=18, act_dim=3)
        obs = torch.randn(4, 18)
        act = torch.randn(4, 3)
        q1, q2 = critic(obs, act)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_twin_q_different(self):
        """Twin Q-networks should give different outputs (different weights)."""
        critic = TwinQCritic(obs_dim=18, act_dim=3)
        obs = torch.randn(4, 18)
        act = torch.randn(4, 3)
        q1, q2 = critic(obs, act)
        assert not torch.allclose(q1, q2)


class TestCustomSACAgent:
    def test_predict_shape(self, env):
        agent = CustomSACAgent(env=env, config_path="config.yaml")
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        assert action.shape == (3,)

    def test_predict_within_bounds(self, env):
        agent = CustomSACAgent(env=env, config_path="config.yaml")
        obs, _ = env.reset()
        for _ in range(20):
            action, _ = agent.predict(obs, deterministic=False)
            assert np.all(action >= env.action_space.low - 1e-6)
            assert np.all(action <= env.action_space.high + 1e-6)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    def test_update_runs(self, env):
        agent = CustomSACAgent(env=env, config_path="config.yaml")
        # Fill buffer with enough samples
        obs, _ = env.reset()
        for _ in range(300):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.replay_buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        agent._update()
        assert "sac/critic_loss" in agent.metrics
        assert "sac/actor_loss" in agent.metrics
        assert "sac/ent_coef" in agent.metrics
        assert "sac/entropy" in agent.metrics

    def test_save_load(self, env, tmp_path):
        agent = CustomSACAgent(env=env, config_path="config.yaml")
        path = tmp_path / "test_model.pt"
        agent.save(path)

        agent2 = CustomSACAgent(env=env, config_path="config.yaml")
        agent2.load(path)

        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        a1 = agent.actor.deterministic(obs_t)
        a2 = agent2.actor.deterministic(obs_t)
        assert torch.allclose(a1, a2)
