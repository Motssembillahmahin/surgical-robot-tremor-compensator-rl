"""Custom SAC (Soft Actor-Critic) implementation from scratch in PyTorch.

Phase 3: Reimplements SAC without Stable-Baselines3 to deepen understanding
of the algorithm internals — actor-critic architecture, automatic entropy
tuning, twin Q-networks, and experience replay.

Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ── Replay Buffer ───────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy learning."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "actions": torch.as_tensor(self.actions[idx], device=device),
            "rewards": torch.as_tensor(self.rewards[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "dones": torch.as_tensor(self.dones[idx], device=device),
        }


# ── Networks ────────────────────────────────────────────────────


class MLPNetwork(nn.Module):
    """Shared MLP backbone for actor and critic networks."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    """Stochastic policy that outputs a squashed Gaussian distribution.

    The action is sampled as: a = tanh(mu + sigma * eps), eps ~ N(0, I).
    Log-probability accounts for the tanh squashing via the change-of-variables formula.
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_std_head = nn.Linear(prev, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and clamped log_std."""
        h = self.backbone(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log-probability with tanh squashing correction."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterisation trick
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Log-prob with tanh correction: log pi(a|s) = log pi(u|s) - sum(log(1 - tanh^2(u)))
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (tanh of mean) for evaluation."""
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class TwinQCritic(nn.Module):
    """Twin Q-networks for clipped double-Q learning."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.q1 = MLPNetwork(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLPNetwork(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ── Custom SAC Agent ────────────────────────────────────────────


class CustomSACAgent:
    """Pure PyTorch SAC implementation with automatic entropy tuning.

    Implements the full SAC algorithm:
    - Twin Q-critics with target networks (soft update)
    - Squashed Gaussian actor with reparameterisation
    - Automatic entropy coefficient tuning
    - Experience replay buffer
    """

    def __init__(
        self,
        env: gym.Env,
        config_path: str = "config.yaml",
    ) -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        sac_cfg = config["sac"]
        seed = config.get("seed", 42)
        torch.manual_seed(seed + 1)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.act_dim = act_dim
        self.env = env

        # Scale actions from [-1, 1] (tanh output) to env action bounds
        self.action_scale = torch.as_tensor(
            (env.action_space.high - env.action_space.low) / 2.0,
            dtype=torch.float32, device=self.device,
        )
        self.action_bias = torch.as_tensor(
            (env.action_space.high + env.action_space.low) / 2.0,
            dtype=torch.float32, device=self.device,
        )

        # Hyperparameters from config
        self.lr = float(sac_cfg["learning_rate"])
        self.gamma = float(sac_cfg["gamma"])
        self.tau = float(sac_cfg["tau"])
        self.batch_size = int(sac_cfg["batch_size"])
        self.buffer_size = int(sac_cfg["buffer_size"])
        self.train_freq = int(sac_cfg["train_freq"])
        self.gradient_steps = int(sac_cfg["gradient_steps"])

        # Networks
        self.actor = SquashedGaussianActor(obs_dim, act_dim).to(self.device)
        self.critic = TwinQCritic(obs_dim, act_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Automatic entropy tuning
        self.target_entropy = -float(act_dim)
        self.log_ent_coef = torch.zeros(1, requires_grad=True, device=self.device)
        self.ent_coef_optim = torch.optim.Adam([self.log_ent_coef], lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, act_dim)

        # Training metrics (exposed for logging)
        self.metrics: dict[str, float] = {}

    @property
    def ent_coef(self) -> float:
        return self.log_ent_coef.exp().item()

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to env bounds."""
        return action * self.action_scale + self.action_bias

    def _unscale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from env bounds to [-1, 1]."""
        return (action - self.action_bias.cpu().numpy()) / self.action_scale.cpu().numpy()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Select action for a single observation."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
            action = self._scale_action(action)
        return action.cpu().numpy().squeeze(0), None

    def _update(self) -> None:
        """Single gradient update on actor, critic, and entropy coefficient."""
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        obs, actions, rewards, next_obs, dones = (
            batch["obs"], batch["actions"], batch["rewards"],
            batch["next_obs"], batch["dones"],
        )

        alpha = self.log_ent_coef.exp().detach()

        # ── Critic update ──────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            next_actions_scaled = self._scale_action(next_actions)
            q1_next, q2_next = self.critic_target(next_obs, next_actions_scaled)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            target_q = rewards.unsqueeze(-1) + (1.0 - dones.unsqueeze(-1)) * self.gamma * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ── Actor update ───────────────────────────────────────
        new_actions, log_prob = self.actor.sample(obs)
        new_actions_scaled = self._scale_action(new_actions)
        q1_new, q2_new = self.critic(obs, new_actions_scaled)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ── Entropy coefficient update ─────────────────────────
        ent_coef_loss = -(self.log_ent_coef * (log_prob.detach() + self.target_entropy)).mean()

        self.ent_coef_optim.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optim.step()

        # ── Soft update target networks ────────────────────────
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

        # Store metrics for logging
        self.metrics = {
            "sac/critic_loss": critic_loss.item(),
            "sac/actor_loss": actor_loss.item(),
            "sac/ent_coef": self.ent_coef,
            "sac/entropy": -log_prob.mean().item(),
            "sac/ent_coef_loss": ent_coef_loss.item(),
        }

    def train(
        self,
        total_timesteps: int,
        logger: Any | None = None,
        log_freq: int = 100,
        checkpoint_dir: str = "checkpoints/",
    ) -> dict[str, list[float]]:
        """Train the agent and return episode metrics history.

        Args:
            total_timesteps: Total environment steps to train.
            logger: Optional TrainingLogger for TensorBoard logging.
            log_freq: Log scalar metrics every N steps.
            checkpoint_dir: Directory for saving checkpoints.

        Returns:
            Dict with lists of per-episode rewards and errors.
        """
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        episode_rewards: list[float] = []
        episode_errors: list[float] = []
        ep_reward = 0.0
        ep_errors: list[float] = []
        episodes_done = 0
        best_error = float("inf")

        obs, _ = self.env.reset()
        for step in range(1, total_timesteps + 1):
            # Collect experience
            if self.replay_buffer.size < self.batch_size:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, action, reward, next_obs, terminated)

            ep_reward += reward
            if "compensation_error_mm" in info:
                ep_errors.append(info["compensation_error_mm"])

            obs = next_obs

            # Train after collecting enough data
            if self.replay_buffer.size >= self.batch_size and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._update()

                # Log SAC internals
                if logger and step % log_freq == 0:
                    logger.log_scalars(self.metrics, step)
                    # Log step-level env metrics
                    for key in ["reward_tracking", "reward_smooth", "reward_safety",
                                "reward_latency", "tissue_proximity_mm"]:
                        if key in info:
                            logger.log_scalar(f"step/{key}", info[key], step)

            if done:
                episodes_done += 1
                avg_error = sum(ep_errors) / len(ep_errors) if ep_errors else 0.0
                episode_rewards.append(ep_reward)
                episode_errors.append(avg_error)

                if logger:
                    logger.log_scalars({
                        "episode/reward": ep_reward,
                        "episode/compensation_error_mm": avg_error,
                    }, step)

                # Progress output every 10 episodes
                if episodes_done % 10 == 0:
                    recent_r = episode_rewards[-20:]
                    recent_e = episode_errors[-20:]
                    print(
                        f"  Episode {episodes_done} | "
                        f"Step {step:,} | "
                        f"Avg Return(20): {sum(recent_r)/len(recent_r):.2f} | "
                        f"Avg Error(20): {sum(recent_e)/len(recent_e):.4f} mm"
                    )

                # Save best model
                if avg_error < best_error and episodes_done > 5:
                    best_error = avg_error
                    self.save(ckpt_dir / "best_model_custom.pt")
                    if logger:
                        logger.log_audit_event(
                            "best_model_saved",
                            {"error_mm": avg_error, "episode": episodes_done, "agent": "custom"},
                        )

                ep_reward = 0.0
                ep_errors = []
                obs, _ = self.env.reset()

        # Training summary
        if episode_errors:
            first_10 = episode_errors[:10]
            last_10 = episode_errors[-10:]
            avg_first = sum(first_10) / len(first_10) if first_10 else 0
            avg_last = sum(last_10) / len(last_10) if last_10 else 0
            improvement = ((avg_first - avg_last) / avg_first * 100) if avg_first > 0 else 0
            print(f"\n--- Custom SAC Training Summary ---")
            print(f"Episodes: {episodes_done}")
            print(f"Avg error (first 10 eps): {avg_first:.4f} mm")
            print(f"Avg error (last 10 eps):  {avg_last:.4f} mm")
            print(f"Best error:               {best_error:.4f} mm")
            print(f"Improvement:              {improvement:.1f}%")

        return {"rewards": episode_rewards, "errors": episode_errors}

    def save(self, path: str | Path) -> None:
        """Save all model weights and optimiser states."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "log_ent_coef": self.log_ent_coef.detach().cpu(),
            "ent_coef_optim": self.ent_coef_optim.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        """Load model weights and optimiser states."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.log_ent_coef = ckpt["log_ent_coef"].to(self.device).requires_grad_(True)
        self.ent_coef_optim = torch.optim.Adam([self.log_ent_coef], lr=self.lr)
        self.ent_coef_optim.load_state_dict(ckpt["ent_coef_optim"])
