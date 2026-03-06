"""Custom Gymnasium environment for surgical tremor compensation.

The agent observes the surgeon's raw and filtered hand input, the robot tip
state, and tissue proximity. It outputs a 3D correction vector that is added
to the surgeon's input to compensate for tremor.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from env.physics_sim import RobotArmSimulation
from env.tremor_generator import TremorGenerator
from utils.signal_processing import compute_dominant_frequency, low_pass_filter


class SurgicalTremorEnv(gym.Env):
    """Gymnasium environment for surgical robot tremor compensation.

    Observation space (18-dim):
        robot_tip_position (3), robot_tip_velocity (3),
        surgeon_input_raw (3), surgeon_input_filtered (3),
        tremor_frequency_band (1), tissue_proximity (1),
        time_in_episode (1), prev_action (3)

    Action space:
        Continuous 3D correction vector clipped to [-max_correction, +max_correction].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str = "config.yaml", use_physics: bool = False) -> None:
        super().__init__()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        env_cfg = config["environment"]
        self.dt = env_cfg["simulation_timestep_ms"] / 1000.0  # convert to seconds
        self.max_steps = env_cfg["episode_length_steps"]
        self.max_correction = env_cfg["max_correction_mm"]

        tissue_pos = env_cfg["tissue_boundary_position"]
        self.tissue_boundary = np.array(
            [tissue_pos["x"], tissue_pos["y"], tissue_pos["z"]], dtype=np.float32
        )

        # Physics simulation (Phase 4)
        self.use_physics = use_physics
        if use_physics:
            self._physics = RobotArmSimulation(
                tissue_position=self.tissue_boundary,
                dt=self.dt,
            )
            self._physics.connect()
        else:
            self._physics = None

        term_cfg = env_cfg["termination"]
        self.terminate_on_perforation = term_cfg["on_tissue_perforation"]
        self.max_consecutive_violations = term_cfg["max_consecutive_violations"]

        # Reward config
        reward_cfg = config["reward"]
        self.tracking_weight = reward_cfg["tracking_weight"]
        self.smoothness_weight = reward_cfg["smoothness_weight"]
        self.safety_penalty_value = reward_cfg["safety_penalty"]
        self.latency_weight = reward_cfg["latency_weight"]
        self.latency_threshold_ms = reward_cfg["latency_threshold_ms"]
        self.human_feedback_weight = reward_cfg["human_feedback_weight"]

        # Safety config
        safety_cfg = config["safety"]
        self.safety_margin_mm = safety_cfg["safety_margin_mm"]

        # Tremor generator
        tremor_type = config["tremor"]["default_type"]
        self.tremor_gen = TremorGenerator(tremor_type=tremor_type, config_path=config_path)

        # Seed
        self._master_seed = config.get("seed", 42)
        self._rng = np.random.default_rng(self._master_seed + 2)

        # Spaces: 18-dim observation, 3-dim action
        obs_high = np.full(18, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        act_high = np.full(3, self.max_correction, dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_high, act_high, dtype=np.float32)

        # Episode state (initialised in reset)
        self._step_count = 0
        self._robot_tip_pos = np.zeros(3, dtype=np.float32)
        self._robot_tip_vel = np.zeros(3, dtype=np.float32)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._intended_trajectory = np.zeros(3, dtype=np.float32)
        self._consecutive_violations = 0
        self._raw_signal_history: list[float] = []
        self._human_feedback_signal = 0.0
        self._sample_rate_hz = 1.0 / self.dt

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._consecutive_violations = 0
        self._raw_signal_history = []
        self._human_feedback_signal = 0.0

        # Reset robot arm (physics or simple mode)
        if self._physics is not None:
            self._robot_tip_pos = self._physics.reset(self._rng)
            self._robot_tip_vel = np.zeros(3, dtype=np.float32)
        else:
            self._robot_tip_pos = np.zeros(3, dtype=np.float32)
            self._robot_tip_vel = np.zeros(3, dtype=np.float32)

        # Generate a smooth intended trajectory starting point
        self._intended_trajectory = self._rng.uniform(-5.0, 5.0, size=3).astype(np.float32)

        # Reset tremor generator phases for this episode
        self.tremor_gen.reset(self._rng)

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one timestep of tremor compensation."""
        action = np.clip(action, -self.max_correction, self.max_correction).astype(
            np.float32
        )

        t = self._step_count * self.dt

        # Surgeon's true intended movement (slow, smooth trajectory)
        self._intended_trajectory += (
            0.01 * np.sin(0.5 * t + np.array([0, 1, 2], dtype=np.float32))
        )

        # Raw surgeon input = intended + tremor
        tremor = self.tremor_gen.generate(t)
        surgeon_raw = self._intended_trajectory + tremor

        # Store for FFT
        self._raw_signal_history.append(float(surgeon_raw[0]))

        # Compensated robot tip position = raw input + agent correction
        prev_pos = self._robot_tip_pos.copy()

        if self._physics is not None:
            # Physics mode: desired position goes through IK → FK pipeline
            desired_pos = surgeon_raw + action
            desired_delta = desired_pos - self._robot_tip_pos
            self._robot_tip_pos = self._physics.apply_action(desired_delta)
            self._robot_tip_vel = self._physics.get_tip_velocity()
            # Tissue proximity from collision mesh
            tissue_proximity = self._physics.get_tissue_proximity()
        else:
            # Simple mode: direct position update
            self._robot_tip_pos = surgeon_raw + action
            self._robot_tip_vel = (self._robot_tip_pos - prev_pos) / self.dt
            tissue_proximity = float(np.linalg.norm(self._robot_tip_pos - self.tissue_boundary))

        # ── Reward computation (clinically grounded) ──────────────────

        # 1. Tracking accuracy: reward for matching intended trajectory
        # Motivates tremor removal without over-correction
        r_tracking = -self.tracking_weight * float(
            np.linalg.norm(self._robot_tip_pos - self._intended_trajectory)
        )

        # 2. Smoothness penalty: penalise jerky compensation
        # Sudden robot movements are dangerous near tissue
        r_smooth = -self.smoothness_weight * float(
            np.linalg.norm(action - self._prev_action)
        )

        # 3. Tissue safety penalty: hard penalty if too close to tissue
        # Simulates perforation risk
        r_safety = (
            self.safety_penalty_value
            if tissue_proximity < self.safety_margin_mm
            else 0.0
        )

        # 4. Latency penalty: penalise slow compensation
        # Real surgery requires sub-20ms response
        compensation_delay_ms = self.dt * 1000  # 1-step delay as baseline
        r_latency = -self.latency_weight * max(
            0.0, compensation_delay_ms - self.latency_threshold_ms
        )

        # 5. Human feedback bonus: sparse signal from evaluator
        # Collected via FastAPI endpoint, 0.0 most steps
        r_human = self.human_feedback_weight * self._human_feedback_signal
        self._human_feedback_signal = 0.0  # Reset after use

        reward = r_tracking + r_smooth + r_safety + r_latency + r_human

        # ── Termination logic ─────────────────────────────────────────
        self._step_count += 1
        self._prev_action = action.copy()

        # Safety violation tracking
        if tissue_proximity < self.safety_margin_mm:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        terminated = False
        if self.terminate_on_perforation and tissue_proximity < 0:
            terminated = True
        if self._consecutive_violations >= self.max_consecutive_violations:
            terminated = True

        truncated = self._step_count >= self.max_steps

        info = {
            "compensation_error_mm": float(
                np.linalg.norm(self._robot_tip_pos - self._intended_trajectory)
            ),
            "tissue_proximity_mm": tissue_proximity,
            "reward_tracking": r_tracking,
            "reward_smooth": r_smooth,
            "reward_safety": r_safety,
            "reward_latency": r_latency,
            "reward_human": r_human,
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def inject_human_feedback(self, signal: float) -> None:
        """Inject a human feedback signal to be used on the next step."""
        self._human_feedback_signal = signal

    def _get_obs(self) -> np.ndarray:
        """Construct the 18-dim observation vector."""
        t = self._step_count * self.dt
        surgeon_raw = self._intended_trajectory + self.tremor_gen.generate(t)

        # Low-pass filtered signal (remove tremor)
        if len(self._raw_signal_history) > 10:
            filtered_x = low_pass_filter(
                np.array(self._raw_signal_history[-100:]),
                cutoff_hz=3.0,
                sample_rate_hz=self._sample_rate_hz,
            )
            surgeon_filtered = np.array(
                [filtered_x[-1], filtered_x[-1] * 0.8, filtered_x[-1] * 0.6],
                dtype=np.float32,
            )
        else:
            surgeon_filtered = surgeon_raw.copy()

        # Dominant tremor frequency from recent history
        if len(self._raw_signal_history) > 50:
            tremor_freq = compute_dominant_frequency(
                np.array(self._raw_signal_history[-200:]),
                sample_rate_hz=self._sample_rate_hz,
            )
        else:
            tremor_freq = self.tremor_gen.dominant_frequency

        tissue_proximity = float(
            np.linalg.norm(self._robot_tip_pos - self.tissue_boundary)
        )
        time_normalized = self._step_count / max(self.max_steps, 1)

        obs = np.concatenate([
            self._robot_tip_pos,           # 3
            self._robot_tip_vel,           # 3
            surgeon_raw,                   # 3
            surgeon_filtered,              # 3
            np.array([tremor_freq], dtype=np.float32),          # 1
            np.array([tissue_proximity], dtype=np.float32),     # 1
            np.array([time_normalized], dtype=np.float32),      # 1
            self._prev_action,             # 3
        ])
        return obs.astype(np.float32)
