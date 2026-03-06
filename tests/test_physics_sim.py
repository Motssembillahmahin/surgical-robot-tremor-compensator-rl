"""Unit tests for the 6-DOF robot arm physics simulation (Phase 4)."""

import numpy as np
import pytest

from env.physics_sim import (
    RobotArmSimulation,
    TissueSurface,
    _dh_transform,
)
from env.surgical_env import SurgicalTremorEnv


class TestDHTransform:
    def test_identity_at_zero(self):
        """Zero DH params should give identity-like transform."""
        T = _dh_transform(0.0, 0.0, 0.0, 0.0)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])

    def test_translation_d(self):
        """Non-zero d should translate along z."""
        T = _dh_transform(0.0, 0.0, 10.0, 0.0)
        assert T[2, 3] == pytest.approx(10.0)

    def test_translation_a(self):
        """Non-zero a should translate along x."""
        T = _dh_transform(5.0, 0.0, 0.0, 0.0)
        assert T[0, 3] == pytest.approx(5.0)


class TestTissueSurface:
    def test_signed_distance_positive(self):
        tissue = TissueSurface(
            position=np.array([0.0, 0.0, 50.0]),
            normal=np.array([0.0, 0.0, -1.0]),
        )
        # Point at z=40 is 10mm away on safe side (normal points toward z=0)
        d = tissue.signed_distance(np.array([0.0, 0.0, 40.0]))
        assert d == pytest.approx(10.0)

    def test_signed_distance_negative(self):
        tissue = TissueSurface(
            position=np.array([0.0, 0.0, 50.0]),
            normal=np.array([0.0, 0.0, -1.0]),
        )
        # Point at z=55 is 5mm past tissue
        d = tissue.signed_distance(np.array([0.0, 0.0, 55.0]))
        assert d == pytest.approx(-5.0)

    def test_contact_force_zero_when_safe(self):
        tissue = TissueSurface(
            position=np.array([0.0, 0.0, 50.0]),
            normal=np.array([0.0, 0.0, -1.0]),
        )
        force = tissue.contact_force(np.array([0.0, 0.0, 40.0]))
        np.testing.assert_allclose(force, [0, 0, 0])

    def test_contact_force_nonzero_when_penetrating(self):
        tissue = TissueSurface(
            position=np.array([0.0, 0.0, 50.0]),
            normal=np.array([0.0, 0.0, -1.0]),
        )
        force = tissue.contact_force(np.array([0.0, 0.0, 55.0]))
        assert np.linalg.norm(force) > 0
        # Force should push back toward safe side (negative z)
        assert force[2] < 0


class TestRobotArmSimulation:
    def test_reset_returns_3d(self):
        sim = RobotArmSimulation()
        sim.connect()
        pos = sim.reset()
        assert pos.shape == (3,)
        assert pos.dtype == np.float32

    def test_home_position_is_deterministic(self):
        sim = RobotArmSimulation()
        sim.connect()
        p1 = sim.reset()
        p2 = sim.reset()
        np.testing.assert_allclose(p1, p2)

    def test_apply_action_moves_tip(self):
        sim = RobotArmSimulation()
        sim.connect()
        pos0 = sim.reset()
        pos1 = sim.apply_action(np.array([1.0, 0.0, 0.0]))
        # Tip should have moved (may not be exactly 1mm due to IK)
        assert not np.allclose(pos0, pos1)

    def test_tissue_proximity_positive_at_home(self):
        sim = RobotArmSimulation(
            tissue_position=np.array([0.0, 0.0, 50.0]),
        )
        sim.connect()
        sim.reset()
        prox = sim.get_tissue_proximity()
        assert prox >= 0  # should be on safe side

    def test_joint_limits_enforced(self):
        sim = RobotArmSimulation()
        sim.connect()
        sim.reset()
        # Apply a very large action repeatedly
        for _ in range(100):
            sim.apply_action(np.array([10.0, 10.0, 10.0]))
        # Joint angles should still be within limits
        for i in range(sim.n_joints):
            assert sim.joint_angles[i] >= sim.joint_limits.lower[i] - 1e-6
            assert sim.joint_angles[i] <= sim.joint_limits.upper[i] + 1e-6

    def test_get_joint_state(self):
        sim = RobotArmSimulation()
        sim.connect()
        sim.reset()
        state = sim.get_joint_state()
        assert "joint_angles" in state
        assert "joint_velocities" in state
        assert state["joint_angles"].shape == (6,)

    def test_tip_velocity(self):
        sim = RobotArmSimulation()
        sim.connect()
        sim.reset()
        vel = sim.get_tip_velocity()
        assert vel.shape == (3,)
        # At rest, velocity should be near zero
        np.testing.assert_allclose(vel, [0, 0, 0], atol=1e-6)


class TestPhysicsIntegration:
    def test_env_with_physics(self):
        """SurgicalTremorEnv should work with physics enabled."""
        env = SurgicalTremorEnv(config_path="config.yaml", use_physics=True)
        obs, info = env.reset()
        assert obs.shape == (18,)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (18,)
        assert "compensation_error_mm" in info
        assert "tissue_proximity_mm" in info

    def test_env_runs_episode_with_physics(self):
        """Full episode should complete with physics enabled."""
        env = SurgicalTremorEnv(config_path="config.yaml", use_physics=True)
        obs, _ = env.reset()
        steps = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps > 0
        assert info["tissue_proximity_mm"] >= 0
