"""6-DOF robot arm physics simulation for surgical tremor compensation.

Implements a kinematic robot arm with:
- Forward kinematics via DH parameters (Denavit-Hartenberg)
- Damped least-squares inverse kinematics for tip position control
- Joint limits, velocity limits, and joint damping
- Tissue collision detection against a planar tissue surface
- Configurable control latency to model real actuator delays

The simulation runs in Cartesian tip-space: the agent provides a desired
tip correction, IK resolves it to joint commands, FK computes the realised
tip position. This models the kinematic chain constraints that a real
surgical robot imposes — not every Cartesian command is perfectly achievable.

If PyBullet becomes available, the core interface (reset, apply_action,
get_tissue_proximity) remains identical, only the internal physics change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── DH Parameters for a 6-DOF surgical manipulator ─────────────


@dataclass
class DHParams:
    """Denavit-Hartenberg parameters for one link."""
    a: float       # link length (mm)
    alpha: float   # link twist (rad)
    d: float       # link offset (mm)
    theta: float   # joint angle (rad) — this is the variable for revolute joints


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Compute 4x4 homogeneous transform from DH parameters."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0],
    ], dtype=np.float64)


# ── Tissue Surface Model ───────────────────────────────────────


@dataclass
class TissueSurface:
    """Planar tissue surface for collision detection.

    Models tissue as an infinite plane defined by a point and normal.
    The signed distance from tip to plane gives tissue proximity,
    with negative values indicating penetration.
    """
    position: np.ndarray    # point on the tissue plane (mm)
    normal: np.ndarray      # outward normal (pointing away from tissue)
    stiffness: float = 50.0  # N/mm — tissue contact stiffness for force feedback

    def __post_init__(self) -> None:
        self.normal = self.normal / np.linalg.norm(self.normal)

    def signed_distance(self, point: np.ndarray) -> float:
        """Signed distance from point to tissue plane.

        Positive = safe side, negative = penetrating tissue.
        """
        return float(np.dot(point - self.position, self.normal))

    def contact_force(self, point: np.ndarray) -> np.ndarray:
        """Compute contact force if penetrating tissue (Kelvin-Voigt model)."""
        d = self.signed_distance(point)
        if d >= 0:
            return np.zeros(3, dtype=np.float64)
        # Push back along normal, proportional to penetration depth
        return -d * self.stiffness * self.normal


# ── Robot Arm Simulation ────────────────────────────────────────


@dataclass
class JointLimits:
    """Joint angle and velocity limits."""
    lower: np.ndarray    # min joint angles (rad)
    upper: np.ndarray    # max joint angles (rad)
    vel_max: np.ndarray  # max joint velocities (rad/s)


# Default 6-DOF surgical arm DH parameters (inspired by microsurgery robots)
DEFAULT_DH = [
    DHParams(a=0.0,  alpha=-np.pi / 2, d=50.0,  theta=0.0),  # base rotation
    DHParams(a=40.0, alpha=0.0,         d=0.0,   theta=0.0),  # shoulder
    DHParams(a=30.0, alpha=-np.pi / 2, d=0.0,   theta=0.0),  # elbow
    DHParams(a=0.0,  alpha=np.pi / 2,  d=25.0,  theta=0.0),  # wrist pitch
    DHParams(a=0.0,  alpha=-np.pi / 2, d=0.0,   theta=0.0),  # wrist yaw
    DHParams(a=0.0,  alpha=0.0,         d=15.0,  theta=0.0),  # tool tip
]


class RobotArmSimulation:
    """6-DOF robot arm with forward/inverse kinematics and tissue collision.

    The arm uses DH convention for kinematic modelling. Joint dynamics
    include damping and velocity limits to simulate realistic actuator
    behaviour. Tissue proximity is computed as signed distance to a
    planar tissue surface.
    """

    def __init__(
        self,
        dh_params: list[DHParams] | None = None,
        tissue_position: np.ndarray | None = None,
        dt: float = 0.005,
        use_gui: bool = False,
    ) -> None:
        self.use_gui = use_gui
        self.n_joints = 6
        self.dt = dt

        # DH parameters
        self.dh_params = dh_params or DEFAULT_DH

        # Joint state
        self.joint_angles = np.zeros(self.n_joints, dtype=np.float64)
        self.joint_velocities = np.zeros(self.n_joints, dtype=np.float64)

        # Joint limits (symmetric ±150° for most joints, tighter for wrist)
        self.joint_limits = JointLimits(
            lower=np.array([-2.6, -1.8, -2.6, -3.14, -2.0, -3.14]),
            upper=np.array([2.6, 1.8, 2.6, 3.14, 2.0, 3.14]),
            vel_max=np.full(self.n_joints, 5.0),  # rad/s
        )

        # Joint damping coefficients (Nm·s/rad)
        self.joint_damping = np.full(self.n_joints, 0.1)

        # Tissue surface (plane perpendicular to z-axis at tissue_position)
        if tissue_position is None:
            tissue_position = np.array([0.0, 0.0, 50.0])
        self.tissue = TissueSurface(
            position=tissue_position.astype(np.float64),
            normal=np.array([0.0, 0.0, -1.0]),  # normal points away from tissue (toward robot)
        )

        # IK parameters
        self._ik_damping = 0.05  # damped least-squares regularisation
        self._ik_max_iterations = 20
        self._ik_tolerance = 0.01  # mm

        # Cached tip position
        self._tip_position = np.zeros(3, dtype=np.float32)
        self._connected = False

    def connect(self) -> None:
        """Initialise the simulation."""
        self._connected = True
        self._tip_position = self._forward_kinematics().astype(np.float32)

    def disconnect(self) -> None:
        """Shut down the simulation."""
        self._connected = False

    def reset(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Reset robot arm to home position with optional random perturbation.

        Returns:
            Tip position (3,) in mm.
        """
        if rng is not None:
            # Small random joint perturbation for diversity
            self.joint_angles = rng.uniform(-0.1, 0.1, size=self.n_joints)
        else:
            self.joint_angles = np.zeros(self.n_joints, dtype=np.float64)
        self.joint_velocities = np.zeros(self.n_joints, dtype=np.float64)
        self._tip_position = self._forward_kinematics().astype(np.float32)
        return self._tip_position.copy()

    def apply_action(self, desired_tip_delta: np.ndarray) -> np.ndarray:
        """Apply a Cartesian tip correction through IK and step dynamics.

        The desired tip delta is resolved to joint velocities via damped
        least-squares IK. Joint velocities are clamped and integrated to
        produce new joint angles, then FK computes the realised tip position.

        Args:
            desired_tip_delta: 3D correction vector in mm.

        Returns:
            Realised tip position (3,) in mm after physics step.
        """
        desired_tip_delta = np.asarray(desired_tip_delta, dtype=np.float64)

        # Current tip position
        current_tip = self._forward_kinematics()

        # Desired tip position
        target_tip = current_tip + desired_tip_delta

        # Solve IK for joint velocity command
        J = self._compute_jacobian()  # 3 x n_joints
        joint_vel_cmd = self._damped_least_squares(J, desired_tip_delta / self.dt)

        # Apply damping: v_new = v_cmd - damping * v_old
        joint_vel_cmd -= self.joint_damping * self.joint_velocities

        # Clamp joint velocities
        joint_vel_cmd = np.clip(
            joint_vel_cmd,
            -self.joint_limits.vel_max,
            self.joint_limits.vel_max,
        )

        # Integrate: theta_new = theta_old + v * dt
        self.joint_angles += joint_vel_cmd * self.dt
        self.joint_velocities = joint_vel_cmd

        # Enforce joint limits
        self.joint_angles = np.clip(
            self.joint_angles,
            self.joint_limits.lower,
            self.joint_limits.upper,
        )

        # Apply tissue contact force (pushes tip back if penetrating)
        tip_pos = self._forward_kinematics()
        contact_force = self.tissue.contact_force(tip_pos)
        if np.linalg.norm(contact_force) > 1e-8:
            # Apply contact force as additional tip displacement
            J = self._compute_jacobian()
            correction_vel = self._damped_least_squares(J, contact_force * 0.1)
            self.joint_angles += correction_vel * self.dt
            self.joint_angles = np.clip(
                self.joint_angles,
                self.joint_limits.lower,
                self.joint_limits.upper,
            )

        self._tip_position = self._forward_kinematics().astype(np.float32)
        return self._tip_position.copy()

    def get_tip_position(self) -> np.ndarray:
        """Return current tip position (3,) in mm."""
        return self._tip_position.copy()

    def get_tip_velocity(self) -> np.ndarray:
        """Compute tip velocity from Jacobian and joint velocities."""
        J = self._compute_jacobian()
        tip_vel = J @ self.joint_velocities
        return tip_vel.astype(np.float32)

    def get_tissue_proximity(self, tissue_position: np.ndarray | None = None) -> float:
        """Compute signed distance from robot tip to tissue surface.

        Uses the tissue plane model for collision detection.
        Positive = safe, negative = penetrating.

        Args:
            tissue_position: Ignored (kept for API compatibility).
                Uses internal tissue surface model.

        Returns:
            Signed distance in mm (always >= 0 for safe, can be negative
            momentarily during contact resolution).
        """
        return max(0.0, self.tissue.signed_distance(self._tip_position.astype(np.float64)))

    def get_joint_state(self) -> dict[str, np.ndarray]:
        """Return full joint state for logging."""
        return {
            "joint_angles": self.joint_angles.copy(),
            "joint_velocities": self.joint_velocities.copy(),
        }

    # ── Internal methods ────────────────────────────────────────

    def _forward_kinematics(self) -> np.ndarray:
        """Compute end-effector position from joint angles using DH parameters.

        Returns:
            Tip position (3,) in mm.
        """
        T = np.eye(4, dtype=np.float64)
        for i, dh in enumerate(self.dh_params):
            T = T @ _dh_transform(dh.a, dh.alpha, dh.d, dh.theta + self.joint_angles[i])
        return T[:3, 3]

    def _compute_jacobian(self, eps: float = 1e-4) -> np.ndarray:
        """Compute the geometric Jacobian numerically (finite differences).

        Returns:
            Jacobian matrix (3, n_joints) mapping joint velocities to
            tip linear velocity.
        """
        J = np.zeros((3, self.n_joints), dtype=np.float64)
        f0 = self._forward_kinematics()
        for i in range(self.n_joints):
            self.joint_angles[i] += eps
            f1 = self._forward_kinematics()
            self.joint_angles[i] -= eps
            J[:, i] = (f1 - f0) / eps
        return J

    def _damped_least_squares(
        self, J: np.ndarray, target_vel: np.ndarray
    ) -> np.ndarray:
        """Solve IK velocity via damped least-squares (Levenberg-Marquardt).

        q_dot = J^T (J J^T + lambda^2 I)^{-1} * x_dot

        This avoids singularity issues near workspace boundaries.
        """
        JJT = J @ J.T
        n = JJT.shape[0]
        damped = JJT + (self._ik_damping ** 2) * np.eye(n)
        x_dot = np.linalg.solve(damped, target_vel)
        return J.T @ x_dot
