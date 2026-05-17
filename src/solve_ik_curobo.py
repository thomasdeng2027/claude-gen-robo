"""GPU-batched IK solver using cuRobo for the Kinova Gen3 7-DOF arm.

Frame convention
----------------
cuRobo builds its kinematic chain from ``base_link``.  The simulation URDF
(GEN3_URDF_V12_with_hand_right.urdf) contains a fixed joint::

    world_to_base_link: xyz=[0.2032, -0.127, 0.1524]  rpy=[0, 0.7854, 0]

When Isaac Sim imports this URDF (``merge_fixed_joints=False``), the robot
prim root maps to the URDF ``world`` link, and ``base_link`` is placed at that
fixed-joint offset inside the scene.  cuRobo ignores the fixed joint and
treats ``base_link`` as its world origin.

Therefore, any position expressed in Isaac world frame must be converted to
cuRobo base_link frame before being passed to the solver, and FK positions
returned by cuRobo are in base_link frame and must be converted back.

This class handles that conversion internally.  All public APIs accept and
return positions in Isaac world frame (env-0, robot at world origin).

Two interfaces
--------------
1. Single-env  — solve_ik_position(target_pos, q_init) → (q_dict, ok, err)

2. Batched GPU — solve_ik_batch(target_positions, seed_joints_batch)
                    target_positions  : (N, 3) Isaac world-frame EE targets
                    seed_joints_batch : (N, 7) warm-start arm joints
                    returns           : q_batch (N, 7), success (N,) bool

Usage:
    from solve_ik_curobo import CuRoboArmIK
    ik = CuRoboArmIK(urdf_path=..., world_to_base_xyz=..., world_to_base_rpy_y=...)
    q_batch, succ = ik.solve_ik_batch(world_targets, seed_joints)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.tool_pose import GoalToolPose
    from motion_planner_batch import BatchMotionPlanner

    _CUROBO_AVAILABLE = True
except ImportError as _e:
    _CUROBO_AVAILABLE = False
    _CUROBO_IMPORT_ERROR = str(_e)


# ── Default hand-joint values ──────────────────────────────────────────────────

_HAND_OPEN: Dict[str, float] = {
    "right_hand_thumb_bend_joint":  0.0,
    "right_hand_thumb_rota_joint1": 0.0,
    "right_hand_thumb_rota_joint2": 0.0,
    "right_hand_index_bend_joint":  0.0,
    "right_hand_index_joint1":      0.0,
    "right_hand_index_joint2":      0.0,
    "right_hand_mid_joint1":        0.0,
    "right_hand_mid_joint2":        0.0,
    "right_hand_ring_joint1":       0.0,
    "right_hand_ring_joint2":       0.0,
    "right_hand_pinky_joint1":      0.0,
    "right_hand_pinky_joint2":      0.0,
}

_HAND_CLOSED: Dict[str, float] = {
    "right_hand_thumb_bend_joint":  1.832,
    "right_hand_thumb_rota_joint1": 1.57,
    "right_hand_thumb_rota_joint2": 1.57,
    "right_hand_index_bend_joint":  0.174,
    "right_hand_index_joint1":      1.919,
    "right_hand_index_joint2":      1.919,
    "right_hand_mid_joint1":        1.919,
    "right_hand_mid_joint2":        1.919,
    "right_hand_ring_joint1":       1.919,
    "right_hand_ring_joint2":       1.919,
    "right_hand_pinky_joint1":      1.919,
    "right_hand_pinky_joint2":      1.919,
}

_DEFAULT_ARM_JOINT_NAMES: List[str] = [f"Actuator{i}" for i in range(1, 8)]

# world_to_base_link in GEN3_URDF_V12_with_hand_right.urdf
# <origin xyz="0.2032 -0.127 0.1524" rpy="0 0.7854 0"/>
_URDF_WORLD_TO_BASE_XYZ  = np.array([0.2032, -0.127, 0.1524])
_URDF_WORLD_TO_BASE_RPY_Y = 0.7854   # pi/4 ≈ 0.7854 rad


def _build_ry(angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix for a rotation of ``angle_rad`` around the Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float64)


class CuRoboArmIK:
    """GPU-batched IK for the Kinova Gen3 arm using cuRobo.

    All public solve/FK methods work in **Isaac world frame** (env-0, robot
    prim at world origin).  The solver internally converts to/from cuRobo
    base_link frame using the ``world_to_base_xyz`` / ``world_to_base_rpy_y``
    transform derived from the URDF's ``world_to_base_link`` fixed joint.

    Two solve modes:
      - Cold-start (``solve_ik_batch``):
          ``num_seeds`` random seeds + 1 user seed.  Broad exploration — use
          for the first approach.
      - Servo / warm-start (``solve_ik_batch_servo``):
          ``num_seeds=1`` only the caller's seed.  Stays on the current IK
          branch — use for the reactive servo loop.

    Args:
        urdf_path:              Absolute path to the full robot URDF.
        ee_link_name:           EE link for IK. Default ``xhand_mount_right``.
        base_link_name:         Root link. Default ``base_link``.
        arm_joint_names:        The 7 arm joint names (Actuator1-7).
        lock_joints:            Hand-joint angles to freeze (open pose default).
        num_seeds:              Random seeds for cold-start IK.
        position_threshold:     Convergence threshold in metres.
        self_collision_check:   Enable self-collision avoidance.
        world_to_base_xyz:      Translation of base_link in Isaac world frame
                                (from the URDF world_to_base_link fixed joint).
                                ``None`` (default) = legacy mode: targets in
                                base_link frame (caller applies frame transform).
                                Pass ``[0.2032, -0.127, 0.1524]`` to enable
                                world-frame mode for the GEN3 URDF.
        world_to_base_rpy_y:    Y-axis rotation (pitch) of base_link in world
                                frame.  Only used when world_to_base_xyz is set.
                                Default: 0.0 (ignored in legacy mode).
    """

    def __init__(
        self,
        urdf_path: str,
        ee_link_name: str = "xhand_mount_right",
        base_link_name: str = "base_link",
        arm_joint_names: Optional[List[str]] = None,
        lock_joints: Optional[Dict[str, float]] = None,
        num_seeds: int = 20,
        position_threshold: float = 0.005,
        self_collision_check: bool = False,
        world_to_base_xyz: Optional[np.ndarray] = None,
        world_to_base_rpy_y: Optional[float] = None,
        max_batch_size: int = 128,
    ):
        if not _CUROBO_AVAILABLE:
            raise ImportError(
                f"cuRobo is not available: {_CUROBO_IMPORT_ERROR}\n"
                "Install it from https://curobo.org"
            )

        self._urdf_path = urdf_path
        self._base_link_name = base_link_name

        self.arm_joint_names: List[str] = (
            arm_joint_names if arm_joint_names is not None
            else list(_DEFAULT_ARM_JOINT_NAMES)
        )
        self.n_arm_dofs: int = len(self.arm_joint_names)
        self.ee_link_name = ee_link_name
        self._position_threshold = position_threshold
        self._num_seeds = num_seeds
        self._self_collision_check = self_collision_check
        self._max_batch_size = max_batch_size

        # ── World-to-base_link frame transform ──────────────────────────────
        self._transform_active = world_to_base_xyz is not None
        if self._transform_active:
            _xyz   = np.asarray(world_to_base_xyz, dtype=np.float64)
            _rpy_y = float(world_to_base_rpy_y) if world_to_base_rpy_y is not None else 0.0
        else:
            _xyz   = np.zeros(3, dtype=np.float64)
            _rpy_y = 0.0
        self._t_wb   = _xyz
        self._R_wb   = _build_ry(_rpy_y)
        self._R_wb_T = self._R_wb.T

        self._device_cfg = DeviceCfg()
        self._device = self._device_cfg.device

        # ── Robot config dict — same format as working in_context_examples ────
        # Collision spheres are REQUIRED for the optimizer to have cost terms.
        # Without them torch.cat([]) crashes inside the MPPI optimizer.
        from pathlib import Path as _Path
        robot_dict = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path":       urdf_path,
                    "asset_root_path": str(_Path(urdf_path).parent),
                    "base_link":       base_link_name,
                    "tool_frames":     [ee_link_name],
                    "grasp_contact_link_names": [ee_link_name],
                    "collision_link_names": [
                        "base_link", "Shoulder_Link", "HalfArm1_Link", "HalfArm2_Link",
                        "ForeArm_Link", "SphericalWrist1_Link", "SphericalWrist2_Link",
                        "Bracelet_Link", "xhand_mount_right", "right_hand_link",
                    ],
                    "collision_spheres": {
                        "base_link":            [{"center": [0., 0., 0.06],  "radius": 0.06}],
                        "Shoulder_Link":        [{"center": [0., 0., -0.10], "radius": 0.06},
                                                 {"center": [0., 0., -0.15], "radius": 0.05}],
                        "HalfArm1_Link":        [{"center": [0.,  0.,    0.], "radius": 0.055},
                                                 {"center": [0., -0.07,  0.], "radius": 0.055},
                                                 {"center": [0., -0.15,  0.], "radius": 0.055}],
                        "HalfArm2_Link":        [{"center": [0.,  0.,    0.], "radius": 0.055},
                                                 {"center": [0.,  0., -0.07], "radius": 0.055},
                                                 {"center": [0.,  0., -0.15], "radius": 0.055}],
                        "ForeArm_Link":         [{"center": [0.,  0.,    0.], "radius": 0.055},
                                                 {"center": [0., -0.07,  0.], "radius": 0.055},
                                                 {"center": [0., -0.17,  0.], "radius": 0.055}],
                        "SphericalWrist1_Link": [{"center": [0., 0.,    0.],  "radius": 0.055},
                                                 {"center": [0., 0., -0.085], "radius": 0.055}],
                        "SphericalWrist2_Link": [{"center": [0.,     0.,    0.], "radius": 0.05},
                                                 {"center": [0., -0.085,    0.], "radius": 0.05}],
                        "Bracelet_Link":        [{"center": [0.,  0.,   -0.05], "radius": 0.04},
                                                 {"center": [0., -0.05, -0.05], "radius": 0.04}],
                        "xhand_mount_right":    [{"center": [0., 0., 0.04], "radius": 0.040}],
                        "right_hand_link":      [{"center": [0.,    0.,    0.02], "radius": 0.050},
                                                 {"center": [0.,    0.,    0.05], "radius": 0.050},
                                                 {"center": [0.04,  0.02,  0.07], "radius": 0.040},
                                                 {"center": [0.04,  0.02,  0.10], "radius": 0.040}],
                    },
                    "collision_sphere_buffer": 0.005,
                    "self_collision_ignore": {
                        "base_link":            ["Shoulder_Link", "HalfArm1_Link"],
                        "Shoulder_Link":        ["HalfArm1_Link", "HalfArm2_Link"],
                        "HalfArm1_Link":        ["HalfArm2_Link", "ForeArm_Link"],
                        "HalfArm2_Link":        ["ForeArm_Link", "SphericalWrist1_Link"],
                        "ForeArm_Link":         ["SphericalWrist1_Link", "SphericalWrist2_Link"],
                        "SphericalWrist1_Link": ["SphericalWrist2_Link", "Bracelet_Link"],
                        "SphericalWrist2_Link": ["Bracelet_Link", "xhand_mount_right", "right_hand_link"],
                        "Bracelet_Link":        ["xhand_mount_right", "right_hand_link"],
                        "xhand_mount_right":    ["right_hand_link"],
                    },
                    "self_collision_buffer": {k: 0.0 for k in [
                        "base_link", "Shoulder_Link", "HalfArm1_Link", "HalfArm2_Link",
                        "ForeArm_Link", "SphericalWrist1_Link", "SphericalWrist2_Link",
                        "Bracelet_Link", "xhand_mount_right", "right_hand_link",
                    ]},
                    "cspace": {
                        "joint_names":            list(self.arm_joint_names),
                        "default_joint_position": [0.0] * self.n_arm_dofs,
                        "null_space_weight":      [1.0] * self.n_arm_dofs,
                        "cspace_distance_weight": [1.0] * self.n_arm_dofs,
                        "max_acceleration":       10.0,
                        "max_jerk":               100.0,
                    },
                    "use_global_cumul": True,
                }
            }
        }

        # ── Cold-start planner (same pattern as cup_lift.py / BatchMotionPlanner)
        _planner_cfg = MotionPlannerCfg.create(
            robot=robot_dict,
            ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
            num_ik_seeds=num_seeds,
            num_trajopt_seeds=4,
            use_cuda_graph=False,
            max_batch_size=max_batch_size,
            max_goalset=1,
            device_cfg=self._device_cfg,
            collision_cache={"cuboid": 1},
            optimizer_collision_activation_distance=0.05,
        )
        self._planner = BatchMotionPlanner(_planner_cfg)
        self._tool_frames = self._planner.ik_solver.tool_frames

        # ── Servo planner (num_seeds=1, warm-start only) ─────────────────────
        _servo_cfg = MotionPlannerCfg.create(
            robot=robot_dict,
            ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
            num_ik_seeds=1,
            num_trajopt_seeds=1,
            use_cuda_graph=False,
            max_batch_size=max_batch_size,
            max_goalset=1,
            device_cfg=self._device_cfg,
            collision_cache={"cuboid": 1},
            optimizer_collision_activation_distance=0.05,
        )
        self._planner_servo = BatchMotionPlanner(_servo_cfg)

        _frame_info = (
            f"world_frame=ON xyz={self._t_wb.tolist()} rpy_y={_rpy_y:.4f}rad"
            if self._transform_active else "world_frame=OFF (base_link frame targets, legacy mode)"
        )
        print(
            f"[CuRoboArmIK] ready — device={self._device} "
            f"arm_joints={self.arm_joint_names} "
            f"num_seeds={num_seeds}(cold)+1(servo) "
            f"max_batch={max_batch_size} "
            f"pos_thresh={position_threshold}m "
            f"{_frame_info}",
            flush=True,
        )

    # ── Frame conversion helpers ───────────────────────────────────────────────

    def world_to_base(self, pos_world: np.ndarray) -> np.ndarray:
        """Convert (N,3) or (3,) world-frame position(s) to cuRobo base_link frame.

        If transform is not active (world_to_base_xyz=None), returns input unchanged.
        """
        if not self._transform_active:
            return np.asarray(pos_world, dtype=np.float64)
        p = np.asarray(pos_world, dtype=np.float64)
        single = p.ndim == 1
        if single:
            p = p[None]
        # Row-vector form: (p - t) @ R_wb  ≡  R_wb.T @ (p - t) in column form
        p_base = (p - self._t_wb) @ self._R_wb
        return p_base[0] if single else p_base

    def base_to_world(self, pos_base: np.ndarray) -> np.ndarray:
        """Convert (N,3) or (3,) base_link-frame position(s) to Isaac world frame.

        If transform is not active (world_to_base_xyz=None), returns input unchanged.
        """
        if not self._transform_active:
            return np.asarray(pos_base, dtype=np.float64)
        p = np.asarray(pos_base, dtype=np.float64)
        single = p.ndim == 1
        if single:
            p = p[None]
        # p_world = R_wb @ p_base + t_wb  →  row form: p_base @ R_wb.T + t_wb
        p_world = p @ self._R_wb_T + self._t_wb
        return p_world[0] if single else p_world

    def _make_joint_state(
        self, q_arr: np.ndarray
    ) -> JointState:
        """Build JointState from (N, 7) or (7,) arm-joint array."""
        q_np = np.asarray(q_arr, dtype=np.float32)
        if q_np.ndim == 1:
            q_np = q_np[None]
        q_t = torch.tensor(q_np, device=self._device)
        return JointState.from_position(q_t, joint_names=self.arm_joint_names)

    def _extract_solution(
        self,
        result,
        seeds_arm: Optional[np.ndarray],
        N: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract (q_batch (N,7), success (N,)) from a v2 cuRobo IK result."""
        # result.js_solution.position shape: (N, return_seeds, n_dof) or (N, n_dof)
        sol_pos = result.js_solution.position
        sol_np = sol_pos.cpu().numpy().astype(np.float64)
        # Collapse seed dim: take first seed per problem
        while sol_np.ndim > 2:
            sol_np = sol_np[:, 0]
        q_batch = sol_np[:N]  # (N, 7) — arm joints in cspace order

        success_np = result.success.cpu().numpy().astype(bool).reshape(-1)[:N]
        if seeds_arm is not None:
            seed_np = np.asarray(seeds_arm, dtype=np.float64)
            q_batch[~success_np] = seed_np[~success_np]

        return q_batch, success_np

    # ── Cold-start batched IK ─────────────────────────────────────────────────

    def solve_ik_batch(
        self,
        target_positions: np.ndarray,
        seed_joints_batch: Optional[np.ndarray] = None,
        target_quats: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve IK for N targets simultaneously (cold-start, num_seeds random seeds).

        Args:
            target_positions:   (N, 3) Isaac world-frame EE target positions.
            seed_joints_batch:  (N, 7) warm-start arm joints.  If None, uses
                                solver's retract config.
            target_quats:       (N, 4) wxyz desired EE orientation.  None =
                                position-only IK (identity quaternion).

        Returns:
            q_batch:  (N, 7) arm joint solutions (Actuator1..7 order).
                      Failed rows contain the seed joints unchanged.
            success:  (N,) bool.
        """
        N = len(target_positions)
        # Convert world → base_link (no-op if transform not active)
        tgt_base = self.world_to_base(np.asarray(target_positions, dtype=np.float32))

        pos_t  = torch.tensor(tgt_base.astype(np.float32), device=self._device)  # (N, 3)
        quat_t = torch.zeros((N, 4), device=self._device, dtype=torch.float32)
        if target_quats is not None:
            q_arr = np.asarray(target_quats, dtype=np.float32)
            norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            quat_t[:] = torch.tensor(q_arr / norms, device=self._device, dtype=torch.float32)
        else:
            quat_t[:, 0] = 1.0

        # v2: GoalToolPose expects (batch, seeds, goalset, links, dim)
        goal = GoalToolPose(
            tool_frames=self._tool_frames,
            position=pos_t.reshape(N, 1, 1, 1, 3),
            quaternion=quat_t.reshape(N, 1, 1, 1, 4),
        )

        current_state = self._make_joint_state(seed_joints_batch) if seed_joints_batch is not None else None
        # same call pattern as bottle_grasp.py: _planner.ik_solver.solve_pose(...)
        result = self._planner.ik_solver.solve_pose(
            goal, current_state=current_state, return_seeds=1
        )
        return self._extract_solution(result, seed_joints_batch, N)

    # ── Servo / warm-start batched IK ─────────────────────────────────────────

    def solve_ik_batch_servo(
        self,
        target_positions: np.ndarray,
        current_joints: np.ndarray,
        target_quats: Optional[np.ndarray] = None,
        retract_joints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warm-start IK (num_seeds=1).  Stays on current IK branch.

        Args:
            target_positions: (N, 3) Isaac world-frame EE targets.
            current_joints:   (N, 7) current arm joints (used as seed).
            target_quats:     (N, 4) wxyz orientation target.  None = pos-only.
            retract_joints:   (N, 7) null-space anchor.  None = no regularisation.

        Returns:
            q_batch:  (N, 7) arm joints.  Failed rows = current_joints unchanged.
            success:  (N,) bool.
        """
        N = len(target_positions)
        tgt_base = self.world_to_base(np.asarray(target_positions, dtype=np.float32))

        pos_t  = torch.tensor(tgt_base.astype(np.float32), device=self._device)
        quat_t = torch.zeros((N, 4), device=self._device, dtype=torch.float32)
        if target_quats is not None:
            q_arr = np.asarray(target_quats, dtype=np.float32)
            norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            quat_t[:] = torch.tensor(q_arr / norms, device=self._device, dtype=torch.float32)
        else:
            quat_t[:, 0] = 1.0

        goal = GoalToolPose(
            tool_frames=self._tool_frames,
            position=pos_t.reshape(N, 1, 1, 1, 3),
            quaternion=quat_t.reshape(N, 1, 1, 1, 4),
        )

        cur_state = self._make_joint_state(np.asarray(current_joints, dtype=np.float32))
        result = self._planner_servo.ik_solver.solve_pose(
            goal, current_state=cur_state, return_seeds=1
        )

        q_batch, success_np = self._extract_solution(result, None, N)
        # Restore current joints for failed envs
        q_batch[~success_np] = np.asarray(current_joints, dtype=np.float64)[~success_np]
        return q_batch, success_np

    # ── Single-env IK (drop-in for legacy callers) ────────────────────────────

    def solve_ik_position(
        self,
        target_pos: np.ndarray,
        q_init: Optional[Dict[str, float]] = None,
        reg_weight: float = 0.01,
        dq_max: Optional[float] = None,
    ) -> Tuple[Dict[str, float], bool, float]:
        """Single-env IK.  target_pos in Isaac world frame.

        Returns:
            (q_dict, success, position_error_m)
        """
        tgt = np.asarray(target_pos, dtype=np.float64).flatten()[:3]
        seed_1d = np.zeros(self.n_arm_dofs, dtype=np.float64)
        if q_init is not None:
            for i, jn in enumerate(self.arm_joint_names):
                seed_1d[i] = float(q_init.get(jn, 0.0))

        q_batch, success = self.solve_ik_batch(tgt[None], seed_1d[None])
        q_arr = q_batch[0]
        ok = bool(success[0])

        # FK error in world frame
        q_dict = {jn: float(q_arr[i]) for i, jn in enumerate(self.arm_joint_names)}
        fk_world = self.fk_pos(q_dict)
        err = float(np.linalg.norm(tgt - fk_world))

        return q_dict, ok, err

    # ── Forward kinematics ────────────────────────────────────────────────────

    def fk_pos(self, q_dict: Dict[str, float]) -> np.ndarray:
        """Return EE position (3,) in Isaac world frame.

        Args:
            q_dict: {joint_name: angle} for arm joints.
        """
        pos, _ = self.fk_pos_and_quat(q_dict)
        return pos

    def _fk_js(self, q_arr_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run FK for a single (7,) arm joint array.  Returns (pos_base (3,), quat_base (4,)) wxyz."""
        js = self._make_joint_state(q_arr_np)
        with torch.no_grad():
            # same call as bottle_grasp.py: _planner.compute_kinematics(js)
            kin_state = self._planner.compute_kinematics(js)
        pos  = kin_state.tool_poses.position[0, 0, 0].cpu().numpy().astype(np.float64)
        quat = kin_state.tool_poses.quaternion[0, 0, 0].cpu().numpy().astype(np.float64)
        return pos, quat

    def fk_pos_and_quat(
        self,
        q_dict: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-env FK: return EE position (3,) and wxyz quaternion (4,) in Isaac world frame.

        Args:
            q_dict: {joint_name: angle} for arm joints.
        Returns:
            pos  : (3,) Isaac world-frame EE position.
            quat : (4,) wxyz EE orientation in Isaac world frame.
        """
        q_arr = np.array([float(q_dict.get(jn, 0.0)) for jn in self.arm_joint_names],
                         dtype=np.float32)
        pos_base, quat = self._fk_js(q_arr)

        pos_world = self.base_to_world(pos_base)
        _q_wb     = _rmat_to_wxyz(self._R_wb)
        quat_world = _qmul(_q_wb, quat)
        return pos_world, quat_world

    def fk_pose_batch(
        self,
        q_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch FK: return EE positions (N,3) and wxyz quaternions (N,4) in Isaac world frame.

        Args:
            q_batch: (N, 7) arm joints in Actuator1..7 order.
        """
        N = len(q_batch)
        js = self._make_joint_state(np.asarray(q_batch, dtype=np.float32))
        with torch.no_grad():
            kin_state = self._planner.compute_kinematics(js)
        # tool_poses.position shape: (N, H, L, 3) — collapse to (N, 3)
        pos_base  = kin_state.tool_poses.position.reshape(N, -1, 3)[:, 0, :].cpu().numpy().astype(np.float64)
        quat_base = kin_state.tool_poses.quaternion.reshape(N, -1, 4)[:, 0, :].cpu().numpy().astype(np.float64)

        # Convert base_link → world
        _q_wb = _rmat_to_wxyz(self._R_wb)
        positions = self.base_to_world(pos_base)  # (N, 3)
        quats = np.stack([_qmul(_q_wb, quat_base[i]) for i in range(N)])
        return positions, quats

    def with_hand_pose(self, lock_joints: Dict[str, float]) -> "CuRoboArmIK":
        """No-op: hand joints not in IK URDF.  Returns self for call-site compat."""
        return self


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _rmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a (4,) wxyz quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


# ── Module-level constants re-exported for convenience ────────────────────────
HAND_OPEN   = _HAND_OPEN
HAND_CLOSED = _HAND_CLOSED
