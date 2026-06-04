# Arm IK solver using pinocchio.
#
# Author: Timothy Yu
# Date: 2/13/2025

import numpy as np
import torch
import pinocchio as pin
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

# URDF world_to_base_link transform: matches <origin xyz="0.2032 -0.127 0.1524" rpy="0 0.7854 0"/>
# The robot base_link is placed at this offset+rotation in the URDF world frame.
# Wrist targets are expressed in base_link frame; they must be mapped to URDF world frame
# before passing to pinocchio IK (which outputs EE positions in URDF world frame).
URDF_BASE_XYZ = np.array([0.2032, -0.127, 0.1524])
URDF_BASE_RPY = np.array([0.0, 0.7854, 0.0])  # roll, pitch, yaw in radians (45° around Y)


class ArmIKSolver:
    """
    IK solver for a robot arm to reach target wrist positions.
    Uses damped least squares (Levenberg-Marquardt) for robust IK.
    """

    def __init__(
        self,
        urdf_path: str,
        ee_link_name: str = "right_hand_link",
        arm_joint_names: list = None,
        position_only: bool = True,
        damping: float = 1e-6,
        max_iters: int = 100,
        eps: float = 1e-4,
        dq_max: float = 0.15,
    ):
        """
        Args:
            urdf_path: Path to the URDF file
            ee_link_name: Name of the end-effector link (wrist)
            arm_joint_names: List of arm joint names (Actuator1-7). If None, auto-detect.
            position_only: If True, only match position (not orientation)
            damping: Damping factor for damped least squares
            max_iters: Maximum IK iterations
            eps: Convergence threshold
        """
        self.urdf_path = urdf_path
        self.ee_link_name = ee_link_name
        self.position_only = position_only
        self.damping = damping
        self.max_iters = max_iters
        self.eps = eps
        self.dq_max = dq_max

        # Load the robot model
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        # Load geometry model for accurate z-constraint checking (mesh vertices, not joint origins).
        # Falls back gracefully if mesh files are missing.
        self.collision_model = None
        self.collision_data = None
        try:
            package_dirs = [str(Path(urdf_path).parent), str(Path(urdf_path).parent.parent)]
            self.collision_model = pin.buildGeomFromUrdf(
                self.model, str(urdf_path), pin.COLLISION, package_dirs=package_dirs
            )
            self.collision_data = self.collision_model.createData()
        except Exception as _e:
            pass  # geometry not available — get_min_z_position will fall back to joint origins

        # Get end-effector frame ID
        self.ee_frame_id = self.model.getFrameId(ee_link_name)
        if self.ee_frame_id == self.model.nframes:
            # Try as joint name
            if ee_link_name in [j.name for j in self.model.joints]:
                joint_id = self.model.getJointId(ee_link_name)
                # Find frame associated with this joint
                for i, frame in enumerate(self.model.frames):
                    if frame.parentJoint == joint_id:
                        self.ee_frame_id = i
                        break

        if self.ee_frame_id == self.model.nframes:
            raise ValueError(f"Could not find frame '{ee_link_name}' in URDF")

        # Identify arm joint indices
        if arm_joint_names is None:
            arm_joint_names = [f"Actuator{i}" for i in range(1, 8)]

        self.arm_joint_names = arm_joint_names
        self.arm_joint_ids = []
        self.arm_dof_ids = []

        for jname in arm_joint_names:
            if jname in self.model.names.tolist():
                jid = self.model.getJointId(jname)
                self.arm_joint_ids.append(jid)
                # Get DOF index for this joint
                dof_idx = self.model.joints[jid].idx_v
                self.arm_dof_ids.append(dof_idx)
            else:
                print(f"Warning: Joint '{jname}' not found in model")

        self.n_arm_dofs = len(self.arm_dof_ids)

        # Get joint limits for arm
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()

        # Default configuration
        self.q_default = pin.neutral(self.model)

        print(f"ArmIKSolver initialized:")
        print(f"  URDF: {urdf_path}")
        print(f"  EE frame: {ee_link_name} (id={self.ee_frame_id})")
        print(f"  Arm joints: {arm_joint_names}")
        print(f"  Arm DOF indices: {self.arm_dof_ids}")
        print(f"  Total model DOFs: {self.model.nv}")

    def get_base_link_pose(self) -> tuple:
        """Get base_link position and rotation in pinocchio world at neutral config.

        The URDF's world_to_base_link fixed joint may include a lab-specific translation
        (e.g. xyz="0.165 -0.461 0.15") that offsets the pinocchio world origin from the
        physical robot base.  This method returns that offset so callers can compensate.

        Returns:
            (translation, rotation_matrix) — both in pinocchio world frame.
        """
        q = self.q_default.copy()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        base_frame_id = self.model.getFrameId("base_link")
        if base_frame_id >= self.model.nframes:
            # Try any frame containing 'base'
            for i, f in enumerate(self.model.frames):
                if "base" in f.name.lower():
                    base_frame_id = i
                    break
        if base_frame_id >= self.model.nframes:
            return np.zeros(3), np.eye(3)
        pose = self.data.oMf[base_frame_id]
        return pose.translation.copy(), pose.rotation.copy()

    def get_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Get end-effector position for a given configuration."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id].translation.copy()

    def get_ee_pose(self, q: np.ndarray) -> pin.SE3:
        """Get end-effector pose (SE3) for a given configuration."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id].copy()

    def solve_ik_position(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray = None,
        fixed_joints: dict = None,
        dq_max: float = None,
        reg_weight: float = 0.01,
    ) -> tuple:
        """
        Solve IK for target position only.

        Args:
            target_pos: Target 3D position [x, y, z]
            q_init: Initial configuration. If None, use default or previous solution.
            fixed_joints: Dict mapping DOF index to fixed value (for hand joints)
            dq_max: Per-iteration step-size limit (radians). Defaults to self.dq_max.
            reg_weight: Tikhonov regularization weight penalizing deviation from q_init.

        Returns:
            (q_solution, success, error)
        """
        if dq_max is None:
            dq_max = self.dq_max

        if q_init is None:
            q = self.q_default.copy()
        else:
            q = q_init.copy()

        # Reference configuration for regularization (bias toward q_init / previous frame)
        q_ref_arm = np.array([q[dof_id] for dof_id in self.arm_dof_ids])

        # Apply fixed joints (e.g., hand joints)
        if fixed_joints is not None:
            for dof_idx, val in fixed_joints.items():
                if dof_idx < len(q):
                    q[dof_idx] = val

        target_pos = np.asarray(target_pos).flatten()[:3]

        for i in range(self.max_iters):
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Current position (in URDF world frame)
            current_pos = self.data.oMf[self.ee_frame_id].translation

            # Position error
            err = target_pos - current_pos
            err_norm = np.linalg.norm(err)

            if err_norm < self.eps:
                return q.copy(), True, err_norm

            # Compute Jacobian (only translation part)
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            # Extract only the arm DOFs from Jacobian (3 x n_arm_dofs)
            J_arm = J_full[:3, self.arm_dof_ids]

            # Tikhonov-regularized least squares: (J^T J + (λ + w_reg) I) dq = J^T e + w_reg (q_ref - q)
            # Biases solution toward q_init (previous frame) for temporal smoothness.
            q_arm = np.array([q[dof_id] for dof_id in self.arm_dof_ids])
            A = J_arm.T @ J_arm + (self.damping + reg_weight) * np.eye(self.n_arm_dofs)
            b = J_arm.T @ err + reg_weight * (q_ref_arm - q_arm)

            try:
                dq_arm = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dq_arm = np.linalg.lstsq(J_arm, err, rcond=None)[0]

            # Clamp per-iteration step size to prevent large jumps
            dq_arm = np.clip(dq_arm, -dq_max, dq_max)

            # Update only arm joints
            for idx, dof_id in enumerate(self.arm_dof_ids):
                q[dof_id] += dq_arm[idx]

            # Clamp to joint limits
            q = np.clip(q, self.q_min, self.q_max)

            # Re-apply fixed joints
            if fixed_joints is not None:
                for dof_idx, val in fixed_joints.items():
                    if dof_idx < len(q):
                        q[dof_idx] = val

        # Return best solution found
        err_final = np.linalg.norm(target_pos - self.data.oMf[self.ee_frame_id].translation)
        return q.copy(), False, err_final

    def solve_ik_pose(
        self,
        target_pose: pin.SE3,
        q_init: np.ndarray = None,
        q_ref: np.ndarray = None,
        fixed_joints: dict = None,
        position_weight: float = 1.0,
        orientation_weight: float = 0.1,
        dq_max: float = None,
        reg_weight: float = 0.01,
    ) -> tuple:
        """
        Solve IK for target pose (position + orientation).

        Args:
            target_pose: Target SE3 pose
            q_init: Initial configuration (starting point for IK)
            q_ref: Reference configuration for regularization. If None, uses q_init.
                   Pass the previous frame's q to regularize toward temporal continuity
                   regardless of which init branch is being tried.
            fixed_joints: Dict mapping DOF index to fixed value
            position_weight: Weight for position error
            orientation_weight: Weight for orientation error (keep low, e.g. 0.1, to reduce instability)
            dq_max: Per-iteration step-size limit (radians). Defaults to self.dq_max.
            reg_weight: Tikhonov regularization weight penalizing deviation from q_ref.

        Returns:
            (q_solution, success, error)
        """
        if dq_max is None:
            dq_max = self.dq_max

        if q_init is None:
            q = self.q_default.copy()
        else:
            q = q_init.copy()

        # Reference for regularization: use q_ref if provided, else fall back to q_init.
        # Passing q_ref = previous frame's config makes ALL retries bias toward the
        # previous frame, preventing branch switches in fresh-init retries.
        _q_ref = q_ref if q_ref is not None else q_init
        if _q_ref is None:
            _q_ref = self.q_default
        q_ref_arm = np.array([_q_ref[dof_id] for dof_id in self.arm_dof_ids])

        if fixed_joints is not None:
            for dof_idx, val in fixed_joints.items():
                if dof_idx < len(q):
                    q[dof_idx] = val

        for i in range(self.max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[self.ee_frame_id]

            # Compute pose error in se3
            err_se3 = pin.log6(current_pose.inverse() * target_pose)
            err = err_se3.vector  # 6D error (linear, angular)

            # Weight the errors
            err[:3] *= position_weight
            err[3:] *= orientation_weight

            err_norm = np.linalg.norm(err)

            if err_norm < self.eps:
                return q.copy(), True, err_norm

            # Compute full Jacobian
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id,
                pin.ReferenceFrame.LOCAL
            )

            # Extract arm DOFs
            J_arm = J_full[:, self.arm_dof_ids].copy()

            # Apply weights to Jacobian rows
            J_arm[:3, :] *= position_weight
            J_arm[3:, :] *= orientation_weight

            # Tikhonov-regularized least squares: (J^T J + (λ + w_reg) I) dq = J^T e + w_reg (q_ref - q)
            q_arm = np.array([q[dof_id] for dof_id in self.arm_dof_ids])
            A = J_arm.T @ J_arm + (self.damping + reg_weight) * np.eye(self.n_arm_dofs)
            b = J_arm.T @ err + reg_weight * (q_ref_arm - q_arm)

            try:
                dq_arm = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dq_arm = np.linalg.lstsq(J_arm, err, rcond=None)[0]

            # Clamp per-iteration step size to prevent large jumps
            dq_arm = np.clip(dq_arm, -dq_max, dq_max)

            # Update arm joints
            for idx, dof_id in enumerate(self.arm_dof_ids):
                q[dof_id] += dq_arm[idx]

            q = np.clip(q, self.q_min, self.q_max)

            if fixed_joints is not None:
                for dof_idx, val in fixed_joints.items():
                    if dof_idx < len(q):
                        q[dof_idx] = val

        err_final = np.linalg.norm(pin.log6(self.data.oMf[self.ee_frame_id].inverse() * target_pose).vector)
        return q.copy(), False, err_final

    def extract_arm_joints(self, q: np.ndarray) -> np.ndarray:
        """Extract arm joint values from full configuration."""
        return np.array([q[dof_id] for dof_id in self.arm_dof_ids])

    def set_arm_joints(self, q: np.ndarray, arm_values: np.ndarray) -> np.ndarray:
        """Set arm joint values in full configuration."""
        q_new = q.copy()
        for idx, dof_id in enumerate(self.arm_dof_ids):
            q_new[dof_id] = arm_values[idx]
        return q_new

    def get_min_z_position(self, q: np.ndarray) -> float:
        """
        Get the minimum z-position across all arm mesh vertices in world frame.
        Falls back to joint origins if collision geometry is not loaded.

        Returns:
            Minimum z-coordinate among arm mesh vertices (or joint origins as fallback)
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        min_z = float('inf')

        if self.collision_model is not None:
            pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, q)
            arm_joint_ids_set = set(self.arm_joint_ids)
            for geom_id, geom_obj in enumerate(self.collision_model.geometryObjects):
                if geom_obj.parentJoint not in arm_joint_ids_set:
                    continue
                placement = self.collision_data.oMg[geom_id]
                geom = geom_obj.geometry
                _verts_raw = geom.vertices() if callable(geom.vertices) else geom.vertices
                verts = np.asarray(_verts_raw) if _verts_raw is not None else np.empty((0, 3))
                if len(verts) > 0:
                    world_z = placement.rotation[2] @ verts.T + placement.translation[2]
                    min_z = min(min_z, world_z.min())

        # Fallback: joint origins
        if min_z == float('inf'):
            for joint_id in self.arm_joint_ids:
                z_pos = self.data.oMi[joint_id].translation[2]
                min_z = min(min_z, z_pos)
            z_pos = self.data.oMf[self.ee_frame_id].translation[2]
            min_z = min(min_z, z_pos)

        return min_z

    def is_arm_above_table(self, q: np.ndarray, z_threshold: float = 0.0) -> bool:
        """
        Check if the entire arm (all joint origins) is above z_threshold.

        Args:
            q: Configuration vector
            z_threshold: Minimum z-position threshold (default 0.0)

        Returns:
            True if all arm joint origins and EE have z > z_threshold
        """
        return self.get_min_z_position(q) > z_threshold


def solve_robot_base_pose_via_ik(
    wrist_positions_floor: np.ndarray,
    ik_solver: "ArmIKSolver",
    scene_pts_floor: np.ndarray = None,
    wrist_orientations_floor: np.ndarray = None,
    robot_base_z: float = 0.0,
    n_wrist_samples: int = 5,
    n_yaw_candidates: int = 36,
    cam_forward_floor_xy: np.ndarray = None,
) -> tuple:
    """
    Find robot base pose (x, y, yaw) in the floor-aligned frame.

    Yaw disambiguation strategy
    ---------------------------
    Priority 0 (egocentric, when cam_forward_floor_xy is provided): align robot +X
    with the camera-forward direction so the arm reaches straight forward, matching
    how the operator's arm extends in front of them.  Palm-normal scoring is bypassed.

    Primary (when wrist_orientations_floor is provided): orientation consistency.
    For each candidate yaw, transform sampled wrist orientations to robot frame
    and run position-only IK to get the actual EE orientation. The offset
    R_mano_to_ee = R_wrist_robot.T @ R_ee should be ~constant across frames for
    the correct yaw. Score = negative angular std of these estimates.

    Secondary (scene point cloud, used as tiebreaker or when no orientations):
    The correct yaw puts the scene centroid in front of the robot (+X).

    After selecting the best yaw, a warm-start arm configuration is found via
    position-only IK on the closest wrist sample.

    Returns
    -------
    base_pos_floor : (3,) [x, y, robot_base_z] in floor frame
    yaw_floor      : float, yaw in radians (robot-base Z-rotation relative to floor)
    q_init_arm     : (n_arm_dofs,) joint angles for the best-yaw warm-start frame
    """
    # Filter to frames where the hand was actually detected.
    valid = (np.all(np.isfinite(wrist_positions_floor), axis=1) &
             np.any(wrist_positions_floor != 0, axis=1))
    valid_wrists = wrist_positions_floor[valid]
    if len(valid_wrists) == 0:
        return np.array([0.0, 0.0, robot_base_z]), 0.0, None

    workspace_xy = valid_wrists[:, :2].mean(axis=0)
    workspace_z  = float(valid_wrists[:, 2].mean())

    n_sample = min(n_wrist_samples, len(valid_wrists))
    sample_idx = np.linspace(0, len(valid_wrists) - 1, n_sample).astype(int)
    wrist_sample = valid_wrists[sample_idx]

    # Sample orientation frames (same valid-frame filter as positions).
    orient_sample = None
    if wrist_orientations_floor is not None and len(wrist_orientations_floor) > 0:
        valid_orients = wrist_orientations_floor[valid] if len(wrist_orientations_floor) == len(wrist_positions_floor) else wrist_orientations_floor
        n_o = min(n_wrist_samples, len(valid_orients))
        oidx = np.linspace(0, len(valid_orients) - 1, n_o).astype(int)
        orient_sample = valid_orients[oidx]  # (K, 3, 3)

    ee_neutral = ik_solver.get_ee_position(ik_solver.q_default.copy())
    use_scene = scene_pts_floor is not None and len(scene_pts_floor) > 0

    # Egocentric override: use camera-forward direction as primary yaw scorer.
    # When cam_forward_floor_xy is provided, palm-normal scoring is skipped entirely
    # because for first-person footage the palm faces the scene (camera-forward direction),
    # which would mislead the palm-normal heuristic into placing the base on the wrong side.
    use_cam_forward = cam_forward_floor_xy is not None
    if use_cam_forward:
        cf_xy = np.asarray(cam_forward_floor_xy, dtype=np.float64)
        cf_norm = float(np.linalg.norm(cf_xy))
        if cf_norm > 1e-6:
            cf_xy = cf_xy / cf_norm
        # robot +Y = cam_forward  →  [-sin(yaw), cos(yaw)] = cf_xy
        # yaw = atan2(-cf_x, cf_y)
        _cf_preferred_yaw = np.degrees(np.arctan2(-float(cf_xy[0]), float(cf_xy[1])))
        print(f"[robot_base_ik] egocentric cam_forward scorer: floor_xy={cf_xy.round(3)} "
              f"→ preferred yaw≈{_cf_preferred_yaw:.1f}° (robot +Y ∥ cam_forward, "
              f"palm-normal scoring skipped)")

    # Check if palm normal has enough horizontal component to be useful for yaw scoring.
    # MANO Z column = palm outward normal (points away from palm surface, toward robot).
    avg_palm_normal_xy = None
    if not use_cam_forward and orient_sample is not None and len(orient_sample) > 0:
        palm_normals_floor = orient_sample[:, :, 2]  # (K, 3): Z col = palm outward normal
        avg_palm_normal = palm_normals_floor.mean(axis=0)
        pn_xy = avg_palm_normal[:2]
        pn_xy_norm = float(np.linalg.norm(pn_xy))
        if pn_xy_norm > 0.25:  # palm has meaningful horizontal component
            avg_palm_normal_xy = pn_xy / pn_xy_norm
            print(f"[robot_base_ik] palm normal in floor frame (avg): {avg_palm_normal.round(3)}, "
                  f"XY magnitude={pn_xy_norm:.3f} — using as PRIMARY yaw scorer")
        else:
            print(f"[robot_base_ik] palm normal XY too small ({pn_xy_norm:.3f}), "
                  f"falling back to scene centroid scoring")

    print(f"[robot_base_ik] EE at neutral config: {ee_neutral.round(3)}, "
          f"workspace_z={workspace_z:.3f}, "
          f"wrist Z range=[{valid_wrists[:,2].min():.3f}, {valid_wrists[:,2].max():.3f}]")
    print(f"[robot_base_ik] scoring: palm_normal={avg_palm_normal_xy is not None}, "
          f"scene_centroid={use_scene}, n_yaw_candidates={n_yaw_candidates}")

    def Rz_neg_mat(byaw):
        c, s = np.cos(byaw), np.sin(byaw)
        return np.array([[ c,  s, 0.],
                         [-s,  c, 0.],
                         [ 0., 0., 1.]])

    def floor_to_robot_pos(bx, by, byaw, pos):
        return Rz_neg_mat(byaw) @ (pos - np.array([bx, by, robot_base_z]))

    def natural_placement(byaw):
        """XY base position so workspace centroid → ee_neutral_XY in robot frame."""
        cy, sy = np.cos(byaw), np.sin(byaw)
        rotated_ee_xy = np.array([cy * ee_neutral[0] - sy * ee_neutral[1],
                                   sy * ee_neutral[0] + cy * ee_neutral[1]])
        return float(workspace_xy[0] - rotated_ee_xy[0]), float(workspace_xy[1] - rotated_ee_xy[1])

    yaw_grid = np.linspace(0.0, 2.0 * np.pi, n_yaw_candidates, endpoint=False)

    best_score = -np.inf  # higher = better
    best_byaw  = 0.0
    best_bx, best_by = natural_placement(0.0)
    scores = []

    # Subsample scene points for speed (keep up to 2000 pts)
    if use_scene:
        idx = np.random.default_rng(0).choice(len(scene_pts_floor),
                                               min(2000, len(scene_pts_floor)),
                                               replace=False)
        scene_sample = scene_pts_floor[idx]

    for byaw_cand in yaw_grid:
        bx_cand, by_cand = natural_placement(byaw_cand)
        Rz = Rz_neg_mat(byaw_cand)

        # --- Primary: palm normal alignment ---
        # The palm outward normal (MANO Z axis in floor frame) should point toward the
        # robot base: dot(palm_normal_xy, base_xy - workspace_xy) > 0 for correct yaw.
        palm_score = 0.0
        if avg_palm_normal_xy is not None:
            base_dir = np.array([bx_cand, by_cand]) - workspace_xy
            bd_norm = float(np.linalg.norm(base_dir))
            if bd_norm > 1e-6:
                base_dir /= bd_norm
            palm_score = float(np.dot(avg_palm_normal_xy, base_dir))

        # --- Secondary: scene centroid score ---
        scene_score = 0.0
        if use_scene:
            base_3d = np.array([bx_cand, by_cand, robot_base_z])
            scene_robot = (Rz @ (scene_sample - base_3d).T).T
            scene_score = float(np.mean(scene_robot[:, 0]))

        # Combine: cam_forward (egocentric) > palm normal > scene centroid > IK error
        if use_cam_forward:
            # Arm extends in robot +Y direction; align robot +Y with cam_forward.
            # robot +Y in floor = [-sin(yaw), cos(yaw)]
            # dot(robot_+Y_floor, cam_fwd) = -sin(yaw)*cf_x + cos(yaw)*cf_y
            # No scene tiebreaker: the scene_score biases toward +X which is the wrong
            # axis for egocentric (would re-introduce the side-approach bias).
            score = float(-np.sin(byaw_cand) * cf_xy[0] + np.cos(byaw_cand) * cf_xy[1])
        elif avg_palm_normal_xy is not None:
            score = palm_score + 0.01 * scene_score
        elif use_scene:
            score = scene_score
        else:
            total_err = 0.0
            for k in range(len(wrist_sample)):
                p_robot = floor_to_robot_pos(bx_cand, by_cand, byaw_cand, wrist_sample[k])
                _, _, err = ik_solver.solve_ik_position(
                    target_pos=p_robot, q_init=ik_solver.q_default.copy(),
                )
                total_err += err
            score = -total_err

        scores.append(score)
        if score > best_score:
            best_score = score
            best_byaw = byaw_cand
            best_bx, best_by = bx_cand, by_cand

    _score_label = ("cam_fwd + 0.01*scene" if use_cam_forward
                    else "palm_score + 0.01*scene" if avg_palm_normal_xy is not None
                    else "scene_score" if use_scene else "ik_error")
    print(f"[robot_base_ik] yaw grid scores ({_score_label}):")
    for yaw_c, sc in zip(yaw_grid, scores):
        marker = " ← best" if abs(yaw_c - best_byaw) < 1e-6 else ""
        print(f"    {np.degrees(yaw_c):6.1f}°  score={sc:.4f}{marker}")
    print(f"[robot_base_ik] selected: bx={best_bx:.3f}, by={best_by:.3f}, "
          f"bz={robot_base_z:.3f}, yaw={np.degrees(best_byaw):.1f}°")

    # Warm-start: position-only IK for the wrist sample closest to EE neutral.
    # Use a "natural reaching" initial configuration (elbow bent down) rather than
    # q_default (all-zeros / arm fully extended), which tends to converge to an
    # upside-down elbow-up branch for low forward-reach targets.
    base_pos_floor = np.array([best_bx, best_by, robot_base_z])
    best_sample_idx = int(np.argmin([
        np.linalg.norm(floor_to_robot_pos(best_bx, best_by, best_byaw, w) - ee_neutral)
        for w in wrist_sample
    ]))
    p0 = floor_to_robot_pos(best_bx, best_by, best_byaw, wrist_sample[best_sample_idx])
    # Natural reaching: only bias Actuator4 (elbow) to -1.5 rad so IK converges to the
    # elbow-down branch instead of the upside-down elbow-up branch.
    q_natural = ik_solver.q_default.copy()
    if len(ik_solver.arm_dof_ids) >= 4:
        q_natural[ik_solver.arm_dof_ids[3]] = -1.5  # Actuator4: elbow bent forward
    q_sol, _, _ = ik_solver.solve_ik_position(
        target_pos=p0,
        q_init=q_natural,
        reg_weight=0.0,
    )
    q_init_arm = ik_solver.extract_arm_joints(q_sol)

    return base_pos_floor, float(best_byaw), q_init_arm


def create_arm_ik_solver(arm_name: str, hand_name: str, side: str = "right", urdf_dir: str = None) -> ArmIKSolver:
    """
    Create an IK solver for any arm+hand combination.

    Args:
        arm_name: Name of the arm (e.g., "kinova", "franka")
        hand_name: Name of the hand (e.g., "xhand", "sharpa_hand")
        side: "left" or "right"
        urdf_dir: Directory containing URDF files. If None, use default.

    Returns:
        ArmIKSolver instance
    """
    if urdf_dir is None:
        # Default path relative to this file
        this_dir = Path(__file__).parent.parent
        urdf_dir = this_dir / "assets" / f"{arm_name}_{hand_name}"
    else:
        urdf_dir = Path(urdf_dir)

    urdf_path = urdf_dir / f"{arm_name}_{hand_name}_{side}.urdf"
    ee_link_name = f"{side}_hand_link"

    return ArmIKSolver(
        urdf_path=str(urdf_path),
        ee_link_name=ee_link_name,
        arm_joint_names=[f"Actuator{i}" for i in range(1, 8)],
        position_only=False,
        damping=1e-3,
        max_iters=200,
        eps=1e-3,
    )


def solve_arm_trajectory(
    ik_solver: ArmIKSolver,
    target_positions: np.ndarray,
    hand_joint_values: np.ndarray = None,
    hand_dof_indices: list = None,
    initial_arm_joints: np.ndarray = None,
    ema_alpha: float = 0.5,
    reg_weight: float = 0.01,
    input_smooth_window: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Solve IK for a trajectory of target wrist positions.
    Uses previous frame's solution as initialization for smoothness.

    Args:
        ik_solver: ArmIKSolver instance
        target_positions: (N, 3) array of target wrist positions
        hand_joint_values: (N, n_hand_joints) array of hand joint values (optional)
        hand_dof_indices: List of DOF indices for hand joints in the full model
        initial_arm_joints: (7,) initial arm joint values for first frame
        ema_alpha: EMA blend factor for output smoothing (0=all previous, 1=all new).
        reg_weight: Tikhonov regularization weight inside IK (penalizes deviation from q_init).
        input_smooth_window: Uniform filter window size for input position smoothing (1=no smoothing).
        verbose: Print progress

    Returns:
        Dict with:
            'arm_joints': (N, 7) array of arm joint solutions
            'success': (N,) boolean array
            'errors': (N,) array of final position errors
    """
    N = target_positions.shape[0]
    arm_joints = np.zeros((N, 7))
    success = np.zeros(N, dtype=bool)
    errors = np.zeros(N)

    # Optionally smooth input target trajectory
    if input_smooth_window > 1:
        target_positions = uniform_filter1d(target_positions, size=input_smooth_window, axis=0, mode='nearest')

    # Initialize
    q = ik_solver.q_default.copy()
    if initial_arm_joints is not None:
        q = ik_solver.set_arm_joints(q, initial_arm_joints)

    prev_arm = ik_solver.extract_arm_joints(q)

    for i in range(N):
        target_pos = target_positions[i]

        # Set fixed hand joints if provided
        fixed_joints = None
        if hand_joint_values is not None and hand_dof_indices is not None:
            fixed_joints = {}
            for j, dof_idx in enumerate(hand_dof_indices):
                if j < hand_joint_values.shape[1]:
                    fixed_joints[dof_idx] = hand_joint_values[i, j]

        # Solve IK using previous solution as init; regularize toward q (previous frame)
        q_sol, ok, err = ik_solver.solve_ik_position(
            target_pos=target_pos,
            q_init=q,
            fixed_joints=fixed_joints,
            reg_weight=reg_weight,
        )

        # EMA temporal smoothing of output joint angles
        curr_arm = ik_solver.extract_arm_joints(q_sol)
        smoothed_arm = ema_alpha * curr_arm + (1.0 - ema_alpha) * prev_arm
        q_sol = ik_solver.set_arm_joints(q_sol, smoothed_arm)

        arm_joints[i] = smoothed_arm
        success[i] = ok
        errors[i] = err

        prev_arm = smoothed_arm
        # Use smoothed solution as init for next frame
        q = q_sol

        if verbose and (i % 50 == 0 or i == N - 1):
            print(f"Frame {i}/{N}: error={err:.4f}, success={ok}")

    return {
        'arm_joints': arm_joints,
        'success': success,
        'errors': errors,
    }


def solve_arm_trajectory_with_pose(
    ik_solver: ArmIKSolver,
    target_positions: np.ndarray,
    target_orientations: np.ndarray,
    initial_arm_joints: np.ndarray = None,
    position_weight: float = 1.0,
    orientation_weight: float = 0.1,
    verbose: bool = False,
    max_retries: int = 20,
    joint_jump_threshold: float = float('inf'),
    position_jump_threshold: float = float('inf'),
    constrain_arm_z: bool = False,
    z_threshold: float = 0.0,
    ema_alpha: float = 0.5,
    reg_weight: float = 0.01,
    input_smooth_window: int = 5,
    valid_mask: np.ndarray = None,
) -> dict:
    """
    Solve IK for a trajectory of target wrist poses (position + orientation).
    Uses previous frame's solution as initialization for smoothness. Includes
    retry logic to avoid joint jumps. Optionally constrains entire arm mesh to stay above table.

    Args:
        ik_solver: ArmIKSolver instance
        target_positions: (N, 3) array of target wrist positions
        target_orientations: (N, 3, 3) rotation matrices OR (N, 4) quaternions (wxyz)
        initial_arm_joints: (7,) initial arm joint values for first frame
        position_weight: Weight for position error
        orientation_weight: Weight for orientation error (keep low, e.g. 0.1)
        verbose: Print progress
        max_retries: Maximum number of retries if solution jumps (default: 10)
        joint_jump_threshold: Max allowed joint angle change (radians) before retry (default: 0.3)
        position_jump_threshold: Max allowed position change (meters) before retry (default: 0.05)
        constrain_arm_z: If True, ensure entire arm mesh stays above z_threshold
        z_threshold: Minimum z-position for arm mesh (default: 0.0)
        ema_alpha: EMA blend factor for output smoothing (0=all previous, 1=all new).
        reg_weight: Tikhonov regularization weight inside IK (penalizes deviation from q_init).
        input_smooth_window: Uniform filter window size for input trajectory smoothing (1=no smoothing).
        valid_mask: Optional (N,) boolean/uint8 array. Frames where valid_mask[i]==0 are skipped
            (the previous joint configuration is held) instead of trying to solve IK on
            garbage wrist positions (e.g. frames before the hand enters the scene).

    Returns:
        Dict with:
            'arm_joints': (N, 7) array of arm joint solutions
            'success': (N,) boolean array
            'errors': (N,) array of final pose errors
            'retries': (N,) array of retry counts per frame
            'min_z_positions': (N,) array of minimum z-positions for each frame
    """
    N = target_positions.shape[0]
    arm_joints = np.zeros((N, 7))
    success = np.zeros(N, dtype=bool)
    errors = np.zeros(N)
    retries = np.zeros(N, dtype=int)
    min_z_positions = np.zeros(N)

    # Smooth input target trajectory to remove noise before running IK
    if input_smooth_window > 1:
        target_positions = uniform_filter1d(target_positions, size=input_smooth_window, axis=0, mode='nearest')
        # Smooth orientations via their rotation vectors
        if target_orientations.shape[-2:] == (3, 3):
            rotvecs = R.from_matrix(target_orientations).as_rotvec()
            rotvecs = uniform_filter1d(rotvecs, size=input_smooth_window, axis=0, mode='nearest')
            target_orientations = R.from_rotvec(rotvecs).as_matrix()

    # Clamp target wrist Z to keep the arm at a reachable height above the floor.
    # When the hand goes close to the table the arm folds severely; clamping the target
    # lets the arm hold a natural posture instead of chasing an unreachable low point.
    # When the arm still can't keep all links above floor at the clamped target, the
    # fallback uses the best unconstrained IK solution (arm tracks trajectory, may briefly
    # have links near floor) rather than freezing at the last valid config.
    if constrain_arm_z:
        target_z_clamp = z_threshold   # only clamp targets literally below floor level
        n_clamped = int((target_positions[:, 2] < target_z_clamp).sum())
        if n_clamped > 0:
            print(f"  [z_clamp] Clamping {n_clamped}/{N} frame wrist targets from below {target_z_clamp:.3f}m")
        target_positions = target_positions.copy()
        target_positions[:, 2] = np.maximum(target_positions[:, 2], target_z_clamp)

    # Initialize
    q = ik_solver.q_default.copy()
    if initial_arm_joints is not None:
        q = ik_solver.set_arm_joints(q, initial_arm_joints)

    prev_arm_joints = ik_solver.extract_arm_joints(q)
    prev_ee_pos = ik_solver.get_ee_position(q)

    for i in range(N):
        target_pos = target_positions[i]
        target_rot = target_orientations[i]

        # Convert quaternion to rotation matrix if needed
        if target_rot.shape == (4,):
            # Quaternion wxyz to rotation matrix
            target_rot = pin.Quaternion(target_rot[0], target_rot[1], target_rot[2], target_rot[3]).toRotationMatrix()

        # Create SE3 pose
        target_pose = pin.SE3(target_rot, target_pos)

        best_q_sol = None
        best_err = float('inf')
        best_ok = False
        best_joint_jump = float('inf')
        best_score = float('inf')
        # Track best solution among all retries ignoring z-check (fallback to avoid freeze).
        best_any_q_sol = None
        best_any_err = float('inf')

        for retry in range(max_retries):
            if retry == 0:
                q_init = q.copy()
            elif retry == 1 and len(ik_solver.arm_dof_ids) >= 4:
                # First retry: try natural-branch init (elbow bent down, Actuator4=-1.5).
                # If the previous frame's q is in the upside-down branch, all random
                # perturbations around it stay upside-down. This forces a natural-branch
                # restart so the IK can find the correct elbow-down configuration.
                q_init = ik_solver.q_default.copy()
                q_init[ik_solver.arm_dof_ids[3]] = -1.5
            elif retry == 2 and constrain_arm_z and len(ik_solver.arm_dof_ids) >= 4:
                # When z-constraint is active, also try elbow-UP (Actuator4=+1.5)
                # with position-only IK. Elbow-up keeps intermediate links above the
                # floor; position-only ensures convergence even when orientation is
                # complex. Subsequent retries starting from this warm-start can refine.
                q_init = ik_solver.q_default.copy()
                q_init[ik_solver.arm_dof_ids[3]] = 1.5
                q_sol, ok, err = ik_solver.solve_ik_pose(
                    target_pose=target_pose,
                    q_init=q_init,
                    position_weight=position_weight,
                    orientation_weight=0.0,
                    reg_weight=reg_weight,
                )
            elif retry == 3 and constrain_arm_z and len(ik_solver.arm_dof_ids) >= 4:
                q_init = best_q_sol.copy() if best_q_sol is not None else q.copy()
                q_init[ik_solver.arm_dof_ids[3]] = max(0.5, ik_solver.extract_arm_joints(q_init)[3])
                q_sol, ok, err = ik_solver.solve_ik_pose(
                    target_pose=target_pose,
                    q_init=q_init,
                    position_weight=position_weight,
                    orientation_weight=orientation_weight * 0.3,
                    reg_weight=reg_weight,
                )
            elif retry == 4 and constrain_arm_z and len(ik_solver.arm_dof_ids) >= 4:
                q_init = ik_solver.q_default.copy()
                q_init[ik_solver.arm_dof_ids[3]] = 2.0
                q_sol, ok, err = ik_solver.solve_ik_pose(
                    target_pose=target_pose,
                    q_init=q_init,
                    position_weight=position_weight,
                    orientation_weight=0.0,
                    reg_weight=reg_weight,
                )
            elif retry == 5 and constrain_arm_z and len(ik_solver.arm_dof_ids) >= 4:
                q_init = ik_solver.q_default.copy()
                q_init[ik_solver.arm_dof_ids[3]] = 0.8
                q_sol, ok, err = ik_solver.solve_ik_pose(
                    target_pose=target_pose,
                    q_init=q_init,
                    position_weight=position_weight,
                    orientation_weight=0.0,
                    reg_weight=reg_weight,
                )
            else:
                q_init = q.copy()
                perturbation = np.random.uniform(-0.3, 0.3, size=len(ik_solver.arm_dof_ids))
                for idx, dof_id in enumerate(ik_solver.arm_dof_ids):
                    q_init[dof_id] += perturbation[idx]
                q_init = np.clip(q_init, ik_solver.q_min, ik_solver.q_max)

            if retry not in (2, 3, 4, 5) or not constrain_arm_z:
                # Retry 0: regularize toward previous frame (q_ref=q) for temporal continuity.
                # Retries 1+: regularize toward their own init (q_ref=None → q_init) so they
                # can freely explore the target region instead of being trapped near prev frame.
                _q_ref = q if retry == 0 else None
                q_sol, ok, err = ik_solver.solve_ik_pose(
                    target_pose=target_pose,
                    q_init=q_init,
                    q_ref=_q_ref,
                    position_weight=position_weight,
                    orientation_weight=orientation_weight,
                    reg_weight=reg_weight,
                )

            curr_arm_joints = ik_solver.extract_arm_joints(q_sol)
            joint_jump = np.max(np.abs(curr_arm_joints - prev_arm_joints))

            curr_ee_pos = ik_solver.get_ee_position(q_sol)
            pos_err = float(np.linalg.norm(curr_ee_pos - target_pos))

            is_smooth = joint_jump < joint_jump_threshold

            # Score and track by POSITION error only.
            # solve_ik_pose returns the full unweighted se3 error when ok=False, which
            # includes orientation error that may be large even when position is well-achieved
            # (e.g. when orientation_weight is low). Using pos_err keeps selection stable.
            if pos_err < best_any_err:
                best_any_q_sol = q_sol.copy()
                best_any_err = pos_err

            # Check if entire arm is above table when constraint is enabled
            is_above_table = True
            if constrain_arm_z:
                is_above_table = ik_solver.is_arm_above_table(q_sol, z_threshold)

            # Prefer natural branch (Actuator4 < 0 = elbow down) unless z-constraint
            # is active — in that case elbow-up may be the only way to keep the arm
            # above the floor, so don't penalize it.
            is_natural_branch = (len(ik_solver.arm_dof_ids) < 4 or
                                 ik_solver.extract_arm_joints(q_sol)[3] < 0)
            branch_penalty = 0.0 if (is_natural_branch or constrain_arm_z) else 0.05
            # Penalize joint discontinuity so the selection prefers smooth transitions
            # among solutions with similar position error (prevents inter-branch oscillation).
            # 0.02 per rad means a 1-rad branch jump costs as much as a 20mm position error —
            # only tolerate branch switches if the new branch is significantly better in position.
            continuity_penalty = min(joint_jump, 3.0) * 0.02
            solution_score = (pos_err + continuity_penalty +
                              (0 if is_above_table else 0.5) +
                              branch_penalty)

            # Early exit: only when position AND continuity are both good.
            # Requiring joint_jump < 0.4 prevents accepting a different branch
            # just because it happens to achieve pos_err < 5mm.
            if is_smooth and is_above_table and pos_err < 0.005 and joint_jump < 0.4:
                best_q_sol = q_sol
                best_err = pos_err
                best_ok = ok
                best_joint_jump = joint_jump
                best_score = solution_score
                retries[i] = retry
                break
            elif is_above_table and solution_score < best_score:
                best_q_sol = q_sol
                best_err = pos_err
                best_ok = ok
                best_joint_jump = joint_jump
                best_score = solution_score
                retries[i] = retry

        if best_q_sol is None:
            # No valid z-safe solution found — use best position-tracking solution.
            if best_any_q_sol is not None:
                best_q_sol = best_any_q_sol
                best_err = best_any_err
                best_ok = False
                best_joint_jump = float('inf')
                retries[i] = max_retries
            else:
                best_q_sol = q.copy()
                best_err = float('inf')
                best_ok = False
                best_joint_jump = float('inf')
                retries[i] = max_retries

        # Error-conditional blending based on POSITION error:
        #   - Good (pos_err < 0.01m / 1cm): use fully
        #   - Bad  (pos_err > 0.15m / 15cm): hold previous frame
        #   - In between: linear blend
        # Using position error (not se3 error) ensures that good position tracking is
        # accepted even when orientation_weight is low and orientation is off.
        curr_arm = ik_solver.extract_arm_joints(best_q_sol)
        if i == 0:
            effective_alpha = 1.0
        else:
            _ee_pos = ik_solver.get_ee_position(best_q_sol)
            blend_err = float(np.linalg.norm(_ee_pos - target_pos))
            _err_good, _err_bad = 0.01, 0.15
            if blend_err <= _err_good:
                effective_alpha = 1.0
            elif blend_err >= _err_bad:
                effective_alpha = 0.0   # hold previous frame
            else:
                effective_alpha = 1.0 - (blend_err - _err_good) / (_err_bad - _err_good)
        smoothed_arm = effective_alpha * curr_arm + (1.0 - effective_alpha) * prev_arm_joints

        # Hard joint-velocity clamp: prevent any single frame from moving more than
        # max_joint_delta radians. Scales the joint delta proportionally so the arm
        # moves toward the target but never teleports. SavGol handles residual jitter.
        # 0.15 rad/frame means a 3-rad branch switch drifts over ~20 frames, not 1 frame.
        if i > 0:
            _max_joint_delta = 0.15   # rad per frame
            _delta = smoothed_arm - prev_arm_joints
            _delta_max = float(np.max(np.abs(_delta)))
            if _delta_max > _max_joint_delta:
                smoothed_arm = prev_arm_joints + _delta * (_max_joint_delta / _delta_max)

        best_q_sol = ik_solver.set_arm_joints(best_q_sol, smoothed_arm)

        arm_joints[i] = smoothed_arm
        min_z = ik_solver.get_min_z_position(best_q_sol)
        min_z_positions[i] = min_z

        # Success: position within 1cm, smooth motion, and (if constrained) EE above table.
        # Uses position error (stored in best_err) rather than full se3 convergence flag,
        # since ok=True requires weighted se3 < eps which is too strict when orientation_weight is low.
        pos_achieved = best_err < 0.01   # 1cm threshold
        if constrain_arm_z:
            success[i] = pos_achieved and (best_joint_jump < joint_jump_threshold) and (min_z > z_threshold)
        else:
            success[i] = pos_achieved and (best_joint_jump < joint_jump_threshold)
        errors[i] = best_err

        prev_arm_joints = smoothed_arm.copy()
        prev_ee_pos = ik_solver.get_ee_position(best_q_sol)
        q = best_q_sol

        if verbose and (i % 50 == 0 or i == N - 1):
            if constrain_arm_z:
                print(f"Frame {i}/{N}: error={best_err:.4f}, success={success[i]}, "
                      f"retries={retries[i]}, joint_jump={best_joint_jump:.3f}, min_z={min_z:.4f}, above_z={min_z > z_threshold}")
            else:
                print(f"Frame {i}/{N}: error={best_err:.4f}, success={success[i]}, "
                      f"retries={retries[i]}, joint_jump={best_joint_jump:.3f}, min_z={min_z:.4f}")

    return {
        'arm_joints': arm_joints,
        'success': success,
        'errors': errors,
        'retries': retries,
        'min_z_positions': min_z_positions,
    }


def euler_to_rotation_matrix(angles, side) -> np.ndarray:
    """Convert a list of euler angles to to rotation matrix."""
    angles[:, 2] *= -1
    mats = []
    for a in angles:
        euler_xyz = R.from_euler("ZXY", a).as_euler("XYZ")
        euler_xyz[1] += np.pi
        euler_xyz[2] = np.pi / 2 - euler_xyz[2]
        if side == "left":
            euler_xyz[2] += np.pi
        mat = R.from_euler("XYZ", euler_xyz).as_matrix()
        mats.append(mat)
    return np.array(mats)



def _get_ee_frame_correction(hand_name: str):
    """
    Return a constant rotation R_correction such that:
        R_target_wuji = R_target_xhand @ R_correction
    where R_target_xhand is the EE orientation from euler_to_rotation_matrix (calibrated
    for the xhand right_hand_link frame) and R_target_wuji is the correct IK target
    for the given hand's right_hand_link frame.

    The euler_to_rotation_matrix function was empirically calibrated against an xhand
    mount convention where EE->mount rpy="1.57 0 3.14" (not "1.57 0 0" as in the current
    URDF). That calibration gives:
      R_xhand_calibrated = Rx(pi/2) @ Rz(pi) @ Rx(-pi/2) @ Ry(pi/2)
                         = [[0,0,-1],[0,1,0],[1,0,0]]
    For wuji (EE->mount rpy="1.57 0 0", mount->hl rpy="0 1.57 1.57"):
      R_wuji_total = [[0,0,1],[0,-1,0],[1,0,0]]
    R_correction = R_xhand_calibrated^T @ R_wuji_total
                 = [[0,0,1],[0,1,0],[-1,0,0]] @ [[0,0,1],[0,-1,0],[1,0,0]]
                 = [[1,0,0],[0,-1,0],[0,0,-1]] = Rx(pi)

    Note: if you recompute from the CURRENT xhand URDF (rpy="1.57 0 0") you get Rz(pi),
    but euler_to_rotation_matrix embeds the OLD calibration, so Rx(pi) is correct here.

    Returns None if no correction is needed (xhand or unknown hand).
    """
    if hand_name is None:
        return None
    h = hand_name.lower()
    if 'wuji' in h:
        # No correction needed: wuji and xhand share the same R_EE_to_right_hand_link
        # (= Rz(pi/2)), so euler_to_rotation_matrix output applies directly to wuji.
        # Diagnostic confirmed: pre-correction Z-axis matches human finger direction;
        # any correction that flips Z (Rx/Ry/Rz pi) reverses the approach direction.
        return None
    return None


def solve_arm_ik_from_floating_hand(
    retar_data: dict,
    retarget_input_cfg: dict,
    hand_name: str,
    floating_hand_retargeter_fname: str | None = None,
    start: int = 0,
    num_envs: int | None = None,
    floating_hand_qpos: dict[str, np.ndarray] | None = None,
    constrain_arm_z: bool = False,
    z_threshold: float = 0.05,
    T_robot_world: np.ndarray | None = None,
    arm_urdf_paths: dict[str, str] | None = None,
    initial_arm_joints: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    orientation_weight: float = 1.0,
) -> dict:
    """
    Solve arm IK using floating hand retargeting/optimization data.

    Floating hand data can be provided in two ways:
    - ``floating_hand_retargeter_fname``: path to a cached .pt file produced by
      parallel_retarget (contains ``["retarget_data"][side]["joint_qpos"]``).
      ``start`` and ``num_envs`` are used to slice the loaded data.
    - ``floating_hand_qpos``: dict mapping side ('left'/'right') to a
      (N, num_dofs) numpy array ordered as [wrist_6dof, hand_joints].
      When provided, ``start``/``num_envs`` are ignored (data is used as-is).

    Args:
        constrain_arm_z: If True, constrain arm joints to keep end-effector above table (z > 0).

    Returns modified retar_data with arm IK solutions written in.
    """
    import os

    print("\n=== Solving IK for arm joints using floating hand data ===")

    if floating_hand_retargeter_fname is None and floating_hand_qpos is None:
        raise ValueError("Must provide either floating_hand_retargeter_fname or floating_hand_qpos")

    # Extract arm and hand names from config
    config_arm_name = retarget_input_cfg.get("arm_name", None)
    config_hand_name = retarget_input_cfg.get("hand_name", None)

    # Parse from hand_name if not in config (format: {arm}_{hand})
    if config_arm_name is None or config_hand_name is None:
        parts = hand_name.rsplit('_', 1)  # Split from right to handle multi-underscore hand names
        if len(parts) == 2:
            config_arm_name = parts[0]
            config_hand_name = parts[1]
        else:
            raise ValueError(f"Cannot parse arm and hand names from '{hand_name}'. Expected format: '{{arm}}_{{hand}}'")

    print(f"Detected arm: '{config_arm_name}', hand: '{config_hand_name}'")

    # Get floating hand DOF joint names from config
    floating_hand_dof_joints = retarget_input_cfg.get("arm_dof_joints", None)
    floating_hand_hand_joints = retarget_input_cfg.get("hand_dof_joints", None)

    if floating_hand_dof_joints is None or floating_hand_hand_joints is None:
        raise ValueError(
            f"Config file must contain 'arm_dof_joints' and 'hand_dof_joints' for arm-based retargeting. "
            f"Please add these fields to the retarget_config.yaml file."
        )

    # load floating hand data from cache
    floating_hand_cached = None
    if floating_hand_qpos is None:
        assert floating_hand_retargeter_fname is not None
        assert os.path.exists(floating_hand_retargeter_fname), (
            f"Cached floating hand retargeter not found: {floating_hand_retargeter_fname}\n"
            f"Run floating hand retargeting first."
        )
        floating_hand_cached = torch.load(floating_hand_retargeter_fname, weights_only=False)["retarget_data"]
        print(f"Loaded cached floating hand retargeter from {floating_hand_retargeter_fname}")

    for side in ['left', 'right']:
        print(f"\nProcessing {side} hand...")

        if floating_hand_qpos is not None and side in floating_hand_qpos:
            # data provided directly as (N, num_dofs) array
            floating_hand_retargeted = floating_hand_qpos[side]
        elif floating_hand_cached is not None and side in floating_hand_cached:
            # load from cached file
            joint_qpos = floating_hand_cached[side]['joint_qpos']
            floating_hand_retargeted = np.array([joint_qpos[key].detach().cpu().numpy() for key in floating_hand_dof_joints[side] + floating_hand_hand_joints[side]]).T
            floating_hand_retargeted = floating_hand_retargeted[start:start+num_envs]
        else:
            print(f"  Skipping {side} hand (no data available)")
            continue

        wrist_6dof = floating_hand_retargeted[:, :6]
        floating_hand_hand_qpos = floating_hand_retargeted[:, 6:]
        wrist_positions = wrist_6dof[:, :3].copy()

        if 'wuji' in (config_hand_name or '').lower():
            # For wuji, the floating hand URDF uses the same 6DOF joint chain
            # (Rz(roll) @ Rx(pitch) @ Ry(-yaw)) and the 6DOF values directly encode the
            # right_hand_link orientation in robot frame.  euler_to_rotation_matrix has
            # xhand-specific pi corrections that scramble this orientation, so bypass it.
            _angles = wrist_6dof[:, 3:6].copy()
            wrist_orientations = np.array([
                R.from_euler("ZXY", [a[0], a[1], -a[2]]).as_matrix()
                for a in _angles
            ])
            print(f"  [orient_diag] Wuji: using direct ZXY 6DOF rotation (bypassing euler_to_rotation_matrix)")
        else:
            wrist_orientations = euler_to_rotation_matrix(wrist_6dof[:, 3:6], side)

        # Log pre-correction frame-0 orientation so we can diagnose correction issues
        R0_pre = wrist_orientations[0]
        print(f"  [orient_diag] Frame-0 BEFORE correction (euler_to_rotation_matrix output):")
        print(f"    X-axis: {R0_pre[:, 0]}")
        print(f"    Y-axis: {R0_pre[:, 1]}")
        print(f"    Z-axis: {R0_pre[:, 2]}")

        # Apply EE-frame correction for hands with different mount convention than xhand.
        _ee_correction = _get_ee_frame_correction(config_hand_name)
        if _ee_correction is not None:
            wrist_orientations = np.stack([R @ _ee_correction for R in wrist_orientations])
            print(f"  [EE frame] Applied correction for hand '{config_hand_name}':")
            print(f"    correction matrix:\n{_ee_correction}")
        R0_post = wrist_orientations[0]
        print(f"  [orient_diag] Frame-0 AFTER correction (IK target):")
        print(f"    X-axis: {R0_post[:, 0]}")
        print(f"    Y-axis: {R0_post[:, 1]}")
        print(f"    Z-axis: {R0_post[:, 2]}")

        # Transform wrist positions and orientations from world frame to robot base frame
        if T_robot_world is not None:
            R_rw = T_robot_world[:3, :3]
            t_rw = T_robot_world[:3, 3]
            wrist_positions = (R_rw @ wrist_positions.T).T + t_rw
            wrist_orientations = np.stack([R_rw @ R for R in wrist_orientations])
            print(f"  [T_robot_world] Applied world->robot transform to wrist positions/orientations")

        actuated_dof_names = retar_data[side]['actuated_dof_names']
        arm_indices = [actuated_dof_names.index(f"Actuator{i}") for i in range(1, 8)]
        hand_indices = [actuated_dof_names.index(hj) for hj in floating_hand_hand_joints[side] if hj in actuated_dof_names]

        hand_qpos_arr = retar_data[side]['hand_qpos']

        for i, hand_idx in enumerate(hand_indices):
            if i < floating_hand_hand_qpos.shape[1]:
                hand_qpos_arr[:, hand_idx] = floating_hand_hand_qpos[:, i]

        if arm_urdf_paths is not None and side in arm_urdf_paths:
            ik_solver = ArmIKSolver(
                urdf_path=str(arm_urdf_paths[side]),
                ee_link_name=f"{side}_hand_link",
                arm_joint_names=[f"Actuator{i}" for i in range(1, 8)],
                position_only=False,
                damping=1e-4,
                max_iters=200,
                eps=1e-3,
            )
        else:
            ik_solver = create_arm_ik_solver(arm_name=config_arm_name, hand_name=config_hand_name, side=side)
        ik_result = solve_arm_trajectory_with_pose(
            ik_solver=ik_solver,
            target_positions=wrist_positions,
            target_orientations=wrist_orientations,
            initial_arm_joints=initial_arm_joints,
            position_weight=5.0,
            orientation_weight=orientation_weight,
            verbose=True,
            constrain_arm_z=constrain_arm_z,
            z_threshold=z_threshold,
            ema_alpha=1.0,         # error-conditional blending handles this internally
            reg_weight=0.1,        # regularization toward prev frame (q_ref) prevents null-space drift without blocking position tracking
            input_smooth_window=1,
            valid_mask=valid_mask,
        )
        arm_joint_solutions = ik_result['arm_joints']
        ik_success = ik_result['success']
        ik_errors = ik_result['errors']
        min_z_positions = ik_result['min_z_positions']
        print(f"  IK success rate: {ik_success.mean()*100:.1f}%")
        print(f"  Mean error: {ik_errors.mean():.4f}, Max error: {ik_errors.max():.4f}")

        # Diagnostic: compare target vs achieved FK orientation for frame 0 and a mid frame
        _diag_frames = [0, min(len(arm_joint_solutions)//2, len(arm_joint_solutions)-1)]
        for _fi in _diag_frames:
            _q_diag = ik_solver.q_default.copy()
            _q_diag[ik_solver.arm_dof_ids] = arm_joint_solutions[_fi]
            _achieved_pose = ik_solver.get_ee_pose(_q_diag)
            _R_achieved = _achieved_pose.rotation
            _R_target = wrist_orientations[_fi]
            print(f"  [orient_diag] Frame {_fi} TARGET  right_hand_link orientation:")
            print(f"    X: {_R_target[:, 0]}  Y: {_R_target[:, 1]}  Z: {_R_target[:, 2]}")
            print(f"  [orient_diag] Frame {_fi} ACHIEVED right_hand_link orientation (FK):")
            print(f"    X: {_R_achieved[:, 0]}  Y: {_R_achieved[:, 1]}  Z: {_R_achieved[:, 2]}")
            _pos_target = wrist_positions[_fi]
            _pos_achieved = _achieved_pose.translation
            print(f"  [orient_diag] Frame {_fi} pos target={_pos_target}  achieved={_pos_achieved}")

        # Offline smoothing: apply Savitzky-Golay filter to the full joint trajectory.
        # This removes per-frame jitter from the IK finding different local solutions
        # without restricting the trajectory range (SavGol preserves peaks better than Gaussian).
        N_frames = arm_joint_solutions.shape[0]
        _savgol_window = min(11, N_frames if N_frames % 2 == 1 else N_frames - 1)
        if _savgol_window >= 5:
            from scipy.signal import savgol_filter
            arm_joint_solutions = savgol_filter(
                arm_joint_solutions, window_length=_savgol_window, polyorder=2, axis=0
            )
            print(f"  Applied SavGol smoothing (window={_savgol_window})")
        if constrain_arm_z:
            above_table_count = (min_z_positions > 0.0).sum()
            print(f"  Arm z-constraint: {above_table_count}/{len(min_z_positions)} frames have all meshes above z=0")
            print(f"  Min z-pos range: {min_z_positions.min():.4f} to {min_z_positions.max():.4f}")

        for i, arm_idx in enumerate(arm_indices):
            hand_qpos_arr[:, arm_idx] = arm_joint_solutions[:, i]

        retar_data[side]['hand_qpos'] = hand_qpos_arr
        retar_data[side]['floating_hand_wrist_6dof'] = wrist_6dof
        retar_data[side]['floating_hand_hand_qpos'] = floating_hand_hand_qpos
        retar_data[side]['arm_ik_result'] = ik_result

    print("\n=== IK solving complete ===\n")

    return retar_data