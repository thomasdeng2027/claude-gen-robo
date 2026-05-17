"""SYSTEM_PROMPT, BOILERPLATE, and prompt builders for the keypoint pipeline.

Pipeline goal: after a full rollout, the object's 4 bbox-corner keypoints must
match the goal keypoints (last waypoint aligned to run-time init pose) within
`SUCCESS_TOLERANCE` metres. The generated script must be a complete, runnable
Python file.  BOILERPLATE is pre-filled by pipeline.py; Claude writes only the
motion-control section after it.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

from task_spec import KeypointTaskSpec
from trajectory_logger import format_for_prompt, load_for_prompt


# =============================================================================
# HAND JOINT SEMANTICS  (derived from retarget_config.yaml)
# =============================================================================

HAND_JOINT_SEMANTICS = """\
=== HAND JOINTS (XHand Right — 19 DOF total) ===
DOF layout: indices 0–6 = Actuator1…7 (arm); indices 7–18 = hand joints in _dof_names order.
  right_hand_thumb_bend_joint, right_hand_thumb_rota_joint1/2,
  right_hand_index_bend_joint, right_hand_index_joint1/2,
  right_hand_mid_joint1/2, right_hand_ring_joint1/2, right_hand_pinky_joint1/2

Grasp targets (both defined in boilerplate — use SELECTED_GRASP_TARGET):
  HAND_CLOSE_TARGET      — power grasp (object cross-section >= 0.04 m)
  PRECISION_GRASP_TARGET — small/flat objects (cross-section < 0.04 m)

Closing sequence (two-stage, REQUIRED):
  Stage 3a (steps 0–74):   ramp thumb_bend, thumb_rota_joint1/2, index_bend, *_joint1
  Stage 3b (steps 75–149): ramp *_joint2 (distal) joints
  thumb_rota_joint1 must reach >= 0.8 rad before transport starts.
"""


# =============================================================================
# MANIPULATION STRATEGIES  (reusable across all tasks)
# =============================================================================

MANIPULATION_STRATEGIES = """\
=== MANIPULATION STRATEGIES ===

PRIORITY ORDER: (1) stable grasp that couples object to hand → (2) trajectory tracking.
A push that moves the object near the goal does NOT count — trajectory_coupled must be True.

--- KEY CONSTRAINTS ---
  Speed limits: 0.004 m/step (Phase 2 & 4). Never assign arm joints directly — always IK.
  Standoffs: _APPROACH_STANDOFF = 0.15 m (open-hand); _GRASP_STANDOFF = 0.07 m (contact anchor).

  PER-ENV APPROACH DIRECTIONS — use APPROACH_DIRS_BATCH from boilerplate (already computed):
    APPROACH_DIRS_BATCH : (N_ENVS, 3) unit vectors pointing FROM robot base TOWARD object.
    Do NOT recompute approach directions. Just alias at the top of your motion code:
      _approach_dirs = APPROACH_DIRS_BATCH  # (N_ENVS, 3), robot-base → object, Z~0.07 tilt

    *** DO NOT compute _approach_dirs from the EE position. The EE starts ~0.15 m PAST the
    object in Y, so EE-derived direction points AWAY from the robot (wrong direction). ***

    IK EE vs simulation EE — TWO DISTINCT MEASUREMENTS (do not confuse them):
      get_ik_ee_poses_all_envs(cmds)                   → FK on IK-solution joints.
        Error vs IK target is always ~0. Only checks IK algebra. NOT physics state.
      get_ik_ee_poses_all_envs(robots.get_joint_positions()) → FK on ACTUAL joints.
        This tells you where xhand_mount_right ACTUALLY IS in physics.
      get_ee_pos_all_envs(stage, N_ENVS)               → knuckle centroid from USD stage.
        This is what reach/grasp scoring uses. Always ~IK_EE_MOUNT_ADJ_M ahead of
        the actual xhand_mount_right position along the approach direction.

    solve_ik_batch_all_envs moves xhand_mount_right, but the simulation measures EE as
    knuckle_centroid. xhand_mount_right is ~IK_EE_MOUNT_ADJ_M (= 0.10 m) behind
    knuckle_centroid along the approach direction. To place knuckle_centroid at standoff d
    from the object, add this adjustment to each IK target:
      _IK_EE_ADJ = IK_EE_MOUNT_ADJ_M   # imported constant = 0.10 m

    Per-env standoff targets (all shape (N,3)) — IK targets for xhand_mount_right.
    *** CRITICAL: SUBTRACT approach direction (not add). Adding places targets PAST the object,
    on the far side from the robot — the arm extends away and never grasps. ***
      _IK_EE_ADJ    = IK_EE_MOUNT_ADJ_M   # 0.10 m — xhand_mount_right is 10cm behind knuckles
      grasp_tgts    = OBJ_INIT_POS_BATCH - _approach_dirs * (_GRASP_STANDOFF    + _IK_EE_ADJ)
      approach_tgts = OBJ_INIT_POS_BATCH - _approach_dirs * (_APPROACH_STANDOFF + _IK_EE_ADJ)
      grasp_anchor  = live_obj_pos        - _approach_dirs * (_GRASP_STANDOFF    + _IK_EE_ADJ)
      # Clamp Z so the arm never targets below 5 cm (prevents floor-chasing knocked objects):
      grasp_tgts[:, 2] = np.maximum(grasp_tgts[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)
      grasp_anchor[:, 2] = np.maximum(grasp_anchor[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)

    IK convergence check — never use FK(cmds) vs IK target as a reach signal (always 0).
    Instead: compute FK(robots.get_joint_positions()) and compare to FK(cmds).
    If the gap > 0.05 m the robot has not converged — run more hold steps, not more IK calls.
    The true reach signal is: knuckle_centroid distance to object < REACH_STANDOFF + 0.05 m.

    Phase 2 — PREFERRED: use plan_motion_all_envs for smooth, collision-free approach.
    It plans per-env trajectories offline (MotionGen), then executes them at 2 steps/waypoint:
      approach_cmds = solve_ik_batch_all_envs(approach_tgts, cur_cmds, closed_hand=False)
      _mg_trajs = plan_motion_all_envs(approach_cmds, cur_cmds)   # (N, T, 7) arm joints
      _arm_cols = [_dof_idx[jn] for jn in ARM_JOINT_NAMES]
      for _wi in range(_mg_trajs.shape[1]):
          cur_cmds[:, _arm_cols] = _mg_trajs[:, _wi, :]
          robots.set_joint_position_targets(cur_cmds)
          world.step(render=False)
          world.step(render=False)
      for _s in range(100):
          robots.set_joint_position_targets(cur_cmds)
          world.step(render=False)

    Phase 2 — FALLBACK (when plan_motion_all_envs unavailable, i.e. MotionGen not loaded):
    *** NEVER use "warm standoff before the object" with linear interpolation. The arm starts with
    EE PAST the object (~0.15m beyond it in Y). Linear interp to a pre-object standoff sweeps
    the arm BACKWARD through the object, knocking it every time. ***
    *** DO NOT use "approach from above" (IK to above_tgts at Z+0.20 then linear interp) —
    empirical testing shows the IK FAILS for all envs for these elevated positions, leaving
    above_cmds = home joints. Linear interpolation to home joints does nothing and wastes steps. ***

    CORRECT Phase 2 — DIRECT INCREMENTAL APPROACH:
    Start from the ACTUAL current EE positions (home position), move incrementally toward
    grasp_tgts at 0.004 m/step. Use 450 steps total (400 incremental + 50 hold).

      grasp_tgts = OBJ_INIT_POS_BATCH - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ)
      grasp_tgts[:, 2] = np.maximum(grasp_tgts[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)
      # Start from actual current EE positions (home)
      _ee_now = get_ee_pos_all_envs(stage, N_ENVS)
      _reach_pos = np.array([
          _ee_now[ei] if _ee_now[ei] is not None else grasp_tgts[ei]
          for ei in range(N_ENVS)
      ], dtype=np.float64)
      _MAX_STEP_2 = 0.004   # m/step
      for _s in range(400):
          _delta = grasp_tgts - _reach_pos
          _dist  = np.linalg.norm(_delta, axis=1, keepdims=True)
          _step  = np.where(_dist > _MAX_STEP_2,
                            _delta * _MAX_STEP_2 / np.maximum(_dist, 1e-8), _delta)
          _reach_pos += _step
          cur_cmds = solve_ik_batch_all_envs(_reach_pos, cur_cmds, closed_hand=False)
          robots.set_joint_position_targets(cur_cmds)
          world.step(render=False)
      for _s in range(50):
          robots.set_joint_position_targets(cur_cmds)
          world.step(render=False)

--- PHASE 2 REACH ---
  Single-phase: incremental IK from current EE positions to grasp standoff (400 steps + 50 hold).
  After: print REACH_SCORE, REACH_X/Y/Z_OFFSET, OBJ_TILT_AFTER_REACH.

--- PHASE 3 GRASP (150 steps) ---
  Hold arm at grasp standoff via IK (LIVE object positions each step). Ramp fingers two-stage.

  GRASP STABILITY: after finger closure, check EE proximity to object:
      ee_to_obj_post3[ei] = norm(ee_pos[ei] - obj_pos[ei])   # measured AFTER Phase 3
      grasp_stable_per_env[ei] = (ee_to_obj_post3[ei] < 0.08) and (obj_tilt_per_env[ei] < 45)
  Do NOT use min_ee_per_env as a proxy — that fires on brief approach contact, not stable grasp.

  OBJECT TILT: measure RELATIVE to initial settled orientation (avoids mesh-frame ambiguity):
      init_up = np.array([_qrotate(OBJ_INIT_ROT_BATCH[ei], [0,0,1]) for ei in range(N_ENVS)])
      curr_up = np.array([_qrotate(quats_wxyz[ei],          [0,0,1]) for ei in range(N_ENVS)])
      obj_tilt_per_env = np.degrees(np.arccos(np.clip(np.sum(init_up*curr_up, axis=1), -1, 1)))

  OBJECT-RELATIVE RECIPE FROM BEST ENV: after grasp, pick best_ei (lowest cp_err or min_ee_to_obj).
    Extract EE offset relative to object center from that env and apply to ALL envs:
      best_ee_obj_offset = ee_pos[best_ei] - obj_pos[best_ei]
      for ei in range(N_ENVS): ee_obj_offset[ei] = best_ee_obj_offset
    Exact arm joints are only IK seeds. The transferable signal is the relative EE pose.

  CRASH CHECK before Phase 3: verify EE-to-obj >= 0.04 m per env; retreat 30 steps if any < 0.04 m.

  BEST_GRASP_STATE (required print — pipeline reads this):
    print("BEST_GRASP_STATE: " + json.dumps(json_safe({
        "env_id": best_ei,
        "all_joints": list(robots.get_joint_positions()[best_ei]),
        "reach_arm_joints": list(reach_arm_joints_per_env[best_ei]),
        "ee_obj_offset": list(ee_pos[best_ei] - obj_pos[best_ei]),
        "max_lift": float(max_lift_per_env[best_ei]),
        "cp_err": float(cp_errs_post[best_ei]) if cp_errs valid else -1.0,
    })))

--- PHASE 4 TRANSPORT (500 steps) ---
  For envs in grasp_gate: ee_tgt[ei] = WAYPOINTS_WORLD_BATCH[ei, wp_idx[ei]] + ee_obj_offset[ei].
  For envs NOT in grasp_gate: ee_tgt[ei] = live_obj_pos[ei] - _approach_dirs[ei] * (_GRASP_STANDOFF + _IK_EE_ADJ).
    Clamp Z: ee_tgt[ei, 2] = max(ee_tgt[ei, 2], OBJ_INIT_POS_BATCH[ei, 2] - 0.02)
  Re-apply SELECTED_GRASP_TARGET fingers every step.
"""



# =============================================================================
# CONTROL POLICY  (per-phase control rules — non-negotiable)
# =============================================================================

CONTROL_POLICY = """\
=== CONTROL POLICY ===
Phase 2: arm IK only (no fingers, no contact). NEVER assign arm joints directly — always IK.
Phase 3: hold arm via IK at grasp standoff; ramp fingers only (proximal first, distal second).
  Fallback gate — use this (never require max_lift, which is often 0 even when grasping):
    grasp_gate[ei] = grasp_stable_per_env[ei] or (ee_to_obj_post3[ei] < 0.12)
  Rationale: if EE is within 12cm of the object after finger closure, treat it as grasped
  and follow waypoints. Requiring max_lift > 0.01m is too strict — the arm rarely lifts
  the object during Phase 3 (only 50ms hold after closure). Use EE proximity as the gate.
Phase 3 IK anchor: LIVE object positions each step — never OBJ_INIT_POS_BATCH (stale after drift).
Phase 4: IK + locked fingers; only advance waypoints for envs in grasp_gate.
  Non-grasped envs: target live_obj - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ) (SUBTRACT, not add).
"""


# =============================================================================
# CONTACT VALIDITY RULE  (replaces all object-specific "do not push" rules)
# =============================================================================

CONTACT_VALIDITY_RULE = """\
=== CONTACT VALIDITY ===
Valid progress requires controlled contact (EE near object + velocities matched).
trajectory_coupled_any = True (lift > 0.02 m AND EE within 15 cm) is required for success.
Pushing the object without lifting it does NOT count.

Penalties (per env, per step):
  KNOCKED OVER (obj_tilt > 45°): severe penalty. Tipping the object is worse than no contact.
  NO CONTACT (EE never reached object): mild penalty. Do not reward noise-driven waypoint drift.

Priority: grasp_stable → no tilt → kp_dist decreasing while coupled → final keypoint distance.
"""


# =============================================================================
# Shared prompt helpers
# =============================================================================

def _write_traj_file(trajectory: list[dict]) -> str:
    """Write trajectory to a temp JSON file; return the path."""
    import json as _json
    import tempfile as _tempfile
    with _tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="kpt_traj_"
    ) as _f:
        _json.dump(trajectory or [], _f)
        return _f.name


def _tag_line(stdout_lines: list[str], tag: str) -> Optional[str]:
    for l in stdout_lines:
        if l.startswith(tag + ":") or l.startswith(tag + " "):
            return l
    return None


def _parse_float(line: Optional[str]) -> Optional[float]:
    if not line:
        return None
    try:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line.split(":", 1)[1])
        if m:
            return float(m.group(0))
    except Exception:
        pass
    return None


# =============================================================================
# IN-CONTEXT EXAMPLE HELPERS
# =============================================================================

_IN_CONTEXT_EXAMPLE_DIR = Path(__file__).parent / "in_context_examples"
_TRAJ_ROOT_ICE = Path(__file__).parent.parent / "trajectories"

# Semantic trajectory aliases.
# Maps task_name -> (semantic_summary, trajectory_source_task)
# The numeric trajectory may be shared; prompt wording is task-specific.
_TRAJ_SEMANTIC_ALIASES: dict[str, tuple[str, str]] = {
    "cylinder_pour": (
        "Object stays near its start position and rotates (tilts) downward about its side axis. "
        "Final orientation has the opening pointing sideways or down — a pouring pose.",
        "cylinder_pour",
    ),
    "cup_lift": (
        "Object rises vertically from the table surface. "
        "Orientation remains approximately constant throughout the lift.",
        "cup_lift",
    ),
    "cube_pull_rotate": (
        "Object pivots about a bottom edge. One face lifts while the opposite face stays in contact "
        "with the table. Final orientation is rotated ~90° from start.",
        "cube_pull_rotate",
    ),
    "cylinder_pour_traj": (
        "Object is grasped, lifted ~0.37 m, then tilted ~45° following a recorded pour trajectory. "
        "KEY PATTERN (generalizes to any pour/lift/tilt trajectory task): "
        "(1) At grasp time record per-env ee_offset = ee_world - obj_world. "
        "    This locks the hand's relative position to the object. "
        "(2) For each waypoint k: ee_tgt = WAYPOINTS_WORLD_BATCH[ei,k] + ee_offset[ei]. "
        "(3) For orientation: pass target_quats=WAYPOINTS_ROT_BATCH[ei,k] to the IK solver "
        "    so the wrist tilts with the object as it pours. "
        "(4) In the parallel pipeline WAYPOINTS_WORLD_BATCH and WAYPOINTS_ROT_BATCH are "
        "    pre-computed per env by the boilerplate — use them directly, do NOT recompute alignment. "
        "(5) After grasp, transfer best-env ee_offset to any env that failed to grasp, "
        "    so they still attempt transport.",
        "cylinder_pour",   # reuses cylinder_pour trajectory JSON
    ),
}

_TASK_DESCRIPTIONS: dict[str, str] = {
    "cylinder_pour": "Pour from a cylinder (tilt the cylinder to pour).",
    "cup_lift":      "Lift a cup vertically off the table.",
    "cube_pull_rotate": "Pull and rotate a cube ~90° about its bottom edge.",
    "cylinder_pour_traj": (
        "Grasp a cylinder then follow the recorded pour trajectory: lift ~0.37 m then tilt ~45°. "
        "Demonstrates trajectory following via IK waypoint tracking."
    ),
}


def load_solution_code(task_name: str, max_chars: int = 13000) -> str:
    """Load the .py solution file for an in-context example task.

    Truncates to *max_chars* to keep prompts token-efficient.
    The full code is on disk; Claude only needs to see the strategy, not the
    full Isaac Sim boilerplate setup.
    """
    p = _IN_CONTEXT_EXAMPLE_DIR / f"{task_name}.py"
    if not p.exists():
        return f"# Solution for {task_name} not found at {p}"
    text = p.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    # Prefer to cut at a line boundary
    cut = text.rfind("\n", 0, max_chars)
    cut = cut if cut > max_chars // 2 else max_chars
    return text[:cut] + f"\n# ... (truncated; full code in {p.name})"


def load_object_trajectory(task_name: str, traj_root: Optional[str] = None) -> list[dict]:
    """Load the most recent saved trajectory for a task.

    Only tasks in _TRAJ_SEMANTIC_ALIASES have trajectories; all others return [].
    """
    root = Path(traj_root) if traj_root else _TRAJ_ROOT_ICE
    src_task = _TRAJ_SEMANTIC_ALIASES.get(task_name, (None, task_name))[1]
    task_dir = root / src_task
    if not task_dir.exists():
        return []
    files = sorted(task_dir.glob("*.json"))
    if not files:
        return []
    with files[-1].open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _downsample_traj(frames: list[dict], n: int = 20) -> list[dict]:
    """Uniform downsample to at most *n* keyframes."""
    if len(frames) <= n:
        return list(frames)
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def format_trajectory_keyframes(frames: list[dict], n: int = 20) -> str:
    """Compact keyframe list for embedding in a Claude prompt."""
    if not frames:
        return "# (no trajectory recorded)"
    sampled = _downsample_traj(frames, n)
    total = len(frames)
    stride = max(1, total // max(len(sampled), 1))
    lines = ["object_trajectory_keyframes = ["]
    for i, fr in enumerate(sampled):
        t = int(i * stride)
        p = fr["pos"]
        r = fr["rot"]
        lines.append(
            f'    {{"t": {t:3d}, '
            f'"pos": [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}], '
            f'"quat_wxyz": [{r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}, {r[3]:+.4f}]}},'
        )
    lines.append("]")
    return "\n".join(lines)


def semantic_trajectory_summary(task_name: str, frames: list[dict]) -> str:
    """Return a human-readable semantic summary of the object trajectory."""
    alias_entry = _TRAJ_SEMANTIC_ALIASES.get(task_name)
    if alias_entry:
        return f"Trajectory summary:\n{alias_entry[0]}"
    if not frames:
        return "Trajectory summary: (no trajectory recorded)"
    p0, pf = frames[0]["pos"], frames[-1]["pos"]
    r0, rf = frames[0]["rot"], frames[-1]["rot"]
    dz  = pf[2] - p0[2]
    dxy = ((pf[0]-p0[0])**2 + (pf[1]-p0[1])**2) ** 0.5
    dot = sum(a*b for a, b in zip(r0, rf))
    rot_deg = math.degrees(math.acos(max(-1.0, min(1.0, 2*dot*dot - 1))))
    return (
        f"Trajectory summary:\n"
        f"Object moves {dz:+.3f} m in Z, {dxy:.3f} m in XY over {len(frames)} frames. "
        f"Orientation change ~{rot_deg:.0f}°."
    )


def format_qa_example(
    task: str,
    initial_pos: list,
    initial_rot_wxyz: list,
    frames: list[dict],
    solution_code: str,
    task_name: str = "",
    max_keyframes: int = 20,
) -> str:
    """Format a single Q/A in-context example in trajectory-tracking format."""
    traj_kf = format_trajectory_keyframes(frames, n=max_keyframes)
    sem     = semantic_trajectory_summary(task_name or task, frames)
    pos_str = f"[{initial_pos[0]:+.4f}, {initial_pos[1]:+.4f}, {initial_pos[2]:+.4f}]"
    rot_str = (f"[{initial_rot_wxyz[0]:+.4f}, {initial_rot_wxyz[1]:+.4f}, "
               f"{initial_rot_wxyz[2]:+.4f}, {initial_rot_wxyz[3]:+.4f}]")
    return (
        f"QUESTION:\n"
        f"Task: {task}\n"
        f"Initial object pose:\n"
        f"  position  (world): {pos_str}\n"
        f"  orientation (wxyz): {rot_str}\n"
        f"Desired object trajectory (GIVEN — make the object follow this):\n"
        f"{traj_kf}\n"
        f"{sem}\n"
        f"\nANSWER:\n"
        f"{solution_code}\n"
    )


def format_new_task_question(
    task: str,
    initial_pos: list,
    initial_rot_wxyz: list,
    frames: list[dict],
    task_name: str = "",
    max_keyframes: int = 20,
) -> str:
    """Format the new-task QUESTION block (no ANSWER — Claude generates it)."""
    traj_kf = format_trajectory_keyframes(frames, n=max_keyframes)
    sem     = semantic_trajectory_summary(task_name or task, frames)
    pos_str = f"[{initial_pos[0]:+.4f}, {initial_pos[1]:+.4f}, {initial_pos[2]:+.4f}]"
    rot_str = (f"[{initial_rot_wxyz[0]:+.4f}, {initial_rot_wxyz[1]:+.4f}, "
               f"{initial_rot_wxyz[2]:+.4f}, {initial_rot_wxyz[3]:+.4f}]")
    return (
        f"QUESTION:\n"
        f"Task: {task}\n"
        f"Initial object pose:\n"
        f"  position  (world): {pos_str}\n"
        f"  orientation (wxyz): {rot_str}\n"
        f"Desired object trajectory (GIVEN — make the object follow this):\n"
        f"{traj_kf}\n"
        f"{sem}\n"
        f"\nANSWER:\n"
    )


def load_in_context_examples(
    example_tasks: Optional[list[str]] = None,
    max_keyframes: int = 20,
) -> str:
    """Load and format in-context Q/A examples for the prompt.

    Only includes tasks that have a recorded trajectory on disk.
    Returns empty string when no trajectories are available yet.
    """
    if example_tasks is None:
        example_tasks = list(_TRAJ_SEMANTIC_ALIASES.keys())
    blocks: list[str] = []
    for tname in example_tasks:
        frames = load_object_trajectory(tname)
        if not frames:
            continue
        code      = load_solution_code(tname)
        task_desc = _TASK_DESCRIPTIONS.get(tname, tname.replace("_", " ").title() + ".")
        init_pos  = frames[0]["pos"]
        init_rot  = frames[0]["rot"]
        block = format_qa_example(
            task=task_desc,
            initial_pos=init_pos,
            initial_rot_wxyz=init_rot,
            frames=frames,
            solution_code=code,
            task_name=tname,
            max_keyframes=max_keyframes,
        )
        blocks.append(block)
    return "\n---\n\n".join(blocks)


def format_compact_iter_delta(
    prev_metrics: Optional[dict],
    cur_metrics:  Optional[dict],
    best_traj_env: Optional[int] = None,
    likely_failure_mode: str = "",
) -> str:
    """Compact delta summary for refinement prompts (token-efficient)."""
    if not prev_metrics and not cur_metrics:
        return ""
    lines = ["\n=== ITERATION DELTA (what changed vs previous run) ==="]

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    lower_better_keys = {
        "traj_pos_err_mean", "traj_rot_err_mean", "traj_pos_err_final",
        "traj_kp_err_final", "reach_failure_rate", "drop_rate", "min_kp_dist",
    }
    higher_better_keys = {"trajectory_following_score", "success_rate"}
    tracked = [
        "traj_pos_err_mean", "traj_rot_err_mean", "traj_pos_err_final",
        "traj_kp_err_final", "trajectory_following_score",
        "success_rate", "min_kp_dist", "reach_failure_rate", "drop_rate",
    ]
    for key in tracked:
        pv = (prev_metrics or {}).get(key)
        cv = (cur_metrics  or {}).get(key)
        if pv is None and cv is None:
            continue
        arrow = ""
        if pv is not None and cv is not None:
            diff = float(cv) - float(pv)
            if abs(diff) > 1e-4:
                if key in lower_better_keys:
                    arrow = "  ↓ IMPROVED" if diff < 0 else "  ↑ WORSENED"
                elif key in higher_better_keys:
                    arrow = "  ↑ IMPROVED" if diff > 0 else "  ↓ WORSENED"
        lines.append(f"  {key}: {_fmt(pv)} → {_fmt(cv)}{arrow}")

    if best_traj_env is not None:
        lines.append(f"  best_traj_env: env{best_traj_env}")
    if likely_failure_mode:
        lines.append(f"  likely_failure_mode: {likely_failure_mode}")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# PARALLEL SYSTEM PROMPT
# =============================================================================

PARALLEL_SYSTEM_PROMPT = '''\
You are an expert robotic manipulation engineer writing PARALLEL control scripts
that run across N_ENVS simultaneous Isaac Sim environments using batch APIs.
The scene, robots, objects, and all helpers are ALREADY SET UP by the BOILERPLATE — do not modify it.

=== PRIMARY OBJECTIVE: OBJECT TRAJECTORY TRACKING ===
The task is given as:
  GIVEN:  a task description, initial object pose, desired object trajectory
  GENERATE: controller code that makes the object follow the given trajectory as closely as possible

The object trajectory is NOT just context — it IS the target behavior.
The generated policy must OPTIMIZE for trajectory tracking, not just final pose.
Do NOT invent the trajectory. Do NOT treat it as a robot trajectory.
Do NOT replay it as joint targets or EE targets. It describes desired OBJECT motion.

Minimize (in order of importance):
  1. object position tracking error over time  (traj_pos_err_mean, traj_pos_err_final)
  2. object orientation tracking error         (traj_rot_err_mean, traj_rot_err_final)
  3. final keypoint distance                   (traj_kp_err_final, kp_dist)
  4. object drift from trajectory / dropping
  5. unstable contact / object tipping

In-context examples show strategy demonstrations. Adapt contact points, grasp type,
approach direction, and transport behavior to the new object and trajectory.
Do NOT copy-paste the closest example — adapt the strategy to the current task.

=== TASK (secondary framing — also required) ===
N_ENVS robots must each grasp an object and transport it along WAYPOINTS_WORLD to GOAL_POS,
ending with all 4 bbox-corner keypoints within SUCCESS_TOL of GOAL_KEYPOINTS.
SUCCESS = at least one env achieves keypoint_max_dist < SUCCESS_TOL.
Optimize for all envs, not just env-0.

=== BATCH API ===
Key variables: robots (ArticulationView), objects (RigidPrimView), N_ENVS, env_offsets (N,3),
  _n_dof (19), _dof_names, _init_joints_1d, get_object_contact_forces() → (N,3).
Object poses: positions, quats_xyzw = objects.get_world_poses()  # quats are [x,y,z,w]
  quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])  # convert before helpers
EE: get_ee_pos_all_envs(stage, N_ENVS) → list[ndarray|None]; get_ee_pos_env0(stage) for diagnostics.
Joint state: robots.get_joint_positions()  # (N, n_dof)

=== IK SOLVERS ===
solve_ik_batch_all_envs(targets_world, seeds_full, closed_hand=False, target_quats=None)
  → (N_ENVS, n_dof). targets_world is (N,3) — one per-env target. Always cold-start.
  Use for: Phase 2a IK target, Phase 3 live grasp anchor, Phase 4 transport IK.
  Use plan_motion_all_envs() for smooth trajectory execution (Phases 2a/2b/2c).
  NEVER tile env-0's IK result to all envs — each env has a different object position.

IK EE FRAME vs SIMULATION EE — critical distinction:
  solve_ik_batch_all_envs controls xhand_mount_right (IK EE link).
  get_ee_pos_all_envs / knuckle_centroid is what reach/grasp scoring uses.
  IK_EE_MOUNT_ADJ_M = 0.10 m — xhand_mount_right is this far BEHIND knuckle_centroid.
  ALWAYS include IK_EE_MOUNT_ADJ_M in standoff when computing IK targets:
    ik_target = obj_pos - approach_dir * (standoff + IK_EE_MOUNT_ADJ_M)
  get_ik_ee_poses_all_envs(cmds) checks IK algebra only (always ~0 error) — NOT physics.
  get_ik_ee_poses_all_envs(robots.get_joint_positions()) checks physical convergence.
  Phase 2 needs 400 incremental IK steps + 50 hold steps to reach grasp standoff from home EE.

=== KEYPOINT HELPERS ===
  keypoint_max_dist_batch(positions, quats_wxyz) → (N,)  # uses per-env GOAL_KEYPOINTS_BATCH

=== TRAJECTORY FORMAT ===
Trajectories are saved as JSON arrays of frames, each with:
  {"pos": [x, y, z], "rot": [w, x, y, z]}   ← world-frame object pose, rot in wxyz

  Orientation convention:
    • Stored / get_world_poses() returns [x,y,z,w] (Isaac Sim xyzw).
    • All trajectory JSON files use wxyz (w first).  Convert before passing to set_world_poses():
        xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]

  Orientation correction when a trajectory was recorded with a different mesh:
    Problem: the stored rotations were recorded with mesh A; you want to replay with mesh B
             (different canonical orientation).
    Solution: compute a correction quaternion once and apply it to every stored frame.
        q_correction = q_override * inv(q_traj_initial)
        q_display[i]  = q_correction * q_traj[i]        # preserves all relative rotations
    Example (bottle upright = 180° around Z):
        q_override   = np.array([0., 0., 1., 0.])        # wxyz for 180° Z
        q_traj_init  = np.array(frames[0]["rot"])
        q_correction = _qmul(q_override, _qinv(q_traj_init))
        # for each frame: wxyz_display = _qmul(q_correction, np.array(frame["rot"]))

  replay_trajectory.py — visualisation / sanity-check tool:
    Loads a trajectory JSON, teleports the object through each pose (kinematic, no robot),
    renders frames, and encodes an MP4 next to the JSON file.
    Usage:
      python src/replay_trajectory.py \\
          --task bottle_pour \\
          --object_usd /path/to/bottle.usd --scale 1.0 --mass 0.3 \\
          --orientation 0 0 1 0          # wxyz override — applies correction automatically
    --orientation W X Y Z applies the correction formula above so the mesh starts at the
    given orientation and all trajectory rotations are preserved relative to it.
    Use this to verify a new trajectory looks correct before running the full pipeline.

=== TRAJECTORY (pre-computed, do not recompute) ===
  WAYPOINTS_WORLD_BATCH (N,T,3), GOAL_POS_BATCH (N,3), GOAL_KEYPOINTS_BATCH (N,4,3)
  OBJ_INIT_POS_BATCH (N,3), OBJ_INIT_ROT_BATCH (N,4) — actual settled poses after boilerplate.

=== PHASE STRUCTURE ===
Phase 1 (10 steps):   diagnostics. Print STARTUP_EE_OBJ_DIST.
Phase 2 (510 steps):  IK reach. Print REACH_SCORE, REACH_X/Y/Z_OFFSET, OBJ_TILT_AFTER_REACH, REACH_ARM_JOINTS.
Phase 3 (150 steps):  grasp. Print GRASP_HOLD_ARM_JOINTS, BEST_GRASP_STATE.
Phase 4 (500 steps):  transport. Print FRAME_STATE every 30 steps.
Phase 5 (60 steps):   hold.

=== REQUIRED OUTPUTS (pipeline parses — print exactly as shown) ===
PER_ENV_STATS after reach:  {"env_id":ei, "phase":"reach", "min_ee_to_obj":float, "obj_knocked":bool}
PER_ENV_STATS after grasp:  {"env_id":ei, "phase":"grasp", "grasp_stable":bool, "obj_knocked_over":bool}
PER_ENV_STATS after final:  {"env_id":ei, "phase":"final", "min_goal_kp":float, "final_goal_kp":float,
                             "max_lift":float, "final_ee_to_obj":float,
                             "trajectory_coupled":bool, "grasp_stable":bool,
                             "phase_failed":"reach"|"grasp"|"transport"|"none"}
BEST_ENV_SUMMARY: {"best_reach_env":int, "best_reach_dist":float, "best_approach_arm_joints":list,
                   "best_approach_obj_pos":list, "best_goal_env":int, "best_goal_kp":float,
                   "grasp_stable_any":bool, "trajectory_coupled_any":bool, "notes":str}
END_MEMORY: {"cur_best_reach":float, "cur_best_goal_kp":float, "grasp_stable_any":bool,
             "trajectory_coupled_any":bool, "any_env_lifted":bool, "best_env":int,
             "stage_reached":"approach"|"grasp"|"grasp_contact"|"transport", "why_better":str}
FRAME_STATE: {"step":int, "phase":"transport", "kp_mean":float, "kp_min":float, "kp_per_env":list,
              "lift_per_env":list, "best_env":int, "coupled_envs":int, "dense_score_mean":float}
BATCH_METRICS: {"success_count":int, "n_envs":int, "success_rate":float, "mean_kp_dist":float,
                "min_kp_dist":float, "reach_failure_rate":float, "drop_rate":float,
                "final_ee_obj_dist_min":float, "final_ee_obj_dist_mean":float,
                "score_lift":float, "score_grasp":float,
                "traj_pos_err_mean":float, "traj_pos_err_min":float, "traj_pos_err_final":float,
                "traj_rot_err_mean":float, "traj_rot_err_final":float,
                "traj_kp_err_mean":float, "traj_kp_err_final":float,
                "best_traj_env":int, "trajectory_following_score":float}
Print "SUCCESS" if success_count > 0, else "FAILURE".

=== TRAJECTORY TRACKING METRICS (compute these during Phase 4 transport) ===
At each transport step, compare current object pose to the nearest trajectory waypoint:
  wp_t = WAYPOINTS_WORLD_BATCH[ei, wp_idx[ei]]    # (3,) position target
  rot_t = WAYPOINTS_ROT_BATCH[ei, wp_idx[ei]]     # (4,) wxyz rotation target
  traj_pos_err[ei] = np.linalg.norm(obj_pos[ei] - wp_t)
  dot = np.clip(np.dot(quats_wxyz[ei], rot_t), -1, 1)
  traj_rot_err[ei] = 2*np.arccos(abs(dot))  # geodesic angle error in radians
Accumulate per-env min/mean/final over the transport phase.
trajectory_following_score = 1 / (1 + traj_pos_err_mean_best_env)   # higher is better
best_traj_env = env with lowest traj_pos_err_mean
traj_kp_err_final = keypoint_max_dist_batch for the best_traj_env at final step.

=== POSE_TRACE (compact per-step pose log — required every 30-50 steps) ===
Print POSE_TRACE only for: best env (lowest traj_pos_err so far) AND env 0.
Do NOT print for every env. Do NOT print every step.
Format:
  print("POSE_TRACE: " + json.dumps(json_safe({
      "step": step, "phase": "transport", "best_env": best_traj_env_so_far,
      "env": ei,
      "obj_pos": obj_pos[ei].tolist(), "obj_quat_wxyz": quats_wxyz[ei].tolist(),
      "target_obj_pos": wp_t.tolist(), "target_obj_quat_wxyz": rot_t.tolist(),
      "ee_pos": ee_pos_ei, "traj_pos_err": float(traj_pos_err[ei]),
      "traj_rot_err": float(traj_rot_err[ei]),
      "kp_dist": float(kp_dists[ei]),
      "ee_obj_dist": float(ee_obj_d), "contact_force": float(cf_mag[ei]),
  })))

JSON: numpy scalars are not json-serialisable — use float()/int()/.tolist()/json_safe().
Wrap all phases in try/except/finally. End: rep.orchestrator.stop(); simulation_app.close()
Return ONLY raw Python code. No markdown fences.
'''


# =============================================================================
# PARALLEL BOILERPLATE
# =============================================================================

PARALLEL_BOILERPLATE = r'''# === VERIFIED PARALLEL BOILERPLATE — do not modify ===
from __future__ import annotations
import os, sys, json, math
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# Repo root must be set up FIRST so our src/ takes priority over D3CB copies.
_REPO_ROOT = "REPO_ROOT_PLACEHOLDER"
_SRC_DIR_BRT = os.path.join(_REPO_ROOT, "src") if _REPO_ROOT else ""
if _SRC_DIR_BRT and _SRC_DIR_BRT not in sys.path:
    sys.path.insert(0, _SRC_DIR_BRT)
if _REPO_ROOT and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# D3CB directory has helper modules we still need (solve_ik_numpy, etc.)
# but append it so our src/ modules take priority.
_BOILERPLATE_RUNTIME_DIR = (
    "/mnt/storage/DDDC/Dynamic-Dexterous-Digital-Cousin-Benchmark"
    "/scripts/keypoint_pipeline_par"
)
if _BOILERPLATE_RUNTIME_DIR not in sys.path:
    sys.path.append(_BOILERPLATE_RUNTIME_DIR)

# Runtime helpers live in boilerplate_runtime.py — never regenerate these functions.
from boilerplate_runtime import (
    _NumpyEncoder, json_safe,
    _qnorm, _qconj, _qmul, _qmat, _qrotate,
    _get_link_world_pos,
    get_ee_pos_env0, get_ee_pos_all_envs,
    get_ee_transforms_all_envs,
    get_ik_ee_poses_all_envs,
    IK_EE_MOUNT_ADJ_M,
    compute_keypoints, compute_keypoints_batch,
    setup_ik, setup_motion_gen, init_render, init_runtime,
    run_ik_diagnostics,
    solve_ik_env0, solve_ik_for_env, solve_ik_batch_all_envs,
    plan_motion_all_envs,
    get_object_contact_forces, keypoint_max_dist_batch,
)

import torch

N_ENVS      = N_ENVS_PLACEHOLDER
ENV_SPACING = ENV_SPACING_PLACEHOLDER
_n_cols     = max(1, int(np.ceil(np.sqrt(N_ENVS))))

from isaacsim import SimulationApp
print("BOILERPLATE: SimulationApp starting...", flush=True)
simulation_app = SimulationApp({"headless": True, "renderer": "RasterizedRendering"})
print("BOILERPLATE: SimulationApp ready", flush=True)

import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation as ArticulationView, RigidPrim as RigidPrimView
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdLux
print("BOILERPLATE: imports done", flush=True)

_frames_dir = os.environ.get("ISAAC_FRAMES_DIR", "/tmp/isaac_frames_par")
Path(_frames_dir).mkdir(parents=True, exist_ok=True)

world = World(stage_units_in_meters=1.0)
try:
    world.scene.add_default_ground_plane()
except Exception as _gp_err:
    print(f"WARN: add_default_ground_plane failed ({_gp_err}), skipping — floor physics from table USD", flush=True)
stage = omni.usd.get_context().get_stage()

# --- Lighting: dome (ambient) + back fill + overhead fill -------------------
_dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
_dome.CreateIntensityAttr(600.0)

_back = UsdLux.DistantLight.Define(stage, "/World/Lights/BackLight")
_back.CreateIntensityAttr(2000.0)
_back.CreateAngleAttr(2.0)
UsdGeom.Xformable(_back).AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 0.0, 0.0))  # from above-behind

_fill = UsdLux.SphereLight.Define(stage, "/World/Lights/FillLight")
_fill.CreateIntensityAttr(3000.0)
_fill.CreateRadiusAttr(1.5)
_grid_cx = (_n_cols - 1) * ENV_SPACING / 2.0
_grid_cy = (int(np.ceil(N_ENVS / _n_cols)) - 1) * ENV_SPACING / 2.0
UsdGeom.Xformable(_fill).AddTranslateOp().Set(Gf.Vec3d(_grid_cx, _grid_cy, 3.5))

# Per-env world offsets (grid layout)
env_offsets = np.zeros((N_ENVS, 3))
for _i in range(N_ENVS):
    _row, _col = _i // _n_cols, _i % _n_cols
    env_offsets[_i] = [_col * ENV_SPACING, _row * ENV_SPACING, 0.0]

_robot_usd           = r"ROBOT_USD_PLACEHOLDER"
_obj_usd             = r"OBJECT_USD_PLACEHOLDER"
_local_obj_pos       = OBJECT_POSITION_PLACEHOLDER
_local_rob_pos       = ROBOT_POSITION_PLACEHOLDER
_obj_orient          = OBJECT_ORIENTATION_PLACEHOLDER  # [w,x,y,z]
_obj_scale           = OBJECT_SCALE_PLACEHOLDER
_obj_mass            = OBJECT_MASS_PLACEHOLDER
# Task-specific extra Z offset so the object falls onto the surface and settles.
# 0.0 for objects whose saved-pose z already rests on the table.
_OBJ_Z_SETTLE_OFFSET = OBJ_Z_SETTLE_OFFSET_PLACEHOLDER

print(f"BOILERPLATE: spawning {N_ENVS} envs...", flush=True)
for _i in range(N_ENVS):
    _off = env_offsets[_i]
    _ep  = f"/World/envs/env_{_i}"

    _rp = _ep + "/Robot"
    add_reference_to_stage(usd_path=_robot_usd, prim_path=_rp)
    _rprim = stage.GetPrimAtPath(_rp)
    _rxf   = UsdGeom.Xformable(_rprim)
    _rpos  = [_local_rob_pos[_j] + _off[_j] for _j in range(3)]
    _r_ops = {op.GetOpName(): op for op in _rxf.GetOrderedXformOps()}
    if "xformOp:translate" in _r_ops:
        _r_ops["xformOp:translate"].Set(Gf.Vec3d(*_rpos))
    else:
        _rxf.AddTranslateOp().Set(Gf.Vec3d(*_rpos))

    _op = _ep + "/Object"
    add_reference_to_stage(usd_path=_obj_usd, prim_path=_op)
    _bp = stage.GetPrimAtPath(_op)
    _xf = UsdGeom.Xformable(_bp)
    _xf.ClearXformOpOrder()
    _opos = [_local_obj_pos[_j] + _off[_j] for _j in range(3)]
    _xf.AddTranslateOp().Set(Gf.Vec3d(*_opos))
    _bq = _obj_orient
    _xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(_bq[0], _bq[1], _bq[2], _bq[3]))
    _xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(_obj_scale, _obj_scale, _obj_scale))
    _mass_api = UsdPhysics.MassAPI.Apply(_bp)
    _mass_api.CreateMassAttr().Set(_obj_mass)

print("BOILERPLATE: USD refs done", flush=True)

# Articulation() calls get_articulation_root_api_prim_path() which BFS-searches the subtree
# for the prim with ArticulationRootAPI and appends the suffix automatically.
_robot_prim_expr = "/World/envs/env_.*/Robot"
print(f"BOILERPLATE: robot_prim_expr={_robot_prim_expr}", flush=True)

robots  = ArticulationView(prim_paths_expr=_robot_prim_expr, name="robots",
                           reset_xform_properties=False)
objects = RigidPrimView(prim_paths_expr="/World/envs/env_.*/Object", name="objects",
                        reset_xform_properties=False, track_contact_forces=True)
world.scene.add(robots)
world.scene.add(objects)

world.reset()
robots.initialize()
world.step(render=False)
print("BOILERPLATE: world reset done", flush=True)

# --- Randomise object XY position per env (different positions, same orientation) -
import random as _random
_OBJ_XY_NOISE = 0.05   # metres (±5 cm)
_rng = _random.Random(42)
_rand_obj_pos  = np.zeros((N_ENVS, 3), dtype=np.float64)
_rand_obj_quat = np.zeros((N_ENVS, 4), dtype=np.float64)  # [w,x,y,z]
for _ei in range(N_ENVS):
    _off = env_offsets[_ei]
    _dx  = _rng.uniform(-_OBJ_XY_NOISE, _OBJ_XY_NOISE)
    _dy  = _rng.uniform(-_OBJ_XY_NOISE, _OBJ_XY_NOISE)
    _rand_obj_pos[_ei] = [_local_obj_pos[0] + _off[0] + _dx,
                          _local_obj_pos[1] + _off[1] + _dy,
                          _local_obj_pos[2] + _off[2] + _OBJ_Z_SETTLE_OFFSET]
    _rand_obj_quat[_ei] = list(_obj_orient)
# set_world_poses expects [x,y,z,w] quaternion convention
_isaac_quat = np.column_stack([
    _rand_obj_quat[:,1], _rand_obj_quat[:,2],
    _rand_obj_quat[:,3], _rand_obj_quat[:,0],
])
objects.set_world_poses(positions=_rand_obj_pos, orientations=_isaac_quat)
# Zero out any velocities the objects inherited from the pre-pose-set physics step
objects.set_linear_velocities(np.zeros((N_ENVS, 3)))
objects.set_angular_velocities(np.zeros((N_ENVS, 3)))
print(f"BOILERPLATE: object poses set (XY noise={_OBJ_XY_NOISE}m, no rotation noise)", flush=True)
# Use json_safe so numpy float64 values are serialised correctly.
print("OBJECT_POSE_INITIAL_ENV0: " + json.dumps(json_safe({
    "pos": _rand_obj_pos[0], "quat_wxyz": _rand_obj_quat[0]
})), flush=True)

_init_joints_by_name = INITIAL_JOINTS_PLACEHOLDER
_dof_names = list(robots.dof_names)
_n_dof     = robots.num_dof
_init_joints_1d = np.zeros(_n_dof)
for _ji, _jname in enumerate(_dof_names):
    if _jname in _init_joints_by_name:
        _init_joints_1d[_ji] = _init_joints_by_name[_jname]
_init_batch = np.tile(_init_joints_1d, (N_ENVS, 1))
robots.set_joint_positions(_init_batch)
robots.set_joint_velocities(np.zeros((N_ENVS, _n_dof)))

# ── IK + MotionGen setup via boilerplate_runtime ────────────────────────────
# ALWAYS use solve_ik_batch_all_envs() for reactive IK.
# Use plan_motion_all_envs() for smooth offline trajectory planning (Phase 2a).
_dof_idx: dict = {n: i for i, n in enumerate(_dof_names)}
_IK_BACKEND, _IK_OPEN, _IK_CLOSED = setup_ik(_dof_idx)
_PIN_IK = _IK_OPEN   # legacy alias so old-style code still compiles
_MOTION_GEN = setup_motion_gen()   # cuRobo MotionGen for collision-free arm trajectories

# cuRobo frame diagnostics — reads actual base_link world pose, runs FK/IK frame checks,
# and auto-detects whether targets should be in world or base_link frame.
run_ik_diagnostics(stage, "/World/envs/env_0/Robot", _init_joints_1d, ik_solver=_IK_OPEN)

try:
    _kps = np.tile(np.array([500.0]*7 + [200.0]*(_n_dof-7)), (N_ENVS, 1))
    _kds = np.tile(np.array([ 80.0]*7 + [ 20.0]*(_n_dof-7)), (N_ENVS, 1))
    robots.set_gains(kps=_kps, kds=_kds)
    print("BOILERPLATE: PD gains set", flush=True)
except Exception as _eg:
    print(f"BOILERPLATE: PD gains warning: {_eg}", flush=True)

world.step(render=False)
print(f"BOILERPLATE: ready. N_ENVS={N_ENVS} n_dof={_n_dof} dof_names={_dof_names}", flush=True)

# Camera positioned near env_0; cap zoom so it doesn't pull back for large grids
_n_rows   = int(np.ceil(N_ENVS / _n_cols))
_grid_r   = max(_n_cols, _n_rows)
_cam_r    = min(_grid_r, 3)  # cap at 3x3 view regardless of N_ENVS
_cam_pos    = (0.0, -(1.2 + 0.6 * _cam_r * ENV_SPACING), 1.5 + 0.55 * _cam_r)
_cam_target = (0.0, 0.5, 0.1)
_camera = rep.create.camera(position=_cam_pos, look_at=_cam_target)
_render_product = rep.create.render_product(_camera, (640, 480))
_rgb_annotator  = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb_annotator.attach([_render_product])
print("BOILERPLATE: camera ready", flush=True)

# Install auto-render world.step wrapper BEFORE the settle loop so settle frames
# are also captured. Claude code just calls world.step(render=False); rendering
# is automatic every _RENDER_EVERY steps.
_RENDER_EVERY = 6   # save a frame every 6 sim steps → ~10 fps at 60 Hz sim
init_render(world, _frames_dir, _rgb_annotator, _RENDER_EVERY)

# Object extents + mesh via trimesh (for keypoint bbox and standoff)
try:
    from get_started.obj_layout.grasp_anchor_claude import usd_to_trimesh
    _mesh = usd_to_trimesh(_obj_usd, scale=_obj_scale)
    OBJECT_EXTENTS = np.asarray(_mesh.extents, dtype=np.float64)
    OBJECT_MESH = _mesh                                        # trimesh.Trimesh, USD-local coords
    OBJECT_MESH_CENTER = np.asarray(_mesh.bounds.mean(axis=0), dtype=np.float64)  # bbox centre
except Exception as _mesh_err:
    print(f"WARN: usd_to_trimesh failed ({_mesh_err}); using extents=[0.1,0.1,0.1]", flush=True)
    OBJECT_EXTENTS = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    OBJECT_MESH = None
    OBJECT_MESH_CENTER = np.zeros(3, dtype=np.float64)
print(f"OBJECT_EXTENTS: {OBJECT_EXTENTS.tolist()}", flush=True)

# --- Physics settle (60 steps): let objects fall onto the table ─────────────────
# MUST run before trajectory alignment so waypoints are anchored to the
# actual post-settle object pose, not the nominal/randomised spawn position.
print("BOILERPLATE: settling (60 steps)...", flush=True)
_settle_cmd = np.tile(_init_joints_1d, (N_ENVS, 1))
for _s in range(60):
    robots.set_joint_position_targets(_settle_cmd)
    world.step(render=False)

# Read actual settled object poses (after XY randomisation + physics drop)
_settled_pos_raw, _settled_qxyzw = objects.get_world_poses()  # (N,3),(N,4) [x,y,z,w]
_settled_pos   = np.asarray(_settled_pos_raw, dtype=np.float64)   # (N,3)
# Convert IsaacSim [x,y,z,w] → wxyz for our quat helpers
_settled_quats = np.column_stack([            # (N,4) wxyz
    _settled_qxyzw[:, 3].astype(np.float64),  # w
    _settled_qxyzw[:, 0].astype(np.float64),  # x
    _settled_qxyzw[:, 1].astype(np.float64),  # y
    _settled_qxyzw[:, 2].astype(np.float64),  # z
])
_nom_pos = np.asarray(OBJECT_POSITION_PLACEHOLDER, dtype=np.float64)
print("SETTLED_OBJ_POSE_ENV0: " + json.dumps(json_safe({
    "pos":              _settled_pos[0].tolist(),
    "quat_wxyz":        _settled_quats[0].tolist(),
    "drift_from_nom_m": round(float(np.linalg.norm(_settled_pos[0] - _nom_pos)), 4),
})), flush=True)

# Actual settled pose arrays (wxyz quats) — for Claude's Phase 1 reference
OBJ_INIT_POS_BATCH = _settled_pos.copy()    # (N,3) world-frame positions after settle
OBJ_INIT_ROT_BATCH = _settled_quats.copy()  # (N,4) wxyz orientations after settle

# --- Per-env trajectory alignment ───────────────────────────────────────────────
# For each env i, compute an alignment transform so that trajectory frame 0
# maps exactly to the actual settled pose of that env's object:
#   q_align_i = qmul(settled_rot_i, qconj(traj_rot_0))
#   p_align_i = settled_pos_i - qrotate(q_align_i, traj_pos_0)
# Then for each trajectory frame k:
#   WAYPOINTS_WORLD_BATCH[i, k] = qrotate(q_align_i, traj_pos[k]) + p_align_i
#   WAYPOINTS_ROT_BATCH[i, k]   = qnorm(qmul(q_align_i, traj_rot[k]))
_OBJECT_ROT_NOM = _qnorm(OBJECT_ORIENTATION_PLACEHOLDER)
with open(r"TRAJECTORY_PATH_PLACEHOLDER") as _traj_f:
    _TRAJECTORY = json.load(_traj_f)

if len(_TRAJECTORY) == 0:
    _T = 1
    WAYPOINTS_WORLD_BATCH = _settled_pos[:, np.newaxis, :].copy()   # (N,1,3)
    WAYPOINTS_ROT_BATCH   = _settled_quats[:, np.newaxis, :].copy() # (N,1,4) wxyz
    GOAL_POS_BATCH        = _settled_pos.copy()                     # (N,3)
    GOAL_ROT_BATCH        = _settled_quats.copy()                   # (N,4) wxyz
else:
    _t0_pos = np.asarray(_TRAJECTORY[0]["pos"], dtype=np.float64)
    _t0_rot = _qnorm(np.asarray(_TRAJECTORY[0]["rot"], dtype=np.float64))
    _T      = len(_TRAJECTORY)
    WAYPOINTS_WORLD_BATCH = np.zeros((N_ENVS, _T, 3), dtype=np.float64)
    WAYPOINTS_ROT_BATCH   = np.zeros((N_ENVS, _T, 4), dtype=np.float64)
    for _ei in range(N_ENVS):
        _q_a = _qmul(_settled_quats[_ei], _qconj(_t0_rot))
        _p_a = _settled_pos[_ei] - _qrotate(_q_a, _t0_pos)
        for _k, _fr in enumerate(_TRAJECTORY):
            _p = np.asarray(_fr["pos"], dtype=np.float64)
            _r = _qnorm(np.asarray(_fr["rot"], dtype=np.float64))
            WAYPOINTS_WORLD_BATCH[_ei, _k] = _qrotate(_q_a, _p) + _p_a
            WAYPOINTS_ROT_BATCH[_ei, _k]   = _qnorm(_qmul(_q_a, _r))
        # Z-floor per env: no waypoint below the settled surface level
        _z_ref   = float(_settled_pos[_ei, 2])
        _z_shift = max(0.0, float(-(WAYPOINTS_WORLD_BATCH[_ei, :, 2] - _z_ref).min()))
        if _z_shift > 0.0:
            WAYPOINTS_WORLD_BATCH[_ei, :, 2] += _z_shift
    GOAL_POS_BATCH = WAYPOINTS_WORLD_BATCH[:, -1, :].copy()   # (N,3)
    GOAL_ROT_BATCH = WAYPOINTS_ROT_BATCH[:, -1, :].copy()     # (N,4) wxyz

# Per-env goal keypoints (N,4,3) — pass extents explicitly (init_runtime not yet called)
GOAL_KEYPOINTS_BATCH = np.zeros((N_ENVS, 4, 3), dtype=np.float64)
for _ei in range(N_ENVS):
    GOAL_KEYPOINTS_BATCH[_ei] = compute_keypoints(
        GOAL_POS_BATCH[_ei], GOAL_ROT_BATCH[_ei], extents=OBJECT_EXTENTS)

# Env-0 single-env aliases (backward compatible; env 0 is also the diagnostic env)
WAYPOINTS_WORLD = WAYPOINTS_WORLD_BATCH[0]   # (T,3)
GOAL_POS        = GOAL_POS_BATCH[0]          # (3,)
GOAL_ROT        = GOAL_ROT_BATCH[0]          # (4,) wxyz
GOAL_KEYPOINTS  = GOAL_KEYPOINTS_BATCH[0]    # (4,3)
NUM_WAYPOINTS   = _T
SUCCESS_TOL     = SUCCESS_TOLERANCE_PLACEHOLDER

# Alignment diagnostics — log the first 4 envs
for _ei in range(min(N_ENVS, 4)):
    print("ALIGN_ENV" + str(_ei) + ": " + json.dumps(json_safe({
        "settled_pos":   _settled_pos[_ei].tolist(),
        "goal_pos":      GOAL_POS_BATCH[_ei].tolist(),
        "goal_rot_wxyz": GOAL_ROT_BATCH[_ei].tolist(),
        "wp0":           WAYPOINTS_WORLD_BATCH[_ei, 0].tolist(),
        "drift_m":       round(float(np.linalg.norm(_settled_pos[_ei] - _nom_pos)), 4),
    })), flush=True)

print(f"NUM_WAYPOINTS: {NUM_WAYPOINTS}", flush=True)
print(f"GOAL_POS: {GOAL_POS.tolist()}", flush=True)
print(f"GOAL_ROT: {GOAL_ROT.tolist()}", flush=True)
print(f"GOAL_KEYPOINTS: {GOAL_KEYPOINTS.tolist()}", flush=True)
print(f"SUCCESS_TOL: {SUCCESS_TOL}", flush=True)

ARM_JOINT_NAMES = ["Actuator1", "Actuator2", "Actuator3", "Actuator4",
                   "Actuator5", "Actuator6", "Actuator7"]
INITIAL_ARM_JOINTS = {j: _init_joints_by_name.get(j, 0.0) for j in ARM_JOINT_NAMES}
HAND_CLOSE_TARGET = {
    "right_hand_thumb_bend_joint":   1.832,
    "right_hand_thumb_rota_joint1":  1.57,
    "right_hand_thumb_rota_joint2":  1.57,
    "right_hand_index_bend_joint":   0.174,
    "right_hand_index_joint1":       1.919,
    "right_hand_index_joint2":       1.919,
    "right_hand_mid_joint1":         1.919,
    "right_hand_mid_joint2":         1.919,
    "right_hand_ring_joint1":        1.919,
    "right_hand_ring_joint2":        1.919,
    "right_hand_pinky_joint1":       1.919,
    "right_hand_pinky_joint2":       1.919,
}
# Precision grasp: full thumb opposition + index/middle close; ring/pinky partial.
# Use for small/flat objects where OBJECT_EXTENTS max < 0.04 m.
# The two-stage closing rule still applies: joint1 first (steps 0-74), joint2 after (75-149).
PRECISION_GRASP_TARGET = {
    "right_hand_thumb_bend_joint":   1.2,
    "right_hand_thumb_rota_joint1":  1.3,   # thumb opposed (>=0.8 mandatory)
    "right_hand_thumb_rota_joint2":  1.0,
    "right_hand_index_bend_joint":   0.0,
    "right_hand_index_joint1":       1.5,
    "right_hand_index_joint2":       1.5,
    "right_hand_mid_joint1":         1.3,
    "right_hand_mid_joint2":         1.3,
    "right_hand_ring_joint1":        0.5,
    "right_hand_ring_joint2":        0.5,
    "right_hand_pinky_joint1":       0.3,
    "right_hand_pinky_joint2":       0.3,
}

# ── Auto-select grasp target based on object cross-section ─────────────────────
# POWER GRASP  (max cross-section >= 0.04 m): cylindrical objects like bottles,
#              bowls, drills — wrap all fingers around the object.
# PRECISION    (max cross-section < 0.04 m): small/thin/flat objects — three-
#              finger pinch with partial ring/pinky engagement.
# This variable is available in your motion code without redefinition.
SELECTED_GRASP_TARGET = (
    HAND_CLOSE_TARGET if max(OBJECT_EXTENTS) >= 0.04 else PRECISION_GRASP_TARGET
)
_GRASP_TYPE = "POWER" if max(OBJECT_EXTENTS) >= 0.04 else "PRECISION"
print(f"GRASP_TYPE_SELECTED: {_GRASP_TYPE}  (max_extent={round(max(OBJECT_EXTENTS),4)}m)", flush=True)

# Wire up IK, contact forces, and keypoint helpers (call AFTER GOAL_KEYPOINTS_BATCH)
init_runtime(
    N_ENVS=N_ENVS, env_offsets=env_offsets, objects=objects,
    _dof_idx_arg=_dof_idx, _IK_OPEN_arg=_IK_OPEN, _IK_CLOSED_arg=_IK_CLOSED,
    _IK_BACKEND_arg=_IK_BACKEND, OBJECT_EXTENTS=OBJECT_EXTENTS,
    GOAL_KEYPOINTS_BATCH=GOAL_KEYPOINTS_BATCH,
)

# Per-env approach directions: robot base XY → object XY, with slight upward Z tilt.
# Identical to how the in-context examples (cylinder_pour.py) compute the approach.
# These are pre-computed here so Claude's motion code always gets the direction right.
_ROBOT_BASE_XY = np.array([_local_rob_pos[0], _local_rob_pos[1]], dtype=np.float64)
APPROACH_DIRS_BATCH = np.zeros((N_ENVS, 3), dtype=np.float64)
for _ei in range(N_ENVS):
    _to_obj_XY = OBJ_INIT_POS_BATCH[_ei, :2] - (env_offsets[_ei, :2] + _ROBOT_BASE_XY)
    _to_obj_XY /= np.linalg.norm(_to_obj_XY) + 1e-8
    APPROACH_DIRS_BATCH[_ei] = [_to_obj_XY[0], _to_obj_XY[1], 0.07]
    APPROACH_DIRS_BATCH[_ei] /= np.linalg.norm(APPROACH_DIRS_BATCH[_ei]) + 1e-8
print(f"APPROACH_DIR_ENV0: {APPROACH_DIRS_BATCH[0].tolist()}", flush=True)

print(f"INITIAL_ARM_JOINTS: {INITIAL_ARM_JOINTS}", flush=True)
print("BOILERPLATE_READY", flush=True)
# === END PARALLEL BOILERPLATE ===
# ── Scene / control handles ───────────────────────────────────────────────────
#   robots, objects    — ArticulationView / RigidPrimView (N_ENVS each)
#   N_ENVS             — int
#   env_offsets        — np.ndarray (N,3) world-frame origin of each env
#   _n_dof             — int (19)
#   _dof_names         — list[str]
#   _init_joints_1d    — np.ndarray (n_dof,)
#   OBJECT_EXTENTS     — np.ndarray (3,) metres
#
# ── Per-env aligned trajectory (anchored to ACTUAL settled pose) ──────────────
#   WAYPOINTS_WORLD_BATCH  — np.ndarray (N_ENVS, T, 3)  world-frame positions
#   WAYPOINTS_ROT_BATCH    — np.ndarray (N_ENVS, T, 4)  wxyz orientations
#   GOAL_POS_BATCH         — np.ndarray (N_ENVS, 3)     final position per env
#   GOAL_ROT_BATCH         — np.ndarray (N_ENVS, 4)     wxyz final rotation per env
#   GOAL_KEYPOINTS_BATCH   — np.ndarray (N_ENVS, 4, 3)  per-env goal keypoints
#   OBJ_INIT_POS_BATCH     — np.ndarray (N_ENVS, 3)     actual settled positions
#   OBJ_INIT_ROT_BATCH     — np.ndarray (N_ENVS, 4)     wxyz actual settled rots
#   NUM_WAYPOINTS          — int (T)
#   SUCCESS_TOL            — float
#
# ── Env-0 single-env aliases (backward compatible) ───────────────────────────
#   WAYPOINTS_WORLD = WAYPOINTS_WORLD_BATCH[0]   (T,3)
#   GOAL_POS        = GOAL_POS_BATCH[0]           (3,)
#   GOAL_ROT        = GOAL_ROT_BATCH[0]           (4,) wxyz
#   GOAL_KEYPOINTS  = GOAL_KEYPOINTS_BATCH[0]     (4,3)
#   ARM_JOINT_NAMES, INITIAL_ARM_JOINTS
#   HAND_CLOSE_TARGET       — power grasp target (max extent >= 0.04 m)
#   PRECISION_GRASP_TARGET  — precision grasp target (max extent < 0.04 m)
#   SELECTED_GRASP_TARGET   — auto-selected from above based on OBJECT_EXTENTS
#                             USE THIS in Phase 3 instead of hardcoding HAND_CLOSE_TARGET
#
# ── MotionGen — collision-free offline trajectory planning ───────────────────
#   plan_motion_all_envs(goal_cmds_full, start_cmds_full) -> (N_ENVS, T_max, 7)
#     Plans a smooth, self-collision-free arm trajectory for EVERY env in one call.
#     Uses FK(goal_arm_joints) as the goal pose for each env's MotionGen plan.
#     Falls back to linear interpolation per env if MotionGen fails.
#
#     Args:
#       goal_cmds_full  : (N_ENVS, n_dof) — full joint target (from solve_ik_batch_all_envs)
#       start_cmds_full : (N_ENVS, n_dof) — current full joint positions (e.g. cur_cmds)
#
#     Returns:
#       (N_ENVS, T_max, 7) arm trajectories (Actuator1-7), padded to T_max.
#
#     CORRECT USAGE — Phase 2a approach (replaces 150-step linear interpolation):
#       _arm_cols = [_dof_idx[jn] for jn in ARM_JOINT_NAMES]
#       # SUBTRACT approach direction — places target BEFORE object (between robot and object)
#       _approach_tgts = OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (_APPROACH_STANDOFF + IK_EE_MOUNT_ADJ_M)
#       warm_cmds   = solve_ik_batch_all_envs(_approach_tgts, cur_cmds, closed_hand=False)
#       _mg_trajs   = plan_motion_all_envs(warm_cmds, cur_cmds)   # (N, T, 7)
#       for _wi in range(_mg_trajs.shape[1]):
#           cur_cmds[:, _arm_cols] = _mg_trajs[:, _wi, :]
#           robots.set_joint_position_targets(cur_cmds)
#           world.step(render=False)
#           world.step(render=False)   # 2 steps per waypoint = ~33 ms at 50 Hz plan
#
#     _MOTION_GEN is initialised in the boilerplate — do NOT call setup_motion_gen() again.
#
# ── IK helpers — ALWAYS use solve_ik_batch_all_envs (batched GPU IK) ────────
#   solve_ik_batch_all_envs(targets_world, seeds_full, closed_hand=False) -> (N_ENVS, n_dof)
#     Batched IK for ALL envs in one GPU call. Subtracts env_offsets internally.
#     targets_world : (N_ENVS, 3) world-frame EE targets (one per env).
#     seeds_full    : (N_ENVS, n_dof) current joint commands.
#     closed_hand   : False -> open-hand solver (phases 2/3); True -> closed-hand (phase 4).
#     Returns updated cmds with arm joints solved per-env; hand joints unchanged.
#
#     CORRECT USAGE:
#       ee_tgts = WAYPOINTS_WORLD_BATCH[np.arange(N_ENVS), wp_idx] + ee_obj_offset
#       ee_tgts[:, 2] = np.maximum(ee_tgts[:, 2], 0.05)
#       transport_cmds = solve_ik_batch_all_envs(ee_tgts, transport_cmds, closed_hand=True)
#       for jn, jv in HAND_CLOSE_TARGET.items():
#           if jn in _dof_idx: transport_cmds[:, _dof_idx[jn]] = jv
#       robots.set_joint_position_targets(transport_cmds)
#
#     NEVER use np.tile(solve_ik_env0(...), (N_ENVS,1)) — broadcasts env-0's solution
#     to all envs and ignores each env's own object position.
#
#   solve_ik_for_env(env_i, target_world, target_quat_wxyz, seed_joints_1d) -> (n_dof,)
#     Legacy single-env wrapper. Kept for compatibility; prefer solve_ik_batch_all_envs.
#   solve_ik_env0(target_pos, target_quat_wxyz, seed_joints_1d) -> (n_dof,)
#     Low-level env-0 solver. Use directly only for diagnostics.
#
# ── Keypoint helpers ─────────────────────────────────────────────────────────
#   compute_keypoints(pos, quat_wxyz) -> (4,3)
#   compute_keypoints_batch(positions, quats_wxyz) -> (N,4,3)
#   keypoint_max_dist_batch(positions, quats_wxyz)               -> (N,)  uses GOAL_KEYPOINTS_BATCH
#   keypoint_max_dist_batch(positions, quats_wxyz, GOAL_KEYPOINTS_BATCH) -> (N,)  per-env (same)
#   keypoint_max_dist_batch(positions, quats_wxyz, GOAL_KEYPOINTS)        -> (N,)  shared env-0 goal
#   get_ee_pos_env0(stage) -> np.ndarray or None
#   json_safe(obj)         — strip numpy scalars/arrays for json.dumps
#   _NumpyEncoder          — json.JSONEncoder subclass for numpy
#
# ── Batch control ────────────────────────────────────────────────────────────
#   cmd_all = np.tile(cmd_1d, (N_ENVS, 1))
#   robots.set_joint_position_targets(cmd_all)
#   world.step(render=False)  <- auto-renders every _RENDER_EVERY steps
#
# ── Object state — IMPORTANT QUATERNION CONVENTION ───────────────────────────
#   positions, quats_xyzw = objects.get_world_poses()   # (N,3),(N,4) [x,y,z,w]
#   # IsaacSim returns [x,y,z,w]. Convert to wxyz before keypoint helpers:
#   quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])
#   kp_dists = keypoint_max_dist_batch(positions, quats_wxyz)  # per-env goals
#
# ── Reach target: use OBJ_INIT_POS_BATCH, NOT nominal OBJECT_POS ─────────────
#   The boilerplate already ran 60 settle steps. Objects are at OBJ_INIT_POS_BATCH.
#   In Phase 1 just read fresh positions for diagnostics; don't run more settle steps.
#   For Phase 2 reach, target OBJ_INIT_POS_BATCH[ei] per env — NOT the nominal pos.
#
# ── APPROACH_DIRS_BATCH — pre-computed in boilerplate, do NOT recompute ───────
#   APPROACH_DIRS_BATCH : (N_ENVS, 3) unit vectors pointing FROM robot base TOWARD object.
#   Direction: robot_base_XY → object_XY, with slight upward Z=0.07 tilt (matches cylinder_pour.py).
#
#   STANDOFF TARGETS: subtract approach direction to place the target BEFORE the object
#   (between the robot and the object). This is the correct "approach from the robot side":
#     _IK_EE_ADJ        = IK_EE_MOUNT_ADJ_M   # 0.10 m — xhand_mount is behind knuckle centroid
#     _WARM_STANDOFF     = 0.20   # m — desired knuckle-to-object distance at warm standoff
#     _APPROACH_STANDOFF = 0.15   # m
#     _GRASP_STANDOFF    = 0.07   # m — knuckles ~7 cm from object (≈ bottle surface)
#
#     warm_tgts     = OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (_WARM_STANDOFF + _IK_EE_ADJ)
#     approach_tgts = OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (_APPROACH_STANDOFF + _IK_EE_ADJ)
#     grasp_tgts    = OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (_GRASP_STANDOFF + _IK_EE_ADJ)
#
#   *** CRITICAL: SUBTRACT approach direction (not add). Adding places targets PAST the object
#   on the far side from the robot — the arm extends away from the robot and never grasps. ***
#
#   Diagnostics: print APPROACH_DIR_CORRECTED as APPROACH_DIRS_BATCH[0].tolist() (already printed
#   by boilerplate as APPROACH_DIR_ENV0).
#
# ── Already printed by boilerplate (do NOT reprint) ──────────────────────────
#   OBJECT_POSE_INITIAL_ENV0  — pre-settle spawn position
#   SETTLED_OBJ_POSE_ENV0     — post-settle actual position (this is what matters)
#   ALIGN_ENV0..3             — per-env alignment diagnostics
#
# ── JSON serialisation — CRITICAL ────────────────────────────────────────────
#   numpy.float32 scalars (e.g. quats_xyzw[0,3]) are NOT json.dumps-serialisable.
#   ALWAYS use one of:
#     float(quats_xyzw[0,3])                   # scalar -> float
#     quats_xyzw[0].tolist()                   # row -> Python list
#     json_safe({"k": quats_xyzw[0,3]})        # recursive helper
#
# ── Joint state ──────────────────────────────────────────────────────────────
#   joint_pos = robots.get_joint_positions()  # (N, n_dof)
#
# ── Contact forces on the object ─────────────────────────────────────────────
#   get_object_contact_forces() -> np.ndarray (N_ENVS, 3)
#     Net world-frame contact force [Fx, Fy, Fz] on each object in Newtons.
#     Useful to detect when the hand is gripping (large Fz) or pushing.
#     Returns a zero array gracefully if unavailable.
#   Example — log per-env contact force magnitude during grasp:
#     cf = get_object_contact_forces()   # (N_ENVS, 3)
#     cf_mag = np.linalg.norm(cf, axis=1)  # (N_ENVS,)
#     print("CONTACT_FORCES: " + json.dumps(json_safe({
#         "step": step, "cf_mag_per_env": cf_mag.tolist(),
#         "cf_env0": cf[0].tolist(),
#     })))
'''


# Smoke-test suffix appended to the parallel boilerplate for pre-flight checks
PARALLEL_BOILERPLATE_SMOKE_SUFFIX = r"""
# === PARALLEL SMOKE TEST ===
try:
    cmd_all = np.tile(_init_joints_1d, (N_ENVS, 1))
    for _s in range(60):
        robots.set_joint_position_targets(cmd_all)
        world.step(render=False)

    init_pos, init_quats = objects.get_world_poses()
    print(f"SMOKE_OBJ_POS_ENV0: {init_pos[0].tolist()}", flush=True)
    print(f"SMOKE_OBJECT_EXTENTS: {OBJECT_EXTENTS.tolist()}", flush=True)

    ee0 = get_ee_pos_env0(stage)
    print(f"SMOKE_EE_POS_ENV0: {ee0.tolist() if ee0 is not None else 'NONE'}", flush=True)

    kp_dists = keypoint_max_dist_batch(init_pos, init_quats, GOAL_KEYPOINTS)
    print(f"SMOKE_KP_DIST_INIT_MEAN: {round(float(kp_dists.mean()), 4)}", flush=True)

    jp = robots.get_joint_positions()
    print(f"SMOKE_ARM_JOINTS_ENV0: {[round(float(v),4) for v in jp[0,:7].tolist()]}", flush=True)
    print(f"SMOKE_INITIAL_ARM_JOINTS: {INITIAL_ARM_JOINTS}", flush=True)
    print(f"SMOKE_NUM_WAYPOINTS: {NUM_WAYPOINTS}", flush=True)
    print("SMOKE_OK", flush=True)
finally:
    try:
        rep.orchestrator.stop()
    except Exception:
        pass
    simulation_app.close()
"""


# =============================================================================
# Parallel boilerplate filler
# =============================================================================

def _filled_parallel_boilerplate(
    spec: "KeypointTaskSpec",
    *,
    repo_root: str,
    trajectory: list[dict],
    n_envs: int,
    env_spacing: float = 3.0,
) -> str:
    repls = {
        "REPO_ROOT_PLACEHOLDER":             repo_root,
        "N_ENVS_PLACEHOLDER":                str(n_envs),
        "ENV_SPACING_PLACEHOLDER":           repr(float(env_spacing)),
        "ROBOT_USD_PLACEHOLDER":             spec.robot_usd,
        "OBJECT_USD_PLACEHOLDER":            spec.object_usd,
        "OBJECT_POSITION_PLACEHOLDER":       repr(tuple(spec.object_position)),
        "ROBOT_POSITION_PLACEHOLDER":        repr(tuple(spec.robot_position)),
        "OBJECT_ORIENTATION_PLACEHOLDER":    repr(tuple(spec.object_orientation)),
        "OBJECT_SCALE_PLACEHOLDER":          repr(float(spec.object_scale)),
        "OBJECT_MASS_PLACEHOLDER":           repr(float(spec.object_mass)),
        "INITIAL_JOINTS_PLACEHOLDER":        repr(dict(spec.robot_initial_joints)),
        "TRAJECTORY_PATH_PLACEHOLDER":       _write_traj_file(trajectory),
        "CAMERA_POSITION_PLACEHOLDER":       repr(tuple(spec.camera_position)),
        "CAMERA_TARGET_PLACEHOLDER":         repr(tuple(spec.camera_target)),
        "SUCCESS_TOLERANCE_PLACEHOLDER":     repr(float(spec.success_tolerance)),
        "OBJ_Z_SETTLE_OFFSET_PLACEHOLDER":   repr(float(spec.object_z_settle_offset)),
    }
    s = PARALLEL_BOILERPLATE
    for k, v in repls.items():
        s = s.replace(k, v)
    return s


# =============================================================================
# Parallel prompt builders
# =============================================================================

def _format_cross_iter_stats(
    aggregate_stats: Optional[dict],
    best_iter_stats: Optional[dict],
    best_env_stats:  Optional[dict] = None,
) -> str:
    """Format cross-iteration stats for injection into prompts.

    PRIMARY OBJECTIVE encoded here: maximize success_rate, then minimize
    mean_kp_dist, reach_failure_rate, drop_rate (in that order).
    """
    if not aggregate_stats and not best_iter_stats:
        return ""
    lines = [
        "\n=== CROSS-ITERATION STATISTICS (ALWAYS read these before deciding what to change) ===",
        "  PRIMARY: maximize success_rate  SECONDARY: minimize mean_kp_dist, reach_failure_rate, drop_rate",
    ]

    if aggregate_stats:
        total     = aggregate_stats.get("total_runs", 0)
        succs     = aggregate_stats.get("success_count", 0)
        sr_stats  = aggregate_stats.get("success_rate_stats", {})
        kp_min_s  = aggregate_stats.get("kp_dist_stats", {}).get("min", {})
        kp_mean_s = aggregate_stats.get("kp_dist_stats", {}).get("mean", {})
        reach_s   = aggregate_stats.get("reach_score_stats", {})
        rfail_s   = aggregate_stats.get("reach_failure_rate_stats", {})
        drop_s    = aggregate_stats.get("drop_rate_stats", {})
        grasp_r   = aggregate_stats.get("grasp_stable_rate")
        lines.append(f"  aggregate_stats.json  ({total} iterations so far this run):")
        lines.append(f"    success_count={succs}/{total}  overall_success_rate={aggregate_stats.get('overall_success_rate', 0.0):.1%}")
        if sr_stats:
            lines.append(f"    success_rate:         min={sr_stats.get('min',0):.3f}  mean={sr_stats.get('mean',0):.3f}  std={sr_stats.get('std',0):.3f}  max={sr_stats.get('max',0):.3f}")
        if kp_min_s:
            lines.append(f"    min_kp_dist:          min={kp_min_s.get('min',999):.4f}  mean={kp_min_s.get('mean',999):.4f}  std={kp_min_s.get('std',0):.4f}")
        if kp_mean_s:
            lines.append(f"    mean_kp_dist:         min={kp_mean_s.get('min',999):.4f}  mean={kp_mean_s.get('mean',999):.4f}")
        if reach_s:
            lines.append(f"    reach_score:          min={reach_s.get('min',999):.4f}  mean={reach_s.get('mean',999):.4f}  std={reach_s.get('std',0):.4f}")
        if rfail_s:
            lines.append(f"    reach_failure_rate:   min={rfail_s.get('min',1):.3f}  mean={rfail_s.get('mean',1):.3f}  (fraction envs reach_dist>0.12m)")
        if drop_s:
            lines.append(f"    drop_rate:            min={drop_s.get('min',1):.3f}  mean={drop_s.get('mean',1):.3f}  (fraction envs final_ee_obj>0.15m)")
        if grasp_r is not None:
            lines.append(f"    grasp_stable_rate={grasp_r:.1%}")
        lift_r = aggregate_stats.get("any_env_lifted_rate")
        if lift_r is not None:
            lines.append(f"    any_env_lifted_rate={lift_r:.1%}   ← fraction of iterations where at least 1 env lifted the object")
        # Contact / collision stats
        cf_s  = aggregate_stats.get("reach_max_cf_stats", {})
        hvc_s = aggregate_stats.get("high_vel_contact_stats", {})
        ds_s  = aggregate_stats.get("dense_score_stats", {})
        if cf_s:
            lines.append(f"    reach_max_cf (N):     min={cf_s.get('min',0):.2f}  mean={cf_s.get('mean',0):.2f}  max={cf_s.get('max',0):.2f}  (>5N = EE hitting object during approach)")
        if hvc_s:
            lines.append(f"    high_vel_contact:     min={hvc_s.get('min',0):.3f}  mean={hvc_s.get('mean',0):.3f}  max={hvc_s.get('max',0):.3f}  (>0.1 = fast approach near object)")
        if ds_s:
            lines.append(f"    dense_score_mean:     min={ds_s.get('min',0):.2f}  mean={ds_s.get('mean',0):.2f}  max={ds_s.get('max',0):.2f}  (higher = better overall behavior)")

    if best_iter_stats:
        br = best_iter_stats
        lines.append(f"  best_iter_stats.json (best iteration so far this run):")
        lines.append(f"    iteration={br.get('iteration')}  success={br.get('success')}")
        lines.append(f"    success_rate={br.get('success_rate', 0):.1%}  n_envs={br.get('n_envs')}")
        lines.append(f"    reach_score={br.get('reach_score')}  min_kp_dist={br.get('min_kp_dist')}  final_kp={br.get('final_kp')}")
        lines.append(f"    reach_failure_rate={br.get('reach_failure_rate')}  drop_rate={br.get('drop_rate')}")
        lines.append(f"    grasp_stable_any={br.get('grasp_stable_any')}  failure_mode={br.get('failure_mode')}")
        lines.append(f"    any_env_lifted={br.get('any_env_lifted')}")
        lines.append(f"    reach_max_cf={br.get('reach_max_cf')}N  max_high_vel_contact={br.get('max_high_vel_contact')}  dense_score_mean={br.get('dense_score_mean')}")
        if br.get("reach_arm_joints"):
            lines.append(f"    best_reach_arm_joints: {br['reach_arm_joints']}")
        if br.get("code_snippet"):
            lines.append(f"    (best code snippet preserved in best_iter_stats.json for reference)")

    if best_env_stats:
        be = best_env_stats
        lines.append(f"  best_env_stats.json (best single env across ALL iterations):")
        lines.append(f"    from iteration={be.get('iteration')}  env_id={be.get('env_id')}")
        lines.append(f"    final_kp_dist={be.get('final_kp_dist')}  ← THIS is the behavior to replicate and scale up")
        lines.append(f"    reach_min_ee_to_obj={be.get('reach_min_ee_to_obj')}  approach_type={be.get('approach_type')}  obj_knocked_at_reach={be.get('obj_knocked_at_reach')}")
        lines.append(f"    grasp_stable={be.get('grasp_stable')}  max_lift={be.get('max_lift')}")
        if be.get("arm_joints"):
            lines.append(f"    arm_joints: {be['arm_joints']}")
        lines.append(f"    ACTION: Study what made env {be.get('env_id')} succeed. Replicate its approach/grasp in ALL envs.")

    # Trajectory tracking stats from best iteration
    if best_iter_stats:
        br = best_iter_stats
        _te_mean = br.get("traj_pos_err_mean")
        _te_fin  = br.get("traj_pos_err_final")
        _tr_mean = br.get("traj_rot_err_mean")
        _tfs     = br.get("trajectory_following_score")
        if any(v is not None for v in [_te_mean, _te_fin, _tr_mean, _tfs]):
            lines.append(f"  best_iter trajectory tracking:")
            if _te_mean is not None:
                lines.append(f"    traj_pos_err_mean={_te_mean:.4f}m  traj_pos_err_final={_te_fin}")
            if _tr_mean is not None:
                lines.append(f"    traj_rot_err_mean={_tr_mean:.4f}rad")
            if _tfs is not None:
                lines.append(f"    trajectory_following_score={_tfs:.4f}  (higher=better)")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# FORMAT-B  —  object-trajectory-conditioned prompt section
# =============================================================================

def _build_trajectory_demo_section(
    spec: "KeypointTaskSpec",
    frames: list[dict],
    max_frames: int = 30,
) -> str:
    """Build the Format-B block injected into the Claude prompt.

    Returns an empty string when *frames* is empty (backward-compatible with
    Format-A / EE-goal examples that have no trajectory demo).

    Format A (no trajectory):   prompt contains only task description + EE goal
    Format B (with trajectory): prompt additionally contains this block, which
      describes OBJECT motion the controller must reproduce.
    """
    if not frames:
        return ""

    pos0 = spec.object_position
    rot0 = spec.object_orientation  # wxyz

    traj_block = format_for_prompt(frames, max_frames=max_frames)

    return f"""\
--- OBJECT TRAJECTORY DEMO (Format B) ---
The following recorded trajectory shows the DESIRED object motion for this task.
Use it to infer the correct contact strategy, push/pull direction, and motion
primitive.  This is NOT a robot / EE trajectory — do not replay it as joint targets.

Initial object pose:
  position:    [{pos0[0]:+.4f}, {pos0[1]:+.4f}, {pos0[2]:+.4f}]
  orientation: [{rot0[0]:+.4f}, {rot0[1]:+.4f}, {rot0[2]:+.4f}, {rot0[3]:+.4f}]  (wxyz)

{traj_block}
--- END OBJECT TRAJECTORY DEMO ---
"""


def build_initial_prompt_parallel(
    spec: "KeypointTaskSpec",
    *,
    repo_root: str,
    trajectory: list[dict],
    n_envs: int,
    aggregate_stats: Optional[dict] = None,
    best_iter_stats: Optional[dict] = None,
    best_env_stats:  Optional[dict] = None,
) -> str:
    boilerplate = _filled_parallel_boilerplate(
        spec, repo_root=repo_root, trajectory=trajectory,
        n_envs=n_envs, env_spacing=spec.env_spacing,
    )
    n_traj = len(trajectory)
    first = trajectory[0]["pos"] if trajectory else list(spec.object_position)
    last  = trajectory[-1]["pos"] if trajectory else list(spec.object_position)

    rx, ry, rz = spec.robot_position
    ox, oy, oz = spec.object_position
    dx, dy = ox - rx, oy - ry
    dist_horiz = (dx**2 + dy**2) ** 0.5

    task_desc_section = (
        f"\n--- TASK ---\n{spec.task_description}\n"
        if spec.task_description else ""
    )

    # Load in-context examples (Q/A format with trajectories).
    _ice_block = load_in_context_examples(max_keyframes=20)
    if _ice_block:
        ice_section = (
            "=== IN-CONTEXT EXAMPLES (strategy demonstrations — ADAPT, do not copy) ===\n"
            "Each example shows: given task + initial pose + desired object trajectory → solution code.\n"
            "Use these examples to understand the manipulation strategy.\n"
            "For your new task: adapt contact points, grasp type, approach direction, and transport\n"
            "behavior to match the new object and trajectory. Do NOT copy-paste the closest example.\n\n"
            + _ice_block
            + "\n\n=== END IN-CONTEXT EXAMPLES ===\n"
        )
    else:
        ice_section = ""

    # Build the new task QUESTION in trajectory-tracking format.
    _task_question = format_new_task_question(
        task=spec.task_description or spec.task_name or "Manipulation task",
        initial_pos=list(spec.object_position),
        initial_rot_wxyz=list(spec.object_orientation),
        frames=trajectory,
        task_name=spec.task_name,
        max_keyframes=20,
    )

    # Format-B: also include legacy trajectory demo section if spec supplies one.
    _demo_frames = load_for_prompt(spec.object_trajectory_demo_path)
    traj_demo_section = _build_trajectory_demo_section(spec, _demo_frames)

    cross_run_section = _format_cross_iter_stats(aggregate_stats, best_iter_stats, best_env_stats)

    # Pre-escape the APPROACH_DIR_CORRECTED print example so it survives the outer f-string.
    _approach_dir_print_example = (
        'print(f"APPROACH_DIR_CORRECTED: {_approach_dirs[0].tolist()}", flush=True)'
    )

    return f"""\
{ice_section}
=== YOUR NEW TASK ===
{_task_question}
Write the motion-control section for the PARALLEL ({n_envs} envs) task below.
Return the FULL SCRIPT (boilerplate verbatim, then your motion code appended).
{task_desc_section}
{traj_demo_section}
{CONTROL_POLICY}
{CONTACT_VALIDITY_RULE}
{HAND_JOINT_SEMANTICS}
{MANIPULATION_STRATEGIES}
{cross_run_section}
--- TASK PARAMETERS ---
  N_ENVS:                {n_envs}
  Object init pos:       {list(spec.object_position)} (NOMINAL — each env gets ±5cm XY noise, no rotation noise)
  Object init rot(wxyz): {list(spec.object_orientation)}
  Trajectory frames:     {n_traj}  (first_pos={list(first)}, last_pos={list(last)})
  Success tolerance:     {spec.success_tolerance} m  (max keypoint distance to goal)

--- SPATIAL FACTS ---
  Robot base:    ({rx:.3f}, {ry:.3f}, {rz:.3f})
  Object pos:    ({ox:.3f}, {oy:.3f}, {oz:.3f})
  Robot→Object:  dx={dx:+.3f}m, dy={dy:+.3f}m  ({dist_horiz:.3f}m horizontal)
  Object z={oz:.3f}m — {"NEAR GROUND (z<0.15): prefer top-down approach" if oz < 0.15 else "above ground level"}

--- SPATIAL APPROACH GUIDE ---
  Phase 2 uses IK — compute per-env approach directions from geometry, not a shared global value.
  REACH_X/Y/Z_OFFSET is diagnostic info for env-0 ONLY; do NOT use it to manually tune a shared direction.
  Compute per-env approach directions (see system prompt KEY CONSTRAINTS for the code template).
  MANDATORY PRINT: after computing _approach_dirs, print env-0's direction for diagnostics:
      {_approach_dir_print_example}
  This is parsed by the pipeline for diagnostics — do NOT skip it.
  UNDERREACH: use DIRECT INCREMENTAL APPROACH — start from home EE position, move 0.004m/step toward grasp_tgt over 400 steps. DO NOT use "approach from above" (IK fails for high-Z targets) and never use warm standoff with linear interp (sweeps arm through object).

--- PHASE BUDGETS ---
  Phase 1 SETTLE    = {spec.t_settle}
  Phase 2 REACH     = {spec.t_approach}
  Phase 3 GRASP     = {spec.t_grasp}
  Phase 4 TRANSPORT = {spec.t_transport}
  Phase 5 HOLD      = {spec.t_hold}

--- BOILERPLATE (copy verbatim at top of your output) ---
{boilerplate}

# === YOUR MOTION CODE STARTS HERE ===
# Key reminders for PARALLEL mode:
#   * Batch control: cmd_all = np.tile(cmd_1d, (N_ENVS, 1))
#                   robots.set_joint_position_targets(cmd_all)
#                   world.step(render=False)  ← auto-renders every _RENDER_EVERY steps
#
#   * Object state — ALWAYS convert quats [x,y,z,w] → wxyz before keypoint helpers:
#       positions, quats_xyzw = objects.get_world_poses()   # (N,3),(N,4) [x,y,z,w]
#       quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])
#   * Joint state:  joint_pos = robots.get_joint_positions()      # (N, n_dof)
#   * EE (env 0):   ee0 = get_ee_pos_env0(stage)
#   * Reach arm joints for hold: hold_arm_joints = list(robots.get_joint_positions()[0, :7])
#
#   CRITICAL — PER-ENV ALIGNMENT:
#   The boilerplate already ran 60 settle steps and computed per-env alignment.
#   Waypoints are anchored to the ACTUAL settled pose, not the nominal position.
#   Use OBJ_INIT_POS_BATCH[0] (not OBJECT_POS) as the reach target for env 0.
#   WAYPOINTS_WORLD_BATCH[env_i] gives the correct world-frame path for each env.
#
#   CRITICAL — JSON SAFETY:  numpy.float32 scalars (e.g. positions[0,2], quats[i,j])
#   are NOT json.dumps-serialisable.  Always use one of:
#     float(positions[0,2])   — convert single scalar
#     positions[0].tolist()   — convert a row to Python list
#     json_safe({{...}})       — recursive helper (available from boilerplate)
#   Example safe FRAME_STATE print (note quats conversion):
#     positions, quats_xyzw = objects.get_world_poses()
#     quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])
#     kp_dists = keypoint_max_dist_batch(positions, quats_wxyz)  # per-env goals
#     print("FRAME_STATE: " + json.dumps(json_safe({{"step": step, "phase": "transport",
#         "kp_mean": round(float(kp_dists.mean()),4),
#         "kp_min": round(float(kp_dists.min()),4),
#         "obj_z_env0": round(float(positions[0,2]),4),
#         "obj_lifted": bool(float(positions[0,2]) > obj_z_init + 0.05)}})))
#
#   * OBJECT_POSE_INITIAL_ENV0 and SETTLED_OBJ_POSE_ENV0 are ALREADY printed.
#     Do NOT print these tags again. Use fresh objects.get_world_poses() for Phase 1
#     diagnostics.
#
#   * Phase 1 DIAGNOSTIC (~10 steps, NOT settle — already settled by boilerplate):
#     positions, quats_xyzw = objects.get_world_poses()
#     quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])
#     obj_z_init = float(positions[0, 2])   # env-0 reference height for lift detection
#
#   * Phase 2 REACH — DIRECT INCREMENTAL APPROACH (MotionGen unavailable):
#     # NOTE: "approach from above" (IK to above_tgts at Z+0.20 then linear interp) is BROKEN —
#     # IK fails for all envs for high-Z positions, leaving above_cmds = home joints (no movement).
#     # Instead, do a single incremental IK approach directly from the current home EE position.
#     # Use APPROACH_DIRS_BATCH from boilerplate — do NOT recompute from EE position.
#     _approach_dirs = APPROACH_DIRS_BATCH  # (N_ENVS,3) robot-base → object, already correct
#     _GRASP_STANDOFF    = 0.07   # m — near-contact standoff for Phase 2 and Phase 3 anchor
#     _IK_EE_ADJ         = IK_EE_MOUNT_ADJ_M   # 0.10 m
#     # Grasp target: SUBTRACT approach_dirs (not add — adding puts arm PAST the object)
#     grasp_tgts = OBJ_INIT_POS_BATCH - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ)
#     grasp_tgts[:, 2] = np.maximum(grasp_tgts[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)
#     # Start from ACTUAL current EE positions (home position after settle)
#     _ee_now = get_ee_pos_all_envs(stage, N_ENVS)
#     _reach_pos = np.array([
#         _ee_now[ei] if _ee_now[ei] is not None else grasp_tgts[ei]
#         for ei in range(N_ENVS)], dtype=np.float64)
#     _MAX_STEP_2 = 0.004
#     for _s in range(400):
#         _delta = grasp_tgts - _reach_pos
#         _dist  = np.linalg.norm(_delta, axis=1, keepdims=True)
#         _step  = np.where(_dist > _MAX_STEP_2, _delta * _MAX_STEP_2 / np.maximum(_dist, 1e-8), _delta)
#         _reach_pos += _step
#         cur_cmds = solve_ik_batch_all_envs(_reach_pos, cur_cmds, closed_hand=False)
#         robots.set_joint_position_targets(cur_cmds); world.step(render=False)
#     for _s in range(50):
#         robots.set_joint_position_targets(cur_cmds); world.step(render=False)
#     # After Phase 2: log reach score
#     # ee0 = get_ee_pos_env0(stage); reach_ref = OBJ_INIT_POS_BATCH[0]
#     # reach_dist = np.linalg.norm(ee0 - reach_ref); reach_rel = reach_ref - ee0
#     # print(f"REACH_SCORE: {{reach_dist:.4f}}"); print(f"REACH_X_OFFSET: {{reach_rel[0]:.4f}}")
#
#   * Phase 4 TRANSPORT — BATCHED IK, per-env waypoint advancement:
#     wp_idx = np.zeros(N_ENVS, dtype=int)
#     prev_best_kp = np.full(N_ENVS, 999.0); transport_cmds = cur_cmds.copy()
#     for step in range(500):
#         positions, quats_xyzw = objects.get_world_poses()
#         obj_pos = np.asarray(positions, dtype=np.float64)
#         quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])
#         for ei in range(N_ENVS):   # waypoint advancement still per-env (conditional)
#             if float(np.linalg.norm(obj_pos[ei]-WAYPOINTS_WORLD_BATCH[ei,wp_idx[ei]])) < 0.05 and wp_idx[ei] < NUM_WAYPOINTS-1:
#                 wp_idx[ei] += 1
#         ee_tgts = WAYPOINTS_WORLD_BATCH[np.arange(N_ENVS), wp_idx] + ee_obj_offset  # (N,3)
#         ee_tgts[:, 2] = np.maximum(ee_tgts[:, 2], 0.05)   # ← KEY: offset keeps hand to side
#         transport_cmds = solve_ik_batch_all_envs(ee_tgts, transport_cmds, closed_hand=True)
#         for jn,jv in SELECTED_GRASP_TARGET.items():   # use SELECTED, not HAND_CLOSE_TARGET
#             if jn in _dof_idx: transport_cmds[:,_dof_idx[jn]] = jv
#         robots.set_joint_position_targets(transport_cmds); world.step(render=False)
#         if step % 30 == 0:
#             kp_dists = keypoint_max_dist_batch(obj_pos, quats_wxyz)
#             progress = np.maximum(prev_best_kp - kp_dists, 0.0)
#             prev_best_kp = np.minimum(prev_best_kp, kp_dists)
#             lift = [round(float(obj_pos[ei,2])-float(OBJ_INIT_POS_BATCH[ei,2]),4) for ei in range(N_ENVS)]
#             print("FRAME_STATE: " + json.dumps(json_safe({{"step": step, "phase": "transport",
#                 "kp_mean": round(float(kp_dists.mean()),4), "kp_min": round(float(kp_dists.min()),4),
#                 "kp_per_env": [round(float(v),4) for v in kp_dists],
#                 "obj_z_env0": round(float(obj_pos[0,2]),4),
#                 "obj_lifted": bool(float(obj_pos[0,2]) > float(OBJ_INIT_POS_BATCH[0,2])+0.05),
#                 "lift_per_env": lift, "wp_idx_env0": int(wp_idx[0]),
#                 "kp_progress": [round(float(v),4) for v in progress],
#                 "best_env": int(np.argmin(kp_dists))}}))))
#
#   * FINAL OUTPUT (mandatory — use per-env goals):
#       final_pos, final_quats_xyzw = objects.get_world_poses()
#       final_quats_wxyz = np.column_stack([final_quats_xyzw[:,3], final_quats_xyzw[:,:3]])
#       kp_dists = keypoint_max_dist_batch(final_pos, final_quats_wxyz)  # per-env GOAL_KEYPOINTS_BATCH
#       success_mask = kp_dists < SUCCESS_TOL
#       success_count = int(success_mask.sum())
#       print(f"FINAL_KP_MAX_DIST_BATCH: {{kp_dists.tolist()}}")
#       print(f"FINAL_KP_MAX_DIST_MEAN: {{round(float(kp_dists.mean()), 4)}}")
#       print(f"FINAL_KP_MAX_DIST_MIN: {{round(float(kp_dists.min()), 4)}}")
#       _ee_all_final = get_ee_pos_all_envs(stage, N_ENVS)
#       _ee_obj_dists = [float(np.linalg.norm(_ee_all_final[ei] - final_pos[ei])) if _ee_all_final[ei] is not None else -1.0 for ei in range(N_ENVS)]
#       _valid_ee = [d for d in _ee_obj_dists if d >= 0]
#       print(f"FINAL_EE_OBJ_DIST_BATCH: {{[round(d,4) for d in _ee_obj_dists]}}")
#       print("BATCH_METRICS: " + json.dumps({{"success_count": success_count,
#           "n_envs": N_ENVS, "success_rate": round(success_count / N_ENVS, 4),
#           "mean_kp_dist": round(float(kp_dists.mean()), 4),
#           "min_kp_dist": round(float(kp_dists.min()), 4),
#           "final_ee_obj_dist_min": round(float(min(_valid_ee)), 4) if _valid_ee else -1.0,
#           "final_ee_obj_dist_mean": round(float(sum(_valid_ee)/len(_valid_ee)), 4) if _valid_ee else -1.0}}))
#       print("SUCCESS" if success_count > 0 else "FAILURE")
#
#   * Wrap all phases in try/except/finally:
#       finally: rep.orchestrator.stop(); simulation_app.close()
# === END OF YOUR SECTION ===

{spec.output_format_instruction}
"""


def build_feedback_prompt_parallel(
    spec: "KeypointTaskSpec",
    prev_code: str,
    stdout: str,
    stderr: str,
    iteration: int,
    frame_states: list[dict],
    best_min_kp_dist: float,
    history: list[dict],
    *,
    repo_root: str,
    trajectory: list[dict],
    n_envs: int,
    best_reach_score: float = 999.0,
    best_reach_arm_joints: Optional[str] = None,
    best_runs_summary: str = "",
    current_stage: str = "approach",
    best_approach_joints: Optional[str] = None,
    best_per_env_stats: Optional[list] = None,
    best_env_summary: Optional[dict] = None,
    last_end_memory: Optional[dict] = None,
    best_final_ee_obj_dist: float = 999.0,
    aggregate_stats: Optional[dict] = None,
    best_iter_stats: Optional[dict] = None,
    best_env_stats:  Optional[dict] = None,
    stuck_counter:   int = 0,
    best_grasp_state: Optional[dict] = None,
) -> str:
    useful = [
        l for l in stdout.splitlines()
        if l.strip() and not l.startswith("2026-") and not l.startswith("20")
        and not l.startswith("[")
    ]

    def tagged(tag: str) -> Optional[str]:
        return _tag_line(useful, tag)

    import json as _json, random as _random, re as _re_p2

    # Extract Phase 2 block from prev_code for verbatim copying in feedback
    def _extract_phase2_block(code: str) -> str:
        """Extract the Phase 2 reach loop from prev_code (up to 55 lines)."""
        # Find the PHASE 2 print marker
        m = _re_p2.search(
            r'(print\("=== PHASE 2:.*?|# *=+ *Phase 2:.*?|# *=+ *PHASE 2.*?)'
            r'(?=print\("=== PHASE 3:|# *=+ *Phase 3:|# *=+ *PHASE 3)',
            code, _re_p2.DOTALL | _re_p2.IGNORECASE,
        )
        if m:
            lines = m.group(0).splitlines()[:55]
            return '\n'.join(lines)
        return ""

    _phase2_from_prev = _extract_phase2_block(prev_code) if prev_code else ""

    # Parse batch metrics
    batch_metrics: Optional[dict] = None
    bm_line = tagged("BATCH_METRICS")
    if bm_line:
        try:
            batch_metrics = _json.loads(bm_line.split(":", 1)[1].strip())
        except Exception:
            pass

    # Extract trajectory tracking metrics from batch_metrics
    _traj_metrics_cur: dict = {}
    if batch_metrics:
        for _k in ("traj_pos_err_mean", "traj_pos_err_min", "traj_pos_err_final",
                   "traj_rot_err_mean", "traj_rot_err_final",
                   "traj_kp_err_mean", "traj_kp_err_final",
                   "best_traj_env", "trajectory_following_score",
                   "success_rate", "min_kp_dist", "reach_failure_rate", "drop_rate"):
            if _k in batch_metrics:
                _traj_metrics_cur[_k] = batch_metrics[_k]
    # Build iteration delta block (compact token-saving summary)
    _prev_metrics: Optional[dict] = None
    if last_end_memory:
        _prev_metrics = {
            "min_kp_dist":          last_end_memory.get("cur_best_goal_kp"),
            "traj_pos_err_mean":    last_end_memory.get("traj_pos_err_mean"),
            "traj_rot_err_mean":    last_end_memory.get("traj_rot_err_mean"),
            "trajectory_following_score": last_end_memory.get("trajectory_following_score"),
        }
    _best_traj_env_cur = batch_metrics.get("best_traj_env") if batch_metrics else None
    _iter_delta_block = format_compact_iter_delta(
        prev_metrics=_prev_metrics,
        cur_metrics=_traj_metrics_cur,
        best_traj_env=_best_traj_env_cur,
    )

    # Trajectory metrics block for display
    _traj_block = ""
    if _traj_metrics_cur:
        _traj_lines = ["\n=== TRAJECTORY TRACKING METRICS (this iteration) ==="]
        for _k in ("traj_pos_err_mean", "traj_pos_err_min", "traj_pos_err_final",
                   "traj_rot_err_mean", "traj_rot_err_final",
                   "traj_kp_err_mean", "traj_kp_err_final",
                   "trajectory_following_score", "best_traj_env"):
            if _k in _traj_metrics_cur:
                _v = _traj_metrics_cur[_k]
                _traj_lines.append(f"  {_k} = {_v}")
        _traj_block = "\n".join(_traj_lines) + "\n"

    # Detect if previous code used np.tile(solve_ik_env0) — the main failure mode
    used_np_tile_ik = "np.tile(new_cmd" in prev_code or "np.tile(solve_ik_env0" in prev_code
    used_solve_ik_for_env = "solve_ik_for_env" in prev_code
    used_solve_ik_batch = "solve_ik_batch_all_envs" in prev_code

    # Parse per-env kp distances and identify best env
    per_env_kp_summary = ""
    best_env_idx: Optional[int] = None
    kp_all_list: list = []
    kp_batch_line = tagged("FINAL_KP_MAX_DIST_BATCH")
    if kp_batch_line:
        try:
            kp_all_list = _json.loads(kp_batch_line.split(":", 1)[1].strip())
            n = len(kp_all_list)
            if n > 0:
                idx_best  = int(min(range(n), key=lambda i: kp_all_list[i]))
                idx_worst = int(max(range(n), key=lambda i: kp_all_list[i]))
                best_env_idx = idx_best
                pool = [i for i in range(n) if i != idx_best and i != idx_worst]
                n_rand = min(3, len(pool))
                rand_idxs = _random.sample(pool, n_rand) if n_rand else []
                sampled = sorted(set([idx_best, idx_worst] + rand_idxs))
                rows = [f"  env{i}: kp_dist={kp_all_list[i]:.4f}m" +
                        (" ← best" if i == idx_best else " ← worst" if i == idx_worst else "")
                        for i in sampled]
                per_env_kp_summary = (
                    f"Per-env final kp_dist ({len(sampled)} of {n} envs — best, worst, {n_rand} random):\n"
                    + "\n".join(rows)
                )
        except Exception:
            pass

    cur_success_rate = batch_metrics.get("success_rate", 0.0) if batch_metrics else 0.0
    cur_success_count = batch_metrics.get("success_count", 0) if batch_metrics else 0
    cur_mean_kp = batch_metrics.get("mean_kp_dist") if batch_metrics else None
    cur_min_kp  = batch_metrics.get("min_kp_dist")  if batch_metrics else None
    cur_ee_obj_dist_min  = batch_metrics.get("final_ee_obj_dist_min")  if batch_metrics else None
    cur_ee_obj_dist_mean = batch_metrics.get("final_ee_obj_dist_mean") if batch_metrics else None

    # ── Regression detection from END_MEMORY ──────────────────────────────
    regression_block = ""
    if last_end_memory:
        prev_reach = float(last_end_memory.get("cur_best_reach", 999))
        prev_goal  = float(last_end_memory.get("cur_best_goal_kp", 999))
        prev_stage = last_end_memory.get("stage_reached", "approach")
        prev_coupled = bool(last_end_memory.get("trajectory_coupled_any", False))
        prev_lifted  = bool(last_end_memory.get("any_env_lifted", False))
        # compare to current run
        cur_reach_val = _parse_float(tagged("REACH_SCORE")) or 999.0
        cur_goal_val  = cur_min_kp if cur_min_kp is not None else 999.0
        reach_regressed = cur_reach_val > prev_reach + 0.02
        goal_regressed  = cur_goal_val  > prev_goal  + 0.02
        regression_block = (
            f"\n=== REGRESSION ANALYSIS (vs previous run) ===\n"
            f"  prev_best_reach={prev_reach:.4f}m  cur_reach={cur_reach_val:.4f}m  "
            f"{'REGRESSED ↑' if reach_regressed else 'OK ↓' if cur_reach_val < prev_reach else '~SAME'}\n"
            f"  prev_best_goal_kp={prev_goal:.4f}m  cur_goal_kp={cur_goal_val:.4f}m  "
            f"{'REGRESSED ↑' if goal_regressed else 'OK ↓' if cur_goal_val < prev_goal else '~SAME'}\n"
            f"  prev_stage_reached={prev_stage}  prev_coupled={prev_coupled}  prev_lifted={prev_lifted}\n"
            f"  best_env_from_prev={last_end_memory.get('best_env', 'N/A')}  "
            f"why={last_end_memory.get('why_better', '')}\n"
        )
        # The canonical working Phase 2 template (always safe to use)
        _phase2_template = (
            "    print(\"=== PHASE 2: REACH ===\", flush=True)\n"
            "    _IK_EE_ADJ = IK_EE_MOUNT_ADJ_M   # 0.10 m\n"
            "    _GRASP_STANDOFF = 0.07; _MAX_STEP_2 = 0.004\n"
            "    _approach_dirs = APPROACH_DIRS_BATCH  # (N_ENVS,3) pre-computed\n"
            "    # SUBTRACT approach direction (not add — adding sends arm PAST the object)\n"
            "    grasp_tgts = OBJ_INIT_POS_BATCH - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ)\n"
            "    grasp_tgts[:, 2] = np.maximum(grasp_tgts[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)\n"
            "    _ee_now = get_ee_pos_all_envs(stage, N_ENVS)\n"
            "    _reach_pos = np.array([\n"
            "        _ee_now[ei] if _ee_now[ei] is not None else grasp_tgts[ei]\n"
            "        for ei in range(N_ENVS)], dtype=np.float64)\n"
            "    for _s in range(400):\n"
            "        _delta = grasp_tgts - _reach_pos\n"
            "        _dist = np.linalg.norm(_delta, axis=1, keepdims=True)\n"
            "        _reach_pos += np.where(_dist > _MAX_STEP_2, _delta * _MAX_STEP_2 / np.maximum(_dist, 1e-8), _delta)\n"
            "        cur_cmds = solve_ik_batch_all_envs(_reach_pos, cur_cmds, closed_hand=False)\n"
            "        robots.set_joint_position_targets(cur_cmds); world.step(render=False)\n"
            "    for _s in range(50):\n"
            "        robots.set_joint_position_targets(cur_cmds); world.step(render=False)\n"
        )
        if reach_regressed:
            regression_block += (
                "  ACTION: Reach REGRESSED. The previous Phase 2 code was WRONG and broke the approach.\n"
                "  DO NOT restore the broken Phase 2 code. Use the canonical DIRECT INCREMENTAL approach:\n"
                "  DO NOT add 'approach from above', DO NOT split into sub-phases, DO NOT modify step size.\n\n"
                "  *** RESTORE PHASE 2 TO THIS EXACT CODE (canonical working template) ***\n\n"
                + _phase2_template +
                "  *** END PHASE 2 TEMPLATE ***\n"
            )
        elif current_stage in ("grasp", "transport") and _phase2_from_prev:
            regression_block += (
                "\n  PHASE 2 APPROACH IS WORKING — DO NOT TOUCH IT.\n"
                "  The reach succeeded in the previous run. DO NOT modify Phase 2 in ANY way.\n"
                "  DO NOT add 'approach from above', DO NOT split into sub-phases 2a/2b.\n"
                "  DO NOT add dense reward tracking, contact force monitoring, or helper functions\n"
                "  to Phase 2. Copy Phase 2 VERBATIM from the previous iteration:\n\n"
                + '\n'.join("  " + l for l in _phase2_from_prev.splitlines()) + "\n"
                "  *** THIS IS THE LOCKED PHASE 2 BLOCK — COPY IT VERBATIM ***\n"
            )
        if not prev_lifted:
            regression_block += (
                "  WARNING: Previous run did NOT lift the object (any_env_lifted=False).\n"
                "  The object is being pushed/dragged, NOT grasped. Improve grasp before transport.\n"
            )

    # ── Stage-specific priority advice ────────────────────────────────────
    stage_block = ""
    if current_stage == "approach":
        stage_block = (
            "\n=== CURRENT OPTIMIZATION STAGE: APPROACH ===\n"
            "The reach is not yet reliable. Do NOT optimize grasp or transport yet.\n"
            "Focus exclusively on:\n"
            "  1. Getting min_ee_to_obj < 0.10m in at least one env.\n"
            "  2. Using solve_ik_batch_all_envs(tgts, cmds, closed_hand=False) — never np.tile, never per-env IK loop.\n"
            "  3. Use APPROACH_DIRS_BATCH from boilerplate — do NOT recompute approach directions.\n"
            "     Standoff targets: OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (standoff + IK_EE_MOUNT_ADJ_M)\n"
            "     (SUBTRACT — places target BEFORE the object, between robot and object).\n"
            "  4. Standoffs: _APPROACH_STANDOFF=0.15m (pregrasp), _GRASP_STANDOFF=0.07m (contact).\n"
            "  5. Track min_ee_per_env[ei] per step and print in FRAME_STATE.\n"
            "  6. Do NOT add new phases or modify transport until approach works.\n"
        )
        if best_approach_joints:
            stage_block += (
                f"\nBEST KNOWN APPROACH ARM JOINTS (arm Actuator1-7 from best reach run):\n"
                f"  _BEST_APPROACH_ARM = {best_approach_joints}\n"
                "\n"
                "MANDATORY PHASE 2 — DIRECT INCREMENTAL APPROACH (copy this block exactly):\n"
                "  # 'approach from above' (IK to Z+0.20 then linear interp) is BROKEN: IK fails\n"
                "  # for all envs at high-Z targets, above_cmds = home joints, nothing happens.\n"
                "  # Instead: incremental IK directly from current home EE toward grasp_tgt.\n"
                "  _approach_dirs = APPROACH_DIRS_BATCH  # (N_ENVS,3) robot-base → object\n"
                "  _GRASP_STANDOFF = 0.07; _IK_EE_ADJ = IK_EE_MOUNT_ADJ_M\n"
                "  # SUBTRACT approach direction (not add — adding sends arm PAST the object)\n"
                "  grasp_tgts = OBJ_INIT_POS_BATCH - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ)\n"
                "  grasp_tgts[:, 2] = np.maximum(grasp_tgts[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)\n"
                "  # Start from ACTUAL current EE positions (home position after settle)\n"
                "  _ee_now = get_ee_pos_all_envs(stage, N_ENVS)\n"
                "  _reach_pos = np.array([\n"
                "      _ee_now[ei] if _ee_now[ei] is not None else grasp_tgts[ei]\n"
                "      for ei in range(N_ENVS)], dtype=np.float64)\n"
                "  for _s in range(400):\n"
                "      _delta = grasp_tgts - _reach_pos\n"
                "      _dist = np.linalg.norm(_delta, axis=1, keepdims=True)\n"
                "      _reach_pos += np.where(_dist > 0.004, _delta * 0.004 / np.maximum(_dist, 1e-8), _delta)\n"
                "      cur_cmds = solve_ik_batch_all_envs(_reach_pos, cur_cmds, closed_hand=False)\n"
                "      robots.set_joint_position_targets(cur_cmds); world.step(render=False)\n"
                "  for _ws in range(50):\n"
                "      robots.set_joint_position_targets(cur_cmds); world.step(render=False)\n"
            )
    elif current_stage == "grasp":
        stage_block = (
            "\n=== CURRENT OPTIMIZATION STAGE: GRASP ===\n"
            "REACH IS WORKING. *** DO NOT MODIFY PHASE 2 IN ANY WAY. ***\n"
            "Do NOT add 'approach from above', do NOT split Phase 2 into sub-phases,\n"
            "do NOT add dense reward functions or contact-force monitors to Phase 2.\n"
            "Copy Phase 2 VERBATIM from the regression block above (locked code).\n"
            "Now focus on establishing a stable side-enclosing grasp.\n"
            "  1. PHASE 2 IS LOCKED — DO NOT CHANGE IT (shown in regression block above).\n"
            "  2. Capture ee_obj_offset per env AFTER Phase 2 (before closing hand) with SANITY CHECK:\n"
            "       ee_obj_offset = np.zeros((N_ENVS, 3))\n"
            "       _ee_post_reach = get_ee_pos_all_envs(stage, N_ENVS)\n"
            "       _obj_post_reach, _ = objects.get_world_poses(); _obj_post_reach = np.asarray(_obj_post_reach, float)\n"
            "       for ei in range(N_ENVS):\n"
            "           if _ee_post_reach[ei] is not None:\n"
            "               _raw = _ee_post_reach[ei] - _obj_post_reach[ei]\n"
            "               # CRITICAL: clamp bad offsets — if EE is > 8cm from object, IK diverged\n"
            "               ee_obj_offset[ei] = _raw if np.linalg.norm(_raw) < 0.08 else -_approach_dirs[ei] * _GRASP_STANDOFF\n"
            "           else:\n"
            "               ee_obj_offset[ei] = -_approach_dirs[ei] * _GRASP_STANDOFF  # SUBTRACT\n"
            "  3. Ramp HAND_CLOSE_TARGET over 150 steps while holding IK at grasp standoff.\n"
            "  4. Detect grasp_stable: EE proximity after finger closure (see GRASP STABILITY in system prompt).\n"
            "     grasp_stable[ei] = ee_to_obj_post3[ei] < 0.08 and obj_tilt[ei] < 45\n"
            "     # CRITICAL: do NOT use 'or min_ee_per_env < 0.08' — that historical minimum causes false positives\n"
            "     # when the EE briefly touched the object during approach but then bounced far away.\n"
            "  5. Print grasp_stable per env in PER_ENV_STATS.\n"
            "\n"
            "PHASE 4 TRANSPORT — mandatory rule (USE grasp_gate, NOT just grasp_stable):\n"
            "  grasp_gate[ei] = grasp_stable_per_env[ei] or (ee_to_obj_post3[ei] < 0.12)\n"
            "  For envs in grasp_gate: advance wp_idx and target WAYPOINTS[ei,wp_idx] + ee_obj_offset[ei]\n"
            "  For envs NOT in grasp_gate: target LIVE object position + approach standoff (re-approach):\n"
            "       live_obj, _ = objects.get_world_poses(); live_obj = np.asarray(live_obj, float)\n"
            "       for ei in range(N_ENVS):\n"
            "           if grasp_gate[ei]:\n"
            "               ee_tgts[ei] = WAYPOINTS_WORLD_BATCH[ei, wp_idx[ei]] + ee_obj_offset[ei]\n"
            "           else:\n"
            "               ee_tgts[ei] = live_obj[ei] - _approach_dirs[ei] * (_GRASP_STANDOFF + _IK_EE_ADJ)  # SUBTRACT\n"
            "               ee_tgts[ei, 2] = max(float(ee_tgts[ei, 2]), float(OBJ_INIT_POS_BATCH[ei, 2]) - 0.02)\n"
            "  NOTE: grasp_gate is PERMISSIVE — EE within 12cm after Phase 3 = in contact = follow trajectory.\n"
            "  Do NOT gate on max_lift: the arm rarely lifts during the brief Phase 3 hold.\n"
        )
        if best_approach_joints:
            stage_block += (
                f"\nBEST KNOWN APPROACH ARM JOINTS (arm Actuator1-7 from best reach run):\n"
                f"  _BEST_APPROACH_ARM = {best_approach_joints}\n"
                "\n"
                "MANDATORY PHASE 2 — DIRECT INCREMENTAL APPROACH (same block as approach stage):\n"
                "  # 'approach from above' is BROKEN — IK fails for high-Z targets, nothing moves.\n"
                "  _approach_dirs = APPROACH_DIRS_BATCH  # (N_ENVS,3) robot-base → object\n"
                "  _GRASP_STANDOFF = 0.07; _IK_EE_ADJ = IK_EE_MOUNT_ADJ_M\n"
                "  # SUBTRACT approach direction (not add — adding sends arm PAST the object)\n"
                "  grasp_tgts = OBJ_INIT_POS_BATCH - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ)\n"
                "  grasp_tgts[:, 2] = np.maximum(grasp_tgts[:, 2], OBJ_INIT_POS_BATCH[:, 2] - 0.02)\n"
                "  _ee_now = get_ee_pos_all_envs(stage, N_ENVS)\n"
                "  _reach_pos = np.array([\n"
                "      _ee_now[ei] if _ee_now[ei] is not None else grasp_tgts[ei]\n"
                "      for ei in range(N_ENVS)], dtype=np.float64)\n"
                "  for _s in range(400):\n"
                "      _delta = grasp_tgts - _reach_pos\n"
                "      _dist = np.linalg.norm(_delta, axis=1, keepdims=True)\n"
                "      _reach_pos += np.where(_dist > 0.004, _delta * 0.004 / np.maximum(_dist, 1e-8), _delta)\n"
                "      cur_cmds = solve_ik_batch_all_envs(_reach_pos, cur_cmds, closed_hand=False)\n"
                "      robots.set_joint_position_targets(cur_cmds); world.step(render=False)\n"
                "  for _ws in range(50):\n"
                "      robots.set_joint_position_targets(cur_cmds); world.step(render=False)\n"
            )
    else:
        stage_block = (
            "\n=== CURRENT OPTIMIZATION STAGE: TRANSPORT ===\n"
            "REACH AND GRASP ARE WORKING. *** DO NOT MODIFY PHASES 2 OR 3 IN ANY WAY. ***\n"
            "Phase 2 locked code is shown in the regression block above — copy it verbatim.\n"
            "Now focus on trajectory tracking.\n"
            "  1. PHASES 2 AND 3 ARE LOCKED — DO NOT CHANGE THEM.\n"
            "  2. Use grasp_gate = grasp_stable_per_env[ei] or (ee_to_obj_post3[ei] < 0.12).\n"
            "     Advance wp_idx when ||obj - wp|| < 0.05m for envs in grasp_gate.\n"
            "  3. ee_obj_offset sanity check MANDATORY (same as grasp stage — compute after Phase 2):\n"
            "       if np.linalg.norm(_raw) >= 0.08: ee_obj_offset[ei] = -_approach_dirs[ei] * _GRASP_STANDOFF  # SUBTRACT\n"
            "  4. Phase 4 transport target per env:\n"
            "       if grasp_gate[ei]: ee_tgts[ei] = WAYPOINTS_WORLD_BATCH[ei, wp_idx[ei]] + ee_obj_offset[ei]\n"
            "       else: ee_tgts[ei] = live_obj[ei] - _approach_dirs[ei] * (_GRASP_STANDOFF + _IK_EE_ADJ)  # SUBTRACT\n"
            "  5. Print kp_per_env, lift_per_env, best_env in every FRAME_STATE.\n"
            "  6. Score = EE stays within 0.10m of object (grasp_stability) + lift + kp_progress.\n"
        )
        if best_grasp_state:
            _bgs_env  = best_grasp_state.get("env_id", "?")
            _bgs_lift = best_grasp_state.get("max_lift", 0.0)
            _bgs_cp   = best_grasp_state.get("cp_err", -1.0)
            _bgs_all  = best_grasp_state.get("all_joints")
            if _bgs_all:
                # Split into arm (first 7) and hand (remaining) for readability
                _bgs_arm  = [round(float(v), 5) for v in _bgs_all[:7]]
                _bgs_hand = [round(float(v), 5) for v in _bgs_all[7:]]
                stage_block += (
                    f"\nBEST KNOWN GRASP JOINT STATE (env={_bgs_env}  max_lift={_bgs_lift:.4f}m"
                    f"  cp_err={_bgs_cp:.4f}):\n"
                    "  These are the joints at Phase 3 END from the best recorded grasp across all iterations.\n"
                    "  arm_joints (Actuator1-7): " + str(_bgs_arm) + "\n"
                    "  hand_joints (indices 7+): " + str(_bgs_hand) + "\n"
                    "\n"
                    "MECHANICAL SEED INSTRUCTION — use arm_joints as Phase 3 IK anchor seed:\n"
                    "  _BEST_GRASP_ARM = " + str(_bgs_arm) + "\n"
                    "  _BEST_GRASP_HAND = " + str(_bgs_hand) + "\n"
                    "  # At Phase 3 start, build the IK anchor seed from best-known arm config:\n"
                    "  grasp_anchor_cmds = np.tile(_init_joints_1d.copy(), (N_ENVS, 1))\n"
                    "  for _j_i, _jn in enumerate(ARM_JOINT_NAMES):\n"
                    "      grasp_anchor_cmds[:, _dof_idx[_jn]] = _BEST_GRASP_ARM[_j_i]\n"
                    "  # Each Phase 3 step: hold arm via IK from this seed, ramp hand joints only:\n"
                    "  grasp_anchor_cmds = solve_ik_batch_all_envs(\n"
                    "      obj_pos - _approach_dirs * (_GRASP_STANDOFF + _IK_EE_ADJ),  # SUBTRACT — live object positions\n"
                    "      grasp_anchor_cmds, closed_hand=False)\n"
                    "  # Then ramp fingers toward target (do NOT touch arm joints after this IK call)\n"
                    "WHY: seeding IK from _BEST_GRASP_ARM keeps all envs in the same elbow configuration\n"
                    "that was verified to produce a successful grasp.\n"
                )

    # ── Best env from BEST_ENV_SUMMARY (current run) ──────────────────────
    best_env_from_run = ""
    if best_env_summary:
        best_env_from_run = (
            f"\n=== BEST ENV THIS RUN (learn from this) ===\n"
            f"  best_reach_env={best_env_summary.get('best_reach_env')}  "
            f"reach_dist={best_env_summary.get('best_reach_dist', 'N/A'):.4f}m\n"
            f"  best_goal_env={best_env_summary.get('best_goal_env')}  "
            f"goal_kp={best_env_summary.get('best_goal_kp', 'N/A'):.4f}m\n"
            f"  grasp_stable_any={best_env_summary.get('grasp_stable_any', False)}\n"
            f"  notes={best_env_summary.get('notes', '')}\n"
            f"  best_approach_joints={best_env_summary.get('best_approach_arm_joints', 'N/A')}\n"
            f"\n"
            f"  KEY: arm joints are only useful as IK seeds — do NOT tile env-0's joints to all envs.\n"
            f"  The transferable info is the object-relative EE offset from the best env:\n"
            f"    best_ee_obj_offset = ee_pos[best_ei] - obj_pos[best_ei]\n"
            f"  Then for every env: ee_obj_offset[ei] = best_ee_obj_offset (same relative pose).\n"
        ) if isinstance(best_env_summary.get('best_reach_dist'), (int, float)) else (
            f"\n=== BEST ENV THIS RUN ===\n{best_env_summary}\n"
        )

    # ── Best per-env stats from best reach iteration ───────────────────────
    best_per_env_block = ""
    if best_per_env_stats:
        final_stats = [s for s in best_per_env_stats if s.get("phase") == "final"]
        if final_stats:
            rows = [f"  env{s['env_id']}: reach={s.get('min_ee_to_obj','?'):.3f}m  "
                    f"kp={s.get('min_goal_kp','?'):.3f}m  "
                    f"final_ee_to_obj={s.get('final_ee_to_obj','?') if isinstance(s.get('final_ee_to_obj'), (int, float)) else '?'}  "
                    f"grasp={s.get('grasp_stable',False)}  "
                    f"failed_at={s.get('phase_failed','?')}"
                    for s in final_stats if isinstance(s.get('min_ee_to_obj'), (int, float))]
            if rows:
                best_per_env_block = (
                    "\n=== PER-ENV STATS FROM BEST REACH ITERATION ===\n"
                    + "\n".join(rows) + "\n"
                )

    # Extract best-env transport progression from FRAME_STATE logs
    best_env_transport_summary = ""
    if best_env_idx is not None and frame_states:
        transport_fs = [fs for fs in frame_states if fs.get("phase") == "transport"]
        if transport_fs:
            best_kp_seq = []
            lift_seq = []
            for fs in transport_fs:
                kp_per = fs.get("kp_per_env", [])
                lift_per = fs.get("lift_per_env", [])
                if kp_per and best_env_idx < len(kp_per):
                    best_kp_seq.append(round(float(kp_per[best_env_idx]), 4))
                if lift_per and best_env_idx < len(lift_per):
                    lift_seq.append(round(float(lift_per[best_env_idx]), 4))
            if best_kp_seq:
                best_env_transport_summary = (
                    f"\nBEST ENV (env{best_env_idx}) transport trace:\n"
                    f"  kp_dist progression: {best_kp_seq[:8]} {'...' if len(best_kp_seq)>8 else ''}\n"
                    f"  lift progression:    {lift_seq[:8]} {'...' if len(lift_seq)>8 else ''}\n"
                    f"  final kp: {kp_all_list[best_env_idx]:.4f}m  (threshold={spec.success_tolerance}m)"
                )

    # Parse reach diagnostics
    reach_score  = _parse_float(tagged("REACH_SCORE"))
    rx_off       = _parse_float(tagged("REACH_X_OFFSET"))
    ry_off       = _parse_float(tagged("REACH_Y_OFFSET"))
    rz_off       = _parse_float(tagged("REACH_Z_OFFSET"))
    reach_too_high = (rz_off is not None and rz_off < -0.10)
    startup_dist = _parse_float(tagged("STARTUP_EE_OBJ_DIST"))
    reach_diverged = (
        reach_score is not None and startup_dist is not None
        and reach_score > startup_dist + 0.05
    )

    diag: list[str] = []
    for tag in (
        "OBJECT_EXTENTS", "GRASP_TYPE_SELECTED", "INITIAL_ARM_JOINTS",
        "NUM_WAYPOINTS", "GOAL_POS", "GOAL_ROT",
        "EE_LINK_USED", "BOILERPLATE_READY",
        "OBJECT_POSE_INITIAL_ENV0", "SETTLED_OBJ_POSE_ENV0",
        "ALIGN_ENV0", "ALIGN_ENV1",
        "STARTUP_EE_OBJ_DIST",
        "REACH_SCORE", "REACH_REL_POS",
        "REACH_X_OFFSET", "REACH_Y_OFFSET", "REACH_Z_OFFSET",
        "REACH_ARM_JOINTS", "GRASP_HOLD_ARM_JOINTS", "BEST_GRASP_STATE",
        "WRIST_LAT_CHECK",
        "CONTACT_PAIR_ERROR", "THUMB_CONTACT_ERROR", "INDEX_CONTACT_ERROR",
        "MIDDLE_CONTACT_ERROR", "PALM_CONTACT_ERROR", "WRIST_ORIENTATION_ERROR",
        "CP_LINKS_USED", "TRANSPORT_CONTACT_PAIR_ERROR",
        "FINAL_KP_MAX_DIST_MEAN", "FINAL_KP_MAX_DIST_MIN",
        "FINAL_EE_OBJ_DIST_BATCH",
        "BATCH_METRICS", "WAYPOINT_Z_SHIFT",
        "REACH_MAX_CF", "OBJ_TILT_AFTER_REACH",
    ):
        line = tagged(tag)
        if line:
            diag.append(line)

    # Inject obj_tilt from first transport frame_state so Claude sees it in DIAGNOSTICS
    if frame_states:
        first_transport = next((fs for fs in frame_states
                                if fs.get("phase") == "transport"), None)
        if first_transport and first_transport.get("obj_tilt") is not None:
            diag.append(f"OBJ_TILT_AT_TRANSPORT_START: {first_transport['obj_tilt']:.1f}°  "
                        f"(>45° means object was knocked over during reach/grasp)")

    # ── Detect object knocked over: obj_tilt > 45° already at transport step 0 ──
    # MUST be computed before _spatial_advice_block (which checks this flag).
    obj_knocked_over_at_start = False
    knocked_over_tilt = None
    # Primary: parse OBJ_TILT_AFTER_REACH from stdout (most reliable — always printed)
    _tilt_line = tagged("OBJ_TILT_AFTER_REACH")
    if _tilt_line:
        import re as _re
        _m = _re.search(r"([\d.]+)", _tilt_line)
        if _m:
            _t = float(_m.group(1))
            if _t > 45.0:
                obj_knocked_over_at_start = True
                knocked_over_tilt = _t
    # Fallback: check frame_states obj_tilt field
    if not obj_knocked_over_at_start and frame_states:
        first_transport = next((fs for fs in frame_states
                                if fs.get("phase") == "transport"), None)
        if first_transport:
            t = first_transport.get("obj_tilt")
            if t is not None and float(t) > 45.0:
                obj_knocked_over_at_start = True
                knocked_over_tilt = float(t)

    # ── Detect reach failure from FRAME_STATE ────────────────────────────────
    reach_failure_detected = False
    if frame_states:
        transport_fs = [fs for fs in frame_states if fs.get("phase") == "transport"]
        if transport_fs:
            all_zero_progress = all(
                all(float(p) < 1e-4 for p in fs.get("kp_progress", [1.0]))
                for fs in transport_fs
                if fs.get("kp_progress")
            )
            all_not_lifted = all(
                all(float(l) < 0.01 for l in fs.get("lift_per_env", [0.1]))
                for fs in transport_fs
                if fs.get("lift_per_env")
            )
            _never_reached = (reach_score is not None and reach_score > 0.20) or reach_diverged
            if all_zero_progress and all_not_lifted and _never_reached:
                reach_failure_detected = True

    # Per-env approach direction reminder (replaces the old single-direction spatial correction).
    # REACH_X/Y/Z_OFFSET from env-0 only is not a valid correction signal for all envs.
    _spatial_advice_block = ""
    if not obj_knocked_over_at_start and rx_off is not None and ry_off is not None and rz_off is not None:
        _spatial_advice_block = (
            f"\n=== REACH DIAGNOSTICS (env-0 only — informational) ===\n"
            f"REACH offsets env-0 (object_pos − ee_pos): X={rx_off:+.4f} Y={ry_off:+.4f} Z={rz_off:+.4f}\n"
            f"NOTE: Use APPROACH_DIRS_BATCH from the boilerplate — it is pre-computed per-env\n"
            f"(robot_base → object direction) and handles all env offsets automatically.\n"
            f"Standoff formula: OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (standoff + IK_EE_MOUNT_ADJ_M)\n"
            f"(SUBTRACT — targets go BEFORE the object, not past it).\n"
        )

    fix: list[str] = []

    # Determine if any env actually lifted in this run (from last_end_memory, which contains
    # the current run's END_MEMORY).  If so, obj_knocked_over is a tilt-formula false positive —
    # the arm clearly grasped and moved the object, so do NOT fire PRIORITY 0.
    _cur_any_lifted = bool(
        (last_end_memory or {}).get("any_env_lifted", False)
    )

    # Flag object knocked over FIRST — reach/grasp is too aggressive.
    # Skip if any env lifted: that proves the arm was holding (not tipping) the bottle,
    # so the tilt reading is a false positive from the absolute-tilt formula bug.
    if obj_knocked_over_at_start and not _cur_any_lifted:
        _knocked_phase2 = ""
        if _phase2_from_prev:
            _knocked_phase2 = (
                "\n  CRITICAL: DO NOT change Phase 2 structure. The approach direction IS correct.\n"
                "  The problem is the EE is going too fast or the standoff is too small.\n"
                "  If reach succeeded in the previous run (check regression block), the Phase 2\n"
                "  code shown above is LOCKED. Only adjust _GRASP_STANDOFF (try 0.10m) or\n"
                "  _MAX_STEP_2 (try 0.003m/step instead of 0.004m/step).\n"
                "  DO NOT add 'approach from above' — IK FAILS at Z+0.20m for ALL envs.\n"
            )
        fix.append(
            f"PRIORITY 0 — OBJECT KNOCKED OVER BEFORE TRANSPORT (obj_tilt={knocked_over_tilt:.1f}° at step 0):\n"
            "  The object was already tipped over when transport began.\n"
            "  This means the EE hit the object body during reach or grasp and tipped it.\n\n"
            "  ROOT CAUSE: The approach is reaching the bottle but hitting it too hard.\n"
            "  DO NOT change the approach structure. DO NOT use 'approach from above'.\n"
            "  ('Approach from above' requires IK at Z+0.20m which FAILS for ALL envs.)\n\n"
            "  FIXES — try in this order:\n"
            "  (1) INCREASE STANDOFF: Change _GRASP_STANDOFF from 0.07m to 0.10m.\n"
            "      This stops the EE 10cm from the bottle surface, reducing collision force.\n"
            "  (2) SLOWER APPROACH: Change _MAX_STEP_2 from 0.004m to 0.003m/step.\n"
            "  (3) CHECK IK POSITION TARGETS: Use solve_ik_batch_all_envs — never assign\n"
            "      arm joints directly (that jumps at full speed and knocks objects over).\n"
            "  (4) GRASP STANDOFF CHAIN: Use larger standoff in Phase 2 (0.12m) and Phase 3 (0.07m):\n"
            "      Phase 2 final position: OBJ_INIT_POS_BATCH - _approach_dirs*(0.12+IK_EE_MOUNT_ADJ_M)\n"
            "      Phase 3 anchor: obj_pos - _approach_dirs*(0.07+IK_EE_MOUNT_ADJ_M) (live object pos)\n"
            + _knocked_phase2
        )

    # Stuck / exploration override — fires when the same knock-over pattern repeats.
    # Suppressed when any env actually lifted the bottle (arm is working; tilt is a false positive).
    if stuck_counter >= 2 and not _cur_any_lifted:
        strategies = [
            "STRATEGY A — LARGER STANDOFF + SLOWER:\n"
            "  _GRASP_STANDOFF = 0.12   # was 0.07, now approach stops 12cm from bottle\n"
            "  _MAX_STEP_2 = 0.003      # slower, was 0.004 m/step\n"
            "  Keep APPROACH_DIRS_BATCH from boilerplate (do NOT recompute directions).\n"
            "  grasp_tgts = OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH * (_GRASP_STANDOFF + IK_EE_MOUNT_ADJ_M)",

            "STRATEGY B — SIDE with PRE-OPEN + SLOW:\n"
            "  Keep _REACH_STANDOFF = 0.15m (larger than usual). _MAX_STEP_2 = 0.002m.\n"
            "  Before moving, set all finger joints to 0 (full open) and hold 30 steps.\n"
            "  Open hand removes the risk of knuckles catching the object during approach.",

            "STRATEGY C — SLIGHTLY HIGHER APPROACH ANGLE:\n"
            "  Increase the Z component of the approach direction to tilt the EE higher:\n"
            "  _approach_dirs_mod = APPROACH_DIRS_BATCH.copy()\n"
            "  _approach_dirs_mod[:, 2] += 0.15   # more upward tilt\n"
            "  _approach_dirs_mod /= np.linalg.norm(_approach_dirs_mod, axis=1, keepdims=True) + 1e-8\n"
            "  grasp_tgts = OBJ_INIT_POS_BATCH - _approach_dirs_mod * (_GRASP_STANDOFF + IK_EE_MOUNT_ADJ_M)",
        ]
        chosen = strategies[min(stuck_counter - 2, len(strategies) - 1)]
        fix.insert(0,
            f"STUCK FOR {stuck_counter} CONSECUTIVE ITERATIONS\n"
            "  The object is knocked over EVERY iteration.\n"
            "  IMPORTANT: Do NOT use 'approach from above' (IK fails at Z+0.20m for ALL envs).\n"
            "  Instead, try increasing the standoff distance or slowing the approach.\n\n"
            f"  Try this strategy (stuck count={stuck_counter}):\n"
            f"  {chosen}\n\n"
            "  Also REQUIRED this iteration:\n"
            "  • Log contact forces: cf_mag = np.linalg.norm(get_object_contact_forces(), axis=1)\n"
            "    Print REACH_MAX_CF: {cf_mag.max():.2f}N\n"
            "  • Print OBJ_TILT_AFTER_REACH after Phase 2 completes.\n"
        )

    # Flag reach failure FIRST — arm never made it to the object
    if reach_failure_detected:
        fix.append(
            "PRIORITY 0 — APPROACH FAILURE (ARM NEVER REACHED THE OBJECT):\n"
            "  REACH_SCORE > 0.20 m (or reach diverged from startup distance).\n"
            "  DIAGNOSIS: The arm did not reach the object at all — this is NOT a pushing exploit.\n"
            "  kp_progress=0 and lift=0 because the EE was never close enough to contact the object.\n\n"
            "  COMMON CAUSES AND REQUIRED FIXES:\n"
            "  (1) CF ABORT THRESHOLD TOO LOW: If Phase 2 aborts on step 0 because contact force\n"
            "      exceeds the threshold (e.g., < 5 N), the arm never advances past the start position.\n"
            "      Fix: raise the Phase 2 CF abort threshold to ≥ 10 N, or remove the abort entirely.\n"
            "  (2) IK DIVERGENCE during approach: reduce step size to 0.002m/step.\n"
            "  (3) REACH TARGETS WRONG: Verify grasp_tgts = OBJ_INIT_POS_BATCH - APPROACH_DIRS_BATCH\n"
            "      * (_GRASP_STANDOFF + IK_EE_MOUNT_ADJ_M). SUBTRACT approach direction (never ADD).\n"
            "  (4) DO NOT use 'approach from above' — IK fails for all envs at Z+0.20m positions.\n"
            "      Use DIRECT INCREMENTAL APPROACH: _ee_now → grasp_tgts over 400 steps.\n"
        )

    # Contact-pair geometry fix (wrist orientation + finger placement)
    _cp_err_par   = _parse_float(tagged("CONTACT_PAIR_ERROR"))
    _wrist_err_par = _parse_float(tagged("WRIST_ORIENTATION_ERROR"))
    _wrist_lat_par = tagged("WRIST_LAT_CHECK")
    _wrist_sep_par = None
    if _wrist_lat_par:
        import re as _re3
        _mw = _re3.search(r"sep=([+-]?\d*\.?\d+)", _wrist_lat_par)
        if _mw:
            _wrist_sep_par = float(_mw.group(1))

    # Contact-pair geometry: only surface the raw diagnostic values; let Claude
    # interpret them from the DIAGNOSTICS block rather than generating verbose advice.
    if _wrist_err_par is not None and _wrist_err_par > 0.04:
        _sep_str = f"{_wrist_sep_par:.3f}" if _wrist_sep_par is not None else "N/A"
        fix.append(
            f"WRIST_ORIENTATION_ERROR={_wrist_err_par:.3f}m (sep={_sep_str}): "
            "thumb and index on same lateral side — adjust per-env approach direction or negate lateral offset."
        )
    elif _cp_err_par is not None and _cp_err_par > 0.08 and (reach_score is None or reach_score < 0.15):
        fix.append(
            f"CONTACT_PAIR_ERROR={_cp_err_par:.3f}m: fingers not straddling object. "
            "Adjust per-env approach direction (_approach_dirs) or reduce _GRASP_STANDOFF."
        )

    # Flag the single most common failure mode first
    if used_np_tile_ik and not used_solve_ik_for_env and not used_solve_ik_batch:
        fix.insert(0,
            "PRIORITY 0 — PER-ENV IK BUG: Previous code used np.tile(solve_ik_env0(...), (N_ENVS,1)). "
            "This broadcasts env-0's IK to ALL envs — every env executes the same arm motion "
            "regardless of its own object's position. This is why only 1 env (env-0) ever succeeds. "
            "FIX: Replace with solve_ik_batch_all_envs(tgts, cmds, closed_hand=False/True). "
            "Compute tgts as (N_ENVS,3) array (vectorised), then call once — no per-env loop needed. "
            "solve_ik_batch_all_envs is already defined in the boilerplate."
        )
    if reach_diverged:
        fix.insert(0,
            f"PRIORITY 0 — REACH DIVERGED: REACH_SCORE={reach_score:.3f}m is WORSE than "
            f"STARTUP_EE_OBJ_DIST={startup_dist:.3f}m. The arm moved AWAY from the object. "
            "Likely cause: approach direction is wrong. "
            "Fix: use APPROACH_DIRS_BATCH from boilerplate. Verify grasp_tgts = OBJ_INIT_POS_BATCH "
            "- APPROACH_DIRS_BATCH * (_GRASP_STANDOFF + IK_EE_MOUNT_ADJ_M) (SUBTRACT, not add). "
            "Use DIRECT INCREMENTAL APPROACH: start from home EE, move 0.004m/step toward grasp_tgts over 400 steps. "
            f"{_spatial_advice_block}"
        )
    # Detect if the reach is stuck far from the object despite multiple iterations
    reach_stuck = (
        reach_score is not None and reach_score > 0.12
        and best_reach_score < 900.0 and reach_score > best_reach_score - 0.01
    )

    if not obj_knocked_over_at_start and reach_too_high:
        fix.append(
            f"PRIORITY 1 — EE TOO HIGH ({abs(rz_off):.3f}m above object). "
            "Clip _approach_dirs[:, 2] further negative (e.g. -0.3→-0.4) to angle the approach downward. "
            f"{_spatial_advice_block}"
        )
    elif not obj_knocked_over_at_start and reach_stuck and not reach_diverged:
        fix.append(
            f"PRIORITY 1 — REACH STUCK at {reach_score:.3f}m (best was {best_reach_score:.3f}m). "
            "Adjust per-env approach directions and/or _GRASP_STANDOFF — do NOT hardcode arm joint arrays.\n"
            f"{_spatial_advice_block}"
        )
    elif not obj_knocked_over_at_start and reach_score is not None and reach_score > 0.15:
        fix.append(
            f"PRIORITY 1 — REACH FAILED (score={reach_score:.3f}m > 0.15m). "
            "Recheck per-env approach direction computation (see system prompt KEY CONSTRAINTS). "
            "Do NOT adjust individual Actuator values.\n"
            f"{_spatial_advice_block}"
        )

    if not fix and cur_ee_obj_dist_min is not None and cur_ee_obj_dist_min > 0.15:
        fix.append(
            f"PRIORITY 1 — ROBOT NOT HOLDING OBJECT AT END: final_ee_obj_dist_min={cur_ee_obj_dist_min:.4f}m "
            f"(mean={cur_ee_obj_dist_mean:.4f}m). The EE is far from the object after transport — "
            "the robot knocked/pushed the object rather than gripping and carrying it. "
            "This means the grasp failed even if kp_dist looks OK (the object may coincidentally "
            "be near the goal without being held). Fix:\n"
            "  (a) Improve grasp: ensure fingertips close around the object before transport.\n"
            "  (b) During Phase 4, verify obj moves with EE (trajectory_coupled=True in PER_ENV_STATS).\n"
            "  (c) Reduce approach speed so the object is not knocked over before grasping."
        )
    if not fix and cur_min_kp is not None and cur_min_kp > spec.success_tolerance:
        fix.append(
            f"PRIORITY 1 — KEYPOINT MISS: best min kp dist = {cur_min_kp:.4f}m "
            f"(need < {spec.success_tolerance}m). "
            "Ensure Phase 4 actually commands the last WAYPOINTS_WORLD waypoint. "
            "Add a Phase 5 HOLD so physics settles before the final check."
        )
    if not fix:
        fix.append(
            "Continue refining reach target joints. "
            f"Best batch success rate so far: {cur_success_rate:.1%}."
        )

    reach_regression = ""
    if (not obj_knocked_over_at_start
            and reach_score is not None
            and best_reach_score < 900.0
            and reach_score > best_reach_score + 0.05):
        reach_regression = (
            f"\n=== REACH OSCILLATION ===\n"
            f"This iter REACH_SCORE={reach_score:.3f}m > best {best_reach_score:.3f}m.\n"
            "You are oscillating. Revert per-env approach directions toward what gave the "
            f"best reach ({best_reach_score:.3f}m). Make only ONE small change to the direction "
            "computation or _GRASP_STANDOFF. Do NOT hardcode arm joint arrays.\n"
            f"{_spatial_advice_block}"
        )

    stderr_clean = "\n".join(
        l for l in stderr.splitlines()
        if l.strip() and "Warp CUDA error" not in l
    )
    stdout_tail = "\n".join(useful[-40:]) if useful else "(none)"
    has_error = any(
        ("Error" in l or "Traceback" in l or "Exception" in l)
        for l in stderr.splitlines() if "Warp" not in l
    )
    verdict = "crashed" if has_error else "ran but did not succeed"

    task_desc_section = (
        f"\n=== TASK ===\n{spec.task_description}\n"
        if spec.task_description else ""
    )

    cross_run_section = _format_cross_iter_stats(aggregate_stats, best_iter_stats, best_env_stats)

    _cur_reach_fail   = batch_metrics.get("reach_failure_rate") if batch_metrics else None
    _cur_drop         = batch_metrics.get("drop_rate")          if batch_metrics else None
    _dense_score_mean = batch_metrics.get("dense_score_mean")   if batch_metrics else None
    _dense_score_best = batch_metrics.get("dense_score_best")   if batch_metrics else None
    _score_kp         = batch_metrics.get("score_kp_progress")  if batch_metrics else None
    _score_lift       = batch_metrics.get("score_lift")          if batch_metrics else None
    _score_grasp      = batch_metrics.get("score_grasp")         if batch_metrics else None
    _score_tilt_pen   = batch_metrics.get("score_tilt_pen")      if batch_metrics else None
    _score_hvc_pen    = batch_metrics.get("score_hvc_pen")       if batch_metrics else None

    # ── Build SCORE CARD block ────────────────────────────────────────────────
    score_card_block = ""
    if _dense_score_mean is not None:
        # Identify the biggest drag (most negative contributor)
        penalties = {
            "score_tilt_pen": (_score_tilt_pen or 0.0, "reduce approach aggression / lower standoff to prevent knock-over", "score_tilt_pen"),
            "score_hvc_pen":  (_score_hvc_pen  or 0.0, "slow the approach (smaller IK step) — EE moving too fast near object", "score_hvc_pen"),
        }
        positives = {
            "score_kp_progress": (_score_kp    or 0.0, "improve trajectory tracking / waypoint progression"),
            "score_lift":        (_score_lift   or 0.0, "improve grasp — object is not being lifted"),
            "score_grasp":       (_score_grasp  or 0.0, "improve proximity + coupling during transport"),
        }
        # biggest penalty drag is the most negative (raw value is already negative contribution)
        worst_pen_key, (worst_pen_val, worst_pen_hint, _) = min(
            penalties.items(), key=lambda kv: kv[1][0]
        )
        # smallest positive contributor
        worst_pos_key, (worst_pos_val, worst_pos_hint) = min(
            positives.items(), key=lambda kv: kv[1][0]
        )

        def _fmt(v):
            return f"{v:+.3f}" if v is not None else "N/A"

        score_lines = [
            f"\n=== SCORE CARD (dense reward breakdown, mean across {n_envs} envs) ===",
            f"  TOTAL dense_score_mean : {_fmt(_dense_score_mean)}   best env: {_fmt(_dense_score_best)}",
            f"",
            f"  POSITIVE TERMS (want HIGH)",
            f"    kp_progress  (+2×) : {_fmt(_score_kp)}",
            f"    lift_progress(+3×) : {_fmt(_score_lift)}",
            f"    grasp_stab   (+5×) : {_fmt(_score_grasp)}",
            f"",
            f"  PENALTY TERMS  (want ZERO)",
            f"    tilt_penalty (-2×) : {_fmt(_score_tilt_pen)}",
            f"    high_vel_con (-3×) : {_fmt(_score_hvc_pen)}",
            f"",
        ]
        if worst_pen_val < -0.5:
            score_lines.append(
                f"  BIGGEST DRAG → {worst_pen_key}={_fmt(worst_pen_val)}: {worst_pen_hint}"
            )
        elif worst_pos_val < 1.0:
            score_lines.append(
                f"  BIGGEST OPPORTUNITY → {worst_pos_key}={_fmt(worst_pos_val)}: {worst_pos_hint}"
            )
        score_card_block = "\n".join(score_lines) + "\n"

    return f"""\
Iteration {iteration} {verdict}.
{task_desc_section}{cross_run_section}{_iter_delta_block}
=== SUCCESS CRITERION ===
PRIMARY (most important): GRASP the object — trajectory_coupled_any=True AND any_env_lifted=True.
  A successful grasp requires: EE within 12 cm of object after Phase 3 AND object lifts > 2 cm in Phase 4.
  If grasp fails, nothing else matters. Fix grasp before transport.
SECONDARY: trajectory coupling — follow WAYPOINTS_WORLD_BATCH after grasping (minimize traj_pos_err_mean).
Optimization rank: (1) reach object (REACH_SCORE < 0.10m) → (2) stable grasp + lift > 2cm → (3) trajectory tracking while coupled → (4) final kp_dist.
kp_dist improvements where trajectory_coupled=False are PUSHING EXPLOITS — they do NOT count as progress.
{_traj_block}
This iter:
  success_count={cur_success_count}/{n_envs}  success_rate={cur_success_rate:.1%}
  mean_kp_dist={cur_mean_kp}  min_kp_dist={cur_min_kp}
  reach_failure_rate={_cur_reach_fail}  (fraction envs where reach dist > 0.12m)
  drop_rate={_cur_drop}  (fraction envs where final_ee_obj_dist > 0.15m — object dropped)
  final_ee_obj_dist_min={cur_ee_obj_dist_min}  final_ee_obj_dist_mean={cur_ee_obj_dist_mean}
  dense_score_mean={_dense_score_mean}  dense_score_best={_dense_score_best}
  (dense = 2×kp_progress_gated + 3×lift_progress + 5×grasp_stability − 2×tilt − 3×high_vel_contact)
{score_card_block}
Best min kp dist across all iterations = {best_min_kp_dist:.4f} m
Best reach score = {best_reach_score:.4f} m
Best final_ee_obj_dist across all iterations = {best_final_ee_obj_dist:.4f} m  (lower = robot held object)
{regression_block}
{stage_block}
{best_env_from_run}
{best_per_env_block}
{reach_regression}
{best_runs_summary}
{per_env_kp_summary}
{best_env_transport_summary}
{_spatial_advice_block}

=== DIAGNOSTICS ===
{chr(10).join(diag) if diag else "(no tagged output — script likely crashed early)"}

=== PRIORITY FIX ===
{chr(10).join(fix)}

=== STDOUT TAIL ===
{stdout_tail}

=== STDERR ===
{stderr_clean[:1500] if stderr_clean.strip() else "(none)"}

=== INVARIANTS ===
- {n_envs} parallel envs. Object init poses are RANDOMISED per env (±5 cm XY).
- world.step(render=False) auto-renders via wrapper.
- CONTROL POLICY (non-negotiable):
    P2 REACH: IK only, speed-limited (0.005m/step), NO direct joint assignment, NO finger closure.
    P3 GRASP: Freeze arm via IK; DIRECT finger joints ramped in two stages. Verify grasp_stable.
    P4 TRANSPORT: IK speed-limited (0.004m/step) + locked SELECTED_GRASP_TARGET every step.
- CONTACT VALIDITY: Only kp_dist reductions WITH control_score > 0.3 count. Others are exploits.
- CRITICAL — PER-ENV IK: NEVER use np.tile(solve_ik_env0(...), (N_ENVS,1)).
  This broadcasts env-0's solution to all envs. Use solve_ik_batch_all_envs(tgts, cmds, closed_hand=...).
- TRANSPORT OFFSET: ee_tgt = WAYPOINTS_WORLD_BATCH[ei,wp] + ee_obj_offset[ei].
  Never set ee_tgt = waypoint directly — that drives hand into object centre.
- GRASP TARGET: use SELECTED_GRASP_TARGET (auto-set in boilerplate), NOT hardcoded HAND_CLOSE_TARGET.
  Re-enforce SELECTED_GRASP_TARGET every Phase 4 step (IK may relax fingers).
- HAND CLOSING: two-stage — ramp proximal joints (0–74 steps), then distal (75–149 steps).
  Never close all joints at once. thumb_rota_joint1 MUST reach ≥ 0.8 rad before Phase 4.
- Final output MUST include BATCH_METRICS json (with reach_failure_rate, drop_rate) and "SUCCESS"/"FAILURE".
- BATCH_METRICS must track: success_rate, mean_kp_dist, reach_failure_rate, drop_rate.
- TRAJECTORY TRACKING METRICS (required in BATCH_METRICS every iteration):
    traj_pos_err_mean, traj_pos_err_min, traj_pos_err_final  (meters)
    traj_rot_err_mean, traj_rot_err_final                    (radians, geodesic)
    traj_kp_err_mean, traj_kp_err_final                      (keypoint distances)
    best_traj_env (int), trajectory_following_score (float, higher=better)
  Compute these during Phase 4 by comparing obj_pos/rot to WAYPOINTS_WORLD_BATCH/WAYPOINTS_ROT_BATCH.
- POSE_TRACE: print every 30-50 steps, ONLY for best_traj_env and env0 (not all envs).
  Do NOT print huge arrays every step. POSE_TRACE replaces verbose per-step logging.
- FRAME_STATE must include kp_per_env, lift_per_env, best_env, kp_progress, coupled_envs,
  control_score_env0, obj_vel_env0, control_gate_active so the pipeline can detect pushing exploit.
- JSON SAFETY: NEVER pass bare numpy scalar indexing (arr[i,j]) to json.dumps.
  Use float(arr[i,j]) or arr[i].tolist() or json_safe({{...}}).
- QUATERNION SAFETY: objects.get_world_poses() returns (positions, quats_xyzw) where
  positions is shape (N,3) and quats_xyzw is shape (N,4) [x,y,z,w]. ALWAYS convert:
    quats_wxyz = np.column_stack([quats_xyzw[:,3], quats_xyzw[:,:3]])
  before passing to compute_keypoints_batch or keypoint_max_dist_batch.
  CRITICAL: positions is shape (N,3) — it has NO 4th column. NEVER index positions[:,3]
  (crash) and NEVER try to extract quaternions from positions. To get quats, always use
  the second return value of get_world_poses().
- PER-ENV ALIGNMENT: WAYPOINTS_WORLD_BATCH is already anchored to settled poses. Do NOT recompute.
- Phase 1 is DIAGNOSTIC only. Do NOT run additional settle steps.
- Phase 2 uses DIRECT INCREMENTAL APPROACH (MotionGen is unavailable in this pipeline):
    Do NOT use "approach from above" (IK to above_tgts at Z+0.20 then linear interp) —
    IK fails for all envs for high-Z targets; above_cmds = home joints; nothing happens.
    Instead: read ACTUAL current EE positions (home after settle), then move incrementally
    toward grasp_tgts at 0.004m/step over 400 steps.
    Phase 3 uses solve_ik_batch_all_envs each step to track live object positions.
    Phase 4 uses solve_ik_batch_all_envs each step for transport.
  NEVER use servo=True — it is deprecated and removed.
- IK FAILURE HANDLING:
  Phase 2 starts from ACTUAL EE positions (home):
    _ee_now = get_ee_pos_all_envs(stage, N_ENVS)
    _reach_pos = np.array([
        _ee_now[ei] if _ee_now[ei] is not None else grasp_tgts[ei]
        for ei in range(N_ENVS)], dtype=np.float64)
  IK failure in Phase 2: if solve_ik_batch_all_envs returns the seed for an env (no change),
  the arm stays put for that step. This is acceptable — failed envs catch up on later steps.
  Track per-env reach success: reach_ok[ei] = (min_ee_per_env[ei] < 0.15)
  For Phase 3 grasp, still run for all envs but track which ones are unreachable.
- Phase 4 advances waypoints per-env based on object proximity, not fixed time index.
- TRANSPORT GATE: advance wp_idx[ei] and target WAYPOINTS when ee_to_obj_post3[ei] < 0.12m.
  grasp_gate[ei] = grasp_stable_per_env[ei] or (ee_to_obj_post3[ei] < 0.12)
  Rationale: if EE is within 12cm after finger closure, the arm is in contact — follow trajectory.
  Do NOT require max_lift > 0.01m in this gate — the arm rarely lifts during Phase 3 hold steps.
- LIFT REQUIREMENT: A valid solution MUST show lift_per_env > 0.02m in at least one env.
  kp_progress via pushing (lift ≤ 0, EE far from object) is a reward exploit, NOT success.
- DENSE REWARD (required every iteration): Initialise before ANY phase loop:
    dense_score_per_env = np.zeros(N_ENVS)
    prev_best_kp        = np.full(N_ENVS, 999.0)
    prev_ee_pos_all     = np.zeros((N_ENVS, 3))
    # Relative-tilt reference: object's local-Z direction in world frame at SETTLE time.
    # Using OBJ_INIT_ROT_BATCH avoids the absolute-tilt bug for meshes whose local Z ≠ world Z.
    _init_up_world = np.array([_qrotate(OBJ_INIT_ROT_BATCH[i], [0.0, 0.0, 1.0]) for i in range(N_ENVS)])
  Inside EVERY step of EVERY phase, compute and accumulate reward_i for each env:
    ee_all  = get_ee_pos_all_envs(stage, N_ENVS)
    ee_to_obj = np.array([np.linalg.norm(ee_all[i] - obj_pos[i]) if ee_all[i] is not None else 0.5 for i in range(N_ENVS)])
    ee_speed  = np.array([np.linalg.norm((ee_all[i] if ee_all[i] is not None else prev_ee_pos_all[i]) - prev_ee_pos_all[i]) / 0.005 for i in range(N_ENVS)])
    proximity   = np.clip(1.0 - ee_to_obj / 0.15, 0.0, 1.0)
    coupling    = np.clip(1.0 - np.maximum(np.abs(obj_vel_mags - ee_speed), 0.0) / 0.5, 0.0, 1.0)
    control_score = 0.6 * proximity + 0.4 * coupling
    grasp_stability = proximity * coupling
    kp_progress_raw = np.maximum(prev_best_kp - kp_dists, 0.0)
    kp_progress_gated = kp_progress_raw * control_score
    lift_progress = np.maximum(obj_pos[:, 2] - OBJ_INIT_POS_BATCH[:, 2], 0.0)
    # Relative tilt: deviation from initial settled orientation (0=upright, 90=on side, 180=flipped).
    # Do NOT use arccos(up[:,2]) — that assumes local Z = world-up, which is false for many meshes.
    _curr_up_w = np.array([_qrotate(quats_wxyz[i], [0.0, 0.0, 1.0]) for i in range(N_ENVS)])
    tilt_degs = np.degrees(np.arccos(np.clip(np.sum(_init_up_world * _curr_up_w, axis=1), -1.0, 1.0)))
    tilt_penalty = np.maximum(tilt_degs - 10.0, 0.0) / 90.0
    contact_zone = np.clip(1.0 - ee_to_obj / 0.20, 0.0, 1.0)
    uncontrolled_obj_vel = obj_vel_mags * np.maximum(0.0, 1.0 - control_score)
    high_vel_contact = np.maximum(
        np.maximum(ee_speed - 0.05, 0.0) * contact_zone,   # fast approach
        uncontrolled_obj_vel * contact_zone                  # post-impact knock
    )
    reward_step = (2.0 * kp_progress_gated + 3.0 * lift_progress + 5.0 * grasp_stability
                   - 2.0 * tilt_penalty - 3.0 * high_vel_contact)
    dense_score_per_env += reward_step
    prev_best_kp = np.minimum(prev_best_kp, kp_dists)
    for i in range(N_ENVS):
        if ee_all[i] is not None: prev_ee_pos_all[i] = ee_all[i]
  Include in every FRAME_STATE: dense_score_env0, dense_score_mean, lift_progress_env0,
    grasp_stability_env0, tilt_penalty_env0, high_vel_contact_env0.
- CONTACT FORCE MONITORING (required every iteration): During Phase 2 (reach + approach):
    max_cf_reach = np.zeros(N_ENVS)
    # ... inside the reach loop:
    cf_mag = np.linalg.norm(get_object_contact_forces(), axis=1)
    max_cf_reach = np.maximum(max_cf_reach, cf_mag)
  After Phase 2: print(f"REACH_MAX_CF: {{max_cf_reach.max():.2f}}N (env {{max_cf_reach.argmax()}})")
  If REACH_MAX_CF > 5N OR high_vel_contact_env0 > 0.1, the EE is hitting the object.
  Fix: reduce approach speed (smaller IK step size) or increase standoff distance.
- TILT CHECK (required every iteration): After Phase 2, before Phase 3:
    positions, quats_xyzw_t = objects.get_world_poses()
    quats_wxyz_t = np.column_stack([quats_xyzw_t[:,3], quats_xyzw_t[:,:3]])
    # Relative tilt vs settled orientation — do NOT use arccos(up[:,2]) which is wrong for
    # meshes whose local Z ≠ world-up when upright (e.g. bottle USD with 180° Y spawn rotation).
    _curr_up_t = np.array([_qrotate(quats_wxyz_t[i], [0.0, 0.0, 1.0]) for i in range(N_ENVS)])
    tilts = np.degrees(np.arccos(np.clip(np.sum(_init_up_world * _curr_up_t, axis=1), -1.0, 1.0)))
  print(f"OBJ_TILT_AFTER_REACH: {{tilts.mean():.1f}}° mean, {{tilts.max():.1f}}° max")
  If mean tilt > 20°, approach is knocking object over (increase standoff or reduce speed).

--- PREVIOUS CODE ---
Your previous full script is in your last assistant message above. Apply the PRIORITY FIX to that motion section.

Return ONLY the motion section — everything from and including the line:
  # === MOTION CONTROL CODE ===
Do NOT repeat the boilerplate. No markdown fences, no prose, no explanations.
"""
