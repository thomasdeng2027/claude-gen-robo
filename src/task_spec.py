"""KeypointTaskSpec — config carrier for the keypoint-matching pipeline.

Final goal: after the whole rollout, the object's 4 bbox-corner keypoints must
match the goal keypoints (computed from the last waypoint of the trajectory,
aligned to this run's object init pose) within `success_tolerance` metres.

Changes vs keypoint_pipeline/task_spec.py:
  - from_saved_poses handles any object name (not just "object")
  - n_envs / env_spacing fields for parallel Isaac Sim environments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional



def _to_list(x: Any) -> list[float]:
    if hasattr(x, "tolist"):
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]


@dataclass
class KeypointTaskSpec:
    # --- Identity ----------------------------------------------------------
    task_name: str = ""           # set to the task registry key (e.g. "bottle_lift")

    # --- Assets ------------------------------------------------------------
    robot_usd: str = ""  # set via task DEFAULTS
    robot_urdf: str = (          # absolute URDF path for parallel Isaac Sim boilerplate
        "/juno/u/jingyuny/projects/p_vla/claude-data-gen"
        "/assets/kinova_xhand/urdf"
        "/GEN3_URDF_V12_with_hand_right.urdf"
    )
    object_usd: str = ""
    object_scale: float = 1.0
    object_mass: float = 0.1

    # --- Parallel environments --------------------------------------------
    n_envs: int = 1          # >1 → use parallel Isaac Sim boilerplate
    env_spacing: float = 2.0  # metres between env origins in the grid

    # --- Robot init --------------------------------------------------------
    robot_name: str = "gen3_xhand_right"
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    robot_initial_joints: dict[str, float] = field(default_factory=dict)

    # --- Object init -------------------------------------------------------
    object_name: str = "object"
    object_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
    object_orientation: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

    # --- Trajectory --------------------------------------------------------
    trajectory_path: str = ""
    traj_pause_threshold: float = 0.001

    # --- Success criterion -------------------------------------------------
    success_tolerance: float = 0.05

    # --- Phase budgets (simulation steps at dt*decimation) ----------------
    t_settle: int = 60
    t_approach: int = 300
    t_grasp: int = 150
    t_transport: int = 500
    t_hold: int = 60

    # --- Camera ------------------------------------------------------------
    camera_position: tuple[float, float, float] = (0.0, -1.2, 0.8)
    camera_target: tuple[float, float, float] = (0.0, 0.3, 0.15)

    # --- Video -------------------------------------------------------------
    video_fps: int = 30
    video_frames_dir: str = "/tmp/keypoint_frames"

    # --- Object settle offset (parallel mode only) -------------------------
    # Extra Z metres added to initial object spawn height so the object
    # falls and settles to the surface.  0.0 for objects whose saved-pose
    # z already places them on (or just above) the table.  Set to ~0.04 for
    # tall objects like bottles/drills that need a small drop to contact.
    object_z_settle_offset: float = 0.0

    # --- Standoff override (overrides _APPROACH_STANDOFF in boilerplate) ---
    approach_standoff: Optional[float] = None

    # --- Task description (injected into Claude prompts) ------------------
    task_description: str = ""

    # --- Object trajectory demonstration (Format B, optional) ------------
    # Path to a JSON file saved by TrajectoryLogger (flat list of
    # {"pos":[x,y,z], "rot":[w,x,y,z]} dicts).  When set, the prompt
    # builder injects a Format-B trajectory-conditioned block alongside
    # the task description.  Leave empty ("") for Format-A (EE-goal only).
    object_trajectory_demo_path: str = ""

    # --- Output formatting instruction appended to every Claude prompt ----
    output_format_instruction: str = (
        "Return ONLY the complete Python script (boilerplate + motion code). "
        "No markdown fences, no prose, no explanations."
    )

    # ----------------------------------------------------------------------
    @classmethod
    def from_saved_poses(
        cls,
        saved_poses_path: str,
        *,
        object_usd: str,
        trajectory_path: str = "",
        object_scale: float = 1.0,
        robot_usd: str = "",
        **overrides,
    ) -> "KeypointTaskSpec":
        """Load robot + object init state from a .py produced by object_layout.py.

        Handles saved_poses files where the object key is any name (e.g. "bottle",
        "drill"), not just the literal key "object".
        """
        import importlib.util

        p = Path(saved_poses_path)
        if not p.exists():
            raise FileNotFoundError(f"saved_poses not found: {p}")
        mod_spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if mod_spec is None or mod_spec.loader is None:
            raise RuntimeError(f"could not load module from {p}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)

        poses = getattr(mod, "poses", None)
        if poses is None:
            raise AttributeError(f"{p} has no `poses` dict")

        # --- Find the primary object entry ---------------------------------
        # Try canonical key "object" first; fall back to first non-trajectory
        # key (trajectory markers contain "_traj_" in their name).
        objects_dict = poses.get("objects", {})
        obj_entry = objects_dict.get("object")
        obj_name = "object"
        if obj_entry is None:
            for k, v in objects_dict.items():
                if "_traj_" not in k:
                    obj_entry = v
                    obj_name = k
                    break
        if obj_entry is None:
            raise KeyError(
                f"{p} poses['objects'] has no primary object "
                f"(tried 'object' and all non-trajectory keys; found: {list(objects_dict.keys())})"
            )

        robots = poses.get("robots", {})
        if not robots:
            raise KeyError(f"{p} poses['robots'] is empty")
        robot_name = next(iter(robots.keys()))
        robot_entry = robots[robot_name]

        if not robot_usd:
            raise ValueError("robot_usd must be set in task DEFAULTS")
        robot_usd = str(Path(robot_usd).resolve())

        kwargs: dict[str, Any] = {
            "robot_usd": robot_usd,
            "object_usd": object_usd,
            "object_scale": float(object_scale),
            "trajectory_path": trajectory_path,
            "robot_name": robot_name,
            "object_name": obj_name,
            "robot_position": tuple(_to_list(robot_entry["pos"])),
            "robot_orientation": tuple(_to_list(robot_entry["rot"])),
            "robot_initial_joints": {
                k: float(v.item() if hasattr(v, "item") else v)
                for k, v in (robot_entry.get("dof_pos") or {}).items()
            },
            "object_position": tuple(_to_list(obj_entry["pos"])),
            "object_orientation": tuple(_to_list(obj_entry["rot"])),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
