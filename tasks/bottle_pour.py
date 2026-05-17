"""Task spec defaults for bottle_pour.

Task: Grasp bottle, lift it, tilt ~80° to simulate pouring.
Object: bottle twin (SAM-reconstructed).
Trajectory: recorded from cylinder_pour (pour motion: lift + wrist tilt).
"""

from pathlib import Path

_REPO = Path(__file__).parent.parent
_ROBOT_USD = "/mnt/storage/DDDC/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd"

DEFAULTS = {
    "robot_usd":            _ROBOT_USD,
    "saved_poses":           str(_REPO / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_REPO / "assets/objects/bottle/bottle_twin/bottle_twin.usd"),
    "trajectory_json":       str(_REPO / "trajectories/bottle_pour/20260507_122208.json"),
    "object_scale":          1.0,
    "object_mass":           0.3,
    "success_tolerance":     0.1,
    # 180° rotation around Z so the USD mesh (baked cap-up) stands upright.
    "object_orientation":    (0.0, 0.0, 1.0, 0.0),
    # Small drop so the bottle falls and settles onto the table surface.
    "object_z_settle_offset": 0.04,
    "task_description": (
        "Object: upright plastic bottle, cylindrical (~0.07 m diameter, ~0.22 m tall). "
        "Goal: grasp the bottle with a power grasp, lift it, then tilt it ~45° following "
        "the recorded pouring trajectory (wrist-tilt motion), ending with all keypoints "
        "within SUCCESS_TOL of GOAL_KEYPOINTS."
    ),
}
