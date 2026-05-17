"""Task spec defaults for flip_bottle.

Task: Grasp bottle, lift it, flip 180° so it stands on its cap.
Object: bottle twin (SAM-reconstructed).
Trajectory: no trajectory — Claude generates the primitive flip skill.
"""

from pathlib import Path

_REPO = Path(__file__).parent.parent
_ROBOT_USD = "/mnt/storage/DDDC/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd"

DEFAULTS = {
    "robot_usd":            _ROBOT_USD,
    "saved_poses":           str(_REPO / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_REPO / "assets/objects/bottle/bottle_twin/bottle_twin.usd"),
    "trajectory_json":       "",   # no trajectory — Claude generates primitive flip skill
    "object_scale":          1.0,
    "object_mass":           0.3,
    "success_tolerance":     0.05,
    # 180° rotation around Z so the USD mesh (baked cap-up) stands upright.
    "object_orientation":    (0.0, 0.0, 1.0, 0.0),
    "object_z_settle_offset": 0.04,
    "task_description": (
        "Object: plastic bottle, cylindrical (~0.07 m diameter, ~0.22 m tall), upright. "
        "Goal: grasp the bottle, flip it ~180° (cap-up → cap-down), and end with "
        "all keypoints within SUCCESS_TOL of GOAL_KEYPOINTS at the inverted pose."
    ),
}
