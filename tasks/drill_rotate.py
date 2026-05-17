"""Task spec defaults for drill_rotate.

Task: Grasp drill, lift it, rotate in-hand along recorded trajectory.
Object: drill twin (SAM-reconstructed).
Trajectory: record locally and set trajectory_json before running.
"""

from pathlib import Path

_REPO = Path(__file__).parent.parent
_ROBOT_USD = "/mnt/storage/DDDC/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd"

DEFAULTS = {
    "robot_usd":            _ROBOT_USD,
    "saved_poses":           str(_REPO / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_REPO / "assets/objects/drill_0423/twin/twin.usd"),
    "trajectory_json":       "",   # TODO: record drill rotation trajectory
    "object_scale":          1.0,
    "object_mass":           0.4,
    "success_tolerance":     0.05,
    "task_description": (
        "Grasp the power drill by its handle with a power grasp, lift it, then carry "
        "it through a recorded in-hand rotation trajectory (wrist-dominated motion). "
        "Goal: match GOAL_KEYPOINTS at the trajectory end pose."
    ),
}
