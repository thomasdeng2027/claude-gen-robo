"""Task spec defaults for pour_water.

Task: Grasp cup/bottle, lift it, tilt ~60° to simulate pouring.
Object: pourwater twin (SAM-reconstructed).
Trajectory: record locally and set trajectory_json before running.
"""

from pathlib import Path

_REPO = Path(__file__).parent.parent
_ROBOT_USD = "/mnt/storage/DDDC/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd"

DEFAULTS = {
    "robot_usd":             _ROBOT_USD,
    "saved_poses":            str(_REPO / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":             str(_REPO / "assets/objects/pourwater/twin/twin.usd"),
    "trajectory_json":        "",   # TODO: record pourwater tilting trajectory
    "object_scale":           1.0,
    "object_mass":            0.3,
    "success_tolerance":      0.05,
    "task_description": (
        "Grasp the cup or bottle (cylindrical) with a power grasp, lift it, then tilt "
        "it ~60° following the recorded pouring trajectory (wrist-tilt motion). "
        "Goal: match GOAL_KEYPOINTS at the tilted end pose."
    ),
}
