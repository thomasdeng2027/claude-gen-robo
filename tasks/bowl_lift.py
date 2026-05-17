"""Task spec defaults for bowl_lift.

Task: Grasp bowl, lift >= 10 cm from initial Z, transport along recorded trajectory.
Object: bowl twin (SAM-reconstructed).
Trajectory: record locally and set trajectory_json before running.
"""

from pathlib import Path

_REPO = Path(__file__).parent.parent
_ROBOT_USD = "/mnt/storage/DDDC/Dynamic-Dexterous-Digital-Cousin-Benchmark/realkinova_xhand/realkinova_xhand_right.usd"

DEFAULTS = {
    "robot_usd":            _ROBOT_USD,
    "saved_poses":           str(_REPO / "tasks/saved_poses_20260414_002010.py"),
    "object_usd":            str(_REPO / "assets/objects/bowl/twin/twin.usd"),
    "trajectory_json":       "",   # TODO: record bowl lifting trajectory
    "object_scale":          1.0,
    "object_mass":           0.2,
    "success_tolerance":     0.05,
    # Identity: check orientation in smoke test and adjust if bowl appears upside down.
    "object_orientation":    (1.0, 0.0, 0.0, 0.0),
    "task_description": (
        "Object: ceramic bowl, wide and shallow (~0.16 m diameter, ~0.06 m tall). "
        "Goal: grasp the bowl and transport it along the recorded lifting trajectory, "
        "ending with all keypoints within SUCCESS_TOL of GOAL_KEYPOINTS."
    ),
}
