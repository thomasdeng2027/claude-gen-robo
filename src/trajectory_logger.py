"""trajectory_logger.py — Record, save, and format object trajectories.

Each in-context example can optionally record the object state over time and
save it to  trajectories/<task_name>/YYYYMMDD_HHMMSS.json  in the same flat-list
format that trajectory.py's load_trajectory() already understands:

    [ {"pos": [x, y, z], "rot": [w, x, y, z]}, ... ]

Saved files can later be loaded and embedded into Claude prompts as Format-B
trajectory-conditioned in-context examples.

Usage inside an example script:

    from trajectory_logger import TrajectoryLogger

    logger = TrajectoryLogger("cup_lift")

    # inside the sim loop or at log_every checkpoints:
    positions, quats_xyzw = objects.get_world_poses()
    logger.record(positions[0], quats_xyzw[0])   # Isaac Sim returns [x,y,z,w]

    # after execution finishes:
    saved_path = logger.save()
    print(f"TRAJECTORY_SAVED: {saved_path}  ({len(logger)} frames)")

Usage in the prompt pipeline (prompts.py):

    from trajectory_logger import load_for_prompt, format_for_prompt
    frames = load_for_prompt("trajectories/cup_lift/20260507_123456.json")
    block  = format_for_prompt(frames, max_frames=30)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Trajectories directory: <repo_root>/trajectories/
_TRAJ_ROOT = Path(__file__).parent.parent / "trajectories"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float_list(arr) -> list[float]:
    if hasattr(arr, "tolist"):
        return [float(v) for v in arr.tolist()]
    return [float(v) for v in arr]


def _xyzw_to_wxyz(xyzw: list[float]) -> list[float]:
    """Isaac Sim returns quaternions as [x, y, z, w]; convert to [w, x, y, z]."""
    return [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]


def _downsample(frames: list[dict], max_frames: int) -> list[dict]:
    n = len(frames)
    if n <= max_frames:
        return list(frames)
    step = n / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


# ---------------------------------------------------------------------------
# TrajectoryLogger
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """Accumulates per-step object state and saves to disk.

    Parameters
    ----------
    task_name : str
        Subdirectory name under trajectories/.  Use the example script name,
        e.g. "cup_lift", "cube_pull_rotate".
    traj_root : str or Path, optional
        Override the default trajectories/ root (useful for testing).
    record_every : int
        Only store one frame every N calls to record().  Matches the
        log_every cadence used in _execute_segment so no extra API calls
        are needed.
    """

    def __init__(
        self,
        task_name: str,
        traj_root: Optional[str] = None,
        record_every: int = 1,
    ):
        self.task_name = task_name
        self._traj_root = Path(traj_root) if traj_root else _TRAJ_ROOT
        self._record_every = record_every
        self._call_count = 0
        self._frames: list[dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, position, orientation_xyzw) -> None:
        """Record one observation.  orientation_xyzw = [x,y,z,w] (Isaac Sim)."""
        self._call_count += 1
        if (self._call_count - 1) % self._record_every != 0:
            return
        pos = _to_float_list(position)
        wxyz = _xyzw_to_wxyz(_to_float_list(orientation_xyzw))
        self._frames.append({
            "pos": [round(v, 4) for v in pos],
            "rot": [round(v, 4) for v in wxyz],
        })

    def record_wxyz(self, position, orientation_wxyz) -> None:
        """Record when the quaternion is already [w,x,y,z]."""
        self._call_count += 1
        if (self._call_count - 1) % self._record_every != 0:
            return
        pos = _to_float_list(position)
        wxyz = _to_float_list(orientation_wxyz)
        self._frames.append({
            "pos": [round(v, 4) for v in pos],
            "rot": [round(v, 4) for v in wxyz],
        })

    def __len__(self) -> int:
        return len(self._frames)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, tag: str = "") -> str:
        """Write frames to  trajectories/<task_name>/<timestamp>[_<tag>].json.

        Returns the absolute path as a string.
        """
        out_dir = self._traj_root / self.task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{tag}.json" if tag else f"{ts}.json"
        out_path = out_dir / fname
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self._frames, f, indent=2)
        return str(out_path)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def to_prompt_block(self, max_frames: int = 30) -> str:
        """Compact text block for embedding in a Claude prompt (Format B)."""
        return format_for_prompt(self._frames, max_frames=max_frames)

    def to_example_dict(
        self,
        task_description: str,
        initial_position: list[float],
        initial_orientation_wxyz: list[float],
        solution_code: str = "",
    ) -> dict:
        """Return a Format-B example dict suitable for JSON serialisation."""
        n = len(self._frames)
        return {
            "task": task_description,
            "initial_object_pose": {
                "position": [round(v, 4) for v in initial_position],
                "orientation": [round(v, 4) for v in initial_orientation_wxyz],
            },
            "object_trajectory": {
                "positions":    [[round(v, 4) for v in f["pos"]] for f in self._frames],
                "orientations": [[round(v, 4) for v in f["rot"]] for f in self._frames],
                "timesteps":    list(range(n)),
            },
            "solution_code": solution_code,
        }


# ---------------------------------------------------------------------------
# Module-level helpers used by prompts.py
# ---------------------------------------------------------------------------

def load_for_prompt(path: str) -> list[dict]:
    """Load a saved trajectory JSON for prompt injection.

    Returns a list of {"pos": [...], "rot": [...]} dicts (same as trajectory.py).
    Returns [] if path is empty or file does not exist.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def load_latest_for_prompt(task_name: str, traj_root: Optional[str] = None) -> list[dict]:
    """Load the most recently saved trajectory for *task_name*."""
    root = Path(traj_root) if traj_root else _TRAJ_ROOT
    task_dir = root / task_name
    if not task_dir.exists():
        return []
    files = sorted(task_dir.glob("*.json"))
    if not files:
        return []
    return load_for_prompt(str(files[-1]))


def format_for_prompt(frames: list[dict], max_frames: int = 30) -> str:
    """Return a compact multiline string describing the trajectory.

    Intended for injection into the Claude prompt as a Format-B demo block.
    """
    if not frames:
        return ""
    sampled = _downsample(frames, max_frames)
    n_total = len(frames)
    n_shown = len(sampled)

    lines = [
        "=== OBJECT TRAJECTORY DEMONSTRATION ===",
        f"Total recorded frames: {n_total}   Shown (downsampled): {n_shown}",
        "",
        "Format: pos=[x, y, z]  rot_wxyz=[w, x, y, z]",
    ]
    for i, fr in enumerate(sampled):
        p = fr["pos"]
        r = fr["rot"]
        lines.append(
            f"  t={i:3d}: pos=[{p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}]"
            f"  rot=[{r[0]:+.4f},{r[1]:+.4f},{r[2]:+.4f},{r[3]:+.4f}]"
        )
    lines += [
        "",
        "INTERPRETATION RULES:",
        "  * This trajectory describes OBJECT motion, NOT robot / EE motion.",
        "  * Infer the contact point, force direction, and motion primitive needed",
        "    to produce this object motion.",
        "  * Do NOT replay these values as robot joint targets.",
        "  * Do NOT assume they represent EE poses.",
        "=== END TRAJECTORY DEMONSTRATION ===",
    ]
    return "\n".join(lines)
