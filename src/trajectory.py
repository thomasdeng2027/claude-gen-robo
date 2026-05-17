"""Trajectory loading for the keypoint pipeline.

Supports three input formats:
  1. D3CB format:  {"object_trajectories": {"cousins": [[ {"pos":[...],"rot":[w,x,y,z]}, ... ]]}}
  2. Flat list:    [ {"pos":[...],"rot":[w,x,y,z]}, ... ]
  3. iPhone/ARKit: [ {"frame":N,"position":[...],"orientation_rpy":[r,p,y]}, ... ]
     (orientation_rpy converted to wxyz quaternion; coordinates are raw camera-frame values —
     the boilerplate handles world-frame alignment the same way it does for formats 1 & 2)

The returned trajectory is a list of {"pos":[...], "rot":[w,x,y,z]} dicts in the
ORIGINAL frame — alignment to the simulation's object init pose happens inside
the BOILERPLATE (so every Claude-generated script sees aligned waypoints
computed at runtime, matching RL's _build_env_waypoints_from_object_init).
"""

from __future__ import annotations

import json
from pathlib import Path


def _rpy_to_wxyz(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert roll-pitch-yaw (radians, intrinsic XYZ) to [w, x, y, z] quaternion."""
    import math
    cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def load_trajectory(path: str, **_kwargs) -> list[dict]:
    """Load + filter pause frames. Returns [] if path is empty/missing."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[trajectory] not found: {p} — using empty trajectory")
        return []

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames: list[dict]
    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict) and "object_trajectories" in data:
        cousins = (data.get("object_trajectories") or {}).get("cousins") or []
        if not cousins or not isinstance(cousins, list) or not cousins[0]:
            raise ValueError(f"{p}: object_trajectories.cousins[0] missing/empty")
        frames = cousins[0]
    else:
        raise ValueError(f"{p}: unsupported trajectory format (top-level must be list or have object_trajectories)")

    out: list[dict] = []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict):
            raise ValueError(f"{p}: frame {i} is not a dict")

        # --- position ---
        pos = fr.get("pos") or fr.get("position")
        if pos is None:
            raise ValueError(f"{p}: frame {i} missing 'pos'/'position'")
        pos = [float(v) for v in pos]

        # --- rotation ---
        rot = fr.get("rot") or fr.get("orientation") or fr.get("quat")
        if rot is None:
            # iPhone/ARKit format: orientation_rpy = [roll, pitch, yaw] in radians
            rpy = fr.get("orientation_rpy")
            if rpy is not None:
                rot = _rpy_to_wxyz(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            else:
                rot = [1.0, 0.0, 0.0, 0.0]
        else:
            rot = [float(v) for v in rot]

        out.append({"pos": pos, "rot": rot})

    print(f"[trajectory] {p.name}: {len(out)} frames")
    if len(out) >= 2:
        z0 = out[0]["pos"][2]
        zN = out[-1]["pos"][2]
        print(f"[trajectory] first=({out[0]['pos']}) last=({out[-1]['pos']}) "
              f"Δz={zN - z0:+.4f}m")
    return out
