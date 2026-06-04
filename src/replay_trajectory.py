"""replay_trajectory.py — Replay a saved object trajectory in Isaac Sim.

Loads a trajectory JSON saved by TrajectoryLogger and drives the object through
each recorded pose by teleporting it (set_world_poses).  Renders frames and
encodes an MP4 next to the trajectory file.

All paths are resolved relative to the repo root (<repo>/claude-data-gen/).
Run from any directory.

Usage
-----
# cup_lift (latest trajectory):
    python src/replay_trajectory.py \\
        --task cup_lift \\
        --object_usd src/assets/cup/cup.usd \\
        --scale 2.0 --mass 0.01

# cube_pull_rotate (latest trajectory):
    python src/replay_trajectory.py \\
        --task cube_pull_rotate \\
        --object_usd assets/objects/blue_cube/cube.usd \\
        --scale 3.0 --mass 0.05

# cylinder_pour — procedural cylinder, no USD file needed:
    python src/replay_trajectory.py \\
        --task cylinder_pour \\
        --object_usd procedural_cylinder \\
        --cyl_radius 0.025 --cyl_height 0.18 --mass 0.2

# Replay a specific file instead of the latest:
    python src/replay_trajectory.py \\
        --trajectory trajectories/cup_lift/20260507_123456.json \\
        --object_usd src/assets/cup/cup.usd --scale 2.0

The object is driven kinematically (set_world_poses overrides physics each step).
No robot is loaded — this is a pure object-motion visualiser.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path

# ── Environment ────────────────────────────────────────────────────────────────
_OMNI_USER_HOME  = "/tmp/isaac_user_jingyuny"
_WARP_CACHE_PATH = "/tmp/warp_cache_jingyuny"
Path(_OMNI_USER_HOME).mkdir(parents=True, exist_ok=True)
Path(_WARP_CACHE_PATH).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMNI_USER_HOME",              _OMNI_USER_HOME)
os.environ.setdefault("WARP_CACHE_PATH",             _WARP_CACHE_PATH)
os.environ.setdefault("XDG_DATA_HOME",               _OMNI_USER_HOME + "/.local/share")
os.environ.setdefault("XDG_CACHE_HOME",              _OMNI_USER_HOME + "/.cache")
os.environ.setdefault("VK_ICD_FILENAMES",            "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY",                     ":0")
os.environ.setdefault("OMNI_STRUCTUREDLOG_ENABLED",  "0")
os.environ.setdefault("CUROBO_KERNEL_BACKEND",       "pybind")

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = str(Path(__file__).parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC_DIR = str(Path(__file__).parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a saved object trajectory in Isaac Sim.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--trajectory", type=str,
                     help="Path to a trajectory JSON file.")
    grp.add_argument("--task", type=str,
                     help="Task name — loads the most recent JSON from trajectories/<task>/.")
    p.add_argument("--object_usd", type=str, required=True,
                   help="Path to object USD (relative to repo root), or "
                        "'procedural_cylinder' to create a cylinder procedurally.")
    p.add_argument("--scale",  type=float, default=1.0,
                   help="Uniform scale applied to the object (default: 1.0). "
                        "Ignored for procedural_cylinder.")
    p.add_argument("--mass",   type=float, default=0.1,
                   help="Object mass in kg (default: 0.1).")
    p.add_argument("--cyl_radius", type=float, default=0.025,
                   help="Cylinder radius in metres (procedural_cylinder only, default: 0.025 "
                        "— matches cylinder_pour.py).")
    p.add_argument("--cyl_height", type=float, default=0.18,
                   help="Cylinder height in metres (procedural_cylinder only, default: 0.18 "
                        "— matches cylinder_pour.py).")
    p.add_argument("--interp_steps", type=int, default=10,
                   help="Number of interpolated sub-frames to insert between each "
                        "recorded frame (default: 10).  Increases smoothness without "
                        "changing the actual trajectory.")
    p.add_argument("--steps_per_frame", type=int, default=2,
                   help="Sim steps to hold each (interpolated) frame pose (default: 2).")
    p.add_argument("--camera_pos", type=float, nargs=3,
                   default=[1.0, -0.5, 1.2],
                   metavar=("X", "Y", "Z"),
                   help="Camera world position (default: 1.0 -0.5 1.2).")
    p.add_argument("--camera_look", type=float, nargs=3,
                   default=[-0.05, 0.45, 0.15],
                   metavar=("X", "Y", "Z"),
                   help="Camera look-at world position (default: -0.05 0.45 0.15).")
    p.add_argument("--render_every", type=int, default=1,
                   help="Render every N sim steps (default: 1 = render every step).")
    p.add_argument("--orientation", type=float, nargs=4,
                   default=None, metavar=("W", "X", "Y", "Z"),
                   help="Override object orientation for every frame (wxyz). "
                        "Useful when the trajectory was recorded with a different object "
                        "and the stored rotations don't match the current mesh. "
                        "Example: --orientation 0 0 1 0  (180° around Z, bottle upright).")
    return p.parse_args()


def _resolve_trajectory(args: argparse.Namespace) -> tuple[str, list[dict]]:
    """Return (path_str, frames) for the chosen trajectory."""
    if args.trajectory:
        path = Path(args.trajectory)
    else:
        traj_dir = Path(_REPO_ROOT) / "trajectories" / args.task
        if not traj_dir.exists():
            raise FileNotFoundError(
                f"No trajectory directory found for task '{args.task}': {traj_dir}"
            )
        files = sorted(traj_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"No trajectory JSON files found in {traj_dir}"
            )
        path = files[-1]
        print(f"[replay] Loading latest trajectory for task '{args.task}': {path}", flush=True)

    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list in {path}")

    print(f"[replay] Loaded {len(data)} frames from {path}", flush=True)
    return str(path), data


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions at fraction t ∈ [0,1].

    Inputs can be any consistent quaternion order (wxyz or xyzw) as long as both
    are the same.  Returns a normalised quaternion in the same order.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    # Ensure shortest path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        # Quaternions nearly identical — fall back to normalised lerp.
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    sin_t0  = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_t0
    s1 = np.sin(theta) / sin_t0
    return (s0 * q0 + s1 * q1) / np.linalg.norm(s0 * q0 + s1 * q1)


def _interpolate_frames(frames: list[dict], n_interp: int) -> list[dict]:
    """Insert n_interp-1 sub-frames between each consecutive pair of frames.

    Positions are linearly interpolated; orientations use SLERP.
    The last recorded frame is always appended unchanged.
    """
    if n_interp <= 1 or len(frames) < 2:
        return list(frames)
    out: list[dict] = []
    for i in range(len(frames) - 1):
        p0 = np.array(frames[i]["pos"],  dtype=np.float64)
        p1 = np.array(frames[i+1]["pos"], dtype=np.float64)
        r0 = np.array(frames[i]["rot"],  dtype=np.float64)  # wxyz
        r1 = np.array(frames[i+1]["rot"], dtype=np.float64)
        for j in range(n_interp):
            t = j / n_interp
            pos = (p0 + t * (p1 - p0)).tolist()
            rot = _slerp(r0, r1, t).tolist()
            out.append({"pos": pos, "rot": rot})
    out.append(frames[-1])
    return out


# ── Parse args before SimulationApp (fast path for --help / errors) ────────────
_args = _parse_args()
_traj_path, _raw_frames = _resolve_trajectory(_args)

# Densify the sparse recorded frames with SLERP interpolation.
_frames = _interpolate_frames(_raw_frames, _args.interp_steps)
print(
    f"[replay] Interpolated: {len(_raw_frames)} recorded frames → "
    f"{len(_frames)} frames  ({_args.interp_steps}× interp)",
    flush=True,
)

# ── SimulationApp ──────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
print("Starting SimulationApp...", flush=True)
simulation_app = SimulationApp({"headless": True, "renderer": "RasterizedRendering"})
print("SimulationApp ready", flush=True)

import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom, UsdPhysics

# ── Constants derived from args ────────────────────────────────────────────────
_PROCEDURAL_CYLINDER = (_args.object_usd == "procedural_cylinder")
if _PROCEDURAL_CYLINDER:
    _OBJ_USD_PATH = None
    _CYL_RADIUS   = _args.cyl_radius
    _CYL_HEIGHT   = _args.cyl_height
else:
    # Resolve relative paths from the repo root, not the CWD.
    _p = Path(_args.object_usd)
    if not _p.is_absolute():
        _p = Path(_REPO_ROOT) / _p
    _OBJ_USD_PATH = str(_p)
_OBJ_SCALE  = _args.scale
_OBJ_MASS   = _args.mass
_STEPS_PER_FRAME = _args.steps_per_frame
_RENDER_EVERY    = _args.render_every

_OBJ_PRIM   = "/World/Object"

# Frames / video output alongside the trajectory file.
_traj_stem  = Path(_traj_path).stem
_task_name  = Path(_traj_path).parent.name
_OUT_DIR    = Path(_traj_path).parent / f"replay_{_traj_stem}"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_VID_PATH   = Path(_traj_path).parent / f"replay_{_traj_stem}.mp4"

print(f"[replay] Frames → {_OUT_DIR}", flush=True)
print(f"[replay] Video  → {_VID_PATH}", flush=True)

# ── Initial pose from first trajectory frame ───────────────────────────────────
# Trajectory stores rot as [w, x, y, z].
_f0       = _frames[0]
_init_pos = np.array(_f0["pos"], dtype=np.float64)
_init_rot_wxyz = np.array(_f0["rot"], dtype=np.float64)  # [w,x,y,z]

print(f"[replay] Initial pose: pos={_init_pos.tolist()}  rot(wxyz)={_init_rot_wxyz.tolist()}", flush=True)

# ── Build scene ────────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
from isaacsim.core.api.objects import GroundPlane as _GroundPlane
world.scene.add(_GroundPlane(prim_path="/World/defaultGroundPlane"))
stage = omni.usd.get_context().get_stage()

if _PROCEDURAL_CYLINDER:
    _cyl_geom = UsdGeom.Cylinder.Define(stage, _OBJ_PRIM)
    _cyl_geom.CreateAxisAttr("Z")
    _cyl_geom.CreateRadiusAttr(float(_CYL_RADIUS))
    _cyl_geom.CreateHeightAttr(float(_CYL_HEIGHT))
    _cyl_xf = UsdGeom.Xformable(_cyl_geom)
    _cyl_xf.AddTranslateOp().Set(Gf.Vec3d(*_init_pos.tolist()))
    _bp = stage.GetPrimAtPath(_OBJ_PRIM)
    print(
        f"[replay] Procedural cylinder: radius={_CYL_RADIUS}m  height={_CYL_HEIGHT}m",
        flush=True,
    )
else:
    add_reference_to_stage(usd_path=_OBJ_USD_PATH, prim_path=_OBJ_PRIM)
    _bp  = stage.GetPrimAtPath(_OBJ_PRIM)
    _xf  = UsdGeom.Xformable(_bp)
    _xf.ClearXformOpOrder()
    _xf.AddTranslateOp().Set(Gf.Vec3d(*_init_pos.tolist()))
    w, x, y, z = _init_rot_wxyz.tolist()
    _xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    _xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(_OBJ_SCALE, _OBJ_SCALE, _OBJ_SCALE))
    print(f"[replay] Object USD: {_OBJ_USD_PATH}  scale={_OBJ_SCALE}", flush=True)

# Make the object KINEMATIC so set_world_poses controls its position exactly and
# PhysX cannot override it with gravity or collision responses.  Without this,
# world.step() integrates physics after each set_world_poses call and the object
# falls away from the recorded trajectory.
_rba = UsdPhysics.RigidBodyAPI.Apply(_bp)
_rba.CreateKinematicEnabledAttr().Set(True)
UsdPhysics.CollisionAPI.Apply(_bp)

objects = RigidPrim(
    prim_paths_expr=_OBJ_PRIM,
    name="objects",
    reset_xform_properties=False,
    track_contact_forces=False,
)
world.scene.add(objects)

# ── Camera ─────────────────────────────────────────────────────────────────────
_camera = rep.create.camera(
    position=tuple(_args.camera_pos),
    look_at=tuple(_args.camera_look),
)
_rp_cam = rep.create.render_product(_camera, (640, 480))
_rgb    = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb.attach([_rp_cam])
_frame_counter = [0]

import cv2 as _cv2

def _step(render: bool = True):
    _frame_counter[0] += 1
    do_render = render and (_frame_counter[0] % _RENDER_EVERY == 0)
    world.step(render=do_render)
    if do_render:
        try:
            rep.orchestrator.step(rt_subframes=1)
            img = _rgb.get_data()
            if img is not None and img.size > 0:
                _cv2.imwrite(
                    str(_OUT_DIR / f"frame_{_frame_counter[0]:06d}.png"),
                    _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR),
                )
        except Exception:
            pass

# ── World reset + initial settle ───────────────────────────────────────────────
world.reset()
# No settle needed — object is kinematic, so physics cannot move it from the
# initial pose set above.  Just run one step to initialise the renderer.
_step(render=False)

# ── Replay loop ────────────────────────────────────────────────────────────────
# Quaternion convention:
#   Trajectory JSON stores rot as [w, x, y, z]  (our canonical storage format).
#   set_world_poses() uses the SAME convention as get_world_poses(): [x, y, z, w].
#   So we must swap: xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]].
print(
    f"\n[replay] Starting replay: {len(_frames)} frames × {_STEPS_PER_FRAME} steps/frame"
    f"  ({len(_raw_frames)} recorded + {_args.interp_steps}× SLERP interp)",
    flush=True,
)

def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    r = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)
    return r / (np.linalg.norm(r) + 1e-12)

def _qinv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (wxyz)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

# Compute the orientation correction quaternion:
#   q_correction = q_override * inv(q_traj_initial)
# Then for each frame: q_display = q_correction * q_stored[i]
# This replaces the initial orientation with q_override while preserving all
# relative rotations in the trajectory (lift, pour tilt, etc.).
_q_correction: np.ndarray | None = None
if _args.orientation is not None:
    _o = np.array(_args.orientation, dtype=np.float64)
    _q_override = _o / (np.linalg.norm(_o) + 1e-12)
    _q_traj_init = np.array(_raw_frames[0]["rot"], dtype=np.float64)
    _q_traj_init /= np.linalg.norm(_q_traj_init) + 1e-12
    _q_correction = _qmul(_q_override, _qinv(_q_traj_init))
    print(f"[replay] Orientation override (wxyz): {_q_override.tolist()}", flush=True)
    print(f"[replay] Traj initial rot   (wxyz): {_q_traj_init.tolist()}", flush=True)
    print(f"[replay] Correction quat    (wxyz): {_q_correction.tolist()}", flush=True)

for frame_idx, frame in enumerate(_frames):
    pos  = np.array(frame["pos"], dtype=np.float64).reshape(1, 3)
    wxyz = np.array(frame["rot"], dtype=np.float64)             # [w,x,y,z] stored
    if _q_correction is not None:
        wxyz = _qmul(_q_correction, wxyz)
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])      # → [x,y,z,w] for set_world_poses
    quat = xyzw.reshape(1, 4)

    for _ in range(_STEPS_PER_FRAME):
        # Re-apply pose every sub-step so physics cannot drift the object.
        objects.set_world_poses(positions=pos, orientations=quat)
        _step(render=True)

    log_every = max(1, len(_frames) // 10)
    if frame_idx % log_every == 0 or frame_idx == len(_frames) - 1:
        print(
            f"  [replay] frame {frame_idx:4d}/{len(_frames)}  "
            f"pos={[round(v,3) for v in pos[0].tolist()]}",
            flush=True,
        )

# ── Final frame ────────────────────────────────────────────────────────────────
_pos_end, _ = objects.get_world_poses()
print(f"\n[replay] Final object pose: {np.asarray(_pos_end)[0].tolist()}", flush=True)

# ── Encode video ───────────────────────────────────────────────────────────────
try:
    import subprocess, glob as _glob
    png_frames = sorted(_glob.glob(str(_OUT_DIR / "frame_*.png")))
    if png_frames:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "30",
            "-pattern_type", "glob",
            "-i", str(_OUT_DIR / "frame_*.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(_VID_PATH),
        ], capture_output=True, check=False)
        if _VID_PATH.exists():
            print(f"[replay] VIDEO: {_VID_PATH}", flush=True)
        else:
            print("[replay] VIDEO_ERROR: ffmpeg produced no output", flush=True)
    else:
        print("[replay] No frames rendered — video skipped.", flush=True)
except Exception as _ve:
    print(f"[replay] VIDEO_ERROR: {_ve}", flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== REPLAY SUMMARY ===", flush=True)
print(f"  Trajectory:     {_traj_path}", flush=True)
print(f"  Recorded frames:{len(_raw_frames)}", flush=True)
print(f"  Interp steps:   {_args.interp_steps}×  →  {len(_frames)} total frames", flush=True)
print(f"  Steps/frame:    {_STEPS_PER_FRAME}", flush=True)
print(f"  Total sim steps:{len(_frames) * _STEPS_PER_FRAME}", flush=True)
print(f"  Video:          {_VID_PATH}", flush=True)
print(f"  Frame dir:      {_OUT_DIR}", flush=True)

# ── Cleanup ────────────────────────────────────────────────────────────────────
try:
    rep.orchestrator.stop()
except Exception:
    pass
try:
    simulation_app.close()
except Exception:
    pass
