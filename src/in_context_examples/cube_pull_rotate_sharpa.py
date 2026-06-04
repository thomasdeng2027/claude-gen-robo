"""
cube_push_rotate.py

Rotate a cube ~90° by pushing the top-face corner with fingertips.
Adapted to use the realkinova sharpa hand instead of the xhand.

Strategy: contact the (+X, +Y) corner of the cube top face from above
(top-down, like cup_lift), then pull backward in -Y (toward robot).

Workspace note:
  The robot base is at world [0.203, -0.127, 0.152] with 45° Y rotation.
  Reachable EE zone: world_x ≈ -0.04 to -0.02 (only the +X side of the cube).
  workspace_z = (world_x + world_z - 0.355) * 0.707 must stay above ≈ -0.15.
  The -X edge at world_x=-0.166 gives workspace_z=-0.21 → 0/64 IK (unreachable).
  The +X edge at world_x=-0.016 gives workspace_z=-0.09 → 64/64 IK ✓.

Torque (CW rotation, looking down):
  Contact r = [+CUBE/2, +CUBE/2, ...] from cube center.
  Pull F = [0, -F, 0].  τ_z = r_x*F_y = (+0.075)*(-F) → negative yaw.

Run:
    python cube_pull_rotate_sharpa.py
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
_OMNI_USER_HOME = "/tmp/isaac_user_jingyuny"
_WARP_CACHE_PATH = "/tmp/warp_cache_jingyuny"
Path(_OMNI_USER_HOME).mkdir(parents=True, exist_ok=True)
Path(_WARP_CACHE_PATH).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMNI_USER_HOME",              _OMNI_USER_HOME)
os.environ.setdefault("WARP_CACHE_PATH",             _WARP_CACHE_PATH)
os.environ.setdefault("XDG_DATA_HOME",               _OMNI_USER_HOME + "/.local/share")
os.environ.setdefault("XDG_CACHE_HOME",              _OMNI_USER_HOME + "/.cache")
os.environ.setdefault("VK_ICD_FILENAMES",            "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("OMNI_STRUCTUREDLOG_ENABLED",  "0")
os.environ.setdefault("CUROBO_KERNEL_BACKEND",       "pybind")

sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_PIPELINE_DIR = _REPO_ROOT + "/src"
_IK_URDF_PATH = _REPO_ROOT + "/assets/realkinova_sharpa_hand/realkinova_sharpa_hand_right.urdf"
_OBJ_USD_PATH = _REPO_ROOT + "/assets/objects/blue_cube/cube.usd"

_CUROBO_V2_ROOT = "/juno/u/jingyuny/curobo"
if _CUROBO_V2_ROOT not in sys.path:
    sys.path.insert(0, _CUROBO_V2_ROOT)
for _p in (_REPO_ROOT, _PIPELINE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaacsim import SimulationApp
print("Starting SimulationApp...", flush=True)
simulation_app = SimulationApp({"headless": True, "renderer": "RasterizedRendering"})
print("SimulationApp ready", flush=True)

import warp_compat  # noqa: F401 — swap bundled warp 1.8.2 → pip warp 1.13 after render init

import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom, UsdPhysics

# -----------------------------------------------------------------------------
# Scene constants
# -----------------------------------------------------------------------------
# cube.usd is approximately 0.05 m per side at scale 1.0.
# Scale 3 gives a 0.15 m cube.
_OBJ_SCALE = 3.0
_CUBE_SIZE = 0.05 * _OBJ_SCALE
_OBJ_POS_INIT = np.array([-0.09, 0.47, 0.18], dtype=np.float64)
_OBJ_ORIENT_WXYZ = (1.0, 0.0, 0.0, 0.0)
_OBJ_MASS = 0.05

_ROBOT_PRIM = "/World/envs/env_0/Robot"
_OBJ_PRIM = "/World/envs/env_0/Object"

_RENDER_EVERY = 8
_FRAMES_DIR = os.environ.get("ISAAC_FRAMES_DIR", "/tmp/isaac_frames_cube_negx_highy_pull_posx")
if Path(_FRAMES_DIR).exists():
    shutil.rmtree(_FRAMES_DIR)
Path(_FRAMES_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Task geometry constants
# -----------------------------------------------------------------------------
# Contact corner: +X (reachable) and +Y (maximizes r_x moment arm for -Y pull).
# The -X corner is NOT reachable from above — see workspace note in the docstring.
_CORNER_X_SIGN = +1.0
_CORNER_Y_SIGN = +1.0

# Pull direction: -Y = toward robot = "backwards", slight +X inward to stay on corner.
# Torque: τ_z = r_x * F_y = (+CUBE/2) * (-F) → CW (negative yaw).
_Y_PULL_BACKWARD = 0.30    # m total pull toward robot
_X_PULL_INWARD   = 0.04    # m slight inward (+X) to track the corner as cube rotates

# Top-down contact setup (EE_z = -Z world, same as cup_lift).
# From cup_lift log: palm Z 0.324, index_tip Z 0.263 → offset ≈ 0.061 m.
_EE_Z_TO_FINGERTIP = 0.08   # m: EE link is this far above the fingertip contact
# Press IK target this far into the top surface so tracking undershoot still lands.
_CONTACT_Z_PENETRATE = 0.03  # m
_STANDOFF_DIST = 0.080
_GRASP_TILT_DEG = -15.0

# IK waypoint density.
_N_IK_APPROACH_KPS = 5
_N_IK_PULL_KPS = 24
_N_INTERP_BETWEEN = 6

# Simulation pacing.
_SIM_STEPS_PER_WP = 8
_SIM_STEPS_PLAN_WP = 6
_SETTLE_STEPS = 10

# -----------------------------------------------------------------------------
# Initial robot configuration
# -----------------------------------------------------------------------------
_INIT_ARM = {
    "Actuator1": -1.529021,
    "Actuator2":  1.200000,
    "Actuator3":  0.164424,
    "Actuator4":  0.600000,
    "Actuator5":  1.254565,
    "Actuator6": -1.063086,
    "Actuator7":  0.235436,
}

_INIT_HAND = {
    "right_thumb_CMC_FE":  0.0,
    "right_thumb_CMC_AA":  0.0,
    "right_thumb_MCP_FE":  0.0,
    "right_thumb_MCP_AA":  0.0,
    "right_thumb_IP":      0.0,
    "right_index_MCP_FE":  0.0,
    "right_index_MCP_AA":  0.0,
    "right_index_PIP":     0.0,
    "right_index_DIP":     0.0,
    "right_middle_MCP_FE": 0.0,
    "right_middle_MCP_AA": 0.0,
    "right_middle_PIP":    0.0,
    "right_middle_DIP":    0.0,
    "right_ring_MCP_FE":   0.0,
    "right_ring_MCP_AA":   0.0,
    "right_ring_PIP":      0.0,
    "right_ring_DIP":      0.0,
    "right_pinky_CMC":     0.0,
    "right_pinky_MCP_FE":  0.0,
    "right_pinky_MCP_AA":  0.0,
    "right_pinky_PIP":     0.0,
    "right_pinky_DIP":     0.0,
}

_HAND_OPEN = {
    "right_thumb_CMC_FE":  0.0,
    "right_thumb_CMC_AA":  0.0,
    "right_thumb_MCP_FE":  0.0,
    "right_thumb_MCP_AA":  0.0,
    "right_thumb_IP":      0.0,
    "right_index_MCP_FE":  0.0,
    "right_index_MCP_AA":  0.0,
    "right_index_PIP":     0.0,
    "right_index_DIP":     0.0,
    "right_middle_MCP_FE": 0.0,
    "right_middle_MCP_AA": 0.0,
    "right_middle_PIP":    0.0,
    "right_middle_DIP":    0.0,
    "right_ring_MCP_FE":   0.0,
    "right_ring_MCP_AA":   0.0,
    "right_ring_PIP":      0.0,
    "right_ring_DIP":      0.0,
    "right_pinky_CMC":     0.0,
    "right_pinky_MCP_FE":  0.0,
    "right_pinky_MCP_AA":  0.0,
    "right_pinky_PIP":     0.0,
    "right_pinky_DIP":     0.0,
}

_HAND_PUSH = {
    "right_thumb_CMC_FE":  0.8,
    "right_thumb_CMC_AA":  0.0,
    "right_thumb_MCP_FE":  0.6,
    "right_thumb_MCP_AA":  0.0,
    "right_thumb_IP":      0.7,
    "right_index_MCP_FE":  0.7,
    "right_index_MCP_AA":  0.0,
    "right_index_PIP":     0.7,
    "right_index_DIP":     0.5,
    "right_middle_MCP_FE": 0.7,
    "right_middle_MCP_AA": 0.0,
    "right_middle_PIP":    0.7,
    "right_middle_DIP":    0.5,
    "right_ring_MCP_FE":   0.7,
    "right_ring_MCP_AA":   0.0,
    "right_ring_PIP":      0.7,
    "right_ring_DIP":      0.5,
    "right_pinky_CMC":     0.1,
    "right_pinky_MCP_FE":  0.7,
    "right_pinky_MCP_AA":  0.0,
    "right_pinky_PIP":     0.7,
    "right_pinky_DIP":     0.5,
}

# -----------------------------------------------------------------------------
# Stage helpers
# -----------------------------------------------------------------------------
def _prim_world_pos(stage, path: str):
    try:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
            t = xf.ExtractTranslation()
            pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
            if np.any(pos != 0):
                return pos
    except Exception:
        pass
    return None


def _ee_link_world(stage, robot_prim=None):
    prefix = (_ROBOT_PRIM if robot_prim is None else robot_prim).rstrip("/") + "/"
    return _prim_world_pos(stage, prefix + "right_hand_C_MC")


def _quat_wxyz_to_yaw(q_wxyz):
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

# -----------------------------------------------------------------------------
# Build scene
# -----------------------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

try:
    _status, _cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    _cfg.merge_fixed_joints = False
    _cfg.fix_base = True
    _cfg.import_inertia_tensor = True
    _cfg.distance_scale = 1.0
    _cfg.create_physics_scene = False
    _ok, _result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=_IK_URDF_PATH,
        import_config=_cfg,
        dest_path="",
    )
    print(f"URDFParseAndImportFile: ok={_ok} result={_result}", flush=True)
    if not _ok:
        raise RuntimeError("URDF import failed")
    _ROBOT_PRIM = str(_result).strip() if _result else _ROBOT_PRIM
    if not stage.GetPrimAtPath(_ROBOT_PRIM).IsValid():
        raise RuntimeError(f"Robot prim not valid at {_ROBOT_PRIM}")
    print(f"ROBOT_LOADED: prim {_ROBOT_PRIM}", flush=True)
except Exception as exc:
    print(f"URDF import failed: {exc}", flush=True)
    raise

add_reference_to_stage(usd_path=_OBJ_USD_PATH, prim_path=_OBJ_PRIM)
_obj_prim = stage.GetPrimAtPath(_OBJ_PRIM)
_xf = UsdGeom.Xformable(_obj_prim)
_xf.ClearXformOpOrder()
_xf.AddTranslateOp().Set(Gf.Vec3d(*_OBJ_POS_INIT.tolist()))
w, x, y, z = _OBJ_ORIENT_WXYZ
_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
_xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(_OBJ_SCALE, _OBJ_SCALE, _OBJ_SCALE))
UsdPhysics.MassAPI.Apply(_obj_prim).CreateMassAttr().Set(float(_OBJ_MASS))

robots = Articulation(prim_paths_expr=_ROBOT_PRIM, name="robots", reset_xform_properties=False)
objects = RigidPrim(prim_paths_expr=_OBJ_PRIM, name="objects", reset_xform_properties=False, track_contact_forces=True)
world.scene.add(robots)
world.scene.add(objects)

_FINGER_TIP_LINKS = {
    "thumb":  "right_thumb_fingertip",
    "index":  "right_index_fingertip",
    "middle": "right_middle_fingertip",
    "ring":   "right_ring_fingertip",
    "pinky":  "right_pinky_fingertip",
}
_finger_contact_prims = {}
for _fname, _ln in _FINGER_TIP_LINKS.items():
    _fpath = _ROBOT_PRIM.rstrip("/") + "/" + _ln
    try:
        _fp = RigidPrim(prim_paths_expr=_fpath, name=f"finger_{_fname}", track_contact_forces=True)
        world.scene.add(_fp)
        _finger_contact_prims[_fname] = _fp
        print(f"Finger contact prim added: {_fname} @ {_fpath}", flush=True)
    except Exception as exc:
        print(f"Finger contact prim '{_fname}' unavailable: {exc}", flush=True)

world.reset()
robots.initialize()
world.step(render=False)

_dof_names = list(robots.dof_names)
_n_dof = robots.num_dof
_dof_idx = {n: i for i, n in enumerate(_dof_names)}
print(f"DOF names ({_n_dof}): {_dof_names}", flush=True)

_ARM_JOINTS = [f"Actuator{i}" for i in range(1, 8)]
_arm_cols = [_dof_idx[jn] for jn in _ARM_JOINTS if jn in _dof_idx]
if len(_arm_cols) != 7:
    raise RuntimeError(f"Expected 7 arm joints, found {len(_arm_cols)}")

_init_joints_1d = np.zeros(_n_dof, dtype=np.float64)
for jn, val in {**_INIT_ARM, **_INIT_HAND}.items():
    if jn in _dof_idx:
        _init_joints_1d[_dof_idx[jn]] = float(val)

robots.set_joint_positions(np.tile(_init_joints_1d, (1, 1)))
robots.set_joint_velocities(np.zeros((1, _n_dof)))

try:
    kps = np.tile(np.array([8000.0] * 7 + [20000.0] * (_n_dof - 7)), (1, 1))
    kds = np.tile(np.array([400.0] * 7 + [600.0] * (_n_dof - 7)), (1, 1))
    robots.set_gains(kps=kps, kds=kds)
except Exception as exc:
    print(f"PD gains warning: {exc}", flush=True)

# -----------------------------------------------------------------------------
# Camera / render
# -----------------------------------------------------------------------------
_camera = rep.create.camera(position=(1.2, 0.8, 1.5), look_at=(-0.05, 0.45, 0.1))
_rp_cam = rep.create.render_product(_camera, (640, 480))
_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb.attach([_rp_cam])
_frame_counter = [0]


_frame_diag_done = [False]

def _step(render_force=False):
    _frame_counter[0] += 1
    if render_force or (_frame_counter[0] % _RENDER_EVERY == 0):
        world.step(render=True)
        try:
            rep.orchestrator.step(rt_subframes=1)
            img = _rgb.get_data()
            if not _frame_diag_done[0]:
                _frame_diag_done[0] = True
                print(f"FRAME_DIAG type: {type(img)}", flush=True)
                if img is not None:
                    print(f"FRAME_DIAG raw:  shape={img.shape} dtype={img.dtype} "
                          f"min={img.min()} max={img.max()} mean={img.mean():.3f}", flush=True)
                    _has_npy = hasattr(img, 'numpy')
                    _dev = getattr(img, 'device', 'N/A')
                    print(f"FRAME_DIAG meta: has_numpy_method={_has_npy} device={_dev}", flush=True)
            if img is not None and img.size > 0:
                if hasattr(img, 'numpy') and not isinstance(img, np.ndarray):
                    img_np = img.numpy()
                else:
                    img_np = np.asarray(img)
                img_cpu = np.ascontiguousarray(
                    img_np[..., :3] if img_np.ndim == 3 and img_np.shape[-1] == 4 else img_np
                )
                if _frame_counter[0] == 8:
                    print(f"FRAME_DIAG cpu:  shape={img_cpu.shape} dtype={img_cpu.dtype} "
                          f"min={img_cpu.min()} max={img_cpu.max()} mean={img_cpu.mean():.3f}", flush=True)
                from PIL import Image as _PILImage
                _PILImage.fromarray(img_cpu).save(
                    f"{_FRAMES_DIR}/frame_{_frame_counter[0]:06d}.png"
                )
        except Exception as _e:
            if not _frame_diag_done[0]:
                _frame_diag_done[0] = True
                print(f"FRAME_DIAG: exception={_e}", flush=True)
    else:
        world.step(render=False)


def _settle(n, cur_cmd):
    for _ in range(n):
        robots.set_joint_position_targets(cur_cmd)
        _step()


def _log_finger_contacts(label: str):
    row = {"event": label}
    for fname, fp in _finger_contact_prims.items():
        try:
            cf = np.asarray(fp.get_net_contact_forces(dt=1.0), dtype=np.float64).flatten()
            mag = float(np.linalg.norm(cf[:3])) if len(cf) >= 3 else 0.0
            row[f"{fname}_N"] = round(mag, 4)
            row[f"{fname}_vec"] = [round(float(v), 3) for v in cf[:3]]
        except Exception:
            row[f"{fname}_N"] = None
    print(json.dumps(row), flush=True)


def _log_cube_state(label: str, initial_yaw: float | None = None):
    pn, on = objects.get_world_poses()
    pos = np.asarray(pn, dtype=np.float64)[0]
    ori = np.asarray(on, dtype=np.float64)[0]
    yaw = _quat_wxyz_to_yaw(ori)
    row = {
        "event": label,
        "cube_pos": [round(float(v), 4) for v in pos.tolist()],
        "cube_qwxyz": [round(float(v), 4) for v in ori.tolist()],
        "cube_yaw_deg": round(np.degrees(yaw), 2),
    }
    if initial_yaw is not None:
        dyaw = (yaw - initial_yaw + np.pi) % (2.0 * np.pi) - np.pi
        row["yaw_change_deg"] = round(np.degrees(dyaw), 2)
    print(json.dumps(row), flush=True)
    return yaw

# -----------------------------------------------------------------------------
# Settle cube and compute requested contact geometry
# -----------------------------------------------------------------------------
print("Settling physics to read cube pose...", flush=True)
_settle_cmd = np.tile(_init_joints_1d, (1, 1))
robots.set_joint_positions(_settle_cmd)
robots.set_joint_velocities(np.zeros((1, _n_dof)))
for _ in range(30):
    robots.set_joint_position_targets(_settle_cmd)
    _step()

_pos_raw, _ori_raw = objects.get_world_poses()
_obj_settled = np.asarray(_pos_raw, dtype=np.float64)[0]
_ori_settled = np.asarray(_ori_raw, dtype=np.float64)[0]
print(f"[settle] Cube position:   {[round(v,4) for v in _obj_settled.tolist()]}", flush=True)
print(f"[settle] Cube orientation:{[round(v,4) for v in _ori_settled.tolist()]}", flush=True)

_cube_cx, _cube_cy, _cube_cz = _obj_settled

# (+X, +Y) corner of the top face — reachable with top-down approach.
_contact_x = _cube_cx + _CORNER_X_SIGN * (_CUBE_SIZE / 2.0)
_contact_y = _cube_cy + _CORNER_Y_SIGN * (_CUBE_SIZE / 2.0)
_contact_z = _cube_cz + _CUBE_SIZE / 2.0 - _CONTACT_Z_PENETRATE
_contact_world = np.array([_contact_x, _contact_y, _contact_z], dtype=np.float64)

# EE_z = -Z world → fingertip is _EE_Z_TO_FINGERTIP below the EE link.
_ee_contact_world = _contact_world + np.array([0.0, 0.0, _EE_Z_TO_FINGERTIP])
_ee_standoff_world = _ee_contact_world + np.array([0.0, 0.0, _STANDOFF_DIST])

# Pull -Y (toward robot = "backwards"), slight +X inward to stay on the rotating corner.
_pull_delta_world = np.array([_X_PULL_INWARD, -_Y_PULL_BACKWARD, 0.0], dtype=np.float64)
_ee_pull_end_world = _ee_contact_world + _pull_delta_world

_ws_z_contact  = (_ee_contact_world[0]  + _ee_contact_world[2]  - 0.355) * 0.707
_ws_z_standoff = (_ee_standoff_world[0] + _ee_standoff_world[2] - 0.355) * 0.707

print("\n=== GEOMETRY ===", flush=True)
print(f"cube center:        {[round(v,4) for v in _obj_settled.tolist()]}", flush=True)
print(f"cube size assumed:  {_CUBE_SIZE:.4f} m", flush=True)
print(f"target corner:      +X / +Y (reachable; -X is outside workspace)", flush=True)
print(f"contact point:      {[round(v,4) for v in _contact_world.tolist()]}", flush=True)
print(f"EE contact target:  {[round(v,4) for v in _ee_contact_world.tolist()]}", flush=True)
print(f"EE standoff target: {[round(v,4) for v in _ee_standoff_world.tolist()]}", flush=True)
print(f"workspace_z contact ≈ {_ws_z_contact:.3f}  standoff ≈ {_ws_z_standoff:.3f}  (run-1 worked at -0.09)", flush=True)
print(f"pull direction:     -Y (toward robot), delta={[round(v,4) for v in _pull_delta_world.tolist()]}", flush=True)
print(f"pull end target:    {[round(v,4) for v in _ee_pull_end_world.tolist()]}", flush=True)
print("motion: standoff -> descend in -Z -> pull in -Y", flush=True)

# -----------------------------------------------------------------------------
# cuRobo imports
# -----------------------------------------------------------------------------
print("\nImporting cuRobo v2...", flush=True)
warp_compat.ensure()  # re-apply swap in case omni extensions re-hooked bundled warp
import torch
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState as CuroboJointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.types.device_cfg import DeviceCfg as CuroboDeviceCfg
from motion_planner_batch import BatchMotionPlanner
print("cuRobo v2 imports OK", flush=True)

# -----------------------------------------------------------------------------
# Transform / quaternion helpers
# -----------------------------------------------------------------------------
_T_WB = np.zeros(3, dtype=np.float64)
_R_WB = np.eye(3, dtype=np.float64)


def _read_base_world_pose(stage, robot_prim_path: str):
    base_path = robot_prim_path.rstrip("/") + "/base_link"
    p = stage.GetPrimAtPath(base_path)
    if not p.IsValid():
        p = stage.GetPrimAtPath(robot_prim_path)
    xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
    t = xf.ExtractTranslation()
    T = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    R = np.array([[xf[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
    return T, R


def _world_to_base(pos_world):
    return (np.asarray(pos_world, dtype=np.float64) - _T_WB) @ _R_WB


def _base_to_world(pos_base):
    return np.asarray(pos_base, dtype=np.float64) @ _R_WB.T + _T_WB


def _rmat_to_wxyz(R):
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def _world_quat_to_base(q_world):
    q_wb = _rmat_to_wxyz(_R_WB)
    q = _qmul(q_wb, np.asarray(q_world, dtype=np.float64))
    return q / (np.linalg.norm(q) + 1e-12)


def _compute_topdown_quat_world(toward_robot_dir_xy, tilt_deg=0.0, toward_cube_xy=None):
    """Top-down orientation like cup_lift.

    local EE_z points along world -Z.
    local EE_y points generally toward the robot.
    Optional tilt aims EE_z slightly toward cube center.
    """
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_axis = np.array([toward_robot_dir_xy[0], toward_robot_dir_xy[1], 0.0], dtype=np.float64)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    if tilt_deg != 0.0 and toward_cube_xy is not None:
        d = np.array([toward_cube_xy[0], toward_cube_xy[1], 0.0], dtype=np.float64)
        d /= np.linalg.norm(d) + 1e-12
        t = np.deg2rad(tilt_deg)
        z_axis = z_axis * np.cos(t) + d * np.sin(t)
        z_axis /= np.linalg.norm(z_axis) + 1e-12
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-12
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis) + 1e-12

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _rmat_to_wxyz(R)

# -----------------------------------------------------------------------------
# Planner setup
# -----------------------------------------------------------------------------
_T_WB, _R_raw = _read_base_world_pose(stage, _ROBOT_PRIM)
_R_WB = _R_raw.T
print(f"Base link world pos: {_T_WB.tolist()}", flush=True)
print(f"Base link world rot R_WB:\n{_R_WB}", flush=True)

print("\nInitializing BatchMotionPlanner...", flush=True)

_ROBOT_CFG_DICT = {
    "robot_cfg": {
        "kinematics": {
            "urdf_path": _IK_URDF_PATH,
            "asset_root_path": str(Path(_IK_URDF_PATH).parent),
            "base_link": "base_link",
            "tool_frames": ["right_hand_C_MC"],
            "grasp_contact_link_names": ["right_hand_C_MC"],
            "collision_link_names": [
                "base_link", "Shoulder_Link", "HalfArm1_Link", "HalfArm2_Link",
                "ForeArm_Link", "SphericalWrist1_Link", "SphericalWrist2_Link",
                "Bracelet_Link", "sharpa_mount_right", "right_hand_C_MC",
            ],
            "collision_spheres": {
                "base_link": [{"center": [0., 0., 0.06], "radius": 0.06}],
                "Shoulder_Link": [{"center": [0., 0., -0.10], "radius": 0.06}, {"center": [0., 0., -0.15], "radius": 0.05}],
                "HalfArm1_Link": [{"center": [0., 0., 0.], "radius": 0.055}, {"center": [0., -0.07, 0.], "radius": 0.055}, {"center": [0., -0.15, 0.], "radius": 0.055}],
                "HalfArm2_Link": [{"center": [0., 0., 0.], "radius": 0.055}, {"center": [0., 0., -0.07], "radius": 0.055}, {"center": [0., 0., -0.15], "radius": 0.055}],
                "ForeArm_Link": [{"center": [0., 0., 0.], "radius": 0.055}, {"center": [0., -0.07, 0.], "radius": 0.055}, {"center": [0., -0.17, 0.], "radius": 0.055}],
                "SphericalWrist1_Link": [{"center": [0., 0., 0.], "radius": 0.055}, {"center": [0., 0., -0.085], "radius": 0.055}],
                "SphericalWrist2_Link": [{"center": [0., 0., 0.], "radius": 0.05}, {"center": [0., -0.085, 0.], "radius": 0.05}],
                "Bracelet_Link": [{"center": [0., 0., -0.05], "radius": 0.04}, {"center": [0., -0.05, -0.05], "radius": 0.04}],
                "sharpa_mount_right": [{"center": [0., 0., 0.04], "radius": 0.040}],
                "right_hand_C_MC": [
                    {"center": [0., 0., 0.02], "radius": 0.050},
                    {"center": [0., 0., 0.05], "radius": 0.050},
                    {"center": [0.04, 0.02, 0.07], "radius": 0.040},
                    {"center": [0.04, 0.02, 0.10], "radius": 0.040},
                ],
            },
            "collision_sphere_buffer": 0.005,
            "self_collision_ignore": {
                "base_link": ["Shoulder_Link", "HalfArm1_Link"],
                "Shoulder_Link": ["HalfArm1_Link", "HalfArm2_Link"],
                "HalfArm1_Link": ["HalfArm2_Link", "ForeArm_Link"],
                "HalfArm2_Link": ["ForeArm_Link", "SphericalWrist1_Link"],
                "ForeArm_Link": ["SphericalWrist1_Link", "SphericalWrist2_Link"],
                "SphericalWrist1_Link": ["SphericalWrist2_Link", "Bracelet_Link"],
                "SphericalWrist2_Link": ["Bracelet_Link", "sharpa_mount_right", "right_hand_C_MC"],
                "Bracelet_Link": ["sharpa_mount_right", "right_hand_C_MC"],
                "sharpa_mount_right": ["right_hand_C_MC"],
            },
            "self_collision_buffer": {k: 0.0 for k in [
                "base_link", "Shoulder_Link", "HalfArm1_Link", "HalfArm2_Link",
                "ForeArm_Link", "SphericalWrist1_Link", "SphericalWrist2_Link",
                "Bracelet_Link", "sharpa_mount_right", "right_hand_C_MC",
            ]},
            "cspace": {
                "joint_names": _ARM_JOINTS,
                "default_joint_position": [float(v) for v in _init_joints_1d[_arm_cols]],
                "null_space_weight": [1.0] * 7,
                "cspace_distance_weight": [1.0] * 7,
                "max_acceleration": 10.0,
                "max_jerk": 100.0,
            },
            "use_global_cumul": True,
        }
    }
}

_device_cfg = CuroboDeviceCfg()
_planner_cfg = MotionPlannerCfg.create(
    robot=_ROBOT_CFG_DICT,
    ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
    num_ik_seeds=128,
    num_trajopt_seeds=12,
    use_cuda_graph=False,
    max_batch_size=1,
    max_goalset=1,
    device_cfg=_device_cfg,
    collision_cache={"cuboid": 5},
    optimizer_collision_activation_distance=0.05,
)
_planner = BatchMotionPlanner(_planner_cfg)
print("BatchMotionPlanner ready.", flush=True)

_device = _device_cfg.device
_q_arm_init = _init_joints_1d[_arm_cols].astype(np.float32)
_js_init = CuroboJointState.from_position(
    torch.tensor(_q_arm_init[None], dtype=torch.float32, device=_device),
    joint_names=_ARM_JOINTS,
)

# Orientation: same general strategy as cup_lift, top-down and tilted toward cube center.
_toward_robot_xy = _T_WB[:2] - _contact_world[:2]
_toward_robot_xy /= np.linalg.norm(_toward_robot_xy) + 1e-12
_toward_cube_xy = _obj_settled[:2] - _ee_contact_world[:2]
_toward_cube_xy /= np.linalg.norm(_toward_cube_xy) + 1e-12
_push_quat_world = _compute_topdown_quat_world(
    _toward_robot_xy,
    tilt_deg=_GRASP_TILT_DEG,
    toward_cube_xy=_toward_cube_xy,
)
_push_quat_base = _world_quat_to_base(_push_quat_world)
_standoff_base = _world_to_base(_ee_standoff_world)

print(f"[wrist] top-down quat world WXYZ: {[round(v,4) for v in _push_quat_world.tolist()]}", flush=True)
print(f"[wrist] tilt_deg={_GRASP_TILT_DEG}, pull=-Y (toward robot), contact=(+X,+Y) corner", flush=True)

# cuRobo always uses BASE frame internally.  Always pass base-frame coordinates.
# (Passing world-frame values to cuRobo makes it plan to base position = world values,
#  which sends the arm to the wrong world location — confirmed by run diagnostics.)
_USE_WORLD_FRAME = False

_ik_diag_base = _planner.ik_solver.solve_pose(
    GoalToolPose(
        tool_frames=_planner.tool_frames,
        position=torch.tensor(_standoff_base, dtype=torch.float32, device=_device).reshape(1, 1, 1, 1, 3),
        quaternion=torch.tensor(_push_quat_base, dtype=torch.float32, device=_device).reshape(1, 1, 1, 1, 4),
    ),
    return_seeds=64,
    current_state=_js_init,
)
_n_base = int(_ik_diag_base.success.sum().item())
print(f"IK BASE standoff: {_n_base}/64", flush=True)
if _n_base == 0:
    print("WARNING: standoff IK returned 0/64 — position may be out of workspace", flush=True)
print(f"=> Using BASE frame for IK goals", flush=True)


def _make_goal(pos_world: np.ndarray, quat_world: np.ndarray) -> GoalToolPose:
    if _USE_WORLD_FRAME:
        pos = pos_world
        quat = quat_world
    else:
        pos = _world_to_base(pos_world)
        quat = _world_quat_to_base(quat_world)
    return GoalToolPose(
        tool_frames=_planner.tool_frames,
        position=torch.tensor(pos, dtype=torch.float32, device=_device).reshape(1, 1, 1, 1, 3),
        quaternion=torch.tensor(quat, dtype=torch.float32, device=_device).reshape(1, 1, 1, 1, 4),
    )


def _js_to_waypoints(js, last_tstep=None):
    pos = js.position.cpu().numpy()
    while pos.ndim > 2:
        pos = pos[0]
    if pos.ndim == 1:
        pos = pos[None]
    if last_tstep is not None:
        t_val = int(np.asarray(last_tstep.cpu().numpy()).flat[0])
        t = max(1, min(t_val + 1, len(pos)))
        pos = pos[:t]
    return pos


def _solve_ik_chain(positions_world, quat_world, start_q_arm, n_seeds=64, label="chain"):
    results = []
    current_q = start_q_arm.astype(np.float64)
    n_ok = 0
    n_fail = 0
    for i, pos_w in enumerate(positions_world):
        js_warm = CuroboJointState.from_position(
            torch.tensor(current_q[None].astype(np.float32), dtype=torch.float32, device=_device),
            joint_names=_ARM_JOINTS,
        )
        ik_r = _planner.ik_solver.solve_pose(_make_goal(pos_w, quat_world), return_seeds=n_seeds, current_state=js_warm)
        if ik_r.success.any():
            sols = ik_r.js_solution.position.reshape(-1, len(_ARM_JOINTS)).detach().cpu().numpy()
            mask = ik_r.success.reshape(-1).detach().cpu().numpy().astype(bool)
            valid = sols[mask]
            d = np.linalg.norm(valid - current_q[None, :], axis=1)
            current_q = valid[int(np.argmin(d))].astype(np.float64)
            n_ok += 1
        else:
            print(f"  WARNING [{label}] IK failed at waypoint {i}: pos={pos_w.tolist()}", flush=True)
            n_fail += 1
        results.append(current_q.copy())
    print(f"  [{label}] IK chain: {n_ok}/{len(positions_world)} ok, {n_fail} fallback", flush=True)
    return results


def _interp_waypoints(ik_solutions, steps_between: int) -> np.ndarray:
    wps = []
    prev = np.asarray(ik_solutions[0], dtype=np.float64)
    for nxt in ik_solutions[1:]:
        nxt = np.asarray(nxt, dtype=np.float64)
        for t in np.linspace(0.0, 1.0, steps_between + 1)[1:]:
            wps.append((1.0 - t) * prev + t * nxt)
        prev = nxt
    if len(wps) == 0:
        return np.asarray([prev])
    return np.stack(wps)


def _execute_segment(label, waypoints, hand_joints, current_cmd, log_every=8, sim_steps_per_wp=None):
    n_sim = sim_steps_per_wp if sim_steps_per_wp is not None else _SIM_STEPS_PER_WP
    print(f"\n=== PHASE: {label.upper()} ({len(waypoints)} waypoints, {n_sim} sim/wp) ===", flush=True)
    for jn, val in hand_joints.items():
        if jn in _dof_idx:
            current_cmd[0, _dof_idx[jn]] = float(val)
    for step_i, q_wp in enumerate(waypoints):
        for k, col in enumerate(_arm_cols):
            current_cmd[0, col] = float(q_wp[k])
        for _ in range(n_sim):
            robots.set_joint_position_targets(current_cmd)
            _step()
        if step_i % log_every == 0:
            ee = _ee_link_world(stage)
            pn, pq = objects.get_world_poses()
            obj = np.asarray(pn, dtype=np.float64)[0]
            _pq0 = np.asarray(pq, dtype=np.float64)[0]  # [x,y,z,w] Isaac Sim convention
            _obj_traj.append({
                "pos": [round(float(v), 4) for v in obj.tolist()],
                "rot": [round(float(_pq0[3]),4), round(float(_pq0[0]),4),
                        round(float(_pq0[1]),4), round(float(_pq0[2]),4)],  # → wxyz
            })
            err = float(np.linalg.norm(q_wp - robots.get_joint_positions()[0][_arm_cols]))
            try:
                cf = np.asarray(objects.get_net_contact_forces(dt=1.0), dtype=np.float64).flatten()
                cf_mag = float(np.linalg.norm(cf[:3])) if len(cf) >= 3 else 0.0
            except Exception:
                cf_mag = 0.0
            print(json.dumps({
                "phase": label,
                "step": step_i,
                "progress": f"{step_i}/{len(waypoints)}",
                "ee_pos": [round(float(v), 4) for v in ee.tolist()] if ee is not None else None,
                "obj_pos": [round(float(v), 4) for v in obj.tolist()],
                "arm_err": round(err, 4),
                "contact_force_N": round(cf_mag, 4),
            }), flush=True)
            _log_finger_contacts(f"{label}_step{step_i}")

# -----------------------------------------------------------------------------
# Plan/solve trajectories
# -----------------------------------------------------------------------------
print("\n=== PLANNING TO STANDOFF ===", flush=True)
_standoff_goal = _make_goal(_ee_standoff_world, _push_quat_world)
_approach_wps = None
_traj_result = _planner.plan_pose(_standoff_goal, _js_init, max_attempts=3)
if _traj_result is not None and bool(_traj_result.success.any(dim=-1)[0].item()):
    _approach_wps = _js_to_waypoints(_traj_result.interpolated_trajectory, _traj_result.interpolated_last_tstep)
    print(f"plan_pose standoff success: {len(_approach_wps)} waypoints", flush=True)
else:
    print("plan_pose failed; trying direct IK interpolation to standoff", flush=True)
    ik_r = _planner.ik_solver.solve_pose(_standoff_goal, return_seeds=128, current_state=_js_init)
    if ik_r.success.any():
        sols = ik_r.js_solution.position.reshape(-1, len(_ARM_JOINTS)).detach().cpu().numpy()
        mask = ik_r.success.reshape(-1).detach().cpu().numpy().astype(bool)
        valid = sols[mask]
        d = np.linalg.norm(valid - _q_arm_init[None, :], axis=1)
        q_standoff = valid[int(np.argmin(d))]
        _approach_wps = np.stack([
            (1.0 - t) * _q_arm_init + t * q_standoff
            for t in np.linspace(1.0 / 40.0, 1.0, 40)
        ])
        print(f"direct IK standoff success: {len(_approach_wps)} waypoints", flush=True)
    else:
        raise RuntimeError("Could not solve standoff IK. Reduce tilt/overreach or move cube closer.")

_q_standoff_final = _approach_wps[-1].copy()

# Descend from standoff to contact.
_approach_kp_positions = np.stack([
    _ee_standoff_world + t * (_ee_contact_world - _ee_standoff_world)
    for t in np.linspace(0.0, 1.0, _N_IK_APPROACH_KPS + 1)[1:]
])
_approach_ik = _solve_ik_chain(
    _approach_kp_positions,
    _push_quat_world,
    _q_standoff_final,
    n_seeds=64,
    label="descend_to_posx_posy_corner",
)
_approach_fine_wps = _interp_waypoints([_q_standoff_final] + _approach_ik, _N_INTERP_BETWEEN)

_q_contact_final = _approach_ik[-1].copy()

# Pull in +X from the (-X,+Y) corner.
_pull_kp_positions = np.stack([
    _ee_contact_world + t * _pull_delta_world
    for t in np.linspace(0.0, 1.0, _N_IK_PULL_KPS + 1)[1:]
])
_pull_ik = _solve_ik_chain(
    _pull_kp_positions,
    _push_quat_world,
    _q_contact_final,
    n_seeds=64,
    label="pull_minus_y",
)
_pull_fine_wps = _interp_waypoints([_q_contact_final] + _pull_ik, _N_INTERP_BETWEEN)

print(f"Descend waypoints: {len(_approach_fine_wps)}", flush=True)
print(f"Pull waypoints:    {len(_pull_fine_wps)}", flush=True)

# -----------------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------------
_cur_cmd = np.tile(_init_joints_1d, (1, 1))

# Object trajectory recording — accumulated at log_every checkpoints in _execute_segment.
# Saved to trajectories/cube_pull_rotate_sharpa/ at the end of execution.
_obj_traj: list[dict] = []

print("\n--- Opening hand and settling ---", flush=True)
for jn, val in _HAND_OPEN.items():
    if jn in _dof_idx:
        _cur_cmd[0, _dof_idx[jn]] = float(val)
_settle(20, _cur_cmd)
_yaw_initial = _log_cube_state("pre-approach")

_actual_arm_q = robots.get_joint_positions()[0][_arm_cols]
_approach_wps = np.concatenate([[_actual_arm_q], _approach_wps])
_execute_segment("Standoff", _approach_wps, _HAND_OPEN, _cur_cmd, sim_steps_per_wp=_SIM_STEPS_PLAN_WP)
_settle(_SETTLE_STEPS, _cur_cmd)
_step(render_force=True)
_log_cube_state("post-standoff", _yaw_initial)

print("\n--- Curling fingers for fingertip contact ---", flush=True)
for jn, val in _HAND_PUSH.items():
    if jn in _dof_idx:
        _cur_cmd[0, _dof_idx[jn]] = float(val)
_settle(20, _cur_cmd)

_execute_segment("DescendContact", _approach_fine_wps, _HAND_PUSH, _cur_cmd, log_every=4)
_settle(_SETTLE_STEPS, _cur_cmd)
_step(render_force=True)
_log_cube_state("post-contact", _yaw_initial)
_log_finger_contacts("post-contact")

_execute_segment("PullMinusY", _pull_fine_wps, _HAND_PUSH, _cur_cmd, log_every=4)
_settle(_SETTLE_STEPS, _cur_cmd)
_step(render_force=True)
_log_finger_contacts("post-pull")
_log_cube_state("post-pull", _yaw_initial)

# -----------------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------------
print("\n=== FINAL REPORT ===", flush=True)
_pos_final, _ori_final = objects.get_world_poses()
_cube_final = np.asarray(_pos_final, dtype=np.float64)[0]
_q_final = np.asarray(_ori_final, dtype=np.float64)[0]
_yaw_final = _quat_wxyz_to_yaw(_q_final)
_delta_yaw = (_yaw_final - _yaw_initial + np.pi) % (2.0 * np.pi) - np.pi
_xy_disp = float(np.linalg.norm(_cube_final[:2] - _obj_settled[:2]))
_z_change = float(_cube_final[2] - _obj_settled[2])

print(f"Cube initial pos:      {[round(float(v),4) for v in _obj_settled.tolist()]}", flush=True)
print(f"Cube final pos:        {[round(float(v),4) for v in _cube_final.tolist()]}", flush=True)
print(f"Cube XY displacement:  {_xy_disp:.4f} m", flush=True)
print(f"Cube ΔZ:               {_z_change:+.4f} m", flush=True)
print(f"Initial yaw:           {np.degrees(_yaw_initial):.2f} deg", flush=True)
print(f"Final yaw:             {np.degrees(_yaw_final):.2f} deg", flush=True)
print(f"Yaw change:            {np.degrees(_delta_yaw):.2f} deg", flush=True)
print(f"Rotation success >45°: {abs(np.degrees(_delta_yaw)) > 45.0}", flush=True)

# -----------------------------------------------------------------------------
# Save object trajectory
# -----------------------------------------------------------------------------
try:
    from datetime import datetime as _dt
    _traj_dir = Path(__file__).parent.parent.parent / "trajectories" / "cube_pull_rotate_sharpa"
    _traj_dir.mkdir(parents=True, exist_ok=True)
    _traj_path = _traj_dir / f"{_dt.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json as _json_traj
    with _traj_path.open("w") as _f:
        _json_traj.dump(_obj_traj, _f, indent=2)
    print(f"TRAJECTORY_SAVED: {_traj_path}  ({len(_obj_traj)} frames)", flush=True)
except Exception as _te:
    print(f"TRAJECTORY_SAVE_ERROR: {_te}", flush=True)

# -----------------------------------------------------------------------------
# Encode video
# -----------------------------------------------------------------------------
try:
    import glob
    frames = sorted(glob.glob(os.path.join(_FRAMES_DIR, "frame_*.png")))
    if frames:
        _vid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cube_negx_highy_pull_posx_rotate_sharpa.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", "30", "-pattern_type", "glob",
                "-i", os.path.join(_FRAMES_DIR, "frame_*.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", _vid,
            ],
        )
        print(f"VIDEO: {_vid}", flush=True)
except Exception as exc:
    print(f"VIDEO_ERROR: {exc}", flush=True)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
try:
    _rgb.detach([_rp_cam])
except Exception:
    pass
try:
    _rp_cam.destroy()
except Exception:
    pass
try:
    _planner.destroy()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
except Exception:
    pass
try:
    rep.orchestrator.stop()
except Exception:
    pass
try:
    simulation_app.close()
except Exception:
    pass
