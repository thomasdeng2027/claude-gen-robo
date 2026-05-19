"""
cylinder_lift.py — Three-phase grasp (approach → grasp → lift) using cuRobo v2
BatchMotionPlanner, with the SAM-reconstructed pourwater twin mesh as the object.

Adapted to use the realkinova wuji hand instead of the xhand.
Side grasp: EE Z → approach dir (robot→object), EE Y → world +Z (thumb up).

Run:
    python cup_lift_wuji.py
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

_OMNI_USER_HOME  = "/tmp/isaac_user_tdeng23"
_WARP_CACHE_PATH = "/tmp/warp_cache_tdeng23"
Path(_OMNI_USER_HOME).mkdir(parents=True, exist_ok=True)
Path(_WARP_CACHE_PATH).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMNI_USER_HOME",              _OMNI_USER_HOME)
os.environ.setdefault("WARP_CACHE_PATH",             _WARP_CACHE_PATH)
os.environ.setdefault("XDG_DATA_HOME",               _OMNI_USER_HOME + "/.local/share")
os.environ.setdefault("XDG_CACHE_HOME",              _OMNI_USER_HOME + "/.cache")
os.environ.setdefault("VK_ICD_FILENAMES",            "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("OMNI_STRUCTUREDLOG_ENABLED",  "0")
os.environ.setdefault("CUROBO_KERNEL_BACKEND",       "pybind")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",        "0")  # prevent MGPU sync timeout on busy multi-GPU nodes

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT    = str(Path(__file__).resolve().parents[2])
_PIPELINE_DIR = _REPO_ROOT + "/src"
_IK_URDF_PATH = _REPO_ROOT + "/assets/realkinova_wuji_hand/realkinova_wuji_hand_right.urdf"
_OBJ_USD_PATH = _REPO_ROOT + "/src/assets/cup/cup.usd"

_CUROBO_V2_ROOT = "/home/tdeng23/projects/curobo"
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

# Disable DLSS before any render products are created.
# On A100 GCP, DLSS downscales (640,480) → (320,240) internally; 240 < 300 minimum
# makes DLSS output all-black frames.  Switch to FXAA (no minimum resolution).
import carb as _carb
_cs = _carb.settings.get_settings()
_cs.set("/rtx/post/aa/op", 2)          # 2 = FXAA in Omniverse RTX
_cs.set("/rtx/post/dlss/execMode", 0)  # belt-and-suspenders: DLSS exec mode off
print(f"[aa] op={_cs.get('/rtx/post/aa/op')}  dlss_exec={_cs.get('/rtx/post/dlss/execMode')}", flush=True)

import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom, UsdPhysics

# ── Scene constants ────────────────────────────────────────────────────────────
_OBJ_POS_INIT    = np.array([-0.094117, 0.476573, 0.0])
_OBJ_ORIENT_WXYZ = (1.0, 0.0, 0.0, 0.0)  # identity
_OBJ_SCALE       = 2.0   # cup.usd is in metres
_OBJ_MASS        = 0.01 # 50 g, matches cup.usd MassAPI

_ROBOT_PRIM = "/World/envs/env_0/Robot"
_OBJ_PRIM   = "/World/envs/env_0/Object"

_RENDER_EVERY = 8
_FRAMES_DIR   = os.environ.get("ISAAC_FRAMES_DIR", "/tmp/isaac_frames_cup_lift_wuji")
import shutil as _shutil
if Path(_FRAMES_DIR).exists():
    _shutil.rmtree(_FRAMES_DIR)
Path(_FRAMES_DIR).mkdir(parents=True, exist_ok=True)

# Grasp parameters — top-down rim pinch.
# All offsets are in world frame: +X = right, +Y = away from robot, +Z = up.
# Tune each axis independently to place fingertips on the near-side rim lip.
_GRASP_OFFSET_X_M          =  0.03 # world X offset from cup center (- = in towards cup)
_GRASP_OFFSET_Y_M          = -0.02  # world Y offset from cup center (+ = away, - = toward robot)
_GRASP_OFFSET_Z_M          =  0.45 # world Z offset from cup center (+ = up, toward rim)
# Tilt the EE from straight-down toward the cup center so fingers angle over the rim.
# 0 = perfectly vertical; 20 = 20° lean inward toward the cup.
_GRASP_TILT_DEG            = -30
_LIFT_HEIGHT_M             = 0.25
_SIM_STEPS_PER_WP          = 8
_SETTLE_STEPS              = 10
_HAND_CLOSE_STEPS          = 80

# ── Initial robot configuration ────────────────────────────────────────────────
_INIT_ARM = {
    "Actuator1": -1.529021, "Actuator2":  1.200000, "Actuator3": 0.164424,
    "Actuator4":  0.600000, "Actuator5":  1.254565, "Actuator6": -1.063086,
    "Actuator7":  0.235436,
}
_INIT_HAND = {
    "right_finger1_joint1": 0.04,  # lower limit is 0.0368
    "right_finger1_joint2": 0.0,
    "right_finger1_joint3": 0.0,
    "right_finger1_joint4": 0.0,
    "right_finger2_joint1": 0.04,
    "right_finger2_joint2": 0.0,
    "right_finger2_joint3": 0.0,
    "right_finger2_joint4": 0.0,
    "right_finger3_joint1": 0.04,
    "right_finger3_joint2": 0.0,
    "right_finger3_joint3": 0.0,
    "right_finger3_joint4": 0.0,
    "right_finger4_joint1": 0.04,
    "right_finger4_joint2": 0.0,
    "right_finger4_joint3": 0.0,
    "right_finger4_joint4": 0.0,
    "right_finger5_joint1": 0.04,
    "right_finger5_joint2": 0.0,
    "right_finger5_joint3": 0.0,
    "right_finger5_joint4": 0.0,
}

_HAND_OPEN = {
    "right_finger1_joint1": 0.04,
    "right_finger1_joint2": 0.0,
    "right_finger1_joint3": 0.0,
    "right_finger1_joint4": 0.0,
    "right_finger2_joint1": 0.04,
    "right_finger2_joint2": 0.0,
    "right_finger2_joint3": 0.0,
    "right_finger2_joint4": 0.0,
    "right_finger3_joint1": 0.04,
    "right_finger3_joint2": 0.0,
    "right_finger3_joint3": 0.0,
    "right_finger3_joint4": 0.0,
    "right_finger4_joint1": 0.04,
    "right_finger4_joint2": 0.0,
    "right_finger4_joint3": 0.0,
    "right_finger4_joint4": 0.0,
    "right_finger5_joint1": 0.04,
    "right_finger5_joint2": 0.0,
    "right_finger5_joint3": 0.0,
    "right_finger5_joint4": 0.0,
}

# All joints close simultaneously.
_HAND_CLOSED = {
    "right_finger1_joint1": 1.4,
    "right_finger1_joint2": 0.7,
    "right_finger1_joint3": 1.2,
    "right_finger1_joint4": 1.2,
    "right_finger2_joint1": 1.4,
    "right_finger2_joint2": 0.7,
    "right_finger2_joint3": 1.2,
    "right_finger2_joint4": 1.2,
    "right_finger3_joint1": 1.4,
    "right_finger3_joint2": 0.7,
    "right_finger3_joint3": 1.2,
    "right_finger3_joint4": 1.2,
    "right_finger4_joint1": 1.4,
    "right_finger4_joint2": 0.7,
    "right_finger4_joint3": 1.2,
    "right_finger4_joint4": 1.2,
    "right_finger5_joint1": 1.4,
    "right_finger5_joint2": 0.7,
    "right_finger5_joint3": 1.2,
    "right_finger5_joint4": 1.2,
}

# Partial pre-grasp: fingers curl ~40 % of full close so they bracket the cup
# without squeezing, allowing the arm to push down and seat the rim.
_HAND_PARTIAL = {
    "right_finger1_joint1": 0.6,
    "right_finger1_joint2": 0.3,
    "right_finger1_joint3": 0.5,
    "right_finger1_joint4": 0.5,
    "right_finger2_joint1": 0.6,
    "right_finger2_joint2": 0.3,
    "right_finger2_joint3": 0.5,
    "right_finger2_joint4": 0.5,
    "right_finger3_joint1": 0.6,
    "right_finger3_joint2": 0.3,
    "right_finger3_joint3": 0.5,
    "right_finger3_joint4": 0.5,
    "right_finger4_joint1": 0.6,
    "right_finger4_joint2": 0.3,
    "right_finger4_joint3": 0.5,
    "right_finger4_joint4": 0.5,
    "right_finger5_joint1": 0.6,
    "right_finger5_joint2": 0.3,
    "right_finger5_joint3": 0.5,
    "right_finger5_joint4": 0.5,
}

_HAND_PARTIAL_STEPS = 40   # sim steps for partial close
_PUSH_DOWN_M        = 0.1  # metres to lower the arm after partial close


# ── Stage helpers ──────────────────────────────────────────────────────────────

def _prim_world_pos(stage, path: str):
    try:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
            t  = xf.ExtractTranslation()
            pos = np.array([float(t[0]), float(t[1]), float(t[2])])
            if np.any(pos != 0):
                return pos
    except Exception:
        pass
    return None


def _ee_link_world(stage, robot_prim=None):
    """Return world position of right_hand_link — the cuRobo tool frame."""
    prefix = (_ROBOT_PRIM if robot_prim is None else robot_prim).rstrip("/") + "/"
    return _prim_world_pos(stage, prefix + "right_hand_link")


def _knuckle_centroid(stage, robot_prim=None):
    prefix = (_ROBOT_PRIM if robot_prim is None else robot_prim).rstrip("/") + "/"
    knuckle_links = [
        "right_finger1_link1", "right_finger2_link1",
        "right_finger3_link1", "right_finger4_link1",
        "right_finger5_link1",
    ]
    wrist_links = ["right_hand_link"]
    knuckle_pos = [p for p in (_prim_world_pos(stage, prefix + ln) for ln in knuckle_links) if p is not None]
    if len(knuckle_pos) >= 2:
        centroid = np.mean(knuckle_pos, axis=0)
        for wl in wrist_links:
            wrist = _prim_world_pos(stage, prefix + wl)
            if wrist is not None:
                return 0.7 * centroid + 0.3 * wrist
        return centroid
    for ln in wrist_links + ["right_finger1_link1"]:
        p = _prim_world_pos(stage, prefix + ln)
        if p is not None:
            return p
    return None


# ── Build scene ────────────────────────────────────────────────────────────────

world = World(stage_units_in_meters=1.0)
# add_default_ground_plane() creates the standard Isaac Sim ground + default lighting/sky.
# The manual GroundPlane(prim_path=...) skips the default lighting setup, causing black frames.
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

try:
    _status, _cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    _cfg.merge_fixed_joints    = False
    _cfg.fix_base              = True
    _cfg.import_inertia_tensor = True
    _cfg.distance_scale        = 1.0
    _cfg.create_physics_scene  = False
    _ok, _result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=_IK_URDF_PATH,
        import_config=_cfg,
        dest_path="",
    )
    print(f"URDFParseAndImportFile: ok={_ok} result={_result}", flush=True)
    if not _ok:
        raise RuntimeError(f"URDFParseAndImportFile returned failure: ok={_ok}")
    _ROBOT_PRIM = str(_result).strip() if _result else _ROBOT_PRIM
    _rob_prim = stage.GetPrimAtPath(_ROBOT_PRIM)
    if not _rob_prim.IsValid():
        raise RuntimeError(f"Robot prim not valid at {_ROBOT_PRIM} after import.")
    print(f"ROBOT_LOADED: prim {_ROBOT_PRIM}", flush=True)
except Exception as _e:
    print(f"URDF import failed: {_e}", flush=True)
    raise

add_reference_to_stage(usd_path=_OBJ_USD_PATH, prim_path=_OBJ_PRIM)
_bp = stage.GetPrimAtPath(_OBJ_PRIM)
_xf = UsdGeom.Xformable(_bp)
_xf.ClearXformOpOrder()
_xf.AddTranslateOp().Set(Gf.Vec3d(*_OBJ_POS_INIT.tolist()))
w, x, y, z = _OBJ_ORIENT_WXYZ
_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
_xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(_OBJ_SCALE, _OBJ_SCALE, _OBJ_SCALE))
UsdPhysics.MassAPI.Apply(_bp).CreateMassAttr().Set(_OBJ_MASS)

robots  = Articulation(prim_paths_expr=_ROBOT_PRIM, name="robots", reset_xform_properties=False)
objects = RigidPrim(prim_paths_expr=_OBJ_PRIM, name="objects",
                    reset_xform_properties=False, track_contact_forces=True)
world.scene.add(robots)
world.scene.add(objects)

# Per-finger tip contact sensing — must be added before world.reset().
_FINGER_TIP_LINKS = {
    "finger1": "right_finger1_tip_link",
    "finger2": "right_finger2_tip_link",
    "finger3": "right_finger3_tip_link",
    "finger4": "right_finger4_tip_link",
    "finger5": "right_finger5_tip_link",
}
_finger_contact_prims = {}
for _fname, _ln in _FINGER_TIP_LINKS.items():
    _fpath = _ROBOT_PRIM.rstrip("/") + "/" + _ln
    try:
        _fp = RigidPrim(prim_paths_expr=_fpath, name=f"finger_{_fname}",
                        track_contact_forces=True)
        world.scene.add(_fp)
        _finger_contact_prims[_fname] = _fp
        print(f"Finger contact prim added: {_fname} @ {_fpath}", flush=True)
    except Exception as _fe:
        print(f"Finger contact prim '{_fname}' unavailable: {_fe}", flush=True)

world.reset()
robots.initialize()
world.step(render=False)

_dof_names = list(robots.dof_names)
_n_dof     = robots.num_dof
_dof_idx   = {n: i for i, n in enumerate(_dof_names)}
print(f"DOF names ({_n_dof}): {_dof_names}", flush=True)

_ARM_JOINTS = [f"Actuator{i}" for i in range(1, 8)]
_arm_cols   = [_dof_idx[jn] for jn in _ARM_JOINTS if jn in _dof_idx]

_init_joints_1d = np.zeros(_n_dof)
for jn, val in {**_INIT_ARM, **_INIT_HAND}.items():
    if jn in _dof_idx:
        _init_joints_1d[_dof_idx[jn]] = val

robots.set_joint_positions(np.tile(_init_joints_1d, (1, 1)))
robots.set_joint_velocities(np.zeros((1, _n_dof)))

try:
    kps = np.tile(np.array([8000.0]*7 + [20000.0]*(_n_dof-7)), (1, 1))
    kds = np.tile(np.array([ 400.0]*7 + [   600.0]*(_n_dof-7)), (1, 1))
    robots.set_gains(kps=kps, kds=kds)
except Exception as _eg:
    print(f"PD gains warning: {_eg}", flush=True)

# ── Camera + renderer ──────────────────────────────────────────────────────────
_camera = rep.create.camera(position=(1.2, 0.8, 1.5), look_at=(-0.05, 0.45, 0.1))
_rp_cam = rep.create.render_product(_camera, (640, 480))
_rgb    = rep.AnnotatorRegistry.get_annotator("rgb")
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
                # Handle warp arrays (GPU-backed): .numpy() syncs & copies to CPU.
                # Plain numpy: ascontiguousarray is sufficient.
                if hasattr(img, 'numpy') and not isinstance(img, np.ndarray):
                    img_np = img.numpy()
                else:
                    img_np = np.asarray(img)
                img_cpu = np.ascontiguousarray(
                    img_np[..., :3] if img_np.ndim == 3 and img_np.shape[-1] == 4 else img_np
                )
                if _frame_counter[0] == 8:  # log the cpu copy on the first rendered frame
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

# ── Settle ─────────────────────────────────────────────────────────────────────
print("Settling...", flush=True)
_settle_cmd = np.tile(_init_joints_1d, (1, 1))
robots.set_joint_positions(_settle_cmd)
robots.set_joint_velocities(np.zeros((1, _n_dof)))
for _ in range(20):
    robots.set_joint_position_targets(_settle_cmd)
    _step()

_pos_raw, _ = objects.get_world_poses()
_obj_settled = np.asarray(_pos_raw, dtype=np.float64)[0]
print(f"Object settled at: {_obj_settled.tolist()}", flush=True)

print("\nImporting cuRobo v2...", flush=True)
import torch
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState as CuroboJointState
from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.types.device_cfg import DeviceCfg as CuroboDeviceCfg
from curobo._src.geom.types import SceneCfg as CuroboSceneCfg, Cuboid as CuroboCuboid
from curobo._src.util.trajectory import TrajInterpolationType
from motion_planner_batch import BatchMotionPlanner
print("cuRobo v2 imports OK", flush=True)

# ── Frame-conversion helpers ───────────────────────────────────────────────────
_T_WB: np.ndarray = np.zeros(3, dtype=np.float64)
_R_WB: np.ndarray = np.eye(3, dtype=np.float64)


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
    trace = R[0,0]+R[1,1]+R[2,2]
    if trace > 0:
        s=0.5/np.sqrt(trace+1); w=0.25/s; x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z],dtype=np.float64); return q/(np.linalg.norm(q)+1e-12)


def _qmul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dtype=np.float64)


def _world_quat_to_base(q_world):
    # q_base = q_{base←world} ⊗ q_world  (compose: first world→EE, then base←world)
    # _R_WB = R_{base←world}, so _rmat_to_wxyz(_R_WB) = q_{base←world}
    q_wb = _rmat_to_wxyz(_R_WB)
    q = _qmul(q_wb, np.asarray(q_world, dtype=np.float64))
    return q / (np.linalg.norm(q) + 1e-12)


def _compute_topdown_grasp_quat_world(toward_robot_dir_xy, tilt_deg=0.0,
                                       toward_cup_xy=None):
    """Top-down grasp: EE Z → world -Z (straight down), EE Y → toward robot.

    tilt_deg > 0 tilts the EE Z axis toward toward_cup_xy (the horizontal
    direction from the EE target to the cup center), so the palm angles
    inward and the fingers hook over the rim rather than passing straight past it.

    toward_cup_xy must be provided when tilt_deg != 0.
    """
    z = np.array([0., 0., -1.])
    y = np.array([toward_robot_dir_xy[0], toward_robot_dir_xy[1], 0.])
    y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z); x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-12

    if tilt_deg != 0.0 and toward_cup_xy is not None:
        # Tilt z toward the cup by rotating around k = cross(z, toward_cup_horiz).
        # Rodrigues gives: z_new = z*cos(t) + toward_cup_horiz*sin(t)
        # which cleanly tilts the approach axis in the cup direction.
        d = np.array([toward_cup_xy[0], toward_cup_xy[1], 0.], dtype=np.float64)
        d /= np.linalg.norm(d) + 1e-12
        t = np.deg2rad(tilt_deg)
        z = z * np.cos(t) + d * np.sin(t)
        z /= np.linalg.norm(z) + 1e-12
        # Re-orthogonalise x and y
        x = np.cross(y, z); x /= np.linalg.norm(x) + 1e-12
        y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-12

    return _rmat_to_wxyz(np.stack([x, y, z], axis=1))


# ── Finger tip position logging ───────────────────────────────────────────────

def _log_finger_tips(label: str):
    """Log world-frame positions of fingertip and palm links for alignment debugging."""
    prefix = _ROBOT_PRIM.rstrip("/") + "/"
    links = {
        "palm":     "right_hand_link",
        "f1_base":  "right_finger1_link1",
        "f1_tip":   "right_finger1_tip_link",
        "f2_base":  "right_finger2_link1",
        "f2_tip":   "right_finger2_tip_link",
        "f3_tip":   "right_finger3_tip_link",
        "f4_tip":   "right_finger4_tip_link",
        "f5_tip":   "right_finger5_tip_link",
    }
    row = {"event": label}
    for name, ln in links.items():
        p = _prim_world_pos(stage, prefix + ln)
        row[name] = [round(float(v), 4) for v in p.tolist()] if p is not None else None
    try:
        _op, _ = objects.get_world_poses()
        row["obj_pos"] = [round(float(v), 4) for v in np.asarray(_op, dtype=np.float64)[0].tolist()]
    except Exception:
        pass
    print(json.dumps(row), flush=True)


# ── Per-finger contact logging ────────────────────────────────────────────────

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


# ── Execution helpers ─────────────────────────────────────────────────────────

def _settle(n, cur_cmd):
    for _ in range(n):
        robots.set_joint_position_targets(cur_cmd)
        _step()


def _execute_segment(label, waypoints, hand_joints, current_cmd, log_every=10,
                     early_stop_joint_err=None, early_stop_ee_to_obj_m=None,
                     planned_ee_world=None, sim_steps_per_wp=None):
    _n_sim = sim_steps_per_wp if sim_steps_per_wp is not None else _SIM_STEPS_PER_WP
    print(f"\n=== PHASE: {label.upper()} ({len(waypoints)} waypoints, {_n_sim} sim/wp) ===\n", flush=True)
    for jn, val in hand_joints.items():
        if jn in _dof_idx:
            current_cmd[0, _dof_idx[jn]] = float(val)
    half = len(waypoints) // 2
    for step, q_wp in enumerate(waypoints):
        for k, col in enumerate(_arm_cols):
            current_cmd[0, col] = float(q_wp[k])
        for _ in range(_n_sim):
            robots.set_joint_position_targets(current_cmd)
            _step()
        ee = _ee_link_world(stage)
        _pn, _pq = objects.get_world_poses()
        obj = np.asarray(_pn, dtype=np.float64)[0]
        d = float(np.linalg.norm(ee - obj)) if ee is not None else -1.0
        if step % log_every == 0:
            _pq0 = np.asarray(_pq, dtype=np.float64)[0]  # [x,y,z,w] Isaac Sim convention
            _obj_traj.append({
                "pos": [round(v, 4) for v in obj.tolist()],
                "rot": [round(float(_pq0[3]),4), round(float(_pq0[0]),4),
                        round(float(_pq0[1]),4), round(float(_pq0[2]),4)],  # → wxyz
            })
            err = float(np.linalg.norm(q_wp - robots.get_joint_positions()[0][_arm_cols]))
            try:
                _cf = objects.get_net_contact_forces(dt=1.0)
                _cf_arr = np.asarray(_cf, dtype=np.float64).flatten()
                _cf_mag = float(np.linalg.norm(_cf_arr[:3])) if len(_cf_arr) >= 3 else 0.0
                _cf_vec = [round(float(v), 3) for v in _cf_arr[:3]]
            except Exception:
                _cf_mag, _cf_vec = 0.0, None
            entry = {
                "phase": label,
                "step": step,
                "progress": f"{step}/{len(waypoints)}",
                "ee_z": round(ee[2], 4) if ee is not None else None,
                "ee_pos": [round(v, 4) for v in ee.tolist()] if ee is not None else None,
                "obj_pos": [round(v, 4) for v in obj.tolist()],
                "ee_to_obj": round(d, 4),
                "arm_err": round(err, 4),
                "contact_force_N": round(_cf_mag, 4),
                "contact_force_vec": _cf_vec,
            }
            if planned_ee_world is not None and step < len(planned_ee_world):
                pee = planned_ee_world[step]
                entry["plan_ee_z"] = round(float(pee[2]), 4)
                entry["plan_ee_pos"] = [round(float(v), 4) for v in pee.tolist()]
                if ee is not None:
                    entry["tracking_err_m"] = round(float(np.linalg.norm(ee - pee)), 4)
            print(json.dumps(entry), flush=True)
            _log_finger_contacts(f"{label}_step{step}")
        if step >= half:
            if early_stop_ee_to_obj_m is not None and d >= 0 and d < early_stop_ee_to_obj_m:
                print(f"  [{label}] Proximity stop at step {step}/{len(waypoints)}: "
                      f"ee_to_obj={d:.4f}m < {early_stop_ee_to_obj_m}m", flush=True)
                break
            if early_stop_joint_err is not None:
                _q_actual = robots.get_joint_positions()[0][_arm_cols]
                _final_err = float(np.linalg.norm(waypoints[-1] - _q_actual))
                if _final_err < early_stop_joint_err:
                    print(f"  [{label}] Joint-err stop at step {step}/{len(waypoints)}: "
                          f"final_wp_err={_final_err:.4f} rad < {early_stop_joint_err}", flush=True)
                    break


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


# ── Populate world↔base transform from stage ─────────────────────────────────
_T_WB, _R_raw = _read_base_world_pose(stage, _ROBOT_PRIM)
_R_WB = _R_raw.T
print(f"Base link world pos: {_T_WB.tolist()}", flush=True)
print(f"Base link world rot (R_WB):\n{_R_WB}", flush=True)

# ── Approach direction + grasp pose (top-down onto cup rim) ───────────────────
# Grasp target: cup center + per-axis world-frame offsets.
_grasp_pos_world = np.array([
    _obj_settled[0] + _GRASP_OFFSET_X_M,
    _obj_settled[1] + _GRASP_OFFSET_Y_M,
    _obj_settled[2] + _GRASP_OFFSET_Z_M,
])

# Orient EE top-down, thumb facing toward robot base (keeps grip stable).
_toward_robot_xy = _T_WB[:2] - _obj_settled[:2]
_toward_robot_xy /= np.linalg.norm(_toward_robot_xy) + 1e-12
# Tilt toward the cup center: direction from EE target to cup center in XY.
_toward_cup_xy = _obj_settled[:2] - _grasp_pos_world[:2]
_toward_cup_xy /= np.linalg.norm(_toward_cup_xy) + 1e-12
print(f"Toward-cup tilt direction (world XY): {[round(v,4) for v in _toward_cup_xy.tolist()]}", flush=True)
_grasp_quat_world = _compute_topdown_grasp_quat_world(
    _toward_robot_xy, tilt_deg=_GRASP_TILT_DEG, toward_cup_xy=_toward_cup_xy)
_grasp_pos_base   = _world_to_base(_grasp_pos_world)
_grasp_quat_base  = _world_quat_to_base(_grasp_quat_world)

print(f"Rim target (world): {[round(v,4) for v in _grasp_pos_world.tolist()]}", flush=True)
print(f"Grasp quat (world): {[round(v,4) for v in _grasp_quat_world.tolist()]}", flush=True)
print(f"Grasp pos  (base):  {[round(v,4) for v in _grasp_pos_base.tolist()]}", flush=True)
print(f"Grasp quat (base):  {[round(v,4) for v in _grasp_quat_base.tolist()]}", flush=True)

# ── BatchMotionPlanner ────────────────────────────────────────────────────────
print("\nInitializing BatchMotionPlanner...", flush=True)

_ROBOT_CFG_DICT = {
    "robot_cfg": {
        "kinematics": {
            "urdf_path":        _IK_URDF_PATH,
            "asset_root_path":  str(Path(_IK_URDF_PATH).parent),
            "base_link":        "base_link",
            "tool_frames":      ["right_hand_link"],
            "grasp_contact_link_names": ["right_hand_link"],
            "collision_link_names": [
                "base_link", "Shoulder_Link", "HalfArm1_Link", "HalfArm2_Link",
                "ForeArm_Link", "SphericalWrist1_Link", "SphericalWrist2_Link",
                "Bracelet_Link", "right_wuji_mount_link", "right_palm_link",
            ],
            "collision_spheres": {
                "base_link":            [{"center":[0.,0.,0.06],"radius":0.06}],
                "Shoulder_Link":        [{"center":[0.,0.,-0.10],"radius":0.06},
                                         {"center":[0.,0.,-0.15],"radius":0.05}],
                "HalfArm1_Link":        [{"center":[0., 0.,  0.],"radius":0.055},
                                         {"center":[0.,-0.07,0.],"radius":0.055},
                                         {"center":[0.,-0.15,0.],"radius":0.055}],
                "HalfArm2_Link":        [{"center":[0.,0., 0.],"radius":0.055},
                                         {"center":[0.,0.,-0.07],"radius":0.055},
                                         {"center":[0.,0.,-0.15],"radius":0.055}],
                "ForeArm_Link":         [{"center":[0., 0.,  0.],"radius":0.055},
                                         {"center":[0.,-0.07,0.],"radius":0.055},
                                         {"center":[0.,-0.17,0.],"radius":0.055}],
                "SphericalWrist1_Link": [{"center":[0.,0.,0.],"radius":0.055},
                                         {"center":[0.,0.,-0.085],"radius":0.055}],
                "SphericalWrist2_Link": [{"center":[0., 0.,   0.],"radius":0.05},
                                         {"center":[0.,-0.085,0.],"radius":0.05}],
                "Bracelet_Link":        [{"center":[0., 0.,-0.05],"radius":0.04},
                                         {"center":[0.,-0.05,-0.05],"radius":0.04}],
                "right_wuji_mount_link": [{"center":[0., 0., 0.04],"radius":0.040}],
                "right_palm_link":      [{"center":[0., 0., 0.02],"radius":0.050},
                                         {"center":[0., 0., 0.05],"radius":0.050},
                                         {"center":[0.04, 0.02, 0.07],"radius":0.040},
                                         {"center":[0.04, 0.02, 0.10],"radius":0.040}],
            },
            "collision_sphere_buffer": 0.005,
            "self_collision_ignore": {
                "base_link":            ["Shoulder_Link", "HalfArm1_Link"],
                "Shoulder_Link":        ["HalfArm1_Link", "HalfArm2_Link"],
                "HalfArm1_Link":        ["HalfArm2_Link", "ForeArm_Link"],
                "HalfArm2_Link":        ["ForeArm_Link", "SphericalWrist1_Link"],
                "ForeArm_Link":         ["SphericalWrist1_Link", "SphericalWrist2_Link"],
                "SphericalWrist1_Link": ["SphericalWrist2_Link", "Bracelet_Link"],
                "SphericalWrist2_Link": ["Bracelet_Link", "right_wuji_mount_link", "right_palm_link"],
                "Bracelet_Link":        ["right_wuji_mount_link", "right_palm_link"],
                "right_wuji_mount_link": ["right_palm_link"],
            },
            "self_collision_buffer": {k:0.0 for k in [
                "base_link","Shoulder_Link","HalfArm1_Link","HalfArm2_Link",
                "ForeArm_Link","SphericalWrist1_Link","SphericalWrist2_Link",
                "Bracelet_Link","right_wuji_mount_link","right_palm_link"]},
            "cspace": {
                "joint_names":            _ARM_JOINTS,
                "default_joint_position": [float(v) for v in _init_joints_1d[_arm_cols]],
                "null_space_weight":      [1.0]*7,
                "cspace_distance_weight": [1.0]*7,
                "max_acceleration":       10.0,
                "max_jerk":               100.0,
            },
            "use_global_cumul": True,
        }
    }
}

_device_cfg  = CuroboDeviceCfg()
_planner_cfg = MotionPlannerCfg.create(
    robot=_ROBOT_CFG_DICT,
    ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
    num_ik_seeds=128, num_trajopt_seeds=12,
    use_cuda_graph=False, max_batch_size=1, max_goalset=1,
    device_cfg=_device_cfg,
    collision_cache={"cuboid": 5},
    optimizer_collision_activation_distance=0.05,
)
_planner = BatchMotionPlanner(_planner_cfg)
print("BatchMotionPlanner ready.", flush=True)

# ── Transform sanity check ────────────────────────────────────────────────────
_device = _device_cfg.device
_q_arm_init = _init_joints_1d[_arm_cols].astype(np.float32)
_js_check = CuroboJointState.from_position(
    torch.tensor(_q_arm_init[None], dtype=torch.float32, device=_device),
    joint_names=_ARM_JOINTS,
)
_kin_check   = _planner.compute_kinematics(_js_check)
_ee_base_chk = _kin_check.tool_poses.position[0, 0, 0].cpu().numpy().astype(np.float64)
_ee_world_chk = _base_to_world(_ee_base_chk)
_ee_sim_chk   = _ee_link_world(stage)
_xfm_err = float(np.linalg.norm(_ee_world_chk - _ee_sim_chk)) if _ee_sim_chk is not None else -1.0
print(f"Transform check — FK→world: {[round(v,4) for v in _ee_world_chk.tolist()]}", flush=True)
print(f"Transform check — sim EE:   {[round(v,4) for v in _ee_sim_chk.tolist()] if _ee_sim_chk is not None else None}", flush=True)
print(f"Transform error (cuRobo tool vs Isaac prim): {_xfm_err:.4f} m "
      f"(expected ~0.11 m offset due to frame definition difference)", flush=True)

print(f"Object center (base-frame, NOT added as obstacle): "
      f"{[round(v,4) for v in _world_to_base(_obj_settled).tolist()]}", flush=True)

# ── GoalToolPose + current JointState ─────────────────────────────────────────
# cuRobo IK operates in BASE frame (FK raw output matches sim EE after base_to_world).
# The earlier "64/64 with world-frame" result was misleading: the arm can reach
# the world-frame coordinates when interpreted as base-frame coords — just the wrong place.
# Base-frame IK was failing only because _world_quat_to_base had the wrong sign (now fixed).
_pos_t  = torch.tensor(_grasp_pos_base,  dtype=torch.float32, device=_device)
_quat_t = torch.tensor(_grasp_quat_base, dtype=torch.float32, device=_device)

_grasp_goal = GoalToolPose(
    tool_frames=_planner.tool_frames,
    position=_pos_t.reshape(1,1,1,1,3),
    quaternion=_quat_t.reshape(1,1,1,1,4),
)

_current_js = CuroboJointState.from_position(
    torch.tensor(_q_arm_init[None], dtype=torch.float32, device=_device),
    joint_names=_ARM_JOINTS,
)

# ── cuRobo frame diagnostic ───────────────────────────────────────────────────
print("\n=== CUROBO FRAME DIAGNOSTIC ===\n", flush=True)

_kin_diag   = _planner.compute_kinematics(_current_js)
_ee_raw     = _kin_diag.tool_poses.position[0, 0, 0].cpu().numpy()
_ee_sim     = _ee_link_world(stage)
print(f"cuRobo FK EE (raw, no transform): {[round(float(v),4) for v in _ee_raw.tolist()]}", flush=True)
print(f"Isaac sim EE (world):             {[round(float(v),4) for v in _ee_sim.tolist()] if _ee_sim is not None else None}", flush=True)
print(f"  raw vs sim diff:  {np.linalg.norm(_ee_raw - _ee_sim):.4f} m  <- if ~0: cuRobo is in world frame", flush=True)
_ee_via_xfm = _base_to_world(_ee_raw)
print(f"  base_to_world FK: {[round(float(v),4) for v in _ee_via_xfm.tolist()]}", flush=True)
print(f"  xfm vs sim diff:  {np.linalg.norm(_ee_via_xfm - _ee_sim):.4f} m  <- if ~0: cuRobo is in base frame", flush=True)

_ik_diag_world = _planner.ik_solver.solve_pose(
    GoalToolPose(
        tool_frames=_planner.tool_frames,
        position=torch.tensor(_grasp_pos_world, dtype=torch.float32, device=_device).reshape(1,1,1,1,3),
        quaternion=torch.tensor(_grasp_quat_world, dtype=torch.float32, device=_device).reshape(1,1,1,1,4),
    ),
    return_seeds=64, current_state=_current_js,
)
_n_world = int(_ik_diag_world.success.sum().item())
_pe_world = _ik_diag_world.position_error.min().item() if hasattr(_ik_diag_world, 'position_error') else -1
print(f"IK with WORLD coords:      {_n_world}/64 success, pos_err={_pe_world:.4f}", flush=True)

_ik_diag_base = _planner.ik_solver.solve_pose(
    GoalToolPose(
        tool_frames=_planner.tool_frames,
        position=torch.tensor(_grasp_pos_base, dtype=torch.float32, device=_device).reshape(1,1,1,1,3),
        quaternion=torch.tensor(_grasp_quat_base, dtype=torch.float32, device=_device).reshape(1,1,1,1,4),
    ),
    return_seeds=64, current_state=_current_js,
)
_n_base = int(_ik_diag_base.success.sum().item())
_pe_base = _ik_diag_base.position_error.min().item() if hasattr(_ik_diag_base, 'position_error') else -1
print(f"IK with BASE-frame coords: {_n_base}/64 success, pos_err={_pe_base:.4f}", flush=True)
print(f"  => cuRobo frame is {'WORLD (skip transform)' if _n_world > _n_base else 'BASE (transform needed)'}", flush=True)


# ── Planning ──────────────────────────────────────────────────────────────────
# Use plan_pose (TrajOpt) instead of geometric IK chaining.
# TrajOpt jointly optimises the full trajectory, so it handles large joint-space
# motions without the Cartesian arcs / ground crashes produced by linear
# joint interpolation between two far-apart IK solutions.
print(f"\n=== PLANNING APPROACH (TrajOpt, max_attempts=3) ===\n", flush=True)
_approach_wps: np.ndarray | None = None
_traj_result = _planner.plan_pose(_grasp_goal, _current_js, max_attempts=3)
if _traj_result is None:
    print("  plan_pose returned None — no trajectory", flush=True)
else:
    _ok = bool(_traj_result.success.any(dim=-1)[0].item())
    print(f"  plan_pose success: {_ok}", flush=True)
    if _ok:
        _approach_wps = _js_to_waypoints(
            _traj_result.interpolated_trajectory,
            _traj_result.interpolated_last_tstep,
        )
        print(f"  Approach: {len(_approach_wps)} waypoints", flush=True)
        if len(_approach_wps) > 1:
            _deltas = np.linalg.norm(np.diff(_approach_wps, axis=0), axis=1)
            print(f"  Per-step joint delta: max={_deltas.max():.4f} mean={_deltas.mean():.4f} rad", flush=True)
    else:
        print("  TrajOpt failed — check collision config or goal reachability", flush=True)

# ── Direct IK debug ────────────────────────────────────────────────────────────
print("\n=== DIRECT IK DEBUG ===\n", flush=True)
_ik_result = _planner.ik_solver.solve_pose(
    _grasp_goal, return_seeds=8, current_state=_current_js,
)
print(f"IK success:         {_ik_result.success}", flush=True)
print(f"IK solve_time:      {_ik_result.solve_time:.3f}s", flush=True)
if hasattr(_ik_result, 'position_error'):
    _pe = _ik_result.position_error
    print(f"IK position_error:  min={_pe.min().item():.4f} max={_pe.max().item():.4f}", flush=True)
if hasattr(_ik_result, 'rotation_error'):
    _re = _ik_result.rotation_error
    print(f"IK rotation_error:  min={_re.min().item():.4f} max={_re.max().item():.4f}", flush=True)
_ik_pos = _ik_result.js_solution.position
print(f"IK js_solution shape: {_ik_pos.shape}", flush=True)
if _ik_pos.numel() > 0:
    _ik_joints = _ik_pos.reshape(-1, _ik_pos.shape[-1])[0].cpu().numpy()
    print(f"IK best joints (q): {[round(float(v),4) for v in _ik_joints]}", flush=True)

print(f"Planning summary (mode=trajopt):", flush=True)
print(f"  approach_wps: {len(_approach_wps) if _approach_wps is not None else None}", flush=True)
print(f"  lift_wps: re-planned after hand close", flush=True)

# ── Knocked-over detection helper ────────────────────────────────────────────
def _check_knocked_over(phase_name: str,
                        z_drop_thresh: float = 0.025,
                        xy_disp_thresh: float = 0.06) -> bool:
    _cn, _ = objects.get_world_poses()
    _cur = np.asarray(_cn, dtype=np.float64)[0]
    z_drop  = _obj_settled[2] - _cur[2]
    xy_disp = float(np.linalg.norm(_cur[:2] - _obj_settled[:2]))
    knocked = (z_drop > z_drop_thresh) or (xy_disp > xy_disp_thresh)
    status  = "KNOCKED OVER" if knocked else "upright"
    print(
        f"[{phase_name}] object {status}  "
        f"z={_cur[2]:.4f} (Δz={_cur[2]-_obj_settled[2]:+.4f})  "
        f"xy_disp={xy_disp:.4f}",
        flush=True,
    )
    return knocked


# ── Execute ───────────────────────────────────────────────────────────────────
_cur_cmd = np.tile(_init_joints_1d, (1,1))

# Object trajectory recording — accumulated at log_every checkpoints in _execute_segment.
# Saved to trajectories/cup_lift_wuji/ at the end of execution.
_obj_traj: list[dict] = []

print("\n--- Pre-approach: opening hand and settling ---", flush=True)
for jn, val in _HAND_OPEN.items():
    if jn in _dof_idx:
        _cur_cmd[0, _dof_idx[jn]] = float(val)
_settle(30, _cur_cmd)
print("--- Hand open and settled ---", flush=True)

# Phase 1: Approach
if _approach_wps is not None:
    _actual_arm_q = robots.get_joint_positions()[0][_arm_cols]
    _approach_wps = np.concatenate([[_actual_arm_q], _approach_wps])
    print(f"Approach: {len(_approach_wps)} waypoints", flush=True)
    # For top-down approach, EE arrives from above. Stop when 0.12 m from object
    # center — at that point EE is roughly at rim level, ready for hand close.
    _execute_segment("Approach", _approach_wps, _HAND_OPEN, _cur_cmd,
                     early_stop_joint_err=0.08, early_stop_ee_to_obj_m=0.12)
    _settle(_SETTLE_STEPS, _cur_cmd)
    _step(render_force=True)
else:
    print("WARNING: no approach trajectory", flush=True)
_check_knocked_over("post-approach")
_log_finger_tips("post-approach-tips")

# ── Phase B: Push arm straight down ───────────────────────────────────────────
# Lower the EE so the cup rim seats between thumb and index finger.
print("\n--- Phase B: Pushing arm down ---", flush=True)
_q_pre_push = robots.get_joint_positions()[0][_arm_cols]
_js_pre_push = CuroboJointState.from_position(
    torch.tensor(_q_pre_push[None], dtype=torch.float32, device=_device),
    joint_names=_ARM_JOINTS,
)
_kin_pre_push    = _planner.compute_kinematics(_js_pre_push)
_ee_pp_base      = _kin_pre_push.tool_poses.position[0, 0, 0].cpu().numpy().astype(np.float64)
_ee_pp_qbase     = _kin_pre_push.tool_poses.quaternion[0, 0, 0].cpu().numpy().astype(np.float64)
_ee_pp_world     = _base_to_world(_ee_pp_base)
_push_tgt_world  = _ee_pp_world - np.array([0., 0., _PUSH_DOWN_M])
_push_tgt_base   = _world_to_base(_push_tgt_world)
print(f"  EE before push (world): {[round(v,4) for v in _ee_pp_world.tolist()]}", flush=True)
print(f"  Push target   (world):  {[round(v,4) for v in _push_tgt_world.tolist()]}", flush=True)

_push_goal = GoalToolPose(
    tool_frames=_planner.tool_frames,
    position=torch.tensor(_push_tgt_base, dtype=torch.float32, device=_device).reshape(1,1,1,1,3),
    quaternion=torch.tensor(_ee_pp_qbase,  dtype=torch.float32, device=_device).reshape(1,1,1,1,4),
)
_push_ik = _planner.ik_solver.solve_pose(_push_goal, return_seeds=32, current_state=_js_pre_push)
if _push_ik.success.any():
    _pb = int(_push_ik.success.reshape(-1).float().argmax().item())
    _q_push_end = _push_ik.js_solution.position.reshape(-1, len(_ARM_JOINTS))[_pb].cpu().numpy()
    _N_PUSH = 40
    _push_wps = np.stack([(1 - t) * _q_pre_push + t * _q_push_end
                          for t in np.linspace(1. / _N_PUSH, 1., _N_PUSH)])
    _execute_segment("PushDown", _push_wps, _HAND_OPEN, _cur_cmd,
                     log_every=10, sim_steps_per_wp=10)
    _settle(20, _cur_cmd)
    _log_finger_tips("post-push-down-tips")
else:
    print("  WARNING: push-down IK FAILED — continuing without push", flush=True)
_check_knocked_over("post-push-down")

# ── Phase C: Full close ────────────────────────────────────────────────────────
print("\n--- Phase C: Full hand close ---", flush=True)
for jn, val in _HAND_CLOSED.items():
    if jn in _dof_idx:
        _cur_cmd[0, _dof_idx[jn]] = float(val)
for _ in range(_HAND_CLOSE_STEPS):
    robots.set_joint_position_targets(_cur_cmd)
    _step()
_settle(30, _cur_cmd)

# Contact + joint state snapshot after close.
try:
    _cf_close = np.asarray(objects.get_net_contact_forces(dt=1.0), dtype=np.float64).flatten()
    _cf_close_mag = float(np.linalg.norm(_cf_close[:3])) if len(_cf_close) >= 3 else 0.0
    print(f"[hand-close] net contact force on cup: {[round(float(v),3) for v in _cf_close[:3]]}  |F|={_cf_close_mag:.3f} N", flush=True)
except Exception as _ce:
    print(f"[hand-close] contact force unavailable: {_ce}", flush=True)
_hand_jnames = list(_HAND_CLOSED.keys())
_q_hand_actual = {jn: round(float(robots.get_joint_positions()[0][_dof_idx[jn]]), 4)
                  for jn in _hand_jnames if jn in _dof_idx}
_q_hand_target = {jn: round(float(_HAND_CLOSED[jn]), 4) for jn in _hand_jnames}
_hand_err = {jn: round(_q_hand_actual[jn] - _q_hand_target[jn], 4)
             for jn in _q_hand_actual}
print(f"[hand-close] joint targets : {_q_hand_target}", flush=True)
print(f"[hand-close] joint actual  : {_q_hand_actual}", flush=True)
print(f"[hand-close] joint error   : {_hand_err}  (negative = didn't reach target, blocked by cup)", flush=True)
_log_finger_contacts("post-hand-close")
_log_finger_tips("post-hand-close-tips")

# Sync _cur_cmd arm to actual position and re-settle.
_q_pg = robots.get_joint_positions()[0][_arm_cols]
_cur_cmd[0, _arm_cols] = _q_pg
_settle(10, _cur_cmd)
_check_knocked_over("post-hand-close")

# Phase 2: Lift — re-plan from actual post-grasp joint state.
print("\n--- Re-planning lift from actual post-grasp state ---", flush=True)
_q_pg = robots.get_joint_positions()[0][_arm_cols]
_js_pg = CuroboJointState.from_position(
    torch.tensor(_q_pg[None], dtype=torch.float32, device=_device),
    joint_names=_ARM_JOINTS,
)
_kin_pg      = _planner.compute_kinematics(_js_pg)
_ee_pg_base  = _kin_pg.tool_poses.position[0, 0, 0].cpu().numpy().astype(np.float64)
_ee_pg_qbase = _kin_pg.tool_poses.quaternion[0, 0, 0].cpu().numpy().astype(np.float64)
_ee_pg_world = _base_to_world(_ee_pg_base)
_lift_tgt_base = _world_to_base(_ee_pg_world + np.array([0., 0., _LIFT_HEIGHT_M]))
print(f"  EE (world): {[round(v,4) for v in _ee_pg_world.tolist()]}", flush=True)
print(f"  Lift target (base): {[round(v,4) for v in _lift_tgt_base.tolist()]}", flush=True)

_lgoal_end = GoalToolPose(
    tool_frames=_planner.tool_frames,
    position=torch.tensor(_lift_tgt_base, dtype=torch.float32, device=_device).reshape(1,1,1,1,3),
    quaternion=torch.tensor(_ee_pg_qbase, dtype=torch.float32, device=_device).reshape(1,1,1,1,4),
)
_lr_end = _planner.ik_solver.solve_pose(_lgoal_end, return_seeds=32, current_state=_js_pg)
if _lr_end.success.any():
    _lb = int(_lr_end.success.reshape(-1).float().argmax().item())
    _q_lift_end = _lr_end.js_solution.position.reshape(-1, len(_ARM_JOINTS))[_lb].cpu().numpy()

    _N_LIFT = 60
    _ts = np.linspace(1. / _N_LIFT, 1., _N_LIFT)
    _lift_wps_exec = np.stack([(1 - t) * _q_pg + t * _q_lift_end for t in _ts])

    _lift_js_batch = CuroboJointState.from_position(
        torch.tensor(_lift_wps_exec, dtype=torch.float32, device=_device),
        joint_names=_ARM_JOINTS,
    )
    _lift_kin_batch = _planner.compute_kinematics(_lift_js_batch)
    _lift_ee_base_all = _lift_kin_batch.tool_poses.position[:, 0, 0].cpu().numpy().astype(np.float64)
    _lift_planned_ee_world = np.stack([_base_to_world(p) for p in _lift_ee_base_all])

    print(f"  Lift: {len(_lift_wps_exec)} wps (linear joint interp, IK endpoint)", flush=True)
    _execute_segment("Lift", _lift_wps_exec, _HAND_CLOSED, _cur_cmd,
                     log_every=5, planned_ee_world=_lift_planned_ee_world,
                     sim_steps_per_wp=10)
else:
    print("  WARNING: lift IK FAILED — skipping lift", flush=True)
_check_knocked_over("post-lift")

# ── Final report ───────────────────────────────────────────────────────────────
print("\n=== FINAL GRASP REPORT ===", flush=True)
_pos_final, _ = objects.get_world_poses()
_obj_final    = np.asarray(_pos_final, dtype=np.float64)[0]
_ee_final     = _ee_link_world(stage)
_lift_success = bool(_obj_final[2] > _obj_settled[2] + 0.05)
print(f"Object initial Z: {_obj_settled[2]:.4f} m", flush=True)
print(f"Object final Z:   {_obj_final[2]:.4f} m  (Δ={_obj_final[2]-_obj_settled[2]:.4f})", flush=True)
print(f"EE final pos:     {[round(v,4) for v in _ee_final.tolist()] if _ee_final is not None else None}", flush=True)
print(f"Lift succeeded:   {_lift_success}  (object rose > 5 cm)", flush=True)

# ── Save object trajectory ────────────────────────────────────────────────────
try:
    from datetime import datetime as _dt
    _traj_dir = Path(__file__).parent.parent.parent / "trajectories" / "cup_lift_wuji"
    _traj_dir.mkdir(parents=True, exist_ok=True)
    _traj_path = _traj_dir / f"{_dt.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json as _json_traj
    with _traj_path.open("w") as _f:
        _json_traj.dump(_obj_traj, _f, indent=2)
    print(f"TRAJECTORY_SAVED: {_traj_path}  ({len(_obj_traj)} frames)", flush=True)
except Exception as _te:
    print(f"TRAJECTORY_SAVE_ERROR: {_te}", flush=True)

# ── Encode video ───────────────────────────────────────────────────────────────
try:
    import glob as _glob, av as _av, imageio.v2 as _iio
    frames = sorted(_glob.glob(os.path.join(_FRAMES_DIR, "frame_*.png")))
    if frames:
        _vid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cup_lift_wuji.mp4")
        _img0 = _iio.imread(frames[0])[:, :, :3]
        _h, _w = _img0.shape[:2]
        with _av.open(_vid, mode="w") as _con:
            _st = _con.add_stream("libx264", rate=30)
            _st.width = _w; _st.height = _h; _st.pix_fmt = "yuv420p"
            for _fp in frames:
                _fr = _av.VideoFrame.from_ndarray(_iio.imread(_fp)[:, :, :3], format="rgb24")
                for _pkt in _st.encode(_fr): _con.mux(_pkt)
            for _pkt in _st.encode(): _con.mux(_pkt)
        print(f"VIDEO: {_vid}", flush=True)
except Exception as _ve:
    print(f"VIDEO_ERROR: {_ve}", flush=True)

# ── Cleanup ────────────────────────────────────────────────────────────────────
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
