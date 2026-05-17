"""
cylinder_pour_traj_wuji.py — Grasp a cylinder then follow a recorded trajectory.
Adapted to use the realkinova wuji hand instead of the xhand.

This demonstrates the correct pattern for trajectory-following tasks
(bottle_pour, cylinder_pour, etc.):

  Phase 1 (Approach): incremental IK from home → grasp standoff position.
  Phase 2 (Reach):    incremental IK from standoff → grasp contact.
  Phase 3 (Grasp):    close hand, hold 60 steps.
  Phase 4 (Transport): load the trajectory JSON, align it to the settled
                        object pose, then track each waypoint by IK.

KEY INSIGHT for transport:
  At grasp time record ee_offset = ee_world - obj_world.
  For each aligned trajectory waypoint pos[k]:
      ee_tgt[k] = pos[k] + ee_offset
  Solve IK for ee_tgt[k], execute.  The hand stays in the same relative pose
  to the object throughout the recorded motion.

Object: procedural UsdGeom.Cylinder (r=0.025 m, h=0.18 m).
Trajectory: trajectories/cylinder_pour_wuji/<latest>.json
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
os.environ.setdefault("DISPLAY",                     ":1")
os.environ.setdefault("OMNI_STRUCTUREDLOG_ENABLED",  "0")
os.environ.setdefault("CUROBO_KERNEL_BACKEND",       "pybind")

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_REPO_ROOT    = str(Path(__file__).resolve().parents[2])
_PIPELINE_DIR = _REPO_ROOT + "/src"
_IK_URDF_PATH = _REPO_ROOT + "/assets/realkinova_wuji_hand/realkinova_wuji_hand_right.urdf"

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

import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from pxr import Gf, UsdGeom, UsdPhysics

# ── Scene constants ────────────────────────────────────────────────────────────
_CYL_RADIUS   = 0.025
_CYL_HEIGHT   = 0.18
_OBJ_POS_INIT = np.array([-0.094117, 0.476573, 0.18])
_OBJ_MASS     = 0.2

_ROBOT_PRIM = "/World/envs/env_0/Robot"
_OBJ_PRIM   = "/World/envs/env_0/Object"

_RENDER_EVERY = 8
_FRAMES_DIR   = os.environ.get("ISAAC_FRAMES_DIR", "/tmp/isaac_frames_cyl_pour_traj_wuji")
import shutil as _shutil
if Path(_FRAMES_DIR).exists():
    _shutil.rmtree(_FRAMES_DIR)
Path(_FRAMES_DIR).mkdir(parents=True, exist_ok=True)

# Grasp / approach parameters (matching cylinder_pour_wuji)
IK_EE_MOUNT_ADJ_M      = 0.04   # pull grasp target 4 cm toward robot so palm sits at cylinder surface
_GRASP_HEIGHT_OFFSET_M = 0.01
_APPROACH_STANDOFF_M   = 0.12
_SIM_STEPS_PER_WP      = 8
_SETTLE_STEPS          = 10
_HAND_CLOSE_STEPS      = 60

# Transport: IK interpolation steps between consecutive trajectory frames
_INTERP_STEPS = 8   # each runs _SIM_STEPS_PER_WP physics steps

# ── Initial robot configuration ────────────────────────────────────────────────
_INIT_ARM = {
    "Actuator1": -1.529021, "Actuator2":  1.545096, "Actuator3": 0.164424,
    "Actuator4":  0.946070, "Actuator5":  1.254565, "Actuator6": -1.063086,
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
_HAND_OPEN = {k: (0.04 if k.endswith("_joint1") else 0.0) for k in _INIT_HAND}
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

# ── Quaternion helpers ─────────────────────────────────────────────────────────
def _qnorm(q):
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)

def _qconj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def _qmul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2], dtype=np.float64)

def _qrotate(q, v):
    q = _qnorm(q); v = np.asarray(v, dtype=np.float64)
    qv = np.array([0., v[0], v[1], v[2]])
    r  = _qmul(_qmul(q, qv), _qconj(q))
    return r[1:]

def _slerp_quat(q0, q1, t):
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0: q1, dot = -q1, -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0); return r / (np.linalg.norm(r) + 1e-12)
    theta0 = np.arccos(dot)
    return np.sin(theta0*(1-t))/np.sin(theta0)*q0 + np.sin(theta0*t)/np.sin(theta0)*q1


def _rmat_to_wxyz(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s=0.5/np.sqrt(tr+1); w=0.25/s; x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return _qnorm(np.array([w,x,y,z]))

# ── Build scene ────────────────────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0)
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
    if not _ok:
        raise RuntimeError(f"URDFParseAndImportFile failed: ok={_ok}")
    _ROBOT_PRIM = str(_result).strip() if _result else _ROBOT_PRIM
    print(f"ROBOT_LOADED: {_ROBOT_PRIM}", flush=True)
except Exception as _e:
    print(f"URDF import failed: {_e}", flush=True); raise

_cyl_geom = UsdGeom.Cylinder.Define(stage, _OBJ_PRIM)
_cyl_geom.CreateAxisAttr("Z")
_cyl_geom.CreateRadiusAttr(float(_CYL_RADIUS))
_cyl_geom.CreateHeightAttr(float(_CYL_HEIGHT))
UsdGeom.Xformable(_cyl_geom).AddTranslateOp().Set(Gf.Vec3d(*_OBJ_POS_INIT.tolist()))
_bp = stage.GetPrimAtPath(_OBJ_PRIM)
UsdPhysics.RigidBodyAPI.Apply(_bp)
UsdPhysics.CollisionAPI.Apply(_bp)
UsdPhysics.MassAPI.Apply(_bp).CreateMassAttr().Set(_OBJ_MASS)
print(f"Cylinder created (r={_CYL_RADIUS}m, h={_CYL_HEIGHT}m)", flush=True)

robots  = Articulation(prim_paths_expr=_ROBOT_PRIM, name="robots", reset_xform_properties=False)
objects = RigidPrim(prim_paths_expr=_OBJ_PRIM, name="objects",
                    reset_xform_properties=False, track_contact_forces=True)
world.scene.add(robots); world.scene.add(objects)
world.reset(); robots.initialize(); world.step(render=False)

_dof_names = list(robots.dof_names)
_n_dof     = robots.num_dof
_dof_idx   = {n: i for i, n in enumerate(_dof_names)}
_ARM_JOINTS = [f"Actuator{i}" for i in range(1, 8)]
_arm_cols   = [_dof_idx[jn] for jn in _ARM_JOINTS if jn in _dof_idx]

_init_joints_1d = np.zeros(_n_dof)
for jn, val in {**_INIT_ARM, **_INIT_HAND}.items():
    if jn in _dof_idx:
        _init_joints_1d[_dof_idx[jn]] = val
robots.set_joint_positions(np.tile(_init_joints_1d, (1, 1)))
robots.set_joint_velocities(np.zeros((1, _n_dof)))

try:
    kps = np.tile(np.array([8000.0]*7 + [6000.0]*(_n_dof-7)), (1, 1))
    kds = np.tile(np.array([ 400.0]*7 + [  250.0]*(_n_dof-7)), (1, 1))
    robots.set_gains(kps=kps, kds=kds)
except Exception as _eg:
    print(f"PD gains warning: {_eg}", flush=True)

# ── Camera ─────────────────────────────────────────────────────────────────────
_camera = rep.create.camera(position=(1.2, 0.8, 1.5), look_at=(-0.05, 0.45, 0.1))
_rp_cam = rep.create.render_product(_camera, (640, 480))
_rgb    = rep.AnnotatorRegistry.get_annotator("rgb")
_rgb.attach([_rp_cam])
_frame_counter = [0]

def _step(render_force=False):
    _frame_counter[0] += 1
    if render_force or (_frame_counter[0] % _RENDER_EVERY == 0):
        world.step(render=True)
        try:
            rep.orchestrator.step(rt_subframes=1)
            img = _rgb.get_data()
            if img is not None and img.size > 0:
                if hasattr(img, 'numpy') and not isinstance(img, np.ndarray):
                    img_np = img.numpy()
                else:
                    img_np = np.asarray(img)
                img_cpu = np.ascontiguousarray(
                    img_np[..., :3] if img_np.ndim == 3 and img_np.shape[-1] == 4 else img_np
                )
                from PIL import Image as _PILImage
                _PILImage.fromarray(img_cpu).save(
                    f"{_FRAMES_DIR}/frame_{_frame_counter[0]:06d}.png"
                )
        except Exception:
            pass
    else:
        world.step(render=False)

# ── Settle ─────────────────────────────────────────────────────────────────────
print("Settling...", flush=True)
_settle_cmd = np.tile(_init_joints_1d, (1, 1))
robots.set_joint_positions(_settle_cmd)
robots.set_joint_velocities(np.zeros((1, _n_dof)))
for _ in range(60):
    robots.set_joint_position_targets(_settle_cmd)
    _step()

_pos_raw, _rot_raw = objects.get_world_poses()
_obj_settled     = np.asarray(_pos_raw, dtype=np.float64)[0]
_rot_raw_arr     = np.asarray(_rot_raw, dtype=np.float64)[0]   # xyzw from Isaac
# Convert Isaac xyzw → wxyz
_obj_settled_quat = _qnorm(np.array([_rot_raw_arr[3], _rot_raw_arr[0],
                                      _rot_raw_arr[1], _rot_raw_arr[2]]))
print(f"Object settled at: {_obj_settled.tolist()}", flush=True)
print(f"Object settled quat (wxyz): {_obj_settled_quat.tolist()}", flush=True)

import torch
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState as CuroboJointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.types.device_cfg import DeviceCfg as CuroboDeviceCfg
from motion_planner_batch import BatchMotionPlanner

_device_cfg  = CuroboDeviceCfg()
_device      = _device_cfg.device

_ROBOT_CFG_DICT = {
    "robot_cfg": {
        "kinematics": {
            "urdf_path":        _IK_URDF_PATH,
            "asset_root_path":  str(Path(_IK_URDF_PATH).parent),
            "base_link":        "base_link",
            "tool_frames":      ["right_hand_link"],
            "grasp_contact_link_names": ["right_hand_link"],
            "collision_link_names": [
                "base_link","Shoulder_Link","HalfArm1_Link","HalfArm2_Link",
                "ForeArm_Link","SphericalWrist1_Link","SphericalWrist2_Link",
                "Bracelet_Link","right_wuji_mount_link","right_palm_link",
            ],
            "collision_spheres": {
                "base_link":            [{"center":[0.,0.,0.06],"radius":0.06}],
                "Shoulder_Link":        [{"center":[0.,0.,-0.10],"radius":0.06},
                                         {"center":[0.,0.,-0.15],"radius":0.05}],
                "HalfArm1_Link":        [{"center":[0.,0.,0.],"radius":0.055},
                                         {"center":[0.,-0.07,0.],"radius":0.055},
                                         {"center":[0.,-0.15,0.],"radius":0.055}],
                "HalfArm2_Link":        [{"center":[0.,0.,0.],"radius":0.055},
                                         {"center":[0.,0.,-0.07],"radius":0.055},
                                         {"center":[0.,0.,-0.15],"radius":0.055}],
                "ForeArm_Link":         [{"center":[0.,0.,0.],"radius":0.055},
                                         {"center":[0.,-0.07,0.],"radius":0.055},
                                         {"center":[0.,-0.17,0.],"radius":0.055}],
                "SphericalWrist1_Link": [{"center":[0.,0.,0.],"radius":0.055},
                                         {"center":[0.,0.,-0.085],"radius":0.055}],
                "SphericalWrist2_Link": [{"center":[0.,0.,0.],"radius":0.05},
                                         {"center":[0.,-0.085,0.],"radius":0.05}],
                "Bracelet_Link":        [{"center":[0.,0.,-0.05],"radius":0.04},
                                         {"center":[0.,-0.05,-0.05],"radius":0.04}],
                "right_wuji_mount_link": [{"center":[0.,0.,0.04],"radius":0.040}],
                "right_palm_link":      [{"center":[0.,0.,0.00],"radius":0.040},
                                         {"center":[0.,0.,0.04],"radius":0.035}],
            },
            "collision_sphere_buffer": 0.005,
            "self_collision_ignore": {
                "base_link":            ["Shoulder_Link","HalfArm1_Link"],
                "Shoulder_Link":        ["HalfArm1_Link","HalfArm2_Link"],
                "HalfArm1_Link":        ["HalfArm2_Link","ForeArm_Link"],
                "HalfArm2_Link":        ["ForeArm_Link","SphericalWrist1_Link"],
                "ForeArm_Link":         ["SphericalWrist1_Link","SphericalWrist2_Link"],
                "SphericalWrist1_Link": ["SphericalWrist2_Link","Bracelet_Link"],
                "SphericalWrist2_Link": ["Bracelet_Link","right_wuji_mount_link","right_palm_link"],
                "Bracelet_Link":        ["right_wuji_mount_link","right_palm_link"],
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
_planner_cfg = MotionPlannerCfg.create(
    robot=_ROBOT_CFG_DICT,
    ik_optimizer_configs=["ik/particle_ik.yml", "ik/lbfgs_ik.yml"],
    num_ik_seeds=200, num_trajopt_seeds=12,
    use_cuda_graph=False, max_batch_size=1, max_goalset=1,
    device_cfg=_device_cfg,
    collision_cache={"cuboid": 5},
    optimizer_collision_activation_distance=0.05,
)
_planner = BatchMotionPlanner(_planner_cfg)
print("BatchMotionPlanner ready.", flush=True)

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
    prefix = (_ROBOT_PRIM if robot_prim is None else robot_prim).rstrip("/") + "/"
    return _prim_world_pos(stage, prefix + "right_hand_link")


def _ee_link_world_quat(stage, robot_prim=None):
    prefix = (_ROBOT_PRIM if robot_prim is None else robot_prim).rstrip("/") + "/"
    path = prefix + "right_hand_link"
    try:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0)
            R = np.array([[xf[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
            return _rmat_to_wxyz(R)
    except Exception:
        pass
    return None


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


_T_WB: np.ndarray = np.zeros(3, dtype=np.float64)
_R_WB: np.ndarray = np.eye(3, dtype=np.float64)


def _world_to_base(pos_world):
    return (np.asarray(pos_world, dtype=np.float64) - _T_WB) @ _R_WB


def _base_to_world(pos_base):
    return np.asarray(pos_base, dtype=np.float64) @ _R_WB.T + _T_WB


def _world_quat_to_base(q_world):
    q_wb = _rmat_to_wxyz(_R_WB)
    q_wb_inv = np.array([q_wb[0], -q_wb[1], -q_wb[2], -q_wb[3]])
    q = _qmul(q_wb_inv, np.asarray(q_world, dtype=np.float64))
    return q / (np.linalg.norm(q) + 1e-12)


def _compute_side_grasp_quat_world(approach_dir):
    """EE Z → approach_dir, EE Y → world +Z (thumb up)."""
    z = np.asarray(approach_dir, dtype=np.float64); z /= np.linalg.norm(z) + 1e-12
    y_w = np.array([0., 0., 1.]); x = np.cross(y_w, z)
    if np.linalg.norm(x) < 1e-6: y_w = np.array([1., 0., 0.]); x = np.cross(y_w, z)
    x /= np.linalg.norm(x); y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-12
    return _rmat_to_wxyz(np.stack([x, y, z], axis=1))


# ── IK helper for transport ────────────────────────────────────────────────────

def _solve_ik_pos(pos_world, seed_arm, q_base=None):
    pos_base = _world_to_base(pos_world)
    if q_base is None:
        q_base = np.array([1., 0., 0., 0.], dtype=np.float64)
    seed_js  = CuroboJointState.from_position(
        torch.tensor(seed_arm[None].astype(np.float32), device=_device),
        joint_names=_ARM_JOINTS,
    )
    goal = GoalToolPose(
        tool_frames=_planner.tool_frames,
        position=torch.tensor(pos_base, dtype=torch.float32, device=_device).reshape(1,1,1,1,3),
        quaternion=torch.tensor(q_base, dtype=torch.float32, device=_device).reshape(1,1,1,1,4),
    )
    res = _planner.ik_solver.solve_pose(goal, return_seeds=32, current_state=seed_js)
    if res.success.any():
        best = int(res.success.reshape(-1).float().argmax().item())
        q    = res.js_solution.position.reshape(-1, len(_ARM_JOINTS))[best].cpu().numpy()
        return q.astype(np.float64), True
    return seed_arm.copy(), False


# ── Execution helpers ──────────────────────────────────────────────────────────

def _settle(n, cur_cmd):
    for _ in range(n):
        robots.set_joint_position_targets(cur_cmd)
        _step()


def _execute_segment(label, waypoints, hand_joints, current_cmd, log_every=10,
                     early_stop_joint_err=None, early_stop_ee_to_obj_m=None,
                     early_stop_obj_moved_m=None, obj_pos_init=None):
    print(f"\n=== PHASE: {label.upper()} ({len(waypoints)} waypoints, {_SIM_STEPS_PER_WP} sim/wp) ===\n", flush=True)
    for jn, val in hand_joints.items():
        if jn in _dof_idx:
            current_cmd[0, _dof_idx[jn]] = float(val)
    for step, q_wp in enumerate(waypoints):
        for k, col in enumerate(_arm_cols):
            current_cmd[0, col] = float(q_wp[k])
        for _ in range(_SIM_STEPS_PER_WP):
            robots.set_joint_position_targets(current_cmd)
            _step()
        ee = _ee_link_world(stage)
        _pn, _ = objects.get_world_poses()
        obj = np.asarray(_pn, dtype=np.float64)[0]
        d = float(np.linalg.norm(ee - obj)) if ee is not None else -1.0
        if step % log_every == 0:
            err = float(np.linalg.norm(q_wp - robots.get_joint_positions()[0][_arm_cols]))
            print(json.dumps({
                "phase": label, "step": step, "progress": f"{step}/{len(waypoints)}",
                "ee_z": round(ee[2], 4) if ee is not None else None,
                "obj_pos": [round(v, 4) for v in obj.tolist()],
                "ee_to_obj": round(d, 4), "arm_err": round(err, 4),
            }), flush=True)
        if step >= len(waypoints) // 2:
            if early_stop_ee_to_obj_m is not None and d >= 0 and d < early_stop_ee_to_obj_m:
                print(f"  [{label}] Proximity stop: ee_to_obj={d:.4f}m", flush=True)
                break
            if early_stop_obj_moved_m is not None and obj_pos_init is not None:
                _obj_moved = float(np.linalg.norm(obj - np.asarray(obj_pos_init, dtype=np.float64)))
                if _obj_moved > early_stop_obj_moved_m:
                    print(f"  [{label}] Object displaced {_obj_moved:.4f}m — stopping", flush=True)
                    break
            if early_stop_joint_err is not None:
                _q_actual = robots.get_joint_positions()[0][_arm_cols]
                if float(np.linalg.norm(waypoints[-1] - _q_actual)) < early_stop_joint_err:
                    break


# ── Populate world↔base transform ─────────────────────────────────────────────
_T_WB, _R_raw = _read_base_world_pose(stage, _ROBOT_PRIM)
_R_WB = _R_raw.T
print(f"Base link world pos: {_T_WB.tolist()}", flush=True)

# ── Approach direction + grasp pose ───────────────────────────────────────────
_q_arm_init = _init_joints_1d[_arm_cols].astype(np.float32)

_base_xy = _T_WB[:2]
_approach_dir_xy = _obj_settled[:2] - _base_xy
_approach_dir_xy /= np.linalg.norm(_approach_dir_xy) + 1e-12
_approach_dir = np.array([_approach_dir_xy[0], _approach_dir_xy[1], 0.07])
print(f"Approach dir (world): {[round(v,4) for v in _approach_dir.tolist()]}", flush=True)

_grasp_pos_world  = (_obj_settled + np.array([0., 0., _GRASP_HEIGHT_OFFSET_M])
                    - IK_EE_MOUNT_ADJ_M * _approach_dir)
_grasp_quat_world = _compute_side_grasp_quat_world(_approach_dir)
_grasp_pos_base   = _world_to_base(_grasp_pos_world)
_grasp_quat_base  = _world_quat_to_base(_grasp_quat_world)
print(f"Grasp pos (world): {[round(v,4) for v in _grasp_pos_world.tolist()]}", flush=True)
print(f"Grasp pos (base):  {[round(v,4) for v in _grasp_pos_base.tolist()]}", flush=True)

# ── Plan geometric approach ────────────────────────────────────────────────────

def _plan_geometric_trajectory(n_approach=80, n_seeds=16):
    def _slerp(q0, q1, t):
        q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
        dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if dot < 0: q1, dot = -q1, -dot
        if dot > 0.9995:
            r = q0 + t * (q1 - q0); return r / (np.linalg.norm(r) + 1e-12)
        theta0 = np.arccos(dot)
        return np.sin(theta0*(1-t))/np.sin(theta0)*q0 + np.sin(theta0*t)/np.sin(theta0)*q1

    def _solve_phase(positions, quats, seed_js, label):
        wps, n_fail = [], 0
        cur_seed = seed_js
        for pos, q in zip(positions, quats):
            goal = GoalToolPose(
                tool_frames=_planner.tool_frames,
                position=torch.tensor(pos, dtype=torch.float32, device=_device).reshape(1,1,1,1,3),
                quaternion=torch.tensor(q, dtype=torch.float32, device=_device).reshape(1,1,1,1,4),
            )
            res = _planner.ik_solver.solve_pose(goal, return_seeds=n_seeds, current_state=cur_seed)
            if res.success.any():
                best = int(res.success.reshape(-1).float().argmax().item())
                q_sol = res.js_solution.position.reshape(-1, len(_ARM_JOINTS))[best].cpu().numpy()
                wps.append(q_sol)
                cur_seed = CuroboJointState.from_position(
                    torch.tensor(q_sol[None], dtype=torch.float32, device=_device),
                    joint_names=_ARM_JOINTS,
                )
            else:
                n_fail += 1
                wps.append(wps[-1] if wps else seed_js.position[0].cpu().numpy())
        print(f"  [{label}] {len(wps)} wps, {n_fail} IK failures", flush=True)
        return np.stack(wps)

    _js_init_plan = CuroboJointState.from_position(
        torch.tensor(_q_arm_init[None], dtype=torch.float32, device=_device),
        joint_names=_ARM_JOINTS,
    )
    _kin0 = _planner.compute_kinematics(_js_init_plan)
    p0 = _kin0.tool_poses.position[0, 0, 0].cpu().numpy().astype(np.float64)
    q0 = _kin0.tool_poses.quaternion[0, 0, 0].cpu().numpy().astype(np.float64)

    ts = np.linspace(0.0, 1.0, n_approach)
    pos_wps  = [p0 + t * (_grasp_pos_base - p0) for t in ts]
    quat_wps = [_slerp(q0, _grasp_quat_base, t) for t in ts]

    print(f"  Solving approach IK ({n_approach} wps, {n_seeds} seeds/wp) ...", flush=True)
    return _solve_phase(pos_wps, quat_wps, _js_init_plan, "approach")


# ── Plan and execute approach + grasp ─────────────────────────────────────────
print(f"\n=== PLANNING GEOMETRIC APPROACH ===\n", flush=True)
_approach_wps = _plan_geometric_trajectory()

_cur_cmd = np.tile(_init_joints_1d, (1, 1))
print("\n--- Opening hand and settling ---", flush=True)
for jn, val in _HAND_OPEN.items():
    if jn in _dof_idx:
        _cur_cmd[0, _dof_idx[jn]] = float(val)
_settle(30, _cur_cmd)

_actual_arm_q = robots.get_joint_positions()[0][_arm_cols]
_approach_wps = np.concatenate([[_actual_arm_q], _approach_wps])
_execute_segment("Approach", _approach_wps, _HAND_OPEN, _cur_cmd,
                 early_stop_joint_err=0.08, early_stop_ee_to_obj_m=0.060,
                 early_stop_obj_moved_m=0.015, obj_pos_init=_obj_settled)
_settle(_SETTLE_STEPS, _cur_cmd)

print("\n--- Closing hand ---", flush=True)
for jn, val in _HAND_CLOSED.items():
    if jn in _dof_idx:
        _cur_cmd[0, _dof_idx[jn]] = float(val)
for _ in range(_HAND_CLOSE_STEPS):
    robots.set_joint_position_targets(_cur_cmd)
    _step()
_settle(_SETTLE_STEPS, _cur_cmd)

_q_arm = robots.get_joint_positions()[0][_arm_cols].copy()
_cur_cmd[0, _arm_cols] = _q_arm

_ee_grasp = _ee_link_world(stage)
_pn, _pq  = objects.get_world_poses()
_obj_grasp = np.asarray(_pn, dtype=np.float64)[0]
if _ee_grasp is not None:
    print(f"GRASP: ee={[round(v,4) for v in _ee_grasp.tolist()]}  "
          f"obj={[round(v,4) for v in _obj_grasp.tolist()]}  "
          f"ee_to_obj={np.linalg.norm(_ee_grasp-_obj_grasp):.3f}m", flush=True)

# Record EE orientation and compute its fixed pose relative to the object.
# This lets us track the object's tilt during the pour by reapplying this offset.
_ee_grasp_quat_world = _ee_link_world_quat(stage)
_pq_raw = np.asarray(_pq, dtype=np.float64)[0]   # Isaac returns xyzw
_obj_grasp_quat_world = _qnorm(np.array([_pq_raw[3], _pq_raw[0], _pq_raw[1], _pq_raw[2]]))
_q_ee_rel_obj = (_qmul(_qconj(_obj_grasp_quat_world), _ee_grasp_quat_world)
                 if _ee_grasp_quat_world is not None else None)

# ── PHASE 4: TRANSPORT — follow recorded trajectory ───────────────────────────
# The key idea:
#   1. Compute ee_offset = ee_world_at_grasp - obj_world_at_grasp.
#      This is the vector from the object center to the EE mount in world frame.
#   2. Load the trajectory JSON. Each frame has {pos: [x,y,z], rot: [w,x,y,z]}.
#   3. Align the trajectory to the settled object pose using TRANSLATION ONLY:
#         p_align   = settled_pos - traj_frame0_pos
#         waypoint[k] = traj_pos[k] + p_align
#      (Do NOT apply rotation: rotating the path flips Δx/Δy when the recorded
#       object orientation differs from the settled orientation, producing wrong motion.)
#   4. For each waypoint, compute ee_tgt = waypoint + ee_offset, solve IK.
print("\n=== PHASE 4: TRANSPORT (trajectory following) ===", flush=True)

# 4a. Compute the EE-to-object offset at grasp time (world frame).
if _ee_grasp is not None:
    _ee_offset = _ee_grasp - _obj_grasp
else:
    # Fallback: use approach direction scaled by mount adjustment
    _ee_offset = -_approach_dir * IK_EE_MOUNT_ADJ_M
print(f"  EE offset from obj center: {[round(v,4) for v in _ee_offset.tolist()]}", flush=True)

# 4b. Load trajectory.
_traj_dir   = Path(_REPO_ROOT) / "trajectories" / "cylinder_pour_wuji"
_traj_files = sorted(_traj_dir.glob("*.json"))
if not _traj_files:
    print("  WARNING: no cylinder_pour_wuji trajectory found — skipping transport", flush=True)
    _trajectory = []
else:
    _traj_path = _traj_files[-1]   # use most recent
    print(f"  Loading trajectory: {_traj_path}", flush=True)
    with _traj_path.open() as _tf:
        _trajectory = json.load(_tf)
    print(f"  Trajectory: {len(_trajectory)} frames", flush=True)

if _trajectory:
    # 4c. TRANSLATION-ONLY alignment: shift trajectory so frame-0 coincides with
    #     the settled object position.  We do NOT rotate the path because:
    #       - The trajectory's relative XYZ displacements (Δx, Δy, Δz) are in
    #         world frame and are what the robot needs to reproduce.
    #       - Rotation-based alignment maps the recorded object orientation onto
    #         the settled orientation.  If they differ (e.g. trajectory recorded at
    #         180° around X but object settles at 180° around Y), q_align ≈ 180° Z,
    #         which negates Δx — turning a leftward motion rightward.
    #       - Since world Z is always "up", the upward motion is preserved correctly
    #         with translation-only alignment.
    _t0_pos  = np.asarray(_trajectory[0]["pos"], dtype=np.float64)
    _p_align = _obj_settled - _t0_pos   # pure translation offset

    # Orientation alignment: rotate traj frame-0 onto the settled object orientation.
    _q_traj_rot0 = _qnorm(np.asarray(_trajectory[0]["rot"], dtype=np.float64))  # wxyz
    _q_rot_align = _qmul(_obj_settled_quat, _qconj(_q_traj_rot0))

    _waypoints_world = []
    _waypoints_quat_base = []
    for _fr in _trajectory:
        _p = np.asarray(_fr["pos"], dtype=np.float64)
        _waypoints_world.append(_p + _p_align)
        # Align the recorded object orientation and apply the fixed EE-relative pose.
        _q_obj_aligned = _qnorm(_qmul(_q_rot_align, _qnorm(np.asarray(_fr["rot"], dtype=np.float64))))
        if _q_ee_rel_obj is not None:
            _q_ee_w = _qnorm(_qmul(_q_obj_aligned, _q_ee_rel_obj))
        else:
            _q_ee_w = _qnorm(_qmul(_q_obj_aligned, _rmat_to_wxyz(np.eye(3))))
        _waypoints_quat_base.append(_world_quat_to_base(_q_ee_w))

    print(f"  Aligned waypoints: frame0={[round(v,4) for v in _waypoints_world[0].tolist()]}  "
          f"frame-1={[round(v,4) for v in _waypoints_world[-1].tolist()]}", flush=True)
    print(f"  Z travel: {_waypoints_world[0][2]:.3f} → {_waypoints_world[-1][2]:.3f} m "
          f"(Δ={_waypoints_world[-1][2]-_waypoints_world[0][2]:+.3f}m)", flush=True)

    # 4d. Execute: for each consecutive pair of waypoints, interpolate and IK-track.
    _q_arm = robots.get_joint_positions()[0][_arm_cols].copy()
    _prev_ee_tgt  = _waypoints_world[0] + _ee_offset
    _prev_ee_quat = _waypoints_quat_base[0]

    for _wi, _wp in enumerate(_waypoints_world[1:], start=1):
        _next_ee_tgt  = _wp + _ee_offset
        _next_ee_quat = _waypoints_quat_base[_wi]
        # Interpolate position and orientation between consecutive waypoints.
        for _ii in range(_INTERP_STEPS):
            t = (_ii + 1.0) / _INTERP_STEPS
            _interp_ee   = (1.0 - t) * _prev_ee_tgt + t * _next_ee_tgt
            _interp_quat = _slerp_quat(_prev_ee_quat, _next_ee_quat, t)
            _q_arm, _ok = _solve_ik_pos(_interp_ee, _q_arm, _interp_quat)
            _cur_cmd[0, _arm_cols] = _q_arm
            for _ in range(_SIM_STEPS_PER_WP):
                robots.set_joint_position_targets(_cur_cmd)
                _step()
        _prev_ee_tgt  = _next_ee_tgt
        _prev_ee_quat = _next_ee_quat

        if _wi % 5 == 0 or _wi == len(_waypoints_world) - 1:
            _pn2, _ = objects.get_world_poses()
            _obj_now = np.asarray(_pn2, dtype=np.float64)[0]
            _kp_err  = float(np.linalg.norm(_obj_now - _wp))
            print(f"  wp {_wi:2d}/{len(_waypoints_world)-1}: "
                  f"obj_z={_obj_now[2]:.3f}  target_z={_wp[2]:.3f}  "
                  f"obj_err={_kp_err:.3f}m", flush=True)

    # Hold final pose
    for _ in range(30):
        robots.set_joint_position_targets(_cur_cmd)
        _step()

# ── Final report ───────────────────────────────────────────────────────────────
print("\n=== FINAL REPORT ===", flush=True)
_pn_f, _pq_f = objects.get_world_poses()
_obj_final   = np.asarray(_pn_f, dtype=np.float64)[0]
_ee_final    = _ee_link_world(stage)
_lifted = _obj_final[2] > _obj_settled[2] + 0.05
print(f"Object: settled_z={_obj_settled[2]:.4f}  final_z={_obj_final[2]:.4f}  "
      f"Δz={_obj_final[2]-_obj_settled[2]:+.4f}  lifted={_lifted}", flush=True)
if _trajectory:
    _goal_z = _waypoints_world[-1][2]
    print(f"Target final_z={_goal_z:.4f}  z_err={abs(_obj_final[2]-_goal_z):.4f}", flush=True)

# ── Encode video ────────────────────────────────────────────────────────────────
try:
    import subprocess, glob as _glob
    frames = sorted(_glob.glob(os.path.join(_FRAMES_DIR, "frame_*.png")))
    if frames:
        _vid = str(Path(__file__).with_suffix("")) + ".mp4"
        subprocess.run(["ffmpeg","-y","-framerate","30","-pattern_type","glob",
            "-i",os.path.join(_FRAMES_DIR,"frame_*.png"),
            "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",_vid])
        print(f"VIDEO: {_vid}", flush=True)
except Exception as _ve:
    print(f"VIDEO_ERROR: {_ve}", flush=True)

try: _rgb.detach([_rp_cam])
except Exception: pass
try: _rp_cam.destroy()
except Exception: pass
try: _planner.destroy(); torch.cuda.synchronize(); torch.cuda.empty_cache()
except Exception: pass
try: rep.orchestrator.stop()
except Exception: pass
try: simulation_app.close()
except Exception: pass
