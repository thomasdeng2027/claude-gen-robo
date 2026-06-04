#!/usr/bin/env python3
"""
ONE end-to-end script that:
 (A) Runs real-data pipeline:
     RGB -> Detectron person -> ViTPose -> 2D hand kpts -> HaMeR -> depth alignment -> 3D hand kpts (camera metric)
     (optionally Grounded-SAM mask like your new_demo.py)
 (B) Converts those kpts into the coordinate convention expected by DexMachina/Genesis (Z-up world)
 (C) Feeds them into dexmachina retargeting + Genesis rollout visualization


If you want to use cam_c2w.npy for world alignment, set --use_cam_c2w_world
and we’ll compute world points as c2w @ cam points (then (optional) convert to Genesis axes).
"""


from __future__ import annotations


import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
import cv2
from glob import glob
from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Allow running this file directly ("python src/demo_retarget.py")
# without requiring editable package installation.
if __package__ in (None, ""):
   _THIS_DIR = Path(__file__).resolve().parent
   _REPO_ROOT = _THIS_DIR.parent
   if str(_REPO_ROOT) not in sys.path:
       sys.path.insert(0, str(_REPO_ROOT))
   # Add process-iphone-demos so that detectron/vitpose/hamer imports resolve.
   _PROC_IPHONE_DEMOS = Path("/juno/u/jingyuny/projects/p_vla/process-iphone-demos")
   if _PROC_IPHONE_DEMOS.exists() and str(_PROC_IPHONE_DEMOS) not in sys.path:
       sys.path.insert(0, str(_PROC_IPHONE_DEMOS))


# ----------------------------
# Your real pipeline imports (same as your new_demo.py)
# These are only needed when running from raw RGB (no cached hamer outputs).
# Wrapped in try/except so the script works in envs that only have retargeting deps.
# ----------------------------
try:
    from process_iphone_demos.detectron.default_predictor import load_detectron
    from process_iphone_demos.vitpose.model import ViTPoseModel
    from process_iphone_demos.hamer.configs import CACHE_DIR_HAMER
    from process_iphone_demos.hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
    from process_iphone_demos.hamer.utils import recursive_to
    from process_iphone_demos.hamer.datasets.vitdet_dataset import ViTDetDataset
    from process_iphone_demos.hamer.utils.renderer import Renderer
    _HAMER_AVAILABLE = True
except ImportError as _e:
    print(f"[warn] perception pipeline unavailable ({_e}); "
          "only pre-processed hamer output paths will work.")
    _HAMER_AVAILABLE = False
    load_detectron = ViTPoseModel = None
    CACHE_DIR_HAMER = DEFAULT_CHECKPOINT = None
    download_models = load_hamer = recursive_to = None
    ViTDetDataset = Renderer = None


# Grounded-SAM deps (same as your new_demo.py)
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    GroundingDINOModel = sam_model_registry = SamPredictor = None


# ----------------------------
# DexMachina / Genesis imports (from your dexmachina script)
# ----------------------------
from retargeting_utils import compose_retarget_config, retarget_all_steps, get_ref_val, retarget_one_hand
from solve_ik_arm import solve_arm_ik_from_floating_hand, solve_arm_trajectory, solve_arm_trajectory_with_pose, ArmIKSolver, solve_robot_base_pose_via_ik, euler_to_rotation_matrix


from dex_retargeting.retargeting_config import RetargetingConfig




# ----------------------------
# Constants / conventions
# ----------------------------


REF_HAND_SIZE = 0.14


# Robot base pose in world frame (cam_c2w world frame from MegaSAM).
# When set, hand positions are transformed: camera -> world (cam_c2w) -> robot base frame.
# Workspace centering is skipped. Set both to None to fall back to workspace_center.
# The URDF (GEN3_URDF_V12_with_hand_right.urdf) bakes in its own world_to_base_link
# fixed joint, so pinocchio already knows where base_link is. No external transform needed.
ROBOT_POS_IN_WORLD  = None


# HaMeR overlay color
LIGHT_BLUE = (1, 0.74117647, 0.85882353)


# MANO indices: [thumb, index, middle, ring, pinky, wrist]
MANO6_IDXS = [4, 8, 12, 16, 20, 0]


# OpenCV cam -> Z-up "Genesis/Viser-like" world
# Camera faces the robot from across the scene (opposing viewpoints), so we apply
# a 180° rotation around Z: negate x (mirror) and negate the depth->Y mapping.
R_WORLD_CAM = np.array([
   [-1,  0,  0],   # x_world = -x_cam  (mirror: cam-right → world-left)
   [ 0,  0, -1],   # y_world = -z_cam  (depth maps to opposite side)
   [ 0, -1,  0],   # z_world = -y_cam  (cam-down → world-up)
], dtype=np.float64)




# ============================================================
# Small math helpers (copied/reused from your new_demo.py style)
# ============================================================

# Floor detection & Z-up alignment (same approach as demo.py)

def detect_floor_plane_ransac(
    points: np.ndarray,
    n_iters: int = 200,
    threshold: float = 0.02,
    min_inliers: int = 50,
    world_up: Optional[np.ndarray] = None,
    min_vertical: float = 0.5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """RANSAC plane fit that finds the largest *horizontal* plane.

    world_up: estimated physical 'up' direction in the point cloud frame.
        If provided, candidate planes whose normal has |dot(n, world_up)| < min_vertical
        are skipped — this filters out walls and keeps floors/tables/ceilings.
    min_vertical: minimum |normal · world_up| required to accept a plane (default 0.5 ≈ 60°).
    """
    if len(points) < 10:
        return None, None
    best_normal, best_pt, best_n = None, None, 0
    for _ in range(n_iters):
        idx = np.random.choice(len(points), 3, replace=False)
        s = points[idx]
        v1, v2 = s[1] - s[0], s[2] - s[0]
        n = np.cross(v1, v2)
        nl = np.linalg.norm(n)
        if nl < 1e-8:
            continue
        n /= nl
        # Skip planes that are too vertical (walls) when world_up is given.
        if world_up is not None and abs(float(np.dot(n, world_up))) < min_vertical:
            continue
        dists = np.abs((points - s[0]) @ n)
        cnt = int(np.sum(dists < threshold))
        if cnt > best_n:
            best_n, best_normal, best_pt = cnt, n, s[0]
    if best_n < min_inliers:
        return None, None
    return best_normal, best_pt


def detect_wall_plane_ransac(
    points_zup: np.ndarray,
    n_iters: int = 500,
    threshold: float = 0.05,
    min_inliers: int = 30,
    max_tilt: float = 0.3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """RANSAC for the dominant vertical plane (wall) in a Z-up point cloud.

    Only accepts planes whose normal has |nz| < max_tilt (nearly vertical planes).
    Returns (normal, point_on_plane) in the Z-up frame, or (None, None).
    """
    if len(points_zup) < 10:
        return None, None
    best_normal, best_pt, best_n = None, None, 0
    for _ in range(n_iters):
        idx = np.random.choice(len(points_zup), 3, replace=False)
        s = points_zup[idx]
        v1, v2 = s[1] - s[0], s[2] - s[0]
        n = np.cross(v1, v2)
        nl = np.linalg.norm(n)
        if nl < 1e-8:
            continue
        n /= nl
        if abs(float(n[2])) > max_tilt:
            continue
        dists = np.abs((points_zup - s[0]) @ n)
        cnt = int(np.sum(dists < threshold))
        if cnt > best_n:
            best_n, best_normal, best_pt = cnt, n, s[0]
    if best_n < min_inliers:
        return None, None
    return best_normal, best_pt


def compute_floor_align_transform(
    scene_pts_list: List[np.ndarray],
    hand_pts_ref: Optional[np.ndarray] = None,
    cam_right_megasam: Optional[np.ndarray] = None,
    cam_up_megasam: Optional[np.ndarray] = None,
    cam_forward_megasam: Optional[np.ndarray] = None,
    max_pts: int = 30000,
) -> np.ndarray:
    """
    Aggregate scene points, run RANSAC floor detection, and return a 4×4
    rigid transform that maps the scene to a canonical Z-up frame:
      - Z = up (perpendicular to floor, toward hand)
      - Y = toward wall (camera-forward direction) — wall at +Y
      - X = Z × Y  (camera right)
      - floor sits at Z = 0

    XY orientation priority:
      1. Wall plane detected via RANSAC on Z-up aligned points — wall inward normal → -Y
      2. cam_forward_megasam projected onto floor → +Y  (fallback)
      3. cam_right_megasam projected onto floor → +X  (last resort)

    cam_up_megasam: used to reject vertical planes during floor RANSAC.
    Falls back to identity if floor detection fails.
    """
    chunks = [p for p in scene_pts_list if p is not None and len(p) > 0]
    if not chunks:
        return np.eye(4)
    all_pts = np.concatenate(chunks, axis=0).astype(np.float64)
    if len(all_pts) > max_pts:
        idx = np.random.choice(len(all_pts), max_pts, replace=False)
        all_pts = all_pts[idx]

    # Normalise world_up for use in RANSAC vertical filter.
    world_up = None
    if cam_up_megasam is not None:
        world_up = np.asarray(cam_up_megasam, dtype=np.float64)
        norm = np.linalg.norm(world_up)
        if norm > 1e-6:
            world_up = world_up / norm
        else:
            world_up = None

    floor_normal, floor_pt = detect_floor_plane_ransac(all_pts, world_up=world_up)
    if floor_normal is None and world_up is not None:
        # Retry without the vertical filter if nothing was found.
        print("[floor] No horizontal plane found — retrying RANSAC without vertical constraint")
        floor_normal, floor_pt = detect_floor_plane_ransac(all_pts)
    if floor_normal is None:
        print("[floor] RANSAC floor detection failed — using identity transform")
        return np.eye(4)

    # Orient normal so it points toward the hand (upward from floor)
    if hand_pts_ref is not None and len(hand_pts_ref) > 0:
        hand_dists = np.dot(hand_pts_ref.reshape(-1, 3) - floor_pt, floor_normal)
        if float(np.mean(hand_dists)) < 0:
            floor_normal = -floor_normal

    # Step 1: rotate so floor_normal → +Z
    target_z = np.array([0.0, 0.0, 1.0])
    rot_z, _ = Rotation.align_vectors([target_z], [floor_normal])
    R_z = rot_z.as_matrix()

    T = np.eye(4)
    T[:3, :3] = R_z
    T[2, 3] = -(R_z @ floor_pt)[2]     # shift floor to Z=0

    # Step 2: canonicalise XY orientation.
    # Priority: (1) detected wall inward normal → +X,  (2) camera-right → +X fallback.
    ref_dir_floor = None  # 2-D horizontal unit vector to become +X

    # (1) Wall detection in the Z-up frame.
    pts_zup = (R_z @ all_pts.T).T
    pts_zup[:, 2] += T[2, 3]   # shift so floor is at Z=0
    wall_normal_zup, wall_pt_zup = detect_wall_plane_ransac(pts_zup)
    if wall_normal_zup is not None:
        n_wall_xy = wall_normal_zup[:2].copy()
        norm = np.linalg.norm(n_wall_xy)
        if norm > 0.1:
            n_wall_xy /= norm
            # Flip so the normal points toward the hand/robot (not into the wall).
            if hand_pts_ref is not None and len(hand_pts_ref) > 0:
                hand_zup = (R_z @ hand_pts_ref.reshape(-1, 3).T).T
                hand_xy = hand_zup[:, :2].mean(axis=0)
                if float(np.dot(n_wall_xy, hand_xy - wall_pt_zup[:2])) < 0:
                    n_wall_xy = -n_wall_xy
            # Wall is at +Y (camera forward convention).
            # n_wall_xy points from wall toward robot (inward normal) = should be −Y.
            # To make n_wall_xy → −Y, align its 90°-CW rotation with +X.
            ref_dir_floor = np.array([-n_wall_xy[1], n_wall_xy[0]])
            print(f"[floor] Wall detected — wall inward normal={n_wall_xy.round(3)}, "
                  f"floor +X set to {ref_dir_floor.round(3)} (wall will appear at +Y)")

    # (2) Camera-forward fallback → maps to +Y, so ref_dir for +X = rotate_90_cw(cam_fwd).
    if ref_dir_floor is None and cam_forward_megasam is not None:
        cf = np.asarray(cam_forward_megasam, dtype=np.float64)
        cf /= np.linalg.norm(cf)
        cf_floor = R_z @ cf
        cf_floor[2] = 0.0
        norm = np.linalg.norm(cf_floor)
        if norm > 0.1:
            cf_floor /= norm
            # cf_floor should map to +Y; to achieve this via aligning ref_dir → +X,
            # set ref_dir = rotate_90_cw(cf_floor) = [cf_floor[1], -cf_floor[0]]
            ref_dir_floor = np.array([cf_floor[1], -cf_floor[0]])
            print(f"[floor] No wall found — using cam-forward as +Y fallback, "
                  f"floor +X set to {ref_dir_floor.round(3)}")

    # (3) Camera-right last resort.
    if ref_dir_floor is None and cam_right_megasam is not None:
        cam_right = np.asarray(cam_right_megasam, dtype=np.float64)
        cam_right /= np.linalg.norm(cam_right)
        cr_floor = R_z @ cam_right
        cr_floor[2] = 0.0
        norm = np.linalg.norm(cr_floor)
        if norm > 0.1:
            ref_dir_floor = cr_floor[:2] / norm
            print(f"[floor] No wall found — using camera-right as floor +X last resort")

    if ref_dir_floor is not None:
        yaw = np.arctan2(float(ref_dir_floor[1]), float(ref_dir_floor[0]))
        cy, sy = np.cos(-yaw), np.sin(-yaw)
        Rz_corr = np.array([[cy, -sy, 0, 0],
                             [sy,  cy, 0, 0],
                             [ 0,   0, 1, 0],
                             [ 0,   0, 0, 1]], dtype=np.float64)
        T = Rz_corr @ T

    # Warn if the detected plane looks like a wall (normal has low vertical component).
    if world_up is not None:
        verticality = abs(float(np.dot(floor_normal, world_up)))
        print(f"[floor] floor normal={floor_normal.round(3)}  verticality(|n·up|)={verticality:.2f}"
              f"  →  canonical Z-up T[:3,3]={T[:3,3].round(3)}")
        if verticality < 0.7:
            print(f"[floor] WARNING: detected plane may be a wall (verticality={verticality:.2f} < 0.7). "
                  f"Floor Z values in downstream output will likely be wrong.")
    else:
        print(f"[floor] floor normal={floor_normal.round(3)}  →  canonical Z-up T[:3,3]={T[:3,3].round(3)}")
    return T


def estimate_robot_base_pose_floor(
    wrist_positions_floor: np.ndarray,
    arm_max_reach: float = 0.85,
    arm_min_reach: float = 0.25,
    robot_base_z: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Estimate robot base (x, y, z) position and yaw in the floor-aligned frame
    such that all observed wrist positions are within arm reach.

    The optimisation minimises:
      Σ max(0, dist - arm_max_reach)²  +  Σ max(0, arm_min_reach - dist)²

    Returns (base_pos_floor [3], yaw_floor).
    yaw = angle (rad) that the robot faces toward the workspace centroid.
    """
    from scipy.optimize import minimize

    valid = np.all(np.isfinite(wrist_positions_floor), axis=1) & \
            np.any(wrist_positions_floor != 0, axis=1)
    pts = wrist_positions_floor[valid]
    if len(pts) == 0:
        return np.array([0.0, 0.0, robot_base_z]), 0.0

    centroid_xy = pts[:, :2].mean(axis=0)

    def cost(xy):
        base3 = np.array([xy[0], xy[1], robot_base_z])
        d = np.linalg.norm(pts - base3, axis=1)
        too_far   = np.maximum(0.0, d - arm_max_reach)
        too_close = np.maximum(0.0, arm_min_reach - d)
        return float(np.sum(too_far**2) + 0.5 * np.sum(too_close**2))

    res = minimize(cost, centroid_xy, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-8})
    best_xy = res.x

    direction = centroid_xy - best_xy
    yaw = float(np.arctan2(direction[1], direction[0]))
    base_pos = np.array([best_xy[0], best_xy[1], robot_base_z])

    dists = np.linalg.norm(pts - np.array([base_pos[0], base_pos[1], robot_base_z]), axis=1)
    print(f"[robot_base] estimated base_pos_floor={base_pos.round(3)}, "
          f"yaw={np.degrees(yaw):.1f}°, "
          f"reach range=[{dists.min():.2f}, {dists.max():.2f}]m "
          f"(arm [{arm_min_reach}, {arm_max_reach}]m)")
    return base_pos, yaw


def build_T_robotbase_megasam(
    T_floor_megasam: np.ndarray,
    base_pos_floor: np.ndarray,
    yaw_floor: float,
) -> np.ndarray:
    """
    Build 4×4 T_robotbase_megasam:  MegaSAM world  →  robot base frame.

    Robot base frame:
      origin = base_pos_floor  (in floor-aligned world)
      Z = up  (same as floor frame)
      X/Y = rotated by yaw_floor around Z so robot faces its workspace

    T_robotbase_megasam = T_robotbase_floor  @  T_floor_megasam
    """
    cy, sy = np.cos(-yaw_floor), np.sin(-yaw_floor)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    t = -(Rz @ base_pos_floor)
    T_rb_floor = np.eye(4)
    T_rb_floor[:3, :3] = Rz
    T_rb_floor[:3, 3] = t
    return T_rb_floor @ T_floor_megasam


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
   """Apply 4x4 transform to Nx3 points."""
   points = np.asarray(points)
   if points.size == 0:
       return points.reshape((-1, 3))
   pts_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
   return (T @ pts_h.T).T[:, :3]




def cam_pts_to_genesis_world(P_cam: np.ndarray) -> np.ndarray:
   """OpenCV camera coords -> Z-up world coords."""
   P_cam = np.asarray(P_cam, dtype=np.float64)
   shp = P_cam.shape
   P2 = P_cam.reshape(-1, 3)
   out = (R_WORLD_CAM @ P2.T).T
   return out.reshape(shp)




def compute_mean_fingertip_path_length(keypoints: np.ndarray) -> Optional[float]:
   """Same heuristic as your new_demo.py."""
   try:
       k = np.asarray(keypoints, dtype=float)
       wrist = k[0]
       chains = {
           "thumb":  [1, 2, 3, 4],
           "index":  [5, 6, 7, 8],
           "middle": [9, 10, 11, 12],
           "ring":   [13, 14, 15, 16],
           "pinky":  [17, 18, 19, 20],
       }
       lengths = []
       for idxs in chains.values():
           pts = k[idxs]
           seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() if pts.shape[0] >= 2 else 0.0
           base_to_wrist = np.linalg.norm(pts[0] - wrist)
           lengths.append(float(seg + base_to_wrist))
       return float(np.mean(lengths)) if lengths else None
   except Exception:
       return None




def solve_similarity_transform(source_points: np.ndarray, target_points: np.ndarray):
   """Same as your new_demo.py: scale+translation, rotation fixed identity."""
   assert source_points.shape == target_points.shape
   mu_s = source_points.mean(axis=0)
   mu_t = target_points.mean(axis=0)
   s0 = source_points - mu_s
   t0 = target_points - mu_t
   scale_s = np.linalg.norm(s0, axis=1).mean()
   scale_t = np.linalg.norm(t0, axis=1).mean()
   scale = scale_t / scale_s if scale_s > 1e-6 else 1.0
   trans = mu_t - scale * mu_s
   return scale, np.eye(3), trans




def kabsch_T_from_A_to_B(A: np.ndarray, B: np.ndarray) -> np.ndarray:
   """Return 4x4 T s.t. (R@A+t) best fits B."""
   A = np.asarray(A, dtype=np.float64)
   B = np.asarray(B, dtype=np.float64)
   assert A.shape == B.shape and A.shape[1] == 3 and A.shape[0] >= 3


   muA = A.mean(axis=0)
   muB = B.mean(axis=0)
   Ac = A - muA
   Bc = B - muB


   H = Ac.T @ Bc
   U, S, Vt = np.linalg.svd(H)
   R = Vt.T @ U.T
   if np.linalg.det(R) < 0:
       Vt[-1, :] *= -1
       R = Vt.T @ U.T
   t = muB - R @ muA


   T = np.eye(4, dtype=np.float64)
   T[:3, :3] = R
   T[:3, 3] = t
   return T




def rmse_points(P: np.ndarray, Q: np.ndarray) -> float:
   P = np.asarray(P, dtype=np.float64)
   Q = np.asarray(Q, dtype=np.float64)
   if P.shape != Q.shape or P.size == 0:
       return float("nan")
   return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))




def isfinite_all(x: np.ndarray) -> bool:
   x = np.asarray(x)
   return np.isfinite(x).all()




# ============================================================
# Grounded-SAM (reused)
# ============================================================


def load_grounded_sam(device):
   GROUNDING_DINO_CONFIG_PATH = "_DATA/groundingdino/config/GroundingDINO_SwinT_OGC.py"
   GROUNDING_DINO_CHECKPOINT_PATH = "_DATA/groundingdino/weights/groundingdino_swint_ogc.pth"
   SAM_CHECKPOINT_PATH = "_DATA/groundingdino/weights/sam_vit_h_4b8939.pth"
   SAM_ENCODER_VERSION = "vit_h"


   dino = GroundingDINOModel(
       model_config_path=GROUNDING_DINO_CONFIG_PATH,
       model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
       device=device,
   )
   sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
   predictor = SamPredictor(sam)
   return dino, predictor




def get_hand_mask_grounded_sam(image_rgb: np.ndarray, dino_model, sam_predictor):
   detections, _ = dino_model.predict_with_caption(
       image=image_rgb,
       caption="a person's hand",
       box_threshold=0.35,
       text_threshold=0.25
   )
   if len(detections) == 0:
       return None


   sam_predictor.set_image(image_rgb)
   final_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
   for i in range(len(detections)):
       masks, _, _ = sam_predictor.predict(box=detections.xyxy[i][None, :], multimask_output=False)
       final_mask = np.logical_or(final_mask, masks[0])


   return (final_mask * 255).astype(np.uint8)






def load_models(device):
   detector = load_detectron()
   cpm = ViTPoseModel(device)


   download_models(CACHE_DIR_HAMER)
   model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
   model = model.to(device).eval()


   hand_faces = model.mano.faces
   renderer = Renderer(model_cfg, faces=hand_faces)


   dino_model, sam_predictor = load_grounded_sam(device)
   return detector, cpm, model, model_cfg, renderer, hand_faces, dino_model, sam_predictor




def select_hands(all_detections, hand_choice: str):
   selected = []
   if hand_choice in ["left", "right"]:
       is_right = 1 if hand_choice == "right" else 0
       candidates = [d for d in all_detections if d["is_right"] == is_right]
       if candidates:
           selected.append(max(candidates, key=lambda x: float(np.mean(x["keypoints"][:, 2]))))
   else:  # both
       lefts = [d for d in all_detections if d["is_right"] == 0]
       rights = [d for d in all_detections if d["is_right"] == 1]
       if lefts:
           selected.append(max(lefts, key=lambda x: float(np.mean(x["keypoints"][:, 2]))))
       if rights:
           selected.append(max(rights, key=lambda x: float(np.mean(x["keypoints"][:, 2]))))
   return selected




# ============================================================
# HaMeR alignment -> camera metric 3D kpts (reused)
# ============================================================


def align_kpts_to_depth(
   out, batch, i: int, is_right: int, model_cfg,
   depth: np.ndarray, K: np.ndarray, mask: np.ndarray,
   consistent_hand_size: bool, fixed_hand_size: Optional[float],
   bin_size_m: float = 0.015, depth_tol: Optional[float] = None,
):
   fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
   flip = (2 * is_right) - 1  # left:-1 right:+1


   verts = out["pred_vertices"][i].detach().cpu().numpy()
   verts[:, 0] *= flip


   cam_t = out["pred_cam_t"][i].detach().cpu().numpy()
   cam_t[0] *= flip


   verts_trans = verts + cam_t


   f = model_cfg.EXTRA.FOCAL_LENGTH
   cxy = model_cfg.MODEL.IMAGE_SIZE / 2


   u_crop = f * (verts_trans[:, 0] / verts_trans[:, 2]) + cxy
   v_crop = f * (verts_trans[:, 1] / verts_trans[:, 2]) + cxy


   box_center = batch["box_center"][i].cpu().numpy()
   box_size = batch["box_size"][i].cpu().numpy()


   u = (u_crop / model_cfg.MODEL.IMAGE_SIZE) * box_size + (box_center[0] - box_size / 2)
   v = (v_crop / model_cfg.MODEL.IMAGE_SIZE) * box_size + (box_center[1] - box_size / 2)
   u = np.clip(u.astype(int), 0, depth.shape[1] - 1)
   v = np.clip(v.astype(int), 0, depth.shape[0] - 1)


   z = depth[v, u]
   candidate = (z > 0) & (mask[v, u] > 0)
   cand_idx = np.where(candidate)[0]
   if cand_idx.size == 0:
       return None


   bin_m = float(bin_size_m)
   vx, vy, vz = verts_trans[:, 0], verts_trans[:, 1], verts_trans[:, 2]
   bx = np.floor(vx / bin_m).astype(int)
   by = np.floor(vy / bin_m).astype(int)


   from collections import defaultdict
   bin_to_idxs = defaultdict(list)
   for idx in cand_idx:
       bin_to_idxs[(int(bx[idx]), int(by[idx]))].append(int(idx))


   keep = []
   targets = []
   for _, idxs in bin_to_idxs.items():
       idxs = list(idxs)
       min_idx = int(idxs[int(np.argmin(vz[idxs]))])


       z_meas = z[min_idx]
       if not np.isfinite(z_meas) or z_meas <= 0:
           continue


       if depth_tol is not None:
           if abs(float(verts_trans[min_idx, 2]) - float(z_meas)) > float(depth_tol):
               continue


       u_pix = int(u[min_idx])
       v_pix = int(v[min_idx])
       x_t = (u_pix - cx) * z_meas / fx
       y_t = (v_pix - cy) * z_meas / fy


       keep.append(min_idx)
       targets.append((x_t, y_t, z_meas))


   if len(keep) == 0:
       return None


   keep = np.array(keep, dtype=int)
   source_verts = verts[keep]
   target_verts = np.array(targets, dtype=float)


   kpts = out["pred_keypoints_3d"][i].detach().cpu().numpy()
   kpts[:, 0] *= flip


   if consistent_hand_size:
       src_hand_size = compute_mean_fingertip_path_length(kpts)
       if src_hand_size is None or src_hand_size <= 1e-6:
           scale, rot, trans = solve_similarity_transform(source_verts, target_verts)
       else:
           ref_size = float(fixed_hand_size) if fixed_hand_size is not None else float(REF_HAND_SIZE)
           scale = ref_size / src_hand_size
           mu_s = source_verts.mean(axis=0)
           mu_t = target_verts.mean(axis=0)
           rot = np.eye(3)


           thumb_idx, index_idx = 4, 8
           mid_src = np.mean(kpts[[thumb_idx, index_idx]], axis=0)
           trans = mu_t - (scale * (mu_s - mid_src) + mid_src)
   else:
       scale, rot, trans = solve_similarity_transform(source_verts, target_verts)


   aligned_kpts_cam = (scale * kpts @ rot.T) + trans
   return aligned_kpts_cam




# ============================================================
# ============================================================
# Retargeting setup helpers (no Genesis)
# ============================================================


def prepare_retarget_cfgs(
   hand_name: str,
   side: str,
   retarget_type: str,
   excluded_joint_names=None,
):
   _ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


   if 'mano' in hand_name:
       config_path = str(_ASSETS_DIR / "mano_hand" / "retarget_config.yaml")
       robot_dir = str(_ASSETS_DIR / "mano-urdf")
   else:
       config_path = str(_ASSETS_DIR / hand_name / "retarget_config.yaml")
       robot_dir = str(_ASSETS_DIR / hand_name)


   RetargetingConfig.set_default_urdf_dir(robot_dir)
   with Path(config_path).open('r') as f:
       input_cfg = yaml.safe_load(f)


   low_pass_alpha = input_cfg.get("low_pass_alpha", 1.0)
   scaling_factor = input_cfg.get("scaling_factor", 1.0)
   ignore_mimic_joint = input_cfg.get("ignore_mimic_joint", False)
   add_dummy_free_joint = input_cfg.get("ignore_mimic_joint", False)
   is_arm_based = input_cfg.get("is_arm_based", False)


   side_excluded = excluded_joint_names or []
   config_dict = compose_retarget_config(
       input_cfg[side],
       retarget_type,
       low_pass_alpha,
       scaling_factor,
       add_dummy_free_joint=add_dummy_free_joint,
       ignore_mimic_joint=ignore_mimic_joint,
       excluded_joint_names=side_excluded,
   )
   retarget_cfg = RetargetingConfig.from_dict(deepcopy(config_dict))
   retargeter = retarget_cfg.build()


   return retargeter, is_arm_based, input_cfg, config_path


def read_intrinsics(demo_dir: str) -> np.ndarray:
   K_path = os.path.join(demo_dir, "K.npy")
   if os.path.exists(K_path):
       return np.load(K_path)
   txt_path = os.path.join(demo_dir, "cam_K.txt")
   if os.path.exists(txt_path):
       return np.loadtxt(txt_path).reshape(3, 3)
   raise FileNotFoundError(f"Missing K.npy or cam_K.txt in {demo_dir}")




def read_cam_c2w_if_exists(demo_dir: str) -> Optional[np.ndarray]:
   p = os.path.join(demo_dir, "cam_c2w.npy")
   if os.path.exists(p):
       return np.load(p)
   return None




def filter_bbox_size_outliers(
    bboxes: np.ndarray,
    valid: np.ndarray,
    zscore_threshold: float = 2.0,
) -> np.ndarray:
    """
    Return a copy of `valid` with frames marked False where the bbox area is
    anomalously large (arm captured instead of just the hand).

    Uses median + zscore_threshold * std across all currently-valid frames.
    """
    valid = valid.copy()
    det_idx = np.where(valid)[0]
    if len(det_idx) < 4:
        return valid
    areas = ((bboxes[det_idx, 2] - bboxes[det_idx, 0]) *
             (bboxes[det_idx, 3] - bboxes[det_idx, 1]))
    threshold = np.median(areas) + zscore_threshold * np.std(areas)
    for i, area in zip(det_idx, areas):
        if area > threshold:
            valid[i] = False
    n_removed = int(np.sum(~valid[det_idx]))
    if n_removed:
        print(f"[bbox_filter] removed {n_removed} oversized bbox frames "
              f"(threshold={threshold:.0f}px², median={np.median(areas):.0f}px²)")
    return valid


def interpolate_bboxes(
    bboxes: np.ndarray,
    valid: np.ndarray,
    max_gap: int = 10,
) -> tuple:
    """
    Linearly interpolate bbox coordinates and centers over short gaps.
    Returns (bboxes, valid) with gaps filled in.
    """
    bboxes = bboxes.copy()
    valid  = valid.copy()
    N = len(valid)

    i = 0
    while i < N:
        if not valid[i]:
            # find gap end
            j = i
            while j < N and not valid[j]:
                j += 1
            # gap is [i, j-1]; valid neighbours are i-1 and j
            if i > 0 and j < N and (j - i) <= max_gap:
                start_box = bboxes[i - 1]
                end_box   = bboxes[j]
                gap = j - i
                for k in range(gap):
                    t = (k + 1) / (gap + 1)
                    bboxes[i + k] = (1 - t) * start_box + t * end_box
                    valid[i + k]  = True
            i = j
        else:
            i += 1

    return bboxes, valid


def run_hamer_pipeline_on_range(
   demo_dir: str,
   out_dir: str,
   device: torch.device,
   detector,
   cpm,
   model,
   model_cfg,
   renderer,
   dino_model,
   sam_predictor,
   hand: str,
   rescale_factor: float,
   vit_threshold: float,
   bbox_threshold: float,
   consistent_hand_size: bool,
   fixed_hand_size: Optional[float],
   debug: bool,
   use_grounded_sam_mask: bool,
   start_step: int = 0,
   end_step: Optional[int] = None,
) -> Dict[str, Any]:
   """
   Returns dict with:
     union_indices (T,)
     desired_joint_pos_cam (T,21,3)  (camera metric)
     kpts6_cam (T,6,3)
     kpts6_valid (T,)
     (optional) desired_joint_pos_world_via_c2w (T,21,3) if cam_c2w exists
   """
   os.makedirs(out_dir, exist_ok=True)
   rgb_dir = os.path.join(demo_dir, "rgb")
   depth_dir = os.path.join(demo_dir, "depth")
   assert os.path.isdir(rgb_dir), f"No rgb dir: {rgb_dir}"
   assert os.path.isdir(depth_dir), f"No depth dir: {depth_dir}"


   K = read_intrinsics(demo_dir)
   cam_c2w_all = read_cam_c2w_if_exists(demo_dir)


   if end_step is None:
      end_step = len(os.listdir(rgb_dir))
   frame_idxs = np.arange(start_step, end_step, dtype=np.int32)
   T = int(frame_idxs.shape[0])
   union_indices = frame_idxs.astype(np.int64)


   desired_joint_pos_cam = np.zeros((T, 21, 3), dtype=np.float64)
   kpts6_cam = np.zeros((T, 6, 3), dtype=np.float64)
   kpts6_valid = np.zeros((T,), dtype=np.uint8)
   # Wrist orientation (3×3 rotation matrix) in camera frame from HaMeR global_orient.
   # Used later for 6DOF IK to resolve the left-right yaw ambiguity.
   wrist_orient_cam = np.zeros((T, 3, 3), dtype=np.float64)

   desired_world_via_c2w = np.zeros((T, 21, 3), dtype=np.float64) if cam_c2w_all is not None else None
   world_valid = np.zeros((T,), dtype=np.uint8) if cam_c2w_all is not None else None

   # Auto-detected object position in MegaSAM world frame.
   # Accumulated across ALL frames and resolved after the loop: the hand-held object
   # (cup/bottle) stays near the wrist in 3D for most of the demo, while a stationary
   # target (bowl) is only briefly close — so the minimum 3D distance wins.
   object_pos_megasam_auto: Optional[np.ndarray] = None
   _obj_best_dist3d: float = float('inf')
   _obj_best_world:  Optional[np.ndarray] = None
   _obj_best_cls:    int = -1
   _obj_best_score:  float = 0.0


   for t_i, step in enumerate(tqdm(frame_idxs, desc="Processing frames")):
       img_path   = os.path.join(rgb_dir,   f"{int(step):06d}.png")
       depth_path = os.path.join(depth_dir, f"{int(step):06d}.png")

       image_bgr = cv2.imread(img_path)
       if image_bgr is None:
           continue
       image_rgb = image_bgr[..., ::-1].copy()

       depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
       if depth_raw is None:
           continue
       depth = depth_raw.astype(np.float32) / 1000.0

       if image_rgb.shape[:2] != depth.shape[:2]:
           image_rgb = cv2.resize(image_rgb, (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

       predicted_instances = detector(image_rgb)["instances"]
       valid_idx = (predicted_instances.pred_classes == 0) & (predicted_instances.scores > 0.5)
       pred_bboxes_det = predicted_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
       if len(pred_bboxes_det) == 0:
           continue

       pred_scores = predicted_instances.scores[valid_idx].cpu().numpy()
       predicted_vitposes = cpm.predict_pose(
           image_rgb,
           [np.concatenate([pred_bboxes_det, pred_scores[:, None]], axis=1)]
       )

       all_hand_dets = []
       for vitposes in predicted_vitposes:
           if np.sum(vitposes["keypoints"][-42:-21, 2] > vit_threshold) > 0:
               all_hand_dets.append({"keypoints": vitposes["keypoints"][-42:-21], "is_right": 0,
                                     "wrist_body": vitposes["keypoints"][9]})   # left wrist (body idx 9)
           if np.sum(vitposes["keypoints"][-21:, 2] > vit_threshold) > 0:
               all_hand_dets.append({"keypoints": vitposes["keypoints"][-21:], "is_right": 1,
                                     "wrist_body": vitposes["keypoints"][10]})  # right wrist (body idx 10)

       selected = select_hands(all_hand_dets, hand)
       if not selected:
           continue

       # Use DINO to detect a tight hand bbox directly on the image.
       # This avoids ViTPose keypoints scattering across the arm during occlusion.
       # Fall back to wrist-anchored box if DINO finds nothing.
       image_bgr_for_dino = image_rgb[..., ::-1].copy()
       img_cx = image_rgb.shape[1] / 2
       wrist_pad = image_rgb.shape[1] * 0.06
       bboxes = []
       bbox_sources = []
       try:
           dino_detections, _ = dino_model.predict_with_caption(
               image=image_bgr_for_dino,
               caption="hand",
               box_threshold=0.2,
               text_threshold=0.2,
           )
           dino_boxes = dino_detections.xyxy if len(dino_detections) > 0 else np.empty((0, 4))
           dino_confs = dino_detections.confidence if len(dino_detections) > 0 else np.empty(0)
       except Exception:
           dino_boxes = np.empty((0, 4))
           dino_confs = np.empty(0)

       for det in selected:
           # Compute keypoint-span bbox from ViTPose hand keypoints (hand-specific, not arm).
           # Use median-distance filtering to reject outlier keypoints that land on the arm.
           kpts2 = det["keypoints"]
           vis_kpts = kpts2[kpts2[:, 2] > vit_threshold]
           if len(vis_kpts) >= 3:
               centroid = vis_kpts[:, :2].mean(axis=0)
               dists_from_centroid = np.linalg.norm(vis_kpts[:, :2] - centroid, axis=1)
               median_dist = np.median(dists_from_centroid)
               inlier_mask = dists_from_centroid <= 2.0 * median_dist + 1e-6
               inlier_kpts = vis_kpts[inlier_mask]
               kpt_pad = image_rgb.shape[1] * 0.03
               kpt_box = [
                   inlier_kpts[:, 0].min() - kpt_pad,
                   inlier_kpts[:, 1].min() - kpt_pad,
                   inlier_kpts[:, 0].max() + kpt_pad,
                   inlier_kpts[:, 1].max() + kpt_pad,
               ]
           elif len(vis_kpts) > 0:
               kpt_pad = image_rgb.shape[1] * 0.03
               kpt_box = [
                   vis_kpts[:, 0].min() - kpt_pad,
                   vis_kpts[:, 1].min() - kpt_pad,
                   vis_kpts[:, 0].max() + kpt_pad,
                   vis_kpts[:, 1].max() + kpt_pad,
               ]
           else:
               kpt_box = None

           if len(dino_boxes) > 0:
               # Among DINO detections, prefer the one closest to the hand keypoint centroid.
               if len(vis_kpts) > 0:
                   hand_cx = float(vis_kpts[:, 0].mean())
                   hand_cy = float(vis_kpts[:, 1].mean())
               else:
                   wb = det.get("wrist_body")
                   hand_cx = float(wb[0]) if wb is not None else img_cx
                   hand_cy = float(wb[1]) if wb is not None else image_rgb.shape[0] / 2
               ctrs_x = (dino_boxes[:, 0] + dino_boxes[:, 2]) / 2
               ctrs_y = (dino_boxes[:, 1] + dino_boxes[:, 3]) / 2
               dists = np.sqrt((ctrs_x - hand_cx) ** 2 + (ctrs_y - hand_cy) ** 2)
               best = int(np.argmin(dists))
               dino_box = dino_boxes[best].tolist()
               # Intersect with keypoint bbox to prevent the arm from being included.
               if kpt_box is not None:
                   dino_box = [
                       max(dino_box[0], kpt_box[0]),
                       max(dino_box[1], kpt_box[1]),
                       min(dino_box[2], kpt_box[2]),
                       min(dino_box[3], kpt_box[3]),
                   ]
                   # If intersection is empty, fall back to keypoint bbox.
                   if dino_box[0] >= dino_box[2] or dino_box[1] >= dino_box[3]:
                       dino_box = kpt_box
               bboxes.append(dino_box)
               bbox_sources.append(f"dino+kpt(conf={dino_confs[best]:.2f})")
           else:
               # No DINO detections: use keypoint bbox if available, else body wrist fallback.
               if kpt_box is not None:
                   bboxes.append(kpt_box)
                   bbox_sources.append("kpt_fallback")
               else:
                   wb = det.get("wrist_body")
                   if wb is not None and float(wb[2]) > 0.1:
                       wx, wy = float(wb[0]), float(wb[1])
                       bboxes.append([wx - wrist_pad, wy - wrist_pad, wx + wrist_pad, wy + wrist_pad])
                       bbox_sources.append(f"wrist_fallback(conf={wb[2]:.2f})")
                   else:
                       bboxes.append([0, 0, 0, 0])
                       bbox_sources.append("empty")

           # ── Max-size clamp: if the box is too large it likely captured the arm. ──
           # Fall back to the body wrist keypoint as center with a fixed-size crop.
           # The ViTPose body wrist (index 9/10) is robustly placed at the actual
           # wrist joint even when hand keypoints scatter along the forearm.
           max_side = image_rgb.shape[1] * 0.30
           cur_box = bboxes[-1]
           box_w = cur_box[2] - cur_box[0]
           box_h = cur_box[3] - cur_box[1]
           if max(box_w, box_h) > max_side and bbox_sources[-1] != "empty":
               wb = det.get("wrist_body")
               if wb is not None and float(wb[2]) > 0.1:
                   cx, cy = float(wb[0]), float(wb[1])
               else:
                   cx = (cur_box[0] + cur_box[2]) / 2
                   cy = (cur_box[1] + cur_box[3]) / 2
               half = max_side / 2
               bboxes[-1] = [cx - half, cy - half, cx + half, cy + half]
               bbox_sources[-1] += f"→clamped(wrist)"

       is_right_arr = np.array([det["is_right"] for det in selected], dtype=np.int64)
       boxes = np.array(bboxes)

       dataset = ViTDetDataset(model_cfg, image_rgb, boxes, is_right_arr, rescale_factor=rescale_factor)
       dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False, num_workers=0)
       batch = next(iter(dataloader))
       batch = recursive_to(batch, device)

       with torch.no_grad():
           out = model(batch)

       # optional debug overlays
       if debug:
           try:
               for n in range(len(batch["img"])):
                   regression_img = renderer(
                       out["pred_vertices"][n].detach().cpu().numpy(),
                       out["pred_cam_t"][n].detach().cpu().numpy(),
                       batch["img"][n],
                       mesh_base_color=LIGHT_BLUE,
                       scene_bg_color=(1, 1, 1),
                   )
                   cv2.imwrite(
                       os.path.join(out_dir, f"t{int(step):06d}_hand{n}_2d_overlay.png"),
                       (255 * regression_img[:, :, ::-1]).astype(np.uint8),
                   )
               bbox_vis = image_rgb[..., ::-1].copy()
               for n, (box, src) in enumerate(zip(boxes.tolist(), bbox_sources)):
                   x1, y1, x2, y2 = [int(v) for v in box]
                   color = (0, 255, 0) if src.startswith("wrist") else (0, 0, 255)
                   cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), color, 3)
                   cv2.putText(bbox_vis, f"h{n}:{src}", (x1, max(y1 - 6, 0)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
               cv2.imwrite(os.path.join(out_dir, f"t{int(step):06d}_bbox.png"), bbox_vis)
           except Exception:
               pass

       # mask for depth alignment
       if use_grounded_sam_mask:
           hand_mask = get_hand_mask_grounded_sam(image_rgb, dino_model, sam_predictor)
           if hand_mask is None:
               continue
           eroded_hand_mask = cv2.erode(hand_mask, np.ones((5, 5), np.uint8), iterations=1)
       else:
           eroded_hand_mask = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255


       # take RIGHT if exists else LEFT
       right_cam_this = None
       left_cam_this = None
       right_orient_this = None
       left_orient_this = None

       for i, det in enumerate(selected):
           kpts_cam = align_kpts_to_depth(
               out=out,
               batch=batch,
               i=i,
               is_right=int(det["is_right"]),
               model_cfg=model_cfg,
               depth=depth,
               K=K,
               mask=eroded_hand_mask,
               consistent_hand_size=consistent_hand_size,
               fixed_hand_size=fixed_hand_size,
           )
           if kpts_cam is None:
               if debug:
                   print(f"  [t={int(step)}] align_kpts_to_depth returned None for hand {i} (is_right={det['is_right']})")
               continue
           kpts_cam = np.asarray(kpts_cam, dtype=np.float64)

           # Extract wrist orientation (3×3 rotation matrix in camera frame).
           # HaMeR pred_mano_params['global_orient'] shape: (batch, 1, 3, 3).
           R_wrist = out['pred_mano_params']['global_orient'][i][0].detach().cpu().numpy()  # (3, 3)
           if int(det["is_right"]) == 0:
               # Mirror left-hand orientation across the X axis (same flip as vertex X).
               R_flip = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
               R_wrist = R_flip @ R_wrist @ R_flip

           if int(det["is_right"]) == 1:
               right_cam_this = kpts_cam
               right_orient_this = R_wrist
           else:
               left_cam_this = kpts_cam
               left_orient_this = R_wrist

       base_cam = right_cam_this if right_cam_this is not None else left_cam_this
       if base_cam is None:
           continue
       base_orient = right_orient_this if right_cam_this is not None else left_orient_this

       desired_joint_pos_cam[t_i] = base_cam
       kpts6_cam[t_i] = base_cam[MANO6_IDXS]
       kpts6_valid[t_i] = 1
       if base_orient is not None:
           wrist_orient_cam[t_i] = base_orient

       if cam_c2w_all is not None:
           c2w = cam_c2w_all[int(step)]
           desired_world_via_c2w[t_i] = transform_points(base_cam, c2w)  # NOTE: expects base_cam Nx3, c2w 4x4
           world_valid[t_i] = 1

       # --- Accumulate object candidates across ALL frames using 3D distance ---
       # Run on every frame (not just first): the object the hand holds stays at
       # near-zero 3D distance to the wrist for the majority of the demo, while a
       # stationary target object (e.g. bowl) is only briefly close.  After the
       # loop we pick the world position associated with the minimum 3D distance
       # ever observed — that is the hand-held object (cup/bottle), not the bowl.
       if cam_c2w_all is not None:
           fx, fy, cx_k, cy_k = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
           wrist_cam = base_cam[0]  # (3,) wrist in camera frame
           if wrist_cam[2] > 0:
               obj_mask = (predicted_instances.pred_classes != 0) & (predicted_instances.scores > 0.3)
               if obj_mask.sum() > 0:
                   obj_boxes  = predicted_instances.pred_boxes.tensor[obj_mask].cpu().numpy()
                   obj_scores = predicted_instances.scores[obj_mask].cpu().numpy()
                   h_img, w_img = depth.shape[:2]
                   for bi in range(len(obj_boxes)):
                       x1o, y1o, x2o, y2o = obj_boxes[bi].astype(int)
                       x1o, y1o = max(0, x1o), max(0, y1o)
                       x2o, y2o = min(w_img - 1, x2o), min(h_img - 1, y2o)
                       dc = depth[y1o:y2o, x1o:x2o]
                       vd = dc[(dc > 0.1) & (dc < 5.0)]
                       if len(vd) == 0:
                           continue
                       z_o = float(np.median(vd))
                       ocx = float((obj_boxes[bi, 0] + obj_boxes[bi, 2]) / 2)
                       ocy = float((obj_boxes[bi, 1] + obj_boxes[bi, 3]) / 2)
                       xo = (ocx - cx_k) * z_o / fx
                       yo = (ocy - cy_k) * z_o / fy
                       obj_cam_pt = np.array([xo, yo, z_o])
                       dist3d = float(np.linalg.norm(obj_cam_pt - wrist_cam))
                       if dist3d < _obj_best_dist3d:
                           _obj_best_dist3d = dist3d
                           _obj_best_world  = transform_points(obj_cam_pt[np.newaxis], cam_c2w_all[int(step)])[0]
                           _obj_best_cls    = int(predicted_instances.pred_classes[obj_mask][bi].item())
                           _obj_best_score  = float(obj_scores[bi])


   # ── Resolve auto-detected object: pick the one that was closest in 3D across all frames ──
   if _obj_best_world is not None:
       object_pos_megasam_auto = _obj_best_world
       print(f"[object_detect] auto-detected object: COCO class={_obj_best_cls}  "
             f"score={_obj_best_score:.2f}  min_3d_dist={_obj_best_dist3d:.3f}m  "
             f"pos_megasam={_obj_best_world.round(3)}")

   # ── Post-loop: invalidate frames where HaMeR captured the arm instead of hand ──
   # When the bbox is too large (arm included), HaMeR produces keypoints spread over
   # a much larger volume. We detect this via the XY span of the 21 keypoints in
   # camera space and z-score filter anomalously large frames.
   valid_mask = kpts6_valid.astype(bool)
   if valid_mask.sum() >= 5:
       spans = np.zeros(T, dtype=np.float64)
       for t_i in np.where(valid_mask)[0]:
           kp = desired_joint_pos_cam[t_i]  # (21, 3)
           spans[t_i] = np.linalg.norm(kp.max(axis=0)[:2] - kp.min(axis=0)[:2])
       valid_spans = spans[valid_mask]
       median_span = np.median(valid_spans)
       mad = np.median(np.abs(valid_spans - median_span)) + 1e-6
       # Flag frames whose span is more than 3 MADs above median as arm-capture outliers
       for t_i in np.where(valid_mask)[0]:
           if (spans[t_i] - median_span) / mad > 3.0:
               print(f"[hamer_postfilter] t_i={t_i} span={spans[t_i]:.3f} "
                     f"median={median_span:.3f} → invalidated (arm capture)")
               kpts6_valid[t_i] = 0
               if world_valid is not None:
                   world_valid[t_i] = 0

   # ── Temporal wrist-position outlier filter ───────────────────────────────
   # Catches compact-but-wrong HaMeR outputs (frames where the arm was captured
   # but the keypoint spread looks normal).  A frame is an outlier if its wrist
   # position deviates more than (median + 5×MAD) from the local window median
   # of its neighbouring valid frames.
   _wpos = desired_joint_pos_cam[:, 0, :3].copy()  # (T,3) wrist positions
   _valid_t = np.where(kpts6_valid.astype(bool))[0]
   if len(_valid_t) >= 10:
       _half = 5  # ±5 frame window
       _local_devs = np.zeros(len(_valid_t))
       for idx, t_i in enumerate(_valid_t):
           lo, hi = max(0, t_i - _half), min(T, t_i + _half + 1)
           _nbrs = [j for j in range(lo, hi) if kpts6_valid[j] and j != t_i]
           if len(_nbrs) >= 3:
               _local_med = np.median(_wpos[_nbrs], axis=0)
               _local_devs[idx] = float(np.linalg.norm(_wpos[t_i] - _local_med))
       _dev_median = np.median(_local_devs)
       _dev_mad = np.median(np.abs(_local_devs - _dev_median)) + 1e-6
       _pos_thresh = _dev_median + 5.0 * _dev_mad
       for idx, t_i in enumerate(_valid_t):
           if _local_devs[idx] > _pos_thresh:
               print(f"[wrist_postfilter] t_i={t_i} dev={_local_devs[idx]:.3f} "
                     f"thresh={_pos_thresh:.3f} → invalidated (position outlier)")
               kpts6_valid[t_i] = 0
               if world_valid is not None:
                   world_valid[t_i] = 0

   # ── Post-interpolation overlay: project keypoints onto images after filling gaps ──
   # Forward/backward fill so every frame shows the interpolated trajectory.
   # Green dots = original valid frame; orange dots = interpolated gap frame.
   kpts_filled = desired_joint_pos_cam.copy()
   fill_valid_vis = kpts6_valid.copy().astype(bool)
   last_v = None
   for t_i in range(T):
       if fill_valid_vis[t_i]:
           last_v = t_i
       elif last_v is not None:
           kpts_filled[t_i] = kpts_filled[last_v]
           fill_valid_vis[t_i] = True
   next_v = None
   for t_i in range(T - 1, -1, -1):
       if kpts6_valid[t_i]:
           next_v = t_i
       elif not fill_valid_vis[t_i] and next_v is not None:
           kpts_filled[t_i] = kpts_filled[next_v]
           fill_valid_vis[t_i] = True

   fx, fy, cx_k, cy_k = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
   for t_i, step in enumerate(frame_idxs):
       if not fill_valid_vis[t_i]:
           continue
       img_path = os.path.join(rgb_dir, f"{int(step):06d}.png")
       img_bgr = cv2.imread(img_path)
       if img_bgr is None:
           continue
       overlay = img_bgr.copy()
       was_valid = bool(kpts6_valid[t_i])
       color = (0, 255, 0) if was_valid else (0, 165, 255)  # green=valid, orange=interpolated
       h_img, w_img = overlay.shape[:2]
       for kp in kpts_filled[t_i]:
           if kp[2] > 0:
               u = int(fx * kp[0] / kp[2] + cx_k)
               v = int(fy * kp[1] / kp[2] + cy_k)
               if 0 <= u < w_img and 0 <= v < h_img:
                   cv2.circle(overlay, (u, v), 5, color, -1)
       cv2.imwrite(os.path.join(out_dir, f"t{int(step):06d}_kpts_overlay.png"), overlay)

   return dict(
       union_indices=union_indices,
       desired_joint_pos_cam=desired_joint_pos_cam,
       kpts6_cam=kpts6_cam,
       kpts6_valid=kpts6_valid,
       desired_joint_pos_world_via_c2w=desired_world_via_c2w,
       world_valid=world_valid,
       K=K,
       cam_c2w_all=cam_c2w_all,
       wrist_orient_cam=wrist_orient_cam,  # (T, 3, 3) wrist rotation in camera frame
       object_pos_megasam_auto=object_pos_megasam_auto,
   )




# ============================================================
# Build a DexMachina “demo_data” dict from HaMeR outputs
# ============================================================


def build_demo_data_from_hamer(
   hamer_out: Dict[str, Any],
   side: str = "right",
   use_cam_c2w_world: bool = False,
   T_target_from_megasam: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
   """
   DexMachina retarget expects:
     demo_data["joints.right"] : (T,21,3) points in target frame (robot base or floor-aligned world)
     demo_data["hand_qpos_right"]: (T, something). We'll provide zeros.

   Full pipeline (preferred):
     cam kpts  →  cam_c2w[t]  →  MegaSAM world  →  T_target_from_megasam  →  target frame

   T_target_from_megasam is typically T_robotbase_megasam which includes both the
   floor-alignment and the robot-base-pose estimation.
   Falls back to the old R_WORLD_CAM mapping when None.
   """
   desired_cam = np.asarray(hamer_out["desired_joint_pos_cam"], dtype=np.float64)
   T = desired_cam.shape[0]
   union_indices = hamer_out["union_indices"]
   cam_c2w_all = hamer_out.get("cam_c2w_all", None)

   if T_target_from_megasam is not None:
       # Preferred path: cam → (cam_c2w) → megasam → T_target_from_megasam → robot frame
       joints_world = np.zeros_like(desired_cam)
       for t_i, step in enumerate(union_indices):
           c2w = cam_c2w_all[int(step)] if cam_c2w_all is not None else np.eye(4)
           kpts_megasam = transform_points(desired_cam[t_i], c2w)
           joints_world[t_i] = transform_points(kpts_megasam, T_target_from_megasam)
   elif use_cam_c2w_world and cam_c2w_all is not None:
       world = np.zeros_like(desired_cam)
       for t_i, step in enumerate(union_indices):
           c2w = cam_c2w_all[int(step)]
           world[t_i] = transform_points(desired_cam[t_i], c2w)
       joints_world = world
   else:
       # Legacy fallback: fixed R_WORLD_CAM mapping
       joints_world = cam_pts_to_genesis_world(desired_cam)  # (T,21,3)


   demo_data = {}
   # Fill both sides to satisfy downstream code paths
   demo_data["joints.right"] = joints_world if side == "right" else np.zeros_like(joints_world)
   demo_data["joints.left"] = joints_world if side == "left" else np.zeros_like(joints_world)


   # Provide dummy hand_qpos arrays (your code only uses [0][3:6] for wrist init)
   # We set them zeros; you can later replace with HaMeR wrist rotation if you want.
   demo_data["hand_qpos_right"] = np.zeros((T, 7), dtype=np.float32)  # at least length 6
   demo_data["hand_qpos_left"] = np.zeros((T, 7), dtype=np.float32)


   # No objects for real-world
   return demo_data




# ============================================================
# Retargeting + FK extraction + Kabsch fit (for NPZ viz) don't need
# ============================================================


def compute_base_pose_cam_from_fk_and_kpts6(
   achieved_link_pos_base: np.ndarray,  # (T,6,3)
   kpts6_cam: np.ndarray,               # (T,6,3)
   kpts6_valid: np.ndarray,             # (T,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   """
   Fit T_C_B per frame from FK base points -> camera points using Kabsch.
   Returns:
     right_base_pose_cam (T,4,4)
     base_pose_valid (T,)
     pose_fit_rmse_m (T,)
   """
   achieved_link_pos_base = np.asarray(achieved_link_pos_base, dtype=np.float64)
   kpts6_cam = np.asarray(kpts6_cam, dtype=np.float64)
   kpts6_valid = np.asarray(kpts6_valid, dtype=np.uint8)


   T = achieved_link_pos_base.shape[0]
   right_base_pose_cam = np.repeat(np.eye(4, dtype=np.float64)[None, ...], T, axis=0)
   base_pose_valid = np.zeros((T,), dtype=np.uint8)
   pose_fit_rmse_m = np.full((T,), np.nan, dtype=np.float64)


   for t in range(T):
       if int(kpts6_valid[t]) == 0:
           continue


       A = achieved_link_pos_base[t]  # base frame
       B = kpts6_cam[t]               # camera frame


       if not (isfinite_all(A) and isfinite_all(B)):
           continue


       # Need >=3 valid points
       valid_mask = np.array([np.linalg.norm(a) > 1e-6 for a in A], dtype=bool)
       if valid_mask.sum() < 3:
           continue


       A_f = A[valid_mask]
       B_f = B[valid_mask]


       try:
           T_C_B = kabsch_T_from_A_to_B(A_f, B_f)
           A_cam = transform_points(A_f, T_C_B)
           e = rmse_points(A_cam, B_f)


           right_base_pose_cam[t] = T_C_B
           base_pose_valid[t] = 1
           pose_fit_rmse_m[t] = e
       except Exception:
           continue


   return right_base_pose_cam, base_pose_valid, pose_fit_rmse_m




# ============================================================
# MAIN
# ============================================================


def main():
   ap = argparse.ArgumentParser()


   # Real demo folder with rgb/depth/K/cam_c2w
   ap.add_argument("--demo_dir", required=True, help="Folder containing rgb/, depth/, K.npy/cam_K.txt, optional cam_c2w.npy")
   ap.add_argument("--out_dir", required=True)


   # Frame range
   ap.add_argument("--start_step", type=int, default=0)
   ap.add_argument("--end_step", type=int, default=None)

   # Premanipulation frame: retarget this one frame fresh (no warm-start) for exact joint values
   ap.add_argument("--premanip_frame", type=int, default=None,
                   help="Absolute frame index (matching rgb filenames) to retarget fresh from scratch. "
                        "Joint values are solved independently with zero init — no sequential warm-start. "
                        "Saved to NPZ as premanip_hand_qpos / premanip_frame_t_i.")


   # Hand selection
   ap.add_argument("--hand", choices=["left", "right", "both"], default="right")


   # HaMeR pipeline params
   ap.add_argument("--rescale_factor", type=float, default=2.0)
   ap.add_argument("--vit_threshold", type=float, default=0.0)
   ap.add_argument("--bbox_threshold", type=float, default=0.3,
                   help="Min keypoint confidence for bounding-box computation (should be >= vit_threshold). "
                        "Higher values exclude scattered uncertain keypoints from the bbox, keeping it tight.")
   ap.add_argument("--consistent_hand_size", action="store_true", default=False)
   ap.add_argument("--fixed_hand_size", type=float, default=None)
   ap.add_argument("--debug", action="store_true", default=False)
   ap.add_argument("--no_grounded_sam", action="store_true", default=False)


   # Retargeting params
   ap.add_argument("--robot", type=str, default="inspire_hand")
   ap.add_argument("--retarget_type", type=str, default="vector", choices=["vector", "position"])
   ap.add_argument("--exclude_joints", type=str, default="")
   ap.add_argument("--constrain_arm_z", action="store_true", default=False)
   ap.add_argument("--z_threshold", type=float, default=0.0)


   # Workspace centering fallback (only used when robot base pose estimation is skipped).
   ap.add_argument("--workspace_center", type=str, default="none",
                   help="x,y,z target for mean wrist in world frame (or 'none' to skip centering)")

   # Robot base pose in the floor-aligned frame.
   # Format: "x,y,z,yaw_deg"  where z is the height of the robot base above the floor plane.
   # If not provided, base pose is estimated automatically from arm reach constraints.
   ap.add_argument("--robot_base_pose_floor", type=str, default=None,
                   help="Manual override: robot base pose in floor frame as 'x,y,z,yaw_deg'")
   ap.add_argument("--arm_max_reach", type=float, default=0.85,
                   help="Robot arm max reach in metres (used for auto base-pose estimation)")
   ap.add_argument("--arm_min_reach", type=float, default=0.25,
                   help="Robot arm min reach in metres (used for auto base-pose estimation)")
   ap.add_argument("--robot_base_z", type=float, default=0.0,
                   help="Robot base Z in the detected-plane frame (metres). "
                        "Default 0.0 (robot base at detected plane level). "
                        "Only override if the robot is mounted significantly above or below "
                        "the dominant horizontal surface in the scene.")
   ap.add_argument("--force_yaw", type=float, default=None,
                   help="Force a specific robot yaw (degrees) in the floor frame, "
                        "overriding automatic yaw estimation. Useful for debugging.")
   ap.add_argument("--orientation_weight", type=float, default=1.0,
                   help="Weight for wrist orientation error in the arm IK "
                        "(position_weight is fixed at 5.0). Default 1.0. "
                        "Set to 0.0 for position-only IK — recommended for egocentric "
                        "demos where HAMER orientation estimates are unreliable "
                        "due to close-up, unusual camera angles.")
   ap.add_argument("--object_hand_x", action="store_true", default=True,
                   help="Set robot base +x axis from first-frame object→wrist direction. "
                        "Requires --object_pos_megasam. Only affects yaw; z-up/floor logic unchanged.")
   ap.add_argument("--object_pos_megasam", type=str, default=None,
                   help="First-frame object position in MegaSAM world frame as 'x,y,z'. "
                        "Required when --object_hand_x is used.")
   ap.add_argument("--egocentric", action="store_true", default=False,
                   help="Treat the demo as egocentric (first-person / wrist-mounted camera). "
                        "Overrides palm-normal and object→wrist yaw heuristics: the robot base "
                        "is instead placed so the arm reaches straight in the camera-forward "
                        "direction, matching how the operator's arm extends in front of them.")


   # Saving
   ap.add_argument("--save_npz", action="store_true", default=True)
   ap.add_argument("--save_pointcloud", action="store_true", default=True, help="Save point cloud in NPZ")
   ap.add_argument("--z_max", type=float, default=2.0, help="Max depth for point cloud")

   # Pre-existing HaMeR cache — skip the full perception pipeline
   ap.add_argument("--hamer_cache", type=str, default=None,
                   help="Path to an existing hamer_cache.pkl to load directly, "
                        "skipping detectron/vitpose/hamer (useful when running "
                        "in an env without those dependencies installed).")


   args = ap.parse_args()


   # Build T_robot_world (world -> robot base frame) from constants.
   # Only apply translation — the robot base is aligned with world axes (no rotation).
   T_robot_world = None
   if ROBOT_POS_IN_WORLD is not None:
       pos = np.array(ROBOT_POS_IN_WORLD, dtype=np.float64)
       T_robot_world = np.eye(4, dtype=np.float64)
       T_robot_world[:3, 3] = -pos  # world -> robot base = subtract robot origin
       print(f"[robot_pose] T_robot_world (translation only): pos={pos}")


   os.makedirs(args.out_dir, exist_ok=True)


   # device
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


   # ----------------------
   # (A) Run HaMeR pipeline  (cached — only re-runs when inputs change)
   # ----------------------
   import pickle, hashlib
   if args.hamer_cache is not None:
       # --hamer_cache: load a pre-existing cache file, skip the perception pipeline.
       with open(args.hamer_cache, "rb") as _f:
           _cached_ext = pickle.load(_f)
       hamer_out = _cached_ext.get("hamer_out", _cached_ext)
       print(f"[hamer] Loaded HaMeR output from --hamer_cache {args.hamer_cache}")
   else:
       _hamer_cache_path = os.path.join(args.out_dir, "hamer_cache.pkl")
       # Cache key: parameters that affect hamer_out but NOT the IK/retargeting stage.
       _cache_key = hashlib.md5(str({
           "demo_dir": args.demo_dir,
           "hand": args.hand,
           "start_step": args.start_step,
           "end_step": args.end_step,
           "rescale_factor": args.rescale_factor,
           "vit_threshold": args.vit_threshold,
           "bbox_threshold": args.bbox_threshold,
           "consistent_hand_size": args.consistent_hand_size,
           "fixed_hand_size": args.fixed_hand_size,
           "no_grounded_sam": args.no_grounded_sam,
       }).encode()).hexdigest()

       if os.path.exists(_hamer_cache_path):
           with open(_hamer_cache_path, "rb") as _f:
               _cached = pickle.load(_f)
           if _cached.get("key") == _cache_key:
               print(f"[hamer] Loaded cached HaMeR output from {_hamer_cache_path}")
               hamer_out = _cached["hamer_out"]
           else:
               print(f"[hamer] Cache key mismatch — re-running HaMeR pipeline")
               os.remove(_hamer_cache_path)
               _cached = None
       else:
           _cached = None

       if _cached is None or _cached.get("key") != _cache_key:
           detector, cpm, model, model_cfg, renderer, _faces, dino_model, sam_predictor = load_models(device)
           hamer_out = run_hamer_pipeline_on_range(
               demo_dir=args.demo_dir,
               out_dir=os.path.join(args.out_dir, "hamer_debug"),
               device=device,
               detector=detector,
               cpm=cpm,
               model=model,
               model_cfg=model_cfg,
               renderer=renderer,
               dino_model=dino_model,
               sam_predictor=sam_predictor,
               hand=args.hand,
               start_step=args.start_step,
               end_step=args.end_step,
               rescale_factor=args.rescale_factor,
               vit_threshold=args.vit_threshold,
               bbox_threshold=args.bbox_threshold,
               consistent_hand_size=args.consistent_hand_size,
               fixed_hand_size=args.fixed_hand_size,
               debug=args.debug,
               use_grounded_sam_mask=(not args.no_grounded_sam),
           )
           with open(_hamer_cache_path, "wb") as _f:
               pickle.dump({"key": _cache_key, "hamer_out": hamer_out}, _f)
           print(f"[hamer] Saved HaMeR output cache to {_hamer_cache_path}")

   union_indices = hamer_out["union_indices"]
   T = int(union_indices.shape[0])
   print(f"[hamer] frames={T} valid_kpts6={int(hamer_out['kpts6_valid'].sum())}/{T}")


   # Compute side and hand_name early — needed for IK-based base pose estimation.
   side_for_demo = "right" if args.hand in ["right", "both"] else "left"
   hand_name = args.robot if "hand" in args.robot else f"{args.robot}_hand"

   # IK solver and frame-0 arm config produced by base pose estimation (if arm-based).
   _ik_solver_early: Optional[ArmIKSolver] = None
   _q_init_arm_frame0: Optional[np.ndarray] = None

   # ---------------------------------------
   # (A2) Collect point clouds in MegaSAM world (cam_c2w), run floor detection,
   #      then apply floor_align_T to get Genesis Z-up world.
   # ---------------------------------------
   scene_points_world_seq = None
   scene_colors_seq = None
   floor_align_T = None

   rgb_dir = os.path.join(args.demo_dir, "rgb")
   depth_dir = os.path.join(args.demo_dir, "depth")
   K = hamer_out["K"]
   _cam_c2w_all = hamer_out.get("cam_c2w_all")

   if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
       print("[pointcloud] Collecting MegaSAM-world point clouds for floor detection...")
       scene_pts_megasam_seq: List[np.ndarray] = []
       scene_colors_raw: List[np.ndarray] = []

       # Subsample indices for floor detection — the floor is static so ~50 frames suffice.
       _max_floor_frames = 50
       if len(union_indices) > _max_floor_frames:
           _floor_stride = max(1, len(union_indices) // _max_floor_frames)
           _floor_t_indices = list(range(0, len(union_indices), _floor_stride))
           print(f"[pointcloud] Subsampling to {len(_floor_t_indices)} of {len(union_indices)} frames for floor detection")
       else:
           _floor_t_indices = list(range(len(union_indices)))

       # Collect point clouds for ALL frames (saved to NPZ); floor detection uses only the subsample.
       for t_i, step in enumerate(union_indices):
           img_path = os.path.join(rgb_dir, f"{int(step):06d}.png")
           depth_path = os.path.join(depth_dir, f"{int(step):06d}.png")
           empty_pts = np.zeros((0, 3), dtype=np.float32)
           empty_col = np.zeros((0, 3), dtype=np.float32)

           if not os.path.exists(img_path) or not os.path.exists(depth_path):
               scene_pts_megasam_seq.append(empty_pts)
               scene_colors_raw.append(empty_col)
               continue

           image_bgr = cv2.imread(img_path)
           depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
           if image_bgr is None or depth_raw is None:
               scene_pts_megasam_seq.append(empty_pts)
               scene_colors_raw.append(empty_col)
               continue

           image_rgb = image_bgr[..., ::-1].copy()
           depth = depth_raw.astype(np.float32) / 1000.0  # mm -> m
           if image_rgb.shape[:2] != depth.shape[:2]:
               image_rgb = cv2.resize(image_rgb, (depth.shape[1], depth.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)

           fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
           # Stride the pixels to keep point count manageable for floor detection.
           # 5000 points per frame is plenty for RANSAC; full-res would be millions of pts.
           _stride = max(1, int(np.sqrt(depth.shape[0] * depth.shape[1] / 5000)))
           depth_s = depth[::_stride, ::_stride]
           image_rgb_s = image_rgb[::_stride, ::_stride]
           valid_mask = ((depth_s > 0) & (depth_s < args.z_max)).flatten()
           u_g, v_g = np.meshgrid(np.arange(depth_s.shape[1]), np.arange(depth_s.shape[0]))
           u_g = (u_g.flatten()[valid_mask] * _stride).astype(np.float32)
           v_g = (v_g.flatten()[valid_mask] * _stride).astype(np.float32)
           z_v = depth_s.flatten()[valid_mask]
           x_v = (u_g - cx) * z_v / fx
           y_v = (v_g - cy) * z_v / fy
           points_cam = np.stack((x_v, y_v, z_v), axis=1)
           colors = (image_rgb_s.reshape(-1, 3)[valid_mask] / 255.0).astype(np.float32)

           # cam → MegaSAM world (use cam_c2w if available, else identity)
           c2w = _cam_c2w_all[int(step)].astype(np.float64) if _cam_c2w_all is not None else np.eye(4)
           pts_megasam = transform_points(points_cam.astype(np.float64), c2w).astype(np.float32)

           scene_pts_megasam_seq.append(pts_megasam)
           scene_colors_raw.append(colors)

       print(f"[pointcloud] Collected {len(scene_pts_megasam_seq)} frames in MegaSAM world")

       # --- Build hand keypoints in MegaSAM world for floor detection reference ---
       desired_cam_arr = hamer_out["desired_joint_pos_cam"]   # (T, 21, 3)
       valid_t = hamer_out["kpts6_valid"].astype(bool)
       hand_kpts_megasam_list = []
       for t_i in _floor_t_indices:
           step = union_indices[t_i]
           if not valid_t[t_i]:
               continue
           c2w = _cam_c2w_all[int(step)].astype(np.float64) if _cam_c2w_all is not None else np.eye(4)
           hand_kpts_megasam_list.append(transform_points(desired_cam_arr[t_i].astype(np.float64), c2w))

       hand_kpts_ref = np.concatenate(hand_kpts_megasam_list, axis=0) if hand_kpts_megasam_list else None

       # Camera right and up vectors in MegaSAM world frame (averaged over all frames for robustness).
       # camera +X (right) = first column of c2w rotation.
       # camera "up" in world  = -second column of c2w rotation  (camera Y is physically down).
       if _cam_c2w_all is not None:
           c2w_rots = [_cam_c2w_all[int(s)][:3, :3].astype(np.float64) for s in union_indices]
           cam_right_megasam   = np.mean([R[:, 0] for R in c2w_rots], axis=0)
           cam_up_megasam      = np.mean([-R[:, 1] for R in c2w_rots], axis=0)
           cam_forward_megasam = np.mean([R[:, 2] for R in c2w_rots], axis=0)
           cam_right_megasam   /= np.linalg.norm(cam_right_megasam)
           cam_up_megasam      /= np.linalg.norm(cam_up_megasam)
           cam_forward_megasam /= np.linalg.norm(cam_forward_megasam)
           print(f"[floor] Estimated world-up in MegaSAM frame: {cam_up_megasam.round(3)}")
           print(f"[floor] Estimated cam-forward in MegaSAM frame: {cam_forward_megasam.round(3)}")
       else:
           cam_right_megasam   = np.array([1.0, 0.0, 0.0])
           cam_up_megasam      = None
           cam_forward_megasam = None

       # --- Floor detection with canonical XY (wall at +Y = camera-forward direction) ---
       # Subsample to the floor detection frames so RANSAC isn't slowed by 193 clouds.
       _floor_pts_for_ransac = [scene_pts_megasam_seq[t_i] for t_i in _floor_t_indices]
       # cam_up_megasam constrains RANSAC to only find horizontal planes (no walls).
       print(f"[progress] Running floor RANSAC on {len(_floor_pts_for_ransac)} frames...", flush=True)
       floor_align_T = compute_floor_align_transform(
           _floor_pts_for_ransac, hand_kpts_ref,
           cam_right_megasam=cam_right_megasam,
           cam_up_megasam=cam_up_megasam,
           cam_forward_megasam=cam_forward_megasam,
       )

       print("[progress] Floor alignment done. Computing wrist positions in floor frame...", flush=True)
       # Camera-forward direction projected onto the floor XY plane.
       # Used for egocentric mode: the robot +X should align with this direction so
       # the arm reaches straight forward (same as the operator's reach direction).
       cam_forward_floor_xy = None
       if cam_forward_megasam is not None:
           _R_fl = floor_align_T[:3, :3]
           _cf_floor_3d = _R_fl @ cam_forward_megasam
           _cf_floor_xy = _cf_floor_3d[:2].copy()
           _cf_norm = float(np.linalg.norm(_cf_floor_xy))
           if _cf_norm > 0.1:
               cam_forward_floor_xy = _cf_floor_xy / _cf_norm
               # robot +Y = cam_forward → [-sin(yaw), cos(yaw)] = cf → yaw = atan2(-cf_x, cf_y)
               _cf_pref_yaw = np.degrees(np.arctan2(
                   -float(cam_forward_floor_xy[0]), float(cam_forward_floor_xy[1])))
               print(f"[floor] cam_forward in floor XY: {cam_forward_floor_xy.round(3)} "
                     f"(egocentric preferred yaw≈{_cf_pref_yaw:.1f}° so robot +Y ∥ cam_forward)")
           else:
               print(f"[floor] cam_forward projects near-vertical in floor frame "
                     f"(floor_XY_norm={_cf_norm:.3f}) — egocentric cam_forward override unavailable")

       # --- Robot base pose: manual override or automatic estimation ---
       # Compute wrist positions in floor-aligned frame.
       wrist_floor_list = []
       for kpts_m in hand_kpts_megasam_list:
           wrist_floor_list.append(transform_points(kpts_m[[0]], floor_align_T)[0])
       wrist_floor = np.array(wrist_floor_list) if wrist_floor_list else np.zeros((0, 3))

       # Compute wrist orientations in floor frame (same valid-frame filtering as wrist_floor).
       # Transform chain: MANO camera frame → MegaSAM world (c2w) → floor (floor_align_T).
       wrist_orient_floor = None
       _wrist_orient_cam_for_base = hamer_out.get("wrist_orient_cam")  # (T, 3, 3) or None
       if _wrist_orient_cam_for_base is not None and _cam_c2w_all is not None:
           R_floor_megasam = floor_align_T[:3, :3].astype(np.float64)
           _orient_floor_list = []
           for t_i, step in enumerate(union_indices):
               if not valid_t[t_i]:
                   continue
               c2w_R = _cam_c2w_all[int(step)][:3, :3].astype(np.float64)
               R_w_floor = R_floor_megasam @ c2w_R @ _wrist_orient_cam_for_base[t_i]
               _orient_floor_list.append(R_w_floor)
           if _orient_floor_list:
               wrist_orient_floor = np.stack(_orient_floor_list)  # (N_valid, 3, 3)
               print(f"[robot_base_ik] wrist_orient_floor computed for {len(wrist_orient_floor)} frames")
               # Sanity check: print frame-0 orientation axes
               R0 = wrist_orient_floor[0]
               print(f"[robot_base_ik] frame-0 wrist orientation in floor frame:")
               print(f"  X-axis (thumb): {R0[:,0].round(3)}")
               print(f"  Y-axis (dorsal): {R0[:,1].round(3)}")
               print(f"  Z-axis (palm normal): {R0[:,2].round(3)}")

       # Collect a subsample of scene points in floor frame for yaw disambiguation.
       # The correct robot yaw puts the scene directly in front (+X in robot frame).
       _scene_pts_floor_sample = None
       if scene_pts_megasam_seq is not None:
           _floor_chunks = []
           for pts in scene_pts_megasam_seq[:10]:  # use first 10 frames
               if len(pts) > 0:
                   _floor_chunks.append(transform_points(pts.astype(np.float64), floor_align_T))
           if _floor_chunks:
               _all_scene_floor = np.concatenate(_floor_chunks, axis=0)
               _rng = np.random.default_rng(42)
               _idx = _rng.choice(len(_all_scene_floor), min(5000, len(_all_scene_floor)), replace=False)
               _scene_pts_floor_sample = _all_scene_floor[_idx]
               print(f"[robot_base_ik] scene sample: {len(_scene_pts_floor_sample)} pts in floor frame, "
                     f"centroid=({_scene_pts_floor_sample[:,0].mean():.2f}, "
                     f"{_scene_pts_floor_sample[:,1].mean():.2f}, "
                     f"{_scene_pts_floor_sample[:,2].mean():.2f})")

       if args.robot_base_pose_floor is not None:
           vals = [float(v) for v in args.robot_base_pose_floor.split(',')]
           base_pos_floor = np.array([vals[0], vals[1], vals[2] if len(vals) > 2 else args.robot_base_z])
           yaw_floor = np.radians(vals[3]) if len(vals) > 3 else 0.0
           print(f"[robot_base] using manual pose: base={base_pos_floor.round(3)}, "
                 f"yaw={np.degrees(yaw_floor):.1f}°")
       elif args.force_yaw is not None:
           # --force_yaw overrides only the yaw; XY is still estimated from wrist centroid.
           _assets_dir = Path(__file__).resolve().parent.parent / "assets"
           _cfg_path_early = _assets_dir / hand_name / "retarget_config.yaml"
           if _cfg_path_early.exists() and _ik_solver_early is None:
               with _cfg_path_early.open() as _f_early:
                   _early_cfg = yaml.safe_load(_f_early)
               if _early_cfg.get("is_arm_based", False):
                   _early_arm_urdf = _early_cfg.get("arm_urdf", {}).get(side_for_demo)
                   if _early_arm_urdf is not None:
                       _early_urdf_path = str(_assets_dir / hand_name / _early_arm_urdf)
                       if Path(_early_urdf_path).exists():
                           try:
                               _ik_solver_early = ArmIKSolver(
                                   urdf_path=_early_urdf_path,
                                   ee_link_name=f"{side_for_demo}_hand_link",
                                   arm_joint_names=[f"Actuator{i}" for i in range(1, 8)],
                                   position_only=False,
                                   damping=1e-4, max_iters=200, eps=1e-3,
                               )
                           except Exception as _e:
                               print(f"[robot_base_ik] Warning: could not create IK solver: {_e}")
           # Compute natural-placement XY for the forced yaw.
           forced_yaw_rad = np.radians(args.force_yaw)
           yaw_floor = forced_yaw_rad
           if _ik_solver_early is not None:
               _ee_n = _ik_solver_early.get_ee_position(_ik_solver_early.q_default.copy())
               _wxy = wrist_floor[:, :2].mean(axis=0) if len(wrist_floor) > 0 else np.zeros(2)
               cy, sy = np.cos(forced_yaw_rad), np.sin(forced_yaw_rad)
               _rot_ee = np.array([cy * _ee_n[0] - sy * _ee_n[1],
                                   sy * _ee_n[0] + cy * _ee_n[1]])
               _bx = float(_wxy[0] - _rot_ee[0])
               _by = float(_wxy[1] - _rot_ee[1])
               base_pos_floor = np.array([_bx, _by, args.robot_base_z])
               print(f"[robot_base] --force_yaw={args.force_yaw:.1f}°: "
                     f"base=({_bx:.3f}, {_by:.3f}, {args.robot_base_z:.3f})")
           else:
               base_pos_floor = np.array([0.0, 0.0, args.robot_base_z])
       else:
           # Automatic estimation: use scene-centroid X-maximisation for yaw selection.
           _assets_dir = Path(__file__).resolve().parent.parent / "assets"
           _cfg_path_early = _assets_dir / hand_name / "retarget_config.yaml"
           if _cfg_path_early.exists() and _ik_solver_early is None:
               with _cfg_path_early.open() as _f_early:
                   _early_cfg = yaml.safe_load(_f_early)
               if _early_cfg.get("is_arm_based", False):
                   _early_arm_urdf = _early_cfg.get("arm_urdf", {}).get(side_for_demo)
                   if _early_arm_urdf is not None:
                       _early_urdf_path = str(_assets_dir / hand_name / _early_arm_urdf)
                       if Path(_early_urdf_path).exists():
                           try:
                               _ik_solver_early = ArmIKSolver(
                                   urdf_path=_early_urdf_path,
                                   ee_link_name=f"{side_for_demo}_hand_link",
                                   arm_joint_names=[f"Actuator{i}" for i in range(1, 8)],
                                   position_only=False,  # need 6DOF for per-frame orientation IK
                                   damping=1e-4, max_iters=200, eps=1e-3,
                               )
                           except Exception as _e:
                               print(f"[robot_base_ik] Warning: could not create IK solver: {_e}")

           if _ik_solver_early is not None:
               # In egocentric mode: use camera-forward as primary yaw scorer and skip
               # palm-normal scoring (which is misleading for first-person footage).
               _ego_cf_xy = cam_forward_floor_xy if args.egocentric else None
               _wrist_orients_for_base = None if args.egocentric else wrist_orient_floor
               if args.egocentric and cam_forward_floor_xy is not None:
                   print(f"[robot_base_ik] egocentric mode: using cam_forward_floor_xy="
                         f"{cam_forward_floor_xy.round(3)} as primary yaw scorer "
                         f"(palm-normal scoring disabled)")
               print("[progress] Estimating robot base pose via IK yaw sweep...", flush=True)
               base_pos_floor, yaw_floor, _q_init_arm_frame0 = solve_robot_base_pose_via_ik(
                   wrist_positions_floor=wrist_floor,
                   ik_solver=_ik_solver_early,
                   scene_pts_floor=_scene_pts_floor_sample,
                   wrist_orientations_floor=_wrist_orients_for_base,
                   robot_base_z=args.robot_base_z,
                   cam_forward_floor_xy=_ego_cf_xy,
               )
           else:
               base_pos_floor, yaw_floor = estimate_robot_base_pose_floor(
                   wrist_floor,
                   arm_max_reach=args.arm_max_reach,
                   arm_min_reach=args.arm_min_reach,
                   robot_base_z=args.robot_base_z,
               )

       # --- [--object_hand_x] Candidate scoring: object near (x≈0, y>0), robot +x ∥ object→wrist ---
       # Only active when --object_hand_x is passed.
       #
       # Key idea: for each yaw candidate, analytically place the base so the first-frame object
       # lands at (Xr≈0, Yr_obj) in robot frame.  Then score each candidate:
       #   - penalise |Xr_obj|          (want object near x=0)
       #   - penalise Yr_obj out of reach window   (want object at reachable positive y)
       #   - soft alignment penalty     (prefer robot +x ∥ object→wrist direction)
       #
       # Base placement per candidate (object-centred):
       #   robot +Y in floor = [-sin(yaw), cos(yaw)]
       #   base_floor = object_floor_xy - Yr_obj * robot_+Y_floor
       # This guarantees object_robot_xy = (0, Yr_obj) by construction.
       #
       # Yr_obj chosen so first-frame wrist is at arm_target distance from base:
       #   v_ow = wrist_floor_1st_xy - object_floor_xy   (object→wrist vector in floor XY)
       #   (a, b) = Rz(-yaw) @ v_ow      (object→wrist in robot XY)
       #   wrist_robot_xy = (a, b + Yr_obj)
       #   |wrist_robot|² = a² + (b+Yr_obj)² + z² = arm_target²
       #   → Yr_obj = sqrt(arm_target² - a² - z²) - b
       _pre_ohx_yaw  = yaw_floor
       _pre_ohx_base = base_pos_floor.copy()

       if args.object_hand_x:
           if len(wrist_floor) == 0:
               print("[object_hand_x] WARNING: no valid wrist frames. "
                     "Ignoring and keeping existing yaw/base.")
           else:
               try:
                   if args.object_pos_megasam is not None:
                       import re as _re
                       _obj_megasam = np.array(
                           [float(v) for v in _re.split(r'[\s,]+', args.object_pos_megasam.strip())],
                           dtype=np.float64,
                       )
                       if _obj_megasam.shape != (3,):
                           raise ValueError(f"Expected 3 values, got shape {_obj_megasam.shape}")
                       _obj_floor = transform_points(_obj_megasam.reshape(1, 3), floor_align_T)[0]
                       print(f"[object_hand_x] object_pos_megasam = {_obj_megasam.round(4)}")
                   elif hamer_out.get("object_pos_megasam_auto") is not None:
                       # Use the object position auto-detected by Detectron in the first valid frame.
                       _obj_megasam_auto = hamer_out["object_pos_megasam_auto"]
                       _obj_floor = transform_points(_obj_megasam_auto.reshape(1, 3), floor_align_T)[0]
                       print(f"[object_hand_x] using auto-detected object: "
                             f"megasam={_obj_megasam_auto.round(3)}  floor={_obj_floor.round(3)}")
                   else:
                       print("[object_hand_x] WARNING: no object_pos_megasam and no auto-detected object. "
                             "Keeping existing yaw/base.")
                       raise ValueError("no object position available")

                   # Use the START of the approach phase rather than the closest frame.
                   # When the wrist is literally on top of the object (dist≈0) the
                   # obj→wrist XY vector is noise.  Instead find the first frame where
                   # the wrist enters within 3× the minimum distance — that gives a
                   # meaningful incoming direction before the hand reaches the object.
                   _wrist_dists_to_obj = np.linalg.norm(wrist_floor[:, :2] - _obj_floor[:2], axis=1)
                   _min_dist_val = float(_wrist_dists_to_obj.min())
                   _approach_thresh = max(_min_dist_val * 3.0, 0.08)  # ≥8 cm
                   _approach_frames = np.where(_wrist_dists_to_obj < _approach_thresh)[0]
                   _closest_idx = int(_approach_frames[0]) if len(_approach_frames) > 0 else int(np.argmin(_wrist_dists_to_obj))
                   _wrist_floor_ref = wrist_floor[_closest_idx]

                   # object→wrist in floor XY (desired robot +x direction: arm reach direction)
                   _v_wo_xy  = (_wrist_floor_ref[:2] - _obj_floor[:2]).astype(np.float64)
                   _norm_wo  = float(np.linalg.norm(_v_wo_xy))

                   print("[object_hand_x] --- object_hand_x mode ---")
                   print(f"[object_hand_x] ref frame (closest wrist→obj): idx={_closest_idx}  "
                         f"dist={_wrist_dists_to_obj[_closest_idx]:.3f}m")
                   print(f"[object_hand_x] object_pos_floor         = {_obj_floor.round(4)}")
                   print(f"[object_hand_x] wrist_floor[ref]         = {_wrist_floor_ref.round(4)}")
                   print(f"[object_hand_x] object→wrist XY in floor = {_v_wo_xy.round(4)}  (norm={_norm_wo:.4f})")

                   if _norm_wo < 1e-3:
                       print("[object_hand_x] WARNING: near-zero vector — keeping existing yaw/base.")
                   else:
                       # In egocentric mode the approach direction from the video is just an
                       # artifact of the camera being on the operator's wrist.  Use the
                       # camera-forward direction (= how the arm extends in the real world)
                       # as the desired robot +X instead.
                       _ego_mode = args.egocentric and cam_forward_floor_xy is not None
                       if _ego_mode:
                           _desired_x_floor = cam_forward_floor_xy.copy()
                           print(f"[object_hand_x] egocentric: desired floor +x = cam_forward = "
                                 f"{_desired_x_floor.round(4)}")
                       else:
                           _desired_x_floor = _v_wo_xy / _norm_wo
                           print(f"[object_hand_x] desired floor +x (obj→wrist) = {_desired_x_floor.round(4)}")

                       # wrist height above robot base (for arm-reach formula)
                       _wrist_z_ohx = float(_wrist_floor_ref[2]) - args.robot_base_z
                       # target arm reach: midpoint of [min, max]
                       _arm_target  = 0.5 * (args.arm_max_reach + args.arm_min_reach)

                       # Scoring weights:
                       #   b²        — Y component of (object→wrist) in robot frame; want b≈0
                       #               (replaced by cam_forward alignment in egocentric mode)
                       #   Yr        — Yr_obj outside reachable window
                       #   reach     — full wrist trajectory reach error across all frames,
                       #               prevents picking a yaw good for frame 0 but leaving the
                       #               rest of the trajectory unreachable (e.g. negative-Y wrist)
                       _w_b     = 10.0  # primary: keep object→wrist aligned with +x (b≈0)
                       _w_cf    = 10.0  # primary (egocentric): cam_forward alignment
                       _w_y     =  3.0  # secondary: Yr_obj in [arm_min_reach, arm_max_reach]
                       _w_reach =  0.1  # tertiary: full-trajectory reach penalty (per frame)

                       _n_cands    = 36
                       _best_yaw   = yaw_floor        # fallback = existing estimation
                       _best_base  = base_pos_floor.copy()
                       _best_score = np.inf

                       # Pre-filter wrist_floor to valid frames for trajectory cost
                       _wf_valid = wrist_floor[
                           np.all(np.isfinite(wrist_floor), axis=1) &
                           np.any(wrist_floor != 0, axis=1)
                       ]
                       _n_valid = max(len(_wf_valid), 1)

                       print(f"[object_hand_x] {'yaw':>8}  {'a':>7}  {'b':>7}  "
                             f"{'Yr_obj':>7}  {'reach':>7}  {'score':>8}")

                       for _ci in range(_n_cands):
                           _yaw_c = float(np.radians(_ci * 360.0 / _n_cands))
                           _cy_c  = np.cos(_yaw_c)
                           _sy_c  = np.sin(_yaw_c)

                           # (a, b) = Rz(-yaw) @ v_ow  (object→wrist in robot XY)
                           # Rz(-yaw) = [[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]]
                           _a_c = float( _cy_c * _v_wo_xy[0] + _sy_c * _v_wo_xy[1])  # X
                           _b_c = float(-_sy_c * _v_wo_xy[0] + _cy_c * _v_wo_xy[1])  # Y

                           if not _ego_mode:
                               # Constraint 1 (exocentric): object→wrist must point in +x (a > 0).
                               # Skip candidates where a ≤ 0 (arm would face wrong way).
                               if _a_c <= 0:
                                   continue

                           # Yr_obj: place base so wrist is at arm_target distance.
                           #   Object at (0, Yr_obj) in robot frame (arm extends in +Y).
                           #   wrist_robot_xy = (a, b + Yr_obj)
                           #   a² + (b+Yr_obj)² + z² = arm_target²
                           #   Yr_obj = sqrt(arm_target² - a² - z²) - b
                           _inner_c = _arm_target**2 - _a_c**2 - _wrist_z_ohx**2
                           if _inner_c < 0:
                               _inner_c = args.arm_max_reach**2 - _a_c**2 - _wrist_z_ohx**2
                           if _inner_c < 0:
                               continue  # wrist unreachable at this yaw; skip
                           _Yr_obj_c = float(np.sqrt(_inner_c)) - _b_c

                           # In egocentric mode ensure the object is in front of the base (+Y).
                           if _ego_mode and _Yr_obj_c <= 0:
                               continue

                           # Base placement — object at (0, Yr_obj) by construction.
                           # robot +Y in floor = [-sin(yaw), cos(yaw)]
                           _y_dir_c   = np.array([-_sy_c, _cy_c])
                           _base_xy_c = _obj_floor[:2] - _Yr_obj_c * _y_dir_c
                           _base_c    = np.array([float(_base_xy_c[0]),
                                                  float(_base_xy_c[1]),
                                                  args.robot_base_z])

                           # Yr_obj feasibility penalty
                           _y_pen_c = (max(0.0, args.arm_min_reach - _Yr_obj_c)**2 +
                                       max(0.0, _Yr_obj_c - args.arm_max_reach)**2)

                           # Full-trajectory reach penalty: penalise frames where the wrist is
                           # outside [arm_min_reach, arm_max_reach] from this candidate base.
                           _wf_dists = np.linalg.norm(_wf_valid - _base_c, axis=1)
                           _too_far  = np.maximum(0.0, _wf_dists - args.arm_max_reach)
                           _too_near = np.maximum(0.0, args.arm_min_reach - _wf_dists)
                           _reach_c  = float(np.sum(_too_far**2) + 0.5 * np.sum(_too_near**2)) / _n_valid

                           if _ego_mode:
                               # Primary: maximize alignment of robot +Y with camera-forward.
                               # robot +Y in floor = [-sin(yaw), cos(yaw)]
                               # dot(robot_+Y_floor, cam_fwd) = -sin(yaw)*cf_x + cos(yaw)*cf_y
                               _cf_align_c = float(
                                   -_sy_c * cam_forward_floor_xy[0] + _cy_c * cam_forward_floor_xy[1]
                               )
                               _score_c = _w_cf * (1.0 - _cf_align_c) + _w_y * _y_pen_c + _w_reach * _reach_c
                           else:
                               # Score: penalise b² + Yr out-of-window + full-trajectory reach
                               _score_c = _w_b * _b_c**2 + _w_y * _y_pen_c + _w_reach * _reach_c

                           print(f"[object_hand_x] {np.degrees(_yaw_c):>7.1f}°  "
                                 f"{_a_c:>7.3f}  {_b_c:>7.3f}  "
                                 f"{_Yr_obj_c:>7.3f}  {_reach_c:>7.4f}  {_score_c:>8.4f}")

                           if _score_c < _best_score:
                               _best_score = _score_c
                               _best_yaw   = _yaw_c
                               _best_base  = _base_c

                       print(f"[object_hand_x] ← best: yaw={np.degrees(_best_yaw):.1f}°  "
                             f"base={_best_base.round(3)}  score={_best_score:.4f}")
                       yaw_floor      = _best_yaw
                       base_pos_floor = _best_base

               except Exception as _e_ohx:
                   print(f"[object_hand_x] ERROR: {_e_ohx}. Keeping existing yaw/base.")

           # Recompute frame-0 warm-start for the final yaw/base (which may have been
           # changed by object_hand_x). Without this, _q_init_arm_frame0 was computed
           # for the old yaw/base and is a bad starting point for the new pose.
           if (_ik_solver_early is not None and len(wrist_floor) > 0 and
                   (yaw_floor != _pre_ohx_yaw or not np.allclose(base_pos_floor, _pre_ohx_base))):
               _valid_wf = wrist_floor[
                   np.all(np.isfinite(wrist_floor), axis=1) &
                   np.any(wrist_floor != 0, axis=1)
               ]
               if len(_valid_wf) > 0:
                   _cy_v = np.cos(-yaw_floor); _sy_v = np.sin(-yaw_floor)
                   _Rz_v = np.array([[_cy_v, -_sy_v, 0.], [_sy_v, _cy_v, 0.], [0., 0., 1.]])
                   _base_v = np.array([base_pos_floor[0], base_pos_floor[1], args.robot_base_z])
                   _wf_robot = np.stack([_Rz_v @ (w - _base_v) for w in _valid_wf])
                   # Use a natural reaching warm-start (elbow bent forward) to
                   # stay in the correct configuration branch, falling back to
                   # the prior _q_init_arm_frame0 if it's already in the right branch.
                   _q_warm = _ik_solver_early.q_default.copy()
                   # Apply natural reaching bias: only Actuator4 (elbow) to -1.5 rad
                   if len(_ik_solver_early.arm_dof_ids) >= 4:
                       _q_warm[_ik_solver_early.arm_dof_ids[3]] = -1.5  # Actuator4: elbow down
                   if _q_init_arm_frame0 is not None and _q_init_arm_frame0[3] < 0:
                       # Previous solution already in natural branch — use it as warm-start
                       _q_warm = _ik_solver_early.set_arm_joints(_ik_solver_early.q_default.copy(),
                                                                   _q_init_arm_frame0)
                   _q_sol_r, _, _ = _ik_solver_early.solve_ik_position(
                       _wf_robot[0],
                       q_init=_q_warm,
                       reg_weight=0.0,
                   )
                   _q_init_arm_frame0 = _ik_solver_early.extract_arm_joints(_q_sol_r)
                   print(f"[object_hand_x] Recomputed frame-0 warm-start for new base "
                         f"yaw={np.degrees(yaw_floor):.1f}° from prior configuration branch")

       # --- Build the full MegaSAM → robot-base transform ---
       T_robotbase_megasam = build_T_robotbase_megasam(floor_align_T, base_pos_floor, yaw_floor)
       print(f"[T_robotbase_megasam] rotation:\n{T_robotbase_megasam[:3,:3].round(3)}")
       print(f"[T_robotbase_megasam] translation: {T_robotbase_megasam[:3,3].round(3)}")


       # Diagnostic: wrist positions in floor frame and robot frame
       if len(wrist_floor) > 0:
           _wrist_megasam_arr = np.array([kpts[0] for kpts in hand_kpts_megasam_list])
           _wrist_robot_arr = transform_points(_wrist_megasam_arr.astype(np.float64), T_robotbase_megasam)
           _wz_mean = _wrist_robot_arr[:, 2].mean()
           print(f"[diag] Wrist in floor frame — Z mean={wrist_floor[:,2].mean():.3f} "
                 f"range=[{wrist_floor[:,2].min():.3f}, {wrist_floor[:,2].max():.3f}]")
           print(f"[diag] Wrist in robot frame — Z mean={_wz_mean:.3f} "
                 f"range=[{_wrist_robot_arr[:,2].min():.3f}, {_wrist_robot_arr[:,2].max():.3f}], "
                 f"XY: X=[{_wrist_robot_arr[:,0].min():.3f},{_wrist_robot_arr[:,0].max():.3f}] "
                 f"Y=[{_wrist_robot_arr[:,1].min():.3f},{_wrist_robot_arr[:,1].max():.3f}]")
           if _wz_mean < 0.3:
               print(f"[diag] *** WARNING: wrist mean Z={_wz_mean:.3f}m is LOW — "
                     f"arm will be severely folded. Check floor detection. ***")

       if args.save_pointcloud:
           # Apply T_robotbase_megasam to get points in robot frame
           scene_points_world_seq = [
               transform_points(pts.astype(np.float64), T_robotbase_megasam).astype(np.float32)
               if len(pts) > 0 else pts
               for pts in scene_pts_megasam_seq
           ]
           scene_colors_seq = scene_colors_raw
           print(f"[pointcloud] Applied T_robotbase_megasam to {len(scene_points_world_seq)} frames")
           # Diagnostic: show XYZ range of point cloud in robot frame
           _non_empty = [p for p in scene_points_world_seq if len(p) > 0]
           if _non_empty:
               _all_pts = np.concatenate(_non_empty[:5])  # sample first 5 frames
               print(f"[diag] Point cloud in robot frame — "
                     f"X: [{_all_pts[:,0].min():.3f}, {_all_pts[:,0].max():.3f}]  "
                     f"Y: [{_all_pts[:,1].min():.3f}, {_all_pts[:,1].max():.3f}]  "
                     f"Z: [{_all_pts[:,2].min():.3f}, {np.percentile(_all_pts[:,2],5):.3f} (5th pct), "
                     f"{np.percentile(_all_pts[:,2],50):.3f} (median), {_all_pts[:,2].max():.3f}]")
               print(f"[diag] Scene centroid in robot frame: "
                     f"({_all_pts[:,0].mean():.3f}, {_all_pts[:,1].mean():.3f}, {_all_pts[:,2].mean():.3f})")
   else:
       print(f"[pointcloud] Warning: rgb/depth dirs not found in {args.demo_dir}")
       floor_align_T = None
       T_robotbase_megasam = None
       base_pos_floor = np.zeros(3)
       yaw_floor = 0.0

   # ---------------------------------------
   # (B) Build demo_data: hand keypoints in robot base frame
   # ---------------------------------------
   print("[progress] Transforming hand keypoints to robot frame...", flush=True)
   demo_data = build_demo_data_from_hamer(
       hamer_out=hamer_out,
       side=side_for_demo,
       T_target_from_megasam=T_robotbase_megasam,
   )

   # Forward-fill invalid frames so the IK pre-initializes near the correct configuration.
   # Without this, frames before the hand enters the scene have garbage (zero-keypoint) positions
   # in robot frame. The IK solver drives the arm toward those garbage targets, and EMA smoothing
   # then causes it to lag behind the real trajectory for many frames once valid frames start.
   # Forward-fill replaces invalid frames with the nearest future valid frame's keypoints so the
   # arm starts from the right place.
   _kpts6_valid = hamer_out["kpts6_valid"].astype(bool)
   _joints_key = f"joints.{side_for_demo}"
   if not _kpts6_valid.all() and _kpts6_valid.any():
       _joints_filled = demo_data[_joints_key].copy()
       # Forward-fill: replace each invalid frame with the next valid frame's keypoints.
       _first_valid_kpts = _joints_filled[_kpts6_valid][0]
       for _fi in range(len(_kpts6_valid)):
           if not _kpts6_valid[_fi]:
               _joints_filled[_fi] = _first_valid_kpts
           else:
               _first_valid_kpts = _joints_filled[_fi]
       demo_data[_joints_key] = _joints_filled
       print(f"[fill_invalid] Forward-filled {(~_kpts6_valid).sum()} invalid frames "
             f"with nearest valid keypoints.")

   # Debug: wrist / finger direction in robot frame
   _joints_rb = demo_data[f"joints.{side_for_demo}"]
   _valid_rb = hamer_out["kpts6_valid"].astype(bool)
   if _valid_rb.sum() > 0:
       _fvec = (_joints_rb[_valid_rb, 8, :] - _joints_rb[_valid_rb, 0, :]).mean(axis=0)
       print(f"[robot frame] mean index-finger vec: {_fvec.round(3)}  "
             f"(dominant -Z means fingers pointing down, good for top-down grasps)")
       print(f"[robot frame] wrist range  x=[{_joints_rb[_valid_rb,0,0].min():.3f}, "
             f"{_joints_rb[_valid_rb,0,0].max():.3f}]  "
             f"y=[{_joints_rb[_valid_rb,0,1].min():.3f}, {_joints_rb[_valid_rb,0,1].max():.3f}]  "
             f"z=[{_joints_rb[_valid_rb,0,2].min():.3f}, {_joints_rb[_valid_rb,0,2].max():.3f}]")

   # ---------------------------------------
   # (B2) Optional workspace centering (skipped when robot base pose is available)
   # ---------------------------------------
   scene_offset = np.zeros(3)
   if T_robotbase_megasam is not None:
       print("[robot_base] Skipping workspace centering — using T_robotbase_megasam.")
   elif args.workspace_center.lower() != "none":
       workspace_center = np.array([float(v) for v in args.workspace_center.split(",")], dtype=np.float64)
       joint_positions = demo_data[f"joints.{side_for_demo}"]
       valid_mask = hamer_out["kpts6_valid"].astype(bool)
       if valid_mask.sum() > 0:
           wrist_centroid = joint_positions[valid_mask, 0, :].mean(axis=0)
           scene_offset = workspace_center - wrist_centroid
           print(f"[workspace] wrist centroid={wrist_centroid.round(3)}, "
                 f"target={workspace_center}, offset={scene_offset.round(3)}")
           demo_data[f"joints.{side_for_demo}"] = joint_positions + scene_offset
           if scene_points_world_seq is not None:
               scene_points_world_seq = [pts + scene_offset.astype(np.float32)
                                         for pts in scene_points_world_seq]
       else:
           print("[workspace] No valid frames, skipping workspace centering.")


   num_envs = T


   print("[progress] Building retargeter...", flush=True)
   excluded_joint_names = [n.strip() for n in args.exclude_joints.split(",") if n.strip()]

   # Peek at config to detect is_arm_based before building retargeter so we can
   # auto-exclude arm joints (they will be solved via IK, not kinematic retargeting).
   _config_path_peek = Path(__file__).resolve().parent.parent / "assets" / hand_name / "retarget_config.yaml"
   if _config_path_peek.exists():
       with _config_path_peek.open() as _f:
           _peek_cfg = yaml.safe_load(_f)
       if _peek_cfg.get("is_arm_based", False):
           _arm_joints = [f"Actuator{i}" for i in range(1, 8)]
           excluded_joint_names = list(set(excluded_joint_names + _arm_joints))
           print(f"[arm_based] Auto-excluding arm joints from kinematic retargeting: {_arm_joints}")

   # ---------------------------------------
   # (C) Build retargeter (no Genesis)
   # ---------------------------------------
   retargeter, is_arm_based, retarget_input_cfg, retarget_cfg_path = prepare_retarget_cfgs(
       hand_name=hand_name,
       side=side_for_demo,
       retarget_type=args.retarget_type,
       excluded_joint_names=excluded_joint_names,
   )
   print(f"[retarget] cfg={retarget_cfg_path} is_arm_based={is_arm_based} retarget_type={args.retarget_type}")


   # Get joint info directly from retargeter
   actuated_dof_names = list(retargeter.optimizer.robot.dof_joint_names)
   num_joints = len(actuated_dof_names)
   print(f"[progress] Retargeter ready: {num_joints} DOFs, {num_envs} frames to process.", flush=True)
   actuated_dof_idxs = list(range(num_joints))
   hand_init_qpos = torch.zeros(1, num_joints)
   dof_limits = (
       torch.full((num_joints,), -100.0),
       torch.full((num_joints,),  100.0),
   )


   # ---------------------------------------
   # (D) Retarget active hand only
   # ---------------------------------------
   joint_pos = demo_data[f"joints.{side_for_demo}"]  # (T,21,3)
   if is_arm_based:
       # For arm-based hands, finger retargeting must run AFTER arm IK so that
       # the optimizer uses the correct wrist orientation (see Step 5 below).
       # Skip the initial retarget and create a placeholder that will be filled in.
       print("[arm_based] Skipping initial finger retarget (will run after arm IK in Step 5).")
       _placeholder_qpos = np.zeros((num_envs, num_joints), dtype=np.float32)
       retargeted_vals = {
           'hand_qpos': _placeholder_qpos,
           'wrist_qpos': np.zeros((num_envs, 0), dtype=np.float32),
           'wrist_idxs': np.zeros((num_envs, 0), dtype=np.int64),
           'joint_angles': _placeholder_qpos.copy(),
           'achieved_link_pos': [[] for _ in range(num_envs)],
       }
   else:
       retargeted_vals = retarget_all_steps(
           dof_limits,
           hand_init_qpos=hand_init_qpos,
           actuated_dof_names=actuated_dof_names,
           actuated_dof_idxs=actuated_dof_idxs,
           retargeter=retargeter,
           num_steps=num_envs,
           joint_pos_demo=joint_pos,
           retarget_type=args.retarget_type,
           frame_start=0,
       )
   retar_data: Dict[str, Any] = {side_for_demo: retargeted_vals}
   retar_data[side_for_demo]['actuated_dof_names'] = actuated_dof_names
   retar_data[side_for_demo]['actuated_dof_idxs'] = actuated_dof_idxs

   # Arm-based variables saved for premanip single-frame retarget (filled inside is_arm_based block)
   _pm_xhand_retargeter = None
   _pm_xhand_dof_names = None
   _pm_ordered = None
   _pm_arm_urdf_path = None

   # arm IK via floating hand retargeting (xhand_6dof) + solve_arm_ik_from_floating_hand
   if is_arm_based:
       # Allow per-hand config to override CLI orientation_weight (e.g. 0.0 for position-only IK)
       _ik_orientation_weight = retarget_input_cfg.get("ik_constraints", {}).get(
           "orientation_weight", args.orientation_weight
       )
       if _ik_orientation_weight != args.orientation_weight:
           print(f"[arm_based] Config overrides orientation_weight: {args.orientation_weight} -> {_ik_orientation_weight}")
       print(f"[arm_based] Running floating hand retargeting (xhand) + arm IK ...")
       _cfg_dir = Path(retarget_cfg_path).parent
       _arm_urdf_cfg = retarget_input_cfg.get("arm_urdf", {})
       _rel_urdf = _arm_urdf_cfg.get(side_for_demo)
       if _rel_urdf is None:
           print(f"[arm_based] WARNING: no arm_urdf for side '{side_for_demo}' in config, skipping arm IK")
       else:
           _urdf_path = str(_cfg_dir / _rel_urdf)
           if not Path(_urdf_path).exists():
               print(f"[arm_based] WARNING: URDF not found at {_urdf_path}, skipping arm IK")
           else:
               # Step 1: Build floating hand retargeter.
               # For wuji: use kinova_wuji_hand_float (wuji geometry) so the
               # optimizer finds the wrist orientation that's compatible with wuji
               # finger directions, not xhand. This fixes the 112° thumb misalignment
               # caused by using the xhand-calibrated euler_to_rotation_matrix.
               _float_hand_name = "kinova_wuji_hand_float" if "wuji" in hand_name else "xhand"
               _xhand_retargeter, _, _, _ = prepare_retarget_cfgs(
                   hand_name=_float_hand_name,
                   side=side_for_demo,
                   retarget_type=args.retarget_type,
                   excluded_joint_names=[],
               )
               _xhand_dof_names = list(_xhand_retargeter.optimizer.robot.dof_joint_names)
               _xhand_n = len(_xhand_dof_names)
               _xhand_init_qpos = torch.zeros(1, _xhand_n)
               _xhand_dof_limits = (
                   torch.full((_xhand_n,), -100.0),
                   torch.full((_xhand_n,),  100.0),
               )
               print(f"[arm_based] Floating hand retargeting ({_float_hand_name}) with {_xhand_n} DOFs, {num_envs} frames...", flush=True)

               # Step 2: Run floating hand retargeting on same joint_pos (already in robot frame)
               _xhand_retargeted = retarget_all_steps(
                   _xhand_dof_limits,
                   hand_init_qpos=_xhand_init_qpos,
                   actuated_dof_names=_xhand_dof_names,
                   actuated_dof_idxs=list(range(_xhand_n)),
                   retargeter=_xhand_retargeter,
                   num_steps=num_envs,
                   joint_pos_demo=joint_pos,
                   retarget_type=args.retarget_type,
                   frame_start=0,
               )
               _xhand_qpos = _xhand_retargeted['hand_qpos']  # (N, _xhand_n)

               # Step 3: Reorder columns to [arm_dof_joints, hand_dof_joints] as expected by
               # solve_arm_ik_from_floating_hand (first 6 = wrist tx/ty/tz/roll/pitch/yaw)
               _arm_dof_joints = retarget_input_cfg.get("arm_dof_joints", {}).get(side_for_demo, [])
               _hand_dof_joints = retarget_input_cfg.get("hand_dof_joints", {}).get(side_for_demo, [])
               _ordered = _arm_dof_joints + _hand_dof_joints
               _col_indices = [_xhand_dof_names.index(j) for j in _ordered if j in _xhand_dof_names]
               _floating_hand_qpos = _xhand_qpos[:, _col_indices]  # (N, 18)
               print(f"[arm_based] Floating hand qpos shape: {_floating_hand_qpos.shape}, "
                     f"wrist tx/ty/tz range: {_floating_hand_qpos[:, :3].min(axis=0).round(3)} to "
                     f"{_floating_hand_qpos[:, :3].max(axis=0).round(3)}")

               # Step 3.5: Refine frame-0 warm-start using pose IK on the actual
               # frame-0 target (position + orientation from floating hand data).
               # A position-only warm-start from q_default can land in a subregion of
               # the natural branch that forces joints to limits when orientation is added.
               # Solving pose IK directly from the natural warm-start avoids this.
               try:
                   import pinocchio as pin
                   _ik_refine = ArmIKSolver(
                       urdf_path=_urdf_path,
                       ee_link_name=f"{side_for_demo}_hand_link",
                       arm_joint_names=[f"Actuator{i}" for i in range(1, 8)],
                       position_only=False,
                       damping=1e-4, max_iters=200, eps=1e-3,
                   )
                   _f0_pos = _floating_hand_qpos[0, :3].copy()
                   _f0_euler = _floating_hand_qpos[0:1, 3:6].copy()
                   _f0_rot = euler_to_rotation_matrix(_f0_euler, side_for_demo)[0]
                   # Apply Rx(pi) correction for wuji: fingers in +Z vs xhand -Z
                   from solve_ik_arm import _get_ee_frame_correction
                   _f0_ee_corr = _get_ee_frame_correction(retarget_input_cfg.get("hand_name"))
                   if _f0_ee_corr is not None:
                       _f0_rot = _f0_rot @ _f0_ee_corr
                   _f0_pose = pin.SE3(_f0_rot, _f0_pos)
                   print(f"[arm_based] Frame-0 orientation target (robot frame):")
                   print(f"  X-axis: {_f0_rot[:, 0].round(3)}")
                   print(f"  Y-axis: {_f0_rot[:, 1].round(3)}")
                   print(f"  Z-axis: {_f0_rot[:, 2].round(3)}")

                   # Two-stage warm-start:
                   # Stage 1: position-only → guarantees natural branch (Actuator4<0)
                   # Stage 2: use stage-1 result as init for soft orientation solve
                   # This prevents the orientation gradient from flipping to upside-down branch.
                   _q_default_warm = _ik_refine.q_default.copy()
                   if len(_ik_refine.arm_dof_ids) >= 4:
                       _q_default_warm[_ik_refine.arm_dof_ids[3]] = -1.5
                   # Stage 1: position-only
                   _f0_pose_pos_only = pin.SE3(_f0_rot.copy(), _f0_pos)
                   _q_pos_sol, _pos_ok, _pos_err = _ik_refine.solve_ik_pose(
                       target_pose=_f0_pose_pos_only,
                       q_init=_q_default_warm,
                       position_weight=5.0,
                       orientation_weight=0.0,
                       reg_weight=0.0,
                   )
                   _q_pos_arm = _ik_refine.extract_arm_joints(_q_pos_sol)
                   print(f"[arm_based] Frame-0 Stage-1 (pos-only) warm-start: ok={_pos_ok}, "
                         f"err={_pos_err:.4f}, Actuator4={_q_pos_arm[3]:.3f} "
                         f"({'natural' if _q_pos_arm[3] < 0 else 'upside-down'} branch)")
                   print(f"  joints: {np.round(_q_pos_arm, 3)}")
                   # Stage 2: orientation-constrained, starting from stage-1 result
                   _q_pose_sol, _pose_ok, _pose_err = _ik_refine.solve_ik_pose(
                       target_pose=_f0_pose,
                       q_init=_q_pos_sol,   # start from natural-branch position solution
                       position_weight=5.0,
                       orientation_weight=_ik_orientation_weight,
                       reg_weight=0.0,
                   )
                   _q_pose_arm = _ik_refine.extract_arm_joints(_q_pose_sol)
                   print(f"[arm_based] Frame-0 Stage-2 (orient={_ik_orientation_weight}) warm-start: "
                         f"ok={_pose_ok}, err={_pose_err:.4f}, Actuator4={_q_pose_arm[3]:.3f} "
                         f"({'natural' if _q_pose_arm[3] < 0 else 'upside-down'} branch)")
                   print(f"  joints: {np.round(_q_pose_arm, 3)}")
                   # Accept Stage-2 based on position-only error, not _pose_ok.
                   # _pose_ok requires full weighted SE3 < eps (1e-4), which is never
                   # satisfied when orientation_weight < 1.0 — orientation error alone
                   # pushes the weighted norm >> eps even when position is perfect.
                   _s2_ee_pos = _ik_refine.get_ee_position(_q_pose_sol)
                   _s2_pos_err = float(np.linalg.norm(_s2_ee_pos - _f0_pos))
                   # Accept Stage-2 if position is close enough AND no joints are past limits.
                   # Do NOT require A4 < 0 (natural branch): for side-approach grasps
                   # where fingers point upward, A4 > 0 may be the correct branch.
                   # Only strictly reject if A4 is very far into the upside-down region (>1.0).
                   _s2_ok = (np.max(np.abs(_q_pose_arm)) < np.pi and  # no beyond-limit joints
                             _q_pose_arm[3] < 1.0 and                 # not severely upside-down
                             _s2_pos_err < 0.015)
                   if _s2_ok:
                       _q_init_arm_frame0 = _q_pose_arm
                       print(f"[arm_based] Stage-2 accepted: pos_err={_s2_pos_err:.4f}m, "
                             f"Actuator4={_q_pose_arm[3]:.3f} "
                             f"({'natural' if _q_pose_arm[3] < 0 else 'slight-upside-down'})")
                   elif _q_pos_arm[3] < 0:
                       # Stage-1 is in the natural branch (Actuator4 < 0). Use it even when
                       # _pos_ok=False: solve_ik_pose returns full SE3 error including
                       # orientation mismatch (≈π rad) even with orientation_weight=0, so
                       # _pos_ok=False does NOT mean the position failed to converge.
                       print(f"[arm_based] Stage-1 natural branch (Actuator4={_q_pos_arm[3]:.3f}): "
                             f"using as warm-start (Stage-2 pos_err={_s2_pos_err:.4f}m too large or extreme joints)")
                       _q_init_arm_frame0 = _q_pos_arm
                   elif _pos_ok:
                       _q_init_arm_frame0 = _q_pos_arm
               except Exception as _e_pose:
                   print(f"[arm_based] Frame-0 pose-IK warm-start failed: {_e_pose}")

               # Step 4: Solve arm IK from floating hand wrist pose
               print(f"[arm_based] Solving arm IK for {num_envs} frames...", flush=True)
               retar_data = solve_arm_ik_from_floating_hand(
                   retar_data,
                   retarget_input_cfg,
                   hand_name,
                   floating_hand_qpos={side_for_demo: _floating_hand_qpos},
                   T_robot_world=None,  # joint_pos already in robot base frame
                   arm_urdf_paths={side_for_demo: _urdf_path},
                   initial_arm_joints=_q_init_arm_frame0,
                   orientation_weight=_ik_orientation_weight,
                   constrain_arm_z=args.constrain_arm_z,
                   z_threshold=args.z_threshold,
               )

               # Step 5: Re-run wuji finger retargeting with arm joints fixed to IK solution.
               # The initial finger retarget (step D) ran with arm at default (zero), giving
               # wrong finger orientations. Now that we have the correct arm IK solution, we
               # re-run finger retargeting with those arm joint values as fixed_qpos so the
               # optimizer uses the correct wrist orientation when solving finger angles.
               _fixed_jnames = list(retargeter.optimizer.fixed_joint_names)
               if len(_fixed_jnames) > 0:
                   _arm_joint_vals_ik = retar_data[side_for_demo]['hand_qpos']  # (N, num_joints)
                   _fixed_qpos_seq = np.zeros((num_envs, len(_fixed_jnames)))
                   for _fi, _fname in enumerate(_fixed_jnames):
                       if _fname in actuated_dof_names:
                           _ai = list(actuated_dof_names).index(_fname)
                           _fixed_qpos_seq[:, _fi] = _arm_joint_vals_ik[:, _ai]
                   # Log arm joint values used for finger retarget (frame 0)
                   _arm_dof_names_all = [f"Actuator{i}" for i in range(1, 8)]
                   _arm_f0_vals = {n: float(retar_data[side_for_demo]['hand_qpos'][0, list(actuated_dof_names).index(n)])
                                   for n in _arm_dof_names_all if n in actuated_dof_names}
                   print(f"[arm_based] Frame-0 arm joints for finger retarget: "
                         + " ".join(f"{k}={v:.3f}" for k,v in _arm_f0_vals.items()))
                   print(f"  Actuator4={'natural (<0)' if _arm_f0_vals.get('Actuator4',0)<0 else 'UPSIDE-DOWN (>0)'}")
                   print(f"[arm_based] Re-running finger retargeting with IK arm joints, "
                         f"{num_envs} frames (fixed: {_fixed_jnames[:3]}...)...", flush=True)
                   _finger_init_qpos = torch.zeros(1, num_joints)
                   _finger_retargeted = retarget_all_steps(
                       dof_limits,
                       hand_init_qpos=_finger_init_qpos,
                       actuated_dof_names=actuated_dof_names,
                       actuated_dof_idxs=actuated_dof_idxs,
                       retargeter=retargeter,
                       num_steps=num_envs,
                       joint_pos_demo=joint_pos,
                       retarget_type=args.retarget_type,
                       frame_start=0,
                       fixed_qpos_sequence=_fixed_qpos_seq,
                   )
                   # Update only finger (non-arm) joints; keep arm joints from IK
                   _arm_dof_set = set(_fixed_jnames)
                   _finger_indices = [i for i, n in enumerate(actuated_dof_names)
                                      if n not in _arm_dof_set]
                   retar_data[side_for_demo]['hand_qpos'][:, _finger_indices] = \
                       _finger_retargeted['hand_qpos'][:, _finger_indices]
                   print("[arm_based] Finger retargeting with IK arm joints complete.")

               # Save for premanip single-frame retarget below
               _pm_xhand_retargeter = _xhand_retargeter
               _pm_xhand_dof_names = _xhand_dof_names
               _pm_ordered = _ordered
               _pm_arm_urdf_path = _urdf_path

   # -----------------------------------------------
   # (D2) Fresh single-frame retarget: premanip frame
   # -----------------------------------------------
   # Retargets one specific frame completely from scratch (zero init, no warm-start).
   # This gives exact joint values for that frame independent of trajectory history.
   premanip_hand_qpos_save = None
   premanip_wrist_pos_robot_save = None
   premanip_frame_t_i_save = None
   premanip_frame_abs_save = None

   if args.premanip_frame is not None:
       _pm_abs = int(args.premanip_frame)
       _pm_matches = np.where(union_indices == _pm_abs)[0]
       if len(_pm_matches) == 0:
           print(f"[premanip] WARNING: frame {_pm_abs} not in processed range "
                 f"{union_indices[0]}..{union_indices[-1]}. Skipping.")
       else:
           _pm_t_i = int(_pm_matches[0])
           _pm_joint_pos = joint_pos[_pm_t_i]   # (21, 3) in robot frame
           _pm_wrist_pos = _pm_joint_pos[0].copy()
           print(f"[premanip] Fresh retarget for abs frame {_pm_abs} (t_i={_pm_t_i}), "
                 f"wrist_robot={_pm_wrist_pos.round(3)}")

           # Fresh hand retarget with zero init — no warm-start from neighbouring frames
           _pm_init_qpos = torch.zeros(1, num_joints)
           _pm_indices = retargeter.optimizer.target_link_human_indices
           _pm_ref_val = get_ref_val(_pm_joint_pos, _pm_indices)
           _, _pm_hand_qpos, _, _, _ = retarget_one_hand(
               wrist_pos=_pm_wrist_pos,
               retarget_type=args.retarget_type,
               ref_value=_pm_ref_val,
               retargeter=retargeter,
               hand_init_qpos=_pm_init_qpos,
               actuated_dof_names=actuated_dof_names,
               actuated_dof_idxs=actuated_dof_idxs,
           )
           _pm_hand_qpos = torch.clamp(_pm_hand_qpos, dof_limits[0], dof_limits[1])
           _pm_hand_qpos_np = _pm_hand_qpos[0].cpu().numpy().copy()

           if is_arm_based and _pm_xhand_retargeter is not None:
               # Fresh xhand retarget for this single frame
               _pm_xhand_dof_n = len(_pm_xhand_dof_names)
               _pm_xhand_init = torch.zeros(1, _pm_xhand_dof_n)
               _pm_xhand_indices = _pm_xhand_retargeter.optimizer.target_link_human_indices
               _pm_xhand_ref_val = get_ref_val(_pm_joint_pos, _pm_xhand_indices)
               _, _pm_xhand_qpos_t, _, _, _ = retarget_one_hand(
                   wrist_pos=_pm_wrist_pos,
                   retarget_type=args.retarget_type,
                   ref_value=_pm_xhand_ref_val,
                   retargeter=_pm_xhand_retargeter,
                   hand_init_qpos=_pm_xhand_init,
                   actuated_dof_names=_pm_xhand_dof_names,
                   actuated_dof_idxs=list(range(_pm_xhand_dof_n)),
               )
               _pm_xhand_np = _pm_xhand_qpos_t[0].cpu().numpy()
               _pm_col_idxs = [_pm_xhand_dof_names.index(j) for j in _pm_ordered if j in _pm_xhand_dof_names]
               _pm_floating_qpos_1 = _pm_xhand_np[_pm_col_idxs][np.newaxis, :]  # (1, N_ordered)

               # Fresh arm IK for this single frame — independent solve, no trajectory warm-start
               _pm_mini_retar = {side_for_demo: {
                   'hand_qpos': _pm_hand_qpos_np[np.newaxis, :],
                   'actuated_dof_names': actuated_dof_names,
                   'actuated_dof_idxs': actuated_dof_idxs,
               }}
               _pm_mini_retar = solve_arm_ik_from_floating_hand(
                   _pm_mini_retar,
                   retarget_input_cfg,
                   hand_name,
                   floating_hand_qpos={side_for_demo: _pm_floating_qpos_1},
                   T_robot_world=None,
                   arm_urdf_paths={side_for_demo: _pm_arm_urdf_path},
                   initial_arm_joints=None,  # fresh — no warm-start
                   orientation_weight=_ik_orientation_weight,
               )
               _pm_hand_qpos_np = _pm_mini_retar[side_for_demo]['hand_qpos'][0]

           premanip_hand_qpos_save = _pm_hand_qpos_np.astype(np.float32)
           premanip_wrist_pos_robot_save = _pm_wrist_pos.astype(np.float32)
           premanip_frame_t_i_save = _pm_t_i
           premanip_frame_abs_save = _pm_abs
           print(f"[premanip] Done. hand_qpos={premanip_hand_qpos_save.round(4)}")

   # ---------------------------------------
   # (E) Save outputs (.npz)
   # ---------------------------------------


   if args.save_npz:
       side_data = retar_data[side_for_demo]
       joint_names = np.asarray(side_data.get('actuated_dof_names', []), dtype=str)
       hand_qpos_np = np.asarray(side_data['hand_qpos'], dtype=np.float32)


       demo_name = os.path.basename(os.path.normpath(args.demo_dir))
       npz_path = os.path.join(args.out_dir, f"{demo_name}_retargeted_dexterous_{hand_name}_real.npz")


       # Prepare save dict
       save_dict = {
           'hand_side': np.array([side_for_demo]),
           'hand_qpos': hand_qpos_np,
           'joint_names': joint_names,
           'start_step': args.start_step,
           'end_step': args.end_step,
           'scene_offset': scene_offset.astype(np.float32),
       }
       # Save coordinate transforms so downstream stages can re-use them
       if T_robotbase_megasam is not None:
           save_dict['T_robotbase_megasam'] = T_robotbase_megasam.astype(np.float32)
           save_dict['T_floor_megasam'] = floor_align_T.astype(np.float32)
           save_dict['robot_base_pos_floor'] = base_pos_floor.astype(np.float32)
           save_dict['robot_base_yaw_floor'] = np.float32(yaw_floor)
           print(f"[saved] T_robotbase_megasam and T_floor_megasam")
       elif T_robot_world is not None:
           save_dict['T_robot_world'] = T_robot_world.astype(np.float32)
       else:
           save_dict['workspace_center'] = (
               np.array([float(v) for v in args.workspace_center.split(",")], dtype=np.float32)
               if args.workspace_center.lower() != "none"
               else np.zeros(3, dtype=np.float32)
           )

       # Add cam_c2w if available
       cam_c2w_all = hamer_out.get("cam_c2w_all")
       if cam_c2w_all is not None:
           cam_c2w_arr = np.array([cam_c2w_all[int(step)] for step in union_indices], dtype=np.float32)
           save_dict['cam_c2w'] = cam_c2w_arr
           print(f"[saved] cam_c2w shape: {cam_c2w_arr.shape}")

       # Add point cloud if available
       if scene_points_world_seq is not None and scene_colors_seq is not None:
           save_dict['scene_points_world'] = np.array(scene_points_world_seq, dtype=object)
           save_dict['scene_colors'] = np.array(scene_colors_seq, dtype=object)
           print(f"[pointcloud] Adding point cloud to NPZ ({len(scene_points_world_seq)} frames)")

       np.savez(
           npz_path,
           **save_dict
       )
       print(f"[saved] {npz_path}")

       # Save premanip frame as a separate NPZ alongside the main one
       if premanip_hand_qpos_save is not None:
           premanip_npz_path = os.path.join(
               os.path.dirname(npz_path),
               f"{demo_name}_retargeted_dexterous_{hand_name}_real_premanip_frame{premanip_frame_abs_save}.npz"
           )
           np.savez(
               premanip_npz_path,
               hand_qpos=premanip_hand_qpos_save,
               joint_names=joint_names,
               wrist_pos_robot=premanip_wrist_pos_robot_save,
               frame_t_i=np.int32(premanip_frame_t_i_save),
               frame_abs=np.int32(premanip_frame_abs_save),
               T_robotbase_megasam=T_robotbase_megasam.astype(np.float32) if T_robotbase_megasam is not None else np.eye(4, dtype=np.float32),
           )
           print(f"[premanip] Saved separate NPZ: {premanip_npz_path}")




if __name__ == "__main__":
   main()

