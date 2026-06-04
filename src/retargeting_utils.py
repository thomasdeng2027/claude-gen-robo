import os
import torch
import numpy as np
from tqdm import tqdm
from lxml import etree
from copy import deepcopy
from collections import defaultdict

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor

def get_link_names(urdf_path):
    assert os.path.exists(urdf_path), f"Does not exist: {urdf_path}"
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(urdf_path, parser=lxml_parser)
    robot_elem = tree.getroot()
    assert robot_elem.tag == "robot", "The first element should be a robot element."
    links = []
    for link_elem in robot_elem.findall("link"):
        if link_elem.find("collision") is not None and link_elem.attrib.get("name", None) is not None:
            links.append(link_elem.attrib["name"])
    return sorted(links)

def compose_retarget_config(
        input,
        retarget_type="vector",
        low_pass_alpha=1.0,
        scaling_factor=1.0,
        add_dummy_free_joint=False,
        ignore_mimic_joint=False,
        excluded_joint_names: list | None = None,
        is_arm_based: bool = False,
    ):
    """ write retargeting config given input type and offline cfgs"""
    assert retarget_type in ["position", "vector"], f"Unsupported: {retarget_type}"
    urdf_path = input.get("urdf_path", None)
    assert urdf_path is not None, f"Need input urdf path"

    finger_links = input.get('target_finger_links', [])
    human_finger_idxs = input.get('human_finger_idxs', [])
    assert len(finger_links) == len(human_finger_idxs) and len(finger_links) > 0
    origin_link = input.get('target_origin_link', None)
    human_origin_idx = input.get('human_origin_idx', None)
    assert origin_link is not None and human_origin_idx is not None
    origin_link = str(origin_link)
    human_origin_idx = int(human_origin_idx)
    cfg = dict(
        type=str(retarget_type),
        urdf_path=urdf_path,
        low_pass_alpha=low_pass_alpha,
        scaling_factor=scaling_factor,
        add_dummy_free_joint=add_dummy_free_joint,
        ignore_mimic_joint=ignore_mimic_joint,
    )

    if retarget_type == "position":
        # include wrist position for arm-based to solve IK
        cfg['target_link_names'] = finger_links
        cfg['target_link_human_indices'] = [int(idx) for idx in human_finger_idxs]
    elif retarget_type == "vector":
        num_fingers = len(finger_links)
        cfg['target_origin_link_names'] = [origin_link] * num_fingers
        cfg['target_task_link_names'] = finger_links
        cfg['target_link_human_indices'] = [
            [int(human_origin_idx) for _ in range(num_fingers)],
            [int(idx) for idx in human_finger_idxs]
        ]
    target_joints = input.get("target_joint_names", [])
    if len(target_joints) > 0:
        # If any joints are explicitly excluded, remove them from the
        # target joint list so the kinematic retargeter does not optimize
        # those DOFs (the retargeter will treat them as fixed).
        if excluded_joint_names is not None and len(excluded_joint_names) > 0:
            filtered = [j for j in target_joints if j not in excluded_joint_names]
            cfg['target_joint_names'] = filtered
        else:
            cfg['target_joint_names'] = target_joints
    return cfg

def get_ref_val(joint_pos, indices):
    if len(indices.shape) > 1:
        origin_ind = indices[0, :]
        task_ind = indices[1, :]
        return joint_pos[task_ind, :] - joint_pos[origin_ind, :]
    else: # shape is (num_joints,)
        return joint_pos[indices]

def get_demo_obj_tensors(loaded_data, step, device):
    obj_trans = loaded_data['params']['obj_trans'][step]
    obj_rot = loaded_data['params']['obj_quat'][step]
    obj_arti = loaded_data['params']['obj_arti'][step]
    obj_trans = torch.tensor(obj_trans, dtype=torch.float32, device=device)
    obj_rot = torch.tensor(obj_rot, dtype=torch.float32, device=device)
    obj_arti = torch.tensor(obj_arti, dtype=torch.float32, device=device)
    return obj_trans, obj_rot, obj_arti


def retarget_one_hand(
    wrist_pos,
    retarget_type,
    ref_value,
    retargeter,
    hand_init_qpos,
    actuated_dof_names,
    actuated_dof_idxs,
    fixed_qpos=None,
):
    wrist_pos = wrist_pos.copy() if hasattr(wrist_pos, 'copy') else np.array(wrist_pos)
    if retarget_type == "position":
        wrist_pos *= 0.0
    if fixed_qpos is None:
        fixed_qpos = np.zeros(len(retargeter.optimizer.fixed_joint_names))
    qpos_retargeted = retargeter.retarget(
        ref_value,
        fixed_qpos=fixed_qpos
    )
    
    optimizer_joint_names = retargeter.optimizer.robot.dof_joint_names
    idx_pin2target = retargeter.optimizer.idx_pin2target.tolist()
    if isinstance(retargeter.optimizer.adaptor, MimicJointKinematicAdaptor):
        idx_pin2target.extend(retargeter.optimizer.adaptor.idx_pin2mimic.tolist())

    fixed_joint_names = retargeter.optimizer.fixed_joint_names
    idx_pin2fixed = []
    for fname in fixed_joint_names:
        if fname in optimizer_joint_names:
            idx_pin2fixed.append(list(optimizer_joint_names).index(fname))

    joint_vals = dict()
    hand_qpos = hand_init_qpos.clone()
    wrist_idxs = []
    wrist_qpos = []

    # copy fixed joint values
    for i, idx in enumerate(idx_pin2fixed):
        jname = optimizer_joint_names[idx]
        val = fixed_qpos[i] if fixed_qpos is not None and i < len(fixed_qpos) else 0.0
        if jname in actuated_dof_names:
            joint_idx = list(actuated_dof_names).index(jname)
            hand_qpos[:, joint_idx] = val
            joint_vals[jname] = val

    # copy target joint values
    for idx in idx_pin2target:
        val = qpos_retargeted[idx]
        jname = optimizer_joint_names[idx]
        target_jname = jname
        if 'tx' in jname or "_x" in jname:
            val += wrist_pos[0]
        elif 'ty' in jname or "_y" in jname:
            val += wrist_pos[1]
        elif 'tz' in jname or "_z" in jname: # for mano hand
            val += wrist_pos[2]
        joint_idx = list(actuated_dof_names).index(target_jname)
        hand_qpos[:, joint_idx] = val
        joint_vals[target_jname] = val
        if 'forearm' in target_jname: # wrist qpos
            wrist_qpos.append(val)
            wrist_idxs.append(actuated_dof_idxs[joint_idx])
    
    # Compute FK to get achieved link positions (task links only, not origin links).
    # For VectorOptimizer: use task_link_indices (excludes origin/wrist link so cos
    # comparisons are meaningful).  For PositionOptimizer: use target_link_indices.
    retargeter.optimizer.robot.compute_forward_kinematics(qpos_retargeted)

    # task_link_indices: indices into computed_link_names (NOT pinocchio frame IDs).
    # computed_link_indices: the actual pinocchio frame IDs.
    # To look up FK poses we need computed_link_indices[task_link_idx].
    _computed_link_indices = getattr(retargeter.optimizer, 'computed_link_indices', None)

    if hasattr(retargeter.optimizer, 'task_link_indices'):
        # VectorOptimizer: task_link_indices are offsets into computed_link_indices.
        _task_idx = retargeter.optimizer.task_link_indices
        if _computed_link_indices is not None:
            link_indices = [_computed_link_indices[int(i)] for i in _task_idx]
        else:
            link_indices = _task_idx  # fallback (likely wrong, but avoids crash)
    elif hasattr(retargeter.optimizer, 'target_link_indices'):
        # PositionOptimizer: target_link_indices are direct pinocchio frame IDs.
        link_indices = retargeter.optimizer.target_link_indices
    else:
        print("[warning] Could not find task link indices, using empty list")
        link_indices = []

    # FK positions are in pinocchio world = robot base frame.
    # For floating hand retargeters (wrist at origin), add wrist_pos offset so
    # achieved positions are in the same frame as the human keypoints.
    # For arm-based retargeters (wrist already in world via arm joints), do NOT
    # add wrist_pos (FK already gives world-frame positions).
    is_floating = not any('Actuator' in str(n) for n in retargeter.optimizer.robot.dof_joint_names)
    pos_offset = wrist_pos if is_floating else np.zeros(3)
    achieved_link_pos = [retargeter.optimizer.robot.get_link_pose(int(index))[:3, 3] + pos_offset
                         for index in link_indices]
    
    return joint_vals, hand_qpos, wrist_qpos, wrist_idxs, achieved_link_pos

def control_hand(hand, hand_qpos, wrist_qpos, wrist_idxs, set_finger=False, set_wrist=False):
    """
    NOTE the hand here is genesis entity
    """
    hand_qpos = torch.tensor(hand_qpos).to(hand.init_qpos.device)
    if set_finger and set_wrist:
        hand.set_joint_position(hand_qpos)
    elif set_wrist:
        hand.set_joint_position(
            torch.tensor(wrist_qpos)[None].to(hand_qpos.device),
            joint_idxs=wrist_idxs,
        )
    else:
        # pass
        hand.control_joint_position(hand_qpos)
    return hand_qpos

def retarget_all_steps(
    dof_limits,
    hand_init_qpos,
    actuated_dof_names,
    actuated_dof_idxs,
    retargeter,
    num_steps,
    joint_pos_demo,
    retarget_type,
    frame_start=0,
    fixed_qpos_sequence=None,
):
    """
    Retarget all steps using single-stage optimization.
    For arm-based setups, position mode includes the wrist as a target,
    so the optimizer solves for arm + hand joints together.

    Args:
        fixed_qpos_sequence: Optional array of shape (num_steps, num_fixed_joints).
            If provided, each frame's retargeting uses the corresponding row as
            fixed_qpos (values for the retargeter's fixed/excluded joints).
            Used to retarget fingers with arm joints fixed to IK solution.
    """
    result_keys = ['hand_qpos', 'wrist_qpos', 'wrist_idxs']
    all_ret = {key: [] for key in result_keys}
    joint_angles_list = []
    achieved_link_pos_list = []

    # Log what targets the retargeter is using
    _opt = retargeter.optimizer
    _target_names = (getattr(_opt, 'task_link_names', None)
                     or getattr(_opt, 'target_task_link_names', None)
                     or getattr(_opt, 'target_link_names', None))
    print(f"[retarget_diag] num_target_pairs={len(_target_names) if _target_names else '?'}, "
          f"target_links={_target_names[:6] if _target_names else '?'}...")
    print(f"[retarget_diag] fixed_joints({len(_opt.fixed_joint_names)}): {list(_opt.fixed_joint_names)[:4]}...")

    current_init_qpos = hand_init_qpos.clone()
    for step in range(num_steps):
        if step % 5 == 0 or step == num_steps - 1:
            print(f"[retarget] frame {step}/{num_steps}", flush=True)
        demo_step = step + frame_start
        joint_pos = joint_pos_demo[demo_step]
        wrist_pos = joint_pos[0]

        indices = retargeter.optimizer.target_link_human_indices
        ref_val = get_ref_val(joint_pos, indices)
        fixed_qpos = None if fixed_qpos_sequence is None else fixed_qpos_sequence[step]
        joint_vals, hand_qpos, wrist_qpos, wrist_idxs, achieved_link_pos = retarget_one_hand(
            wrist_pos=wrist_pos,
            retarget_type=retarget_type,
            ref_value=ref_val,
            retargeter=retargeter,
            hand_init_qpos=current_init_qpos,
            actuated_dof_names=actuated_dof_names,
            actuated_dof_idxs=actuated_dof_idxs,
            fixed_qpos=fixed_qpos,
        )
        hand_qpos = torch.clamp(hand_qpos, dof_limits[0].to(hand_qpos.device), dof_limits[1].to(hand_qpos.device))
        current_init_qpos = hand_qpos.detach().clone()
        if step == 0:
            # Log finger joint values for frame 0 to diagnose retargeting quality
            _fj = {n: float(hand_qpos[0, i].cpu()) for i, n in enumerate(actuated_dof_names)
                   if 'finger' in n.lower()}
            print(f"[retarget_diag] frame-0 finger joints: "
                  + ", ".join(f"{k.split('_joint')[0][-2:]}j{k.split('joint')[-1]}={v:.3f}"
                               for k, v in list(_fj.items())[:10]))
            # Log each individual target vector and its norm (for first 5 targets)
            _ref = ref_val if hasattr(ref_val, '__len__') else ref_val
            if hasattr(_ref, 'shape') and len(_ref.shape) == 2:
                print(f"[retarget_diag] frame-0 ref_val per-target (first 6 of {len(_ref)}):")
                for _ti in range(min(6, len(_ref))):
                    _v = _ref[_ti]
                    print(f"  target[{_ti}]: vec={np.round(_v, 3)}  norm={np.linalg.norm(_v):.4f}")
            # Compare human keypoints vs achieved FK for frame 0
            if achieved_link_pos:
                print(f"[retarget_diag] frame-0 FK vs human comparison (first 6 links):")
                _indices = retargeter.optimizer.target_link_human_indices
                if hasattr(_indices, 'shape') and len(_indices.shape) == 2:
                    _task_inds = _indices[1, :]
                else:
                    _task_inds = list(range(len(achieved_link_pos)))
                for _li in range(min(6, len(achieved_link_pos))):
                    _fk_pos = np.array(achieved_link_pos[_li])
                    _fk_vec = _fk_pos - wrist_pos  # FK vector relative to wrist
                    _human_vec = joint_pos[int(_task_inds[_li])] - joint_pos[0]
                    _cos = float(np.dot(_fk_vec, _human_vec) / (np.linalg.norm(_fk_vec) * np.linalg.norm(_human_vec) + 1e-9))
                    print(f"  link[{_li}] FK_vec={np.round(_fk_vec,3)} norm={np.linalg.norm(_fk_vec):.3f} | "
                          f"human_vec={np.round(_human_vec,3)} norm={np.linalg.norm(_human_vec):.3f} | cos={_cos:.3f}")
        all_ret['hand_qpos'].append(hand_qpos[0].cpu().numpy())
        all_ret['wrist_qpos'].append(wrist_qpos)
        all_ret['wrist_idxs'].append(wrist_idxs)
        
        # Store joint angles and achieved link positions
        joint_angles_list.append(hand_qpos[0].cpu().numpy())
        achieved_link_pos_list.append(achieved_link_pos)
        
    all_ret = {key: np.stack(val) for key, val in all_ret.items()}
    all_ret['joint_angles'] = np.stack(joint_angles_list)
    all_ret['achieved_link_pos'] = achieved_link_pos_list
    return all_ret
