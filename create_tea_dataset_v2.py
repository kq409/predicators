import sys
import argparse
import numpy as np
import pybullet as p
import time
import logging
from typing import List, Tuple, Optional, Sequence, Collection, Dict, Any, cast, Set
import random
import json
from collections import defaultdict
from pathlib import Path
import dill as pkl
import os

from PIL import Image as PILImage
import matplotlib.pyplot as plt

from predicators.structs import Action, Array, GroundAtom, Object, State, Type, ParameterizedOption, EnvironmentTask,\
    Image, Predicate, State, Type, Video
from predicators.structs import Dataset, InteractionRequest, \
InteractionResult, Metrics, Response, Task, Video
from predicators import utils
from predicators.settings import CFG
from gymnasium.spaces import Box

#Import core environment methods, robot function etc.

from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions, get_joint_infos, get_joint_positions
from predicators.pybullet_helpers.link import get_link_state
from predicators.envs.kitchen import KitchenEnv
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options, parse_config_included_options, get_gt_nsrts
from predicators.datasets import create_dataset
from predicators.perception import create_perceiver
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations

#Import the functions that are to be tested:

from predicators.pybullet_helpers.motion_planning import run_motion_planning
#The pick/place options to be tested are accessed via the env instance
from predicators.pybullet_helpers.controllers import create_move_end_effector_to_pose_option,\
                                                    create_change_fingers_option

                                                    # 导入planning相关模块



from predicators.execution_monitoring import create_execution_monitor


import copy

import matplotlib
import PIL
from PIL import ImageDraw

try:
    import gymnasium as mujoco_kitchen_gym
    import mujoco
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, \
        get_site_xmat, get_site_xpos
    from gymnasium_robotics.utils.rotations import mat2quat, euler2quat
    from gymnasium_robotics.utils import mujoco_utils
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False
from predicators.envs import BaseEnv
from predicators.envs.kitchen import KitchenEnv
import gymnasium as mujoco_kitchen_gym

script_start = time.perf_counter()

CONTAINERS = ['right_hinge_cabinet', 'slide_cabinet', 'microwave']
OBJECT_NAMES_ORDER = ['mug', 'milk', 'sponge', 'tea']

# Default proposal probabilities used by unconstrained objects.
DEFAULT_OBJECT_CONTAINER_PROBS = {
    'tea': [0.55, 0.15, 0.30],
    'milk': [0.55, 0.15, 0.30],
    'mug': [0.15, 0.55, 0.30],
    'sponge': [0.15, 0.55, 0.30],
}

# Hardcoded strict constraints for final accepted dataset.
# Edit these two values directly if you want different constraints.
STRICT_OBJECTS_HARDCODED = ["mug",
#  "milk",
#   "sponge",
   "tea"]
STRICT_OBJECT_PROBS_HARDCODED = {
    "mug": [0.15, 0.55, 0.30],
    # "milk": [0.55, 0.15, 0.30],
    # "sponge": [0.15, 0.55, 0.30],
    "tea": [0.55, 0.15, 0.30],
}


def _normalize_prob_list(probs: List[float]) -> List[float]:
    if len(probs) != len(CONTAINERS):
        raise ValueError(
            f"Expected {len(CONTAINERS)} probabilities in container order "
            f"{CONTAINERS}, got {len(probs)}"
        )
    arr = np.array(probs, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Probabilities must be non-negative.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Probability sum must be > 0.")
    arr = arr / total
    return arr.tolist()


def _parse_object_prob_overrides(raw: str) -> Dict[str, List[float]]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("strict_object_probs_json must be a JSON object.")
    out: Dict[str, List[float]] = {}
    for obj_name, probs in parsed.items():
        if obj_name not in OBJECT_NAMES_ORDER:
            raise ValueError(
                f"Unknown object '{obj_name}'. Supported: {OBJECT_NAMES_ORDER}"
            )
        if not isinstance(probs, list):
            raise ValueError(
                f"Probabilities for '{obj_name}' must be a list in order {CONTAINERS}."
            )
        out[obj_name] = _normalize_prob_list([float(x) for x in probs])
    return out


def _allocate_counts_from_probs(total_count: int, probs: List[float]) -> Dict[str, int]:
    # Largest remainder method: guarantees exact total_count while matching probs.
    raw = np.array(probs, dtype=float) * total_count
    floor_counts = np.floor(raw).astype(int)
    remainder = raw - floor_counts
    counts = floor_counts.copy()
    need = int(total_count - int(counts.sum()))
    if need > 0:
        order = np.argsort(-remainder)  # descending remainder
        for i in range(need):
            counts[order[i % len(order)]] += 1
    return {CONTAINERS[i]: int(counts[i]) for i in range(len(CONTAINERS))}


def _sample_container_with_remaining(
    remaining_counts: Dict[str, int],
    rng: np.random.Generator,
) -> str:
    valid_containers = [c for c in CONTAINERS if remaining_counts[c] > 0]
    if not valid_containers:
        raise RuntimeError("No remaining container quota to sample from.")
    weights = np.array([remaining_counts[c] for c in valid_containers], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(valid_containers, p=weights))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mujoco tea dataset with optional strict per-object container distributions."
    )
    parser.add_argument(
        "--strict_objects",
        type=str,
        default="",
        help=(
            "Comma-separated objects to enforce exactly in final accepted dataset, "
            "e.g. 'mug,milk'."
        ),
    )
    parser.add_argument(
        "--strict_object_probs_json",
        type=str,
        default="",
        help=(
            "JSON overrides for strict objects. Format: "
            "{\"mug\": [0.15,0.55,0.30], \"milk\": [0.55,0.15,0.30]} "
            f"with order {CONTAINERS}."
        ),
    )
    return parser.parse_args()

#Defining test configuration, and overriding some default ones:
args = {
    "env": "kitchen_v2",
    "approach": "refinement_estimation",
    "seed": 42,
    "use_gui": False,
    "num_test_tasks": 1,
    "kitchen_use_perfect_samplers": True,
    "kitchen_goals": "put_mug_on_countertop",
    "pybullet_sim_steps_per_action": 20,
    "pybullet_camera_width": 1674,
    "pybullet_camera_height": 900,
    "render_state_dpi": 300,
    "video_fps": 10,
    "make_test_videos": False,
    "make_failure_videos": False,
    "log_file": "predicators/logs/predicators_explorer_mujoco.log",
    "log_dir": "predicators/logs",
    "loglevel": logging.DEBUG,
}

utils.reset_config(args)

print(CFG.log_dir)

str_args = " ".join(sys.argv)
# Create logs directory.
os.makedirs(CFG.log_dir, exist_ok=True)
# Log to stderr.
handlers: List[logging.Handler] = [logging.StreamHandler()]
if CFG.log_file:
    handlers.append(logging.FileHandler(CFG.log_file, mode='w'))
logging.basicConfig(level=CFG.loglevel,
                    format="%(message)s",
                    handlers=handlers,
                    force=True)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
if CFG.log_file:
    logging.info(f"Logging to {CFG.log_file}")
logging.info(f"Running command: python {str_args}")
logging.info("Full config:")
logging.info(CFG)

# Create results directory.
os.makedirs(CFG.results_dir, exist_ok=True)
# Create the eval trajectories directory.
os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)

def save_current_view(env, rgb_path: str | Path, depth_path: str | Path) -> None:
    gym_env = env._gym_env
    renderer = gym_env.robot_env.mujoco_renderer

    viewer = renderer.viewer
    cam = viewer.cam

    # RGB viewer
    renderer._get_viewer("rgb_array").vopt.geomgroup[2] = 0
    # Depth viewer
    renderer._get_viewer("depth_array").vopt.geomgroup[2] = 0

    rgb = renderer.render(render_mode="rgb_array", camera_name="fourth_cap")
    depth = renderer.render(render_mode="depth_array", camera_name="fourth_cap")

    renderer._get_viewer("rgb_array").vopt.geomgroup[2] = 1
    renderer._get_viewer("depth_array").vopt.geomgroup[2] = 1

    PILImage.fromarray(rgb).save(rgb_path)

    depth = np.nan_to_num(depth)
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth_norm = (depth - dmin) / (dmax - dmin)
    else:
        depth_norm = np.zeros_like(depth)

    png_path = Path(depth_path).with_suffix(".png")
    plt.imsave(png_path, depth_norm, cmap="gray")

def set_joint(env: KitchenEnv, joint_name: str, value):
    """set the position of the joint, value can be a float (single value) or an array (position + quaternion)"""
    model = env._gym_env.model          # MuJoCo mjModel
    data = env._gym_env.data            # MuJoCo mjData
    mujoco_utils.set_joint_qpos(model, data, joint_name, value)
    mujoco.mj_forward(model, data)

def get_container_position_range(container_name: str) -> Tuple[float, float, float, float, float, float]:
    """get the position range of the container (x_min, x_max, y_min, y_max, z_min, z_max)"""
    ranges = {
        # 新范围：x, y, z 保持原高度
        'microwave': (-0.4, -0.1, 0.8, 0.95, 1.7, 1.7),
        'right_hinge_cabinet': (-0.5, -0.3, 0.68, 0.8, 2.45, 2.45),
        'slide_cabinet': (-0.05, 0.23, 0.69, 0.8, 2.45, 2.45),
    }
    return ranges[container_name]

def sample_position_in_container(container_name: str, rng: np.random.Generator) -> List[float]:
    """sample a position in the specified container"""
    x_min, x_max, y_min, y_max, z_min, z_max = get_container_position_range(container_name)
    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    z = rng.uniform(z_min, z_max)
    return [x, y, z]

def sample_container_for_object(object_name: str, rng: np.random.Generator) -> str:
    """sample a container for the object based on the object name and probability distribution"""
    probs = DEFAULT_OBJECT_CONTAINER_PROBS.get(object_name, [1/3, 1/3, 1/3])
    return rng.choice(CONTAINERS, p=probs)

def get_container_for_position(pos: List[float]) -> Optional[str]:
    """determine which container the object is in based on the position"""
    x, y, z = pos
    
    # check if in microwave
    if -0.4 <= x <= -0.1 and 0.8 <= y <= 0.95 and abs(z - 1.7) < 0.1:
        return 'microwave'
    # check if in right_hinge_cabinet
    elif -0.5 <= x <= -0.3 and 0.68 <= y <= 0.8 and abs(z - 2.45) < 0.1:
        return 'right_hinge_cabinet'
    # check if in slide_cabinet
    elif -0.05 <= x <= 0.23 and 0.69 <= y <= 0.8 and abs(z - 2.45) < 0.1:
        return 'slide_cabinet'
    
    return None

def is_container_open(env: KitchenEnv, container_name: str) -> bool:
    """check if the container door is open"""
    if container_name == 'microwave':
        qpos = mujoco_utils.get_joint_qpos(env._gym_env.model, env._gym_env.data, "microwave")
        return qpos < -0.1  # door is open
    elif container_name == 'right_hinge_cabinet':
        qpos = mujoco_utils.get_joint_qpos(env._gym_env.model, env._gym_env.data, "right_hinge_cabinet")
        return qpos < -0.1  # door is open
    elif container_name == 'slide_cabinet':
        qpos = mujoco_utils.get_joint_qpos(env._gym_env.model, env._gym_env.data, "slide_cabinet")
        return qpos > 0.1  # door is open
    return True

def check_occlusion(env: KitchenEnv, object_positions: Dict[str, List[float]]) -> bool:
    """check if any object is occluded (container door is closed)"""
    for obj_name, pos in object_positions.items():
        if obj_name == 'banana':
            continue
        
        container = get_container_for_position(pos)
        if container and not is_container_open(env, container):
            return True
    
    return False


def get_object_occlusion_status(
    env: KitchenEnv,
    object_positions: Dict[str, List[float]],
) -> Dict[str, bool]:
    """获取每个物体是否被遮挡。返回 { 'mug': bool, 'milk': bool, 'sponge': bool, 'tea': bool }"""
    status = {}
    for obj_name in OBJECT_NAMES_ORDER:
        pos = object_positions.get(obj_name)
        if pos is None:
            status[obj_name] = False
            continue
        container = get_container_for_position(pos)
        if container is None:
            status[obj_name] = False
        else:
            status[obj_name] = not is_container_open(env, container)
    return status


def occlusion_status_to_pattern(status: Dict[str, bool]) -> str:
    """将 per-object 遮挡状态转为 4 位字符串，顺序 mug,milk,sponge,tea。1=遮挡，0=可见"""
    return "".join("1" if status.get(name, False) else "0" for name in OBJECT_NAMES_ORDER)


def pattern_to_occlusion_status(pattern: str) -> Dict[str, bool]:
    """将 4 位遮挡模式字符串转为 per-object 状态"""
    status = {}
    for i, name in enumerate(OBJECT_NAMES_ORDER):
        status[name] = pattern[i] == "1" if i < len(pattern) else False
    return status


def generate_all_occlusion_patterns() -> List[str]:
    """生成全部 16 种遮挡模式（0000 到 1111）"""
    return [f"{i:04b}" for i in range(16)]

# env name must match CFG.env; otherwise options/nsrts can mismatch.
env = create_new_env("kitchen_v2", use_gui=False)
env.action_space.seed(CFG.seed)
assert env.goal_predicates.issubset(env.predicates)

included_preds, excluded_preds = utils.parse_config_excluded_predicates(
        env)
preds = utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds,
        env) if CFG.approach != "oracle" else included_preds

env_train_tasks = env.get_train_tasks()
perceiver = create_perceiver(CFG.perceiver)
train_tasks = [perceiver.reset(t) for t in env_train_tasks]
stripped_train_tasks = [
    utils.strip_task(task, preds) for task in train_tasks
]
approach_train_tasks = [
    task.replace_goal_with_alt_goal() for task in stripped_train_tasks
]
if CFG.option_learner == "no_learning":
    # If we are not doing option learning, pass in all the environment's
    # oracle options.
    options = get_gt_options(env.get_name())
else:
    # Determine from the config which oracle options to include, if any.
    options = parse_config_included_options(env)
# Create the agent (approach).
approach_name = CFG.approach
if CFG.approach_wrapper:
    approach_name = f"{CFG.approach_wrapper}[{approach_name}]"
approach = create_approach(approach_name, preds, options, env.types,
                               env.action_space, approach_train_tasks)

if approach.is_learning_based:
    # Create the offline dataset. Note that this needs to be done using
    # the non-stripped train tasks because dataset generation may need
    # to use the oracle predicates (e.g. demo data generation).
    offline_dataset = create_dataset(env, train_tasks, options, preds)
else:
    offline_dataset = None

# Create the cognitive manager.
execution_monitor = create_execution_monitor(CFG.execution_monitor)
cogman = CogMan(approach, perceiver, execution_monitor)

seed = 42
rng = np.random.default_rng(seed)
_ = _parse_args()  # Keep other script CLI behavior; strict distribution is hardcoded below.

# define the container position range (kept for potential future use)
container_ranges = {
    'microwave': {'x': (-0.4, -0.1), 'y': (0.8, 0.95), 'z': (1.7, 1.7)},
    'right_hinge_cabinet': {'x': (-0.5, -0.3), 'y': (0.68, 0.8), 'z': (2.45, 2.45)},
    'slide_cabinet': {'x': (-0.05, 0.23), 'y': (0.69, 0.8), 'z': (2.45, 2.45)},
}

# create the dataset directory
folder_dir = r'D:\Research\Life_Long_Learning\diffusion_proejct\diffusion\diffusion_behavior\mujoco_tea_dataset_v2'
image_dir = os.path.join(folder_dir, "images")
depth_dir = os.path.join(folder_dir, "depth")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# create the position files and occlusion record
mug_position_file = os.path.join(folder_dir, "mug_positions.txt")
milk_position_file = os.path.join(folder_dir, "milk_positions.txt")
sponge_position_file = os.path.join(folder_dir, "sponge_positions.txt")
tea_position_file = os.path.join(folder_dir, "tea_positions.txt")
camera_position_file = os.path.join(folder_dir, "camera_positions.txt")
camera_rotation_file = os.path.join(folder_dir, "camera_rotations.txt")
# 每个样本的遮挡记录：frame_id, mug_occluded, milk_occluded, sponge_occluded, tea_occluded (0/1)，以及 pattern (4位字符串)
occlusion_record_file = os.path.join(folder_dir, "occlusion_record.txt")

# set the quaternion (all objects use the same quaternion)
euler = (0.0, 0.0, 0.0)
quaternion = euler2quat(euler)

# 按 16 种遮挡模式平衡：每种模式目标样本数
ALL_OCCLUSION_PATTERNS = generate_all_occlusion_patterns()
samples_per_occlusion_pattern = 1000  # 16 * 1000 = 16000
occlusion_pattern_counts = {p: 0 for p in ALL_OCCLUSION_PATTERNS}
distribution_counts = defaultdict(int)
total_target = len(ALL_OCCLUSION_PATTERNS) * samples_per_occlusion_pattern

strict_objects = [s.strip() for s in STRICT_OBJECTS_HARDCODED if s.strip()]
for obj in strict_objects:
    if obj not in OBJECT_NAMES_ORDER:
        raise ValueError(f"Unknown strict object '{obj}', supported: {OBJECT_NAMES_ORDER}")

strict_prob_overrides = {
    obj: _normalize_prob_list(probs)
    for obj, probs in STRICT_OBJECT_PROBS_HARDCODED.items()
}

strict_object_target_counts: Dict[str, Dict[str, int]] = {}
strict_object_remaining_counts: Dict[str, Dict[str, int]] = {}
strict_object_accepted_counts: Dict[str, Dict[str, int]] = {}
for obj in strict_objects:
    probs = strict_prob_overrides.get(
        obj, DEFAULT_OBJECT_CONTAINER_PROBS.get(obj, [1/3, 1/3, 1/3])
    )
    probs = _normalize_prob_list(probs)
    target_counts = _allocate_counts_from_probs(total_target, probs)
    strict_object_target_counts[obj] = target_counts
    strict_object_remaining_counts[obj] = dict(target_counts)
    strict_object_accepted_counts[obj] = {c: 0 for c in CONTAINERS}

if strict_objects:
    print(f"Strict objects: {strict_objects}")
    print("Target counts per strict object:")
    for obj in strict_objects:
        print(f"  {obj}: {strict_object_target_counts[obj]}")

obs = env.reset("test", 0)

# open all files for writing
with open(mug_position_file, "w") as mug_pos_file, \
        open(milk_position_file, "w") as milk_pos_file, \
        open(sponge_position_file, "w") as sponge_pos_file, \
        open(tea_position_file, "w") as tea_pos_file, \
        open(camera_position_file, "w") as cam_pos_file, \
        open(camera_rotation_file, "w") as cam_rot_file, \
        open(occlusion_record_file, "w") as occ_file:

    # 遮挡记录表头：frame_id, 各物体是否被遮挡(0/1), pattern, 三容器开/关(1=开 0=关)
    occ_file.write("frame_id\tmug_occluded\tmilk_occluded\tsponge_occluded\ttea_occluded\tpattern\tmicrowave_open\tright_hinge_cabinet_open\tslide_cabinet_open\n")

    frame_idx = 0
    while any(occlusion_pattern_counts[p] < samples_per_occlusion_pattern for p in ALL_OCCLUSION_PATTERNS):
        # 选一个尚未采够的遮挡模式
        under_sampled = [p for p in ALL_OCCLUSION_PATTERNS if occlusion_pattern_counts[p] < samples_per_occlusion_pattern]
        target_pattern = rng.choice(under_sampled)
        target_status = pattern_to_occlusion_status(target_pattern)

        # 为每个物体采样容器：strict objects按剩余配额采样，其余维持原始概率采样
        object_to_container: Dict[str, str] = {}
        for obj_name in OBJECT_NAMES_ORDER:
            if obj_name in strict_object_remaining_counts:
                remaining = strict_object_remaining_counts[obj_name]
                if sum(remaining.values()) <= 0:
                    raise RuntimeError(
                        f"Object '{obj_name}' strict quota exhausted before dataset finished."
                    )
                object_to_container[obj_name] = _sample_container_with_remaining(remaining, rng)
            else:
                object_to_container[obj_name] = sample_container_for_object(obj_name, rng)

        mug_container = object_to_container['mug']
        milk_container = object_to_container['milk']
        sponge_container = object_to_container['sponge']
        tea_container = object_to_container['tea']

        # 根据目标遮挡模式决定每个容器开/关：若某容器内所有物体在该模式下均为“被遮挡”，则关门；否则开门
        def should_container_be_closed(container_name: str) -> bool:
            objects_inside = [o for o in OBJECT_NAMES_ORDER if object_to_container[o] == container_name]
            if not objects_inside:
                return False
            return all(target_status[o] for o in objects_inside)

        containers_to_close = set()
        for c in ['microwave', 'right_hinge_cabinet', 'slide_cabinet']:
            if should_container_be_closed(c):
                containers_to_close.add(c)

        # 设置门状态
        if 'microwave' in containers_to_close:
            microwave_qpos = 0.0
        elif 'microwave' in {mug_container, milk_container, sponge_container, tea_container}:
            microwave_qpos = rng.uniform(-0.9, -0.45)
        else:
            microwave_qpos = rng.uniform(-0.9, -0.45) if rng.random() < 0.5 else 0.0

        if 'right_hinge_cabinet' in containers_to_close:
            right_hinge_cabinet_qpos = 0.0
        elif 'right_hinge_cabinet' in {mug_container, milk_container, sponge_container, tea_container}:
            right_hinge_cabinet_qpos = rng.uniform(-1.2, -0.8)
        else:
            right_hinge_cabinet_qpos = rng.uniform(-1.2, -0.8) if rng.random() < 0.5 else 0.0

        if 'slide_cabinet' in containers_to_close:
            slide_cabinet_qpos = 0.0
        elif 'slide_cabinet' in {mug_container, milk_container, sponge_container, tea_container}:
            slide_cabinet_qpos = rng.uniform(0.35, 0.55)
        else:
            slide_cabinet_qpos = rng.uniform(0.35, 0.55) if rng.random() < 0.5 else 0.0

        set_joint(env, "microwave", microwave_qpos)
        set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos)
        set_joint(env, "slide_cabinet", slide_cabinet_qpos)

        # 采样物体位置并设置
        mug_pos = sample_position_in_container(mug_container, rng)
        milk_pos = sample_position_in_container(milk_container, rng)
        sponge_pos = sample_position_in_container(sponge_container, rng)
        tea_pos = sample_position_in_container(tea_container, rng)
        # banana_pos = [0.0, 0.0, 0.0]

        object_positions = {
            # 'banana': banana_pos,
            'mug': mug_pos,
            'milk': milk_pos,
            'sponge': sponge_pos,
            'tea': tea_pos,
        }
        for obj_name, pos in object_positions.items():
            qpos_values = np.concatenate([pos, quaternion])
            set_joint(env, obj_name, qpos_values)

        # 校验实际遮挡状态是否与目标一致
        actual_status = get_object_occlusion_status(env, object_positions)
        actual_pattern = occlusion_status_to_pattern(actual_status)
        if actual_pattern != target_pattern:
            continue

        occlusion_pattern_counts[target_pattern] += 1
        distribution_key = f"{mug_container}_{milk_container}_{sponge_container}_{tea_container}"
        distribution_counts[distribution_key] += 1
        for obj_name in strict_objects:
            chosen_container = object_to_container[obj_name]
            strict_object_remaining_counts[obj_name][chosen_container] -= 1
            strict_object_accepted_counts[obj_name][chosen_container] += 1

        rgb_filename = f"frame_{frame_idx:06d}.png"
        depth_filename = f"frame_{frame_idx:06d}.png"
        save_current_view(env,
                          os.path.join(image_dir, rgb_filename),
                          os.path.join(depth_dir, depth_filename))

        mug_pos_file.write(f"{mug_pos[0]:.6f}, {mug_pos[1]:.6f}, {mug_pos[2]:.6f}\n")
        milk_pos_file.write(f"{milk_pos[0]:.6f}, {milk_pos[1]:.6f}, {milk_pos[2]:.6f}\n")
        sponge_pos_file.write(f"{sponge_pos[0]:.6f}, {sponge_pos[1]:.6f}, {sponge_pos[2]:.6f}\n")
        tea_pos_file.write(f"{tea_pos[0]:.6f}, {tea_pos[1]:.6f}, {tea_pos[2]:.6f}\n")
        # 摄像机位姿：pos='0.4 -0.6 2.8' euler='1.1 0.4 0.2'
        cam_pos_file.write(f"{0.4:.6f}, {-0.6:.6f}, {2.8:.6f}\n")
        cam_rot_file.write(f"{1.1:.6f}, {0.4:.6f}, {0.2:.6f}\n")

        # 写入遮挡记录：frame_id, 各物体遮挡, pattern, 三容器开/关(1=开 0=关)
        occ_mug = 1 if actual_status['mug'] else 0
        occ_milk = 1 if actual_status['milk'] else 0
        occ_sponge = 1 if actual_status['sponge'] else 0
        occ_tea = 1 if actual_status['tea'] else 0
        microwave_open = 1 if microwave_qpos < -0.1 else 0
        right_hinge_cabinet_open = 1 if right_hinge_cabinet_qpos < -0.1 else 0
        slide_cabinet_open = 1 if slide_cabinet_qpos > 0.1 else 0
        occ_file.write(
            f"frame_{frame_idx:06d}\t{occ_mug}\t{occ_milk}\t{occ_sponge}\t{occ_tea}\t{actual_pattern}\t"
            f"{microwave_open}\t{right_hinge_cabinet_open}\t{slide_cabinet_open}\n"
        )

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Generated {frame_idx} frames. Pattern counts: min={min(occlusion_pattern_counts.values())}, max={max(occlusion_pattern_counts.values())}")

# 输出每种遮挡模式的样本数
print("\nDataset generation complete!")
print(f"Total frames: {frame_idx}")
print("Samples per occlusion pattern (pattern -> count):")
for p in sorted(ALL_OCCLUSION_PATTERNS):
    print(f"  {p}: {occlusion_pattern_counts[p]}")
print(f"Unique container distributions: {len(distribution_counts)}")
if strict_objects:
    print("\nStrict object distribution check (accepted vs target):")
    for obj in strict_objects:
        print(f"  {obj} accepted={strict_object_accepted_counts[obj]}, target={strict_object_target_counts[obj]}")
        print(f"  {obj} remaining={strict_object_remaining_counts[obj]}")
