import sys
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
    from gymnasium_robotics.utils.rotations import mat2quat
    from gymnasium_robotics.utils import mujoco_utils
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False
from predicators.envs import BaseEnv
from predicators.envs.kitchen import KitchenEnv
import gymnasium as mujoco_kitchen_gym

script_start = time.perf_counter()

#Defining test configuration, and overriding some default ones:
args = {
    "env": "kitchen",
    "approach": "oracle",
    "seed": random.randint(0,10000),
    "use_gui": True,
    "num_test_tasks": 1,
    "kitchen_use_perfect_samplers": True,
    "kitchen_goals": "find_banana",
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
# logging.info(f"Git commit hash: {utils.get_git_commit_hash()}")

# Create results directory.
os.makedirs(CFG.results_dir, exist_ok=True)
# Create the eval trajectories directory.
os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)

def init_kitchen_env(env: KitchenEnv, seed: int, task_idx: int,
                                       train_or_test: str) -> None:
        env._gym_env.reset(seed=seed)

        kettle_x_coord = -0.269
        if train_or_test == "test":
            kettle_x_coord = 0.169
        kettle_y_coord = 0.4
        if CFG.kitchen_randomize_init_state:
            rng = np.random.default_rng(seed)
            # For now, we only randomize the state such that the kettle
            # is anywhere between burners 2 and 4. Later, we might add
            # even more variation.
            kettle_y_coord = rng.uniform(0.4, 0.55)
        env._gym_env.set_body_position(  # type: ignore
            "kettle", (kettle_x_coord, kettle_y_coord, 1.626))


        env._setup_new_objects(seed, train_or_test)
        env.get_object_centric_state_info()

        env._current_task = env.get_task(train_or_test, task_idx)
        print("TASK SET COMPLETE")
        env._current_observation = env._current_task.init_obs
        # Copy to prevent external changes to the environment's state.
        # This default implementation of reset assumes that observations are
        # states. Subclasses with different states should override.
        assert isinstance(env._current_observation, State)
        return env._current_observation.copy()

def save_current_view(env, rgb_path: str | Path, depth_path: str | Path) -> None:
    gym_env = env._gym_env
    renderer = gym_env.robot_env.mujoco_renderer

    viewer = renderer.viewer
    cam = viewer.cam
    # print(cam.distance, cam.azimuth, cam.elevation, cam.lookat)

    # RGB viewer
    renderer._get_viewer("rgb_array").vopt.geomgroup[2] = 0
    # Depth viewer
    renderer._get_viewer("depth_array").vopt.geomgroup[2] = 0

    rgb = renderer.render(render_mode="rgb_array", camera_name="third_cap")
    depth = renderer.render(render_mode="depth_array", camera_name="third_cap")

    renderer._get_viewer("rgb_array").vopt.geomgroup[2] = 1
    renderer._get_viewer("depth_array").vopt.geomgroup[2] = 1

    PILImage.fromarray(rgb).save(rgb_path)
    # np.save(depth_path, depth)

    depth = np.nan_to_num(depth)
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth_norm = (depth - dmin) / (dmax - dmin)
    else:
        depth_norm = np.zeros_like(depth)

    png_path = Path(depth_path).with_suffix(".png")
    plt.imsave(png_path, depth_norm, cmap="gray")

def set_joint(env: KitchenEnv, joint_name: str, value: float):
    model = env._gym_env.model          # MuJoCo mjModel
    data = env._gym_env.data            # MuJoCo mjData
    mujoco_utils.set_joint_qpos(model, data, joint_name, value)
    mujoco.mj_forward(model, data) 

# env = create_new_env("kitchen", use_gui=CFG.use_gui)
env = create_new_env("kitchen", use_gui=False)
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
# np.random.seed(seed)

visible_list = ['visible', 'invisible']
location_list = ['microwave', 'right_hinge_cabinet', 'slide_cabinet', 'tabletop']

folder_dir = r'D:\Research\Life_Long_Learning\diffusion_proejct\diffusion\diffusion_behavior\mujoco_dataset'
image_dir = os.path.join(folder_dir, "images")
depth_dir = os.path.join(folder_dir, "depth")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

banana_position_file = os.path.join(folder_dir, "banana_positions.txt")
camera_position_file = os.path.join(folder_dir, "camera_positions.txt")
camera_rotation_file = os.path.join(folder_dir, "camera_rotations.txt")

rng = np.random.default_rng(seed)

# Microwave
# banana_x_coord = rng.uniform(-0.95, -0.85)
# banana_y_coord = rng.uniform(0.65, 0.85)
# banana_z_coord = 1.7

# Upper right cabinet
# banana_x_coord = rng.uniform(-0.5, -0.35)
# banana_y_coord = rng.uniform(0.85, 1.2)
# banana_z_coord = 2.45

# Slide cabinet
# banana_x_coord = rng.uniform(0.075, 0.2)
# banana_y_coord = rng.uniform(0.85, 1.2)
# banana_z_coord = 2.45

# Tabletop
# banana_x_coord = rng.uniform(-0.55, 0.3)
# banana_y_coord = rng.uniform(0.85, 1.2)
# banana_z_coord = 1.626

# microwave_qpos = rng.uniform(-0.75, -0.45)
# right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
# slide_cabinet_qpos = rng.uniform(0.25, 0.35)
# Start loop

obs = env.reset("test", 0)


with open(camera_position_file, "w") as cam_pos_file, \
            open(camera_rotation_file, "w") as cam_rot_file, \
            open(banana_position_file, "w") as banana_pos_file:

    for condition in visible_list:
        if condition == 'visible':
            for location in location_list:
                if location == 'microwave':
                    for i in range(1000):
                        banana_x_coord = rng.uniform(-0.95, -0.85)
                        banana_y_coord = rng.uniform(0.65, 0.85)
                        banana_z_coord = 1.7
                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=3, dtype=bool)

                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "slide_cabinet", slide_cabinet_qpos if arr[1] else 0)
                        set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos if arr[2] else 0)

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]

                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        # model = env._gym_env.model
                        # data = env._gym_env.data

                        # banana_body_id = env._gym_env.find_body_id("banana")

                        # print(f"Found contacts: {data.ncon}")

                        # for j in range(data.ncon):
                        #     contact = data.contact[j]
                        #     geom1 = contact.geom1
                        #     geom2 = contact.geom2
                            
                        #     # 获取geom对应的body ID
                        #     body1 = model.geom_bodyid[geom1]
                        #     body2 = model.geom_bodyid[geom2]
                            
                        #     # 如果接触涉及香蕉（排除与地面的接触）
                        #     if body1 == banana_body_id or body2 == banana_body_id:
                        #         # 可以进一步检查是否是与地面的正常接触
                        #         # 这里简单返回True表示有碰撞
                        #         other_body_id = body2 if body1 == banana_body_id else body1
                        #         other_body_name = model.names[model.name_bodyadr[other_body_id]:].decode('utf-8').split('\x00', 1)[0]
                        #         print(f"Collision between banana and {other_body_name}")

                        save_current_view(env, os.path.join(image_dir, f"con_{0:04d}_loc_{0:04d}_frame_{i:04d}.png"), os.path.join(depth_dir, f"con_{0:04d}_loc_{0:04d}_frame_{i:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")
                        
                        print(f"Saved frame {i} for condition {condition} and location {location}")

                elif location == 'right_hinge_cabinet':
                    for i in range(1000):
                        banana_x_coord = rng.uniform(-0.5, -0.35)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 2.45
                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=2, dtype=bool)

                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "slide_cabinet", slide_cabinet_qpos if arr[1] else 0)
                        
                        set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos)

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{0:04d}_loc_{1:04d}_frame_{i:04d}.png"), os.path.join(depth_dir, f"con_{0:04d}_loc_{1:04d}_frame_{i:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")
                        
                        print(f"Saved frame {i} for condition {condition} and location {location}")

                elif location == 'slide_cabinet':
                    for i in range(1000):
                        banana_x_coord = rng.uniform(0.075, 0.2)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 2.45
                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.15, 0.35)
                        arr = rng.integers(0, 2, size=2, dtype=bool)

                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos if arr[1] else 0)
                        
                        set_joint(env, "slide_cabinet", slide_cabinet_qpos)

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{0:04d}_loc_{2:04d}_frame_{i:04d}.png"), os.path.join(depth_dir, f"con_{0:04d}_loc_{2:04d}_frame_{i:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")

                        print(f"Saved frame {i} for condition {condition} and location {location}")

                elif location == 'tabletop':

                    for i in range(1000):
                        banana_x_coord = rng.uniform(-0.55, 0.3)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 1.626
                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=3, dtype=bool)

                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos if arr[1] else 0)
                        set_joint(env, "slide_cabinet", slide_cabinet_qpos if arr[2] else 0)

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{0:04d}_loc_{3:04d}_frame_{i:04d}.png"), os.path.join(depth_dir, f"con_{0:04d}_loc_{3:04d}_frame_{i:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")
                        
                        print(f"Saved frame {i} for condition {condition} and location {location}")

        elif condition == 'invisible':
            for location in location_list:
                if location == 'right_hinge_cabinet':
                    for i in range(1000):    # In this case, doors are all closed
                        banana_x_coord = rng.uniform(-0.5, -0.35)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 2.45
                        
                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=1, dtype=bool)

                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "slide_cabinet", 0)              # Slide cabinet is closed
                        set_joint(env, "right_hinge_cabinet", 0)        # Hinge door is closed

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{1:04d}_loc_{0:04d}_frame_{i:04d}.png"), os.path.join(depth_dir, f"con_{1:04d}_loc_{0:04d}_frame_{i:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")
                        
                        print(f"Saved frame {i} for condition {condition} and location {location}")
                    
                    for i in range(1000):
                        banana_x_coord = rng.uniform(-0.5, -0.35)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 2.45
                        
                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=1, dtype=bool)
                        
                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "slide_cabinet", slide_cabinet_qpos)   # Slide cabinet is open           
                        set_joint(env, "right_hinge_cabinet", 0)        # Hinge door is closed
                        
                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{1:04d}_loc_{0:04d}_frame_{i+1000:04d}.png"), os.path.join(depth_dir, f"con_{1:04d}_loc_{0:04d}_frame_{i+1000:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")
                        
                        print(f"Saved frame {i+1000} for condition {condition} and location {location}")

                elif location == 'slide_cabinet':
                    for i in range(1000):
                        banana_x_coord = rng.uniform(0.075, 0.2)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 2.45

                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=1, dtype=bool)
                        
                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "right_hinge_cabinet", 0)    # Hinge door is closed
                        set_joint(env, "slide_cabinet", 0)   # Slide cabinet is closed

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{1:04d}_loc_{1:04d}_frame_{i:04d}.png"), os.path.join(depth_dir, f"con_{1:04d}_loc_{1:04d}_frame_{i:04d}.npy"))

                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")

                        print(f"Saved frame {i} for condition {condition} and location {location}")

                    for i in range(1000):
                        banana_x_coord = rng.uniform(0.075, 0.2)
                        banana_y_coord = rng.uniform(0.85, 1.2)
                        banana_z_coord = 2.45

                        microwave_qpos = rng.uniform(-0.75, -0.45)
                        right_hinge_cabinet_qpos = rng.uniform(0.8, 1.2)
                        slide_cabinet_qpos = rng.uniform(0.25, 0.35)
                        arr = rng.integers(0, 2, size=1, dtype=bool)
                        
                        set_joint(env, "microwave", microwave_qpos if arr[0] else 0)
                        set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos)    # Hinge door is open
                        set_joint(env, "slide_cabinet", 0)   # Slide cabinet is closed

                        banana_position = [banana_x_coord, banana_y_coord, banana_z_coord]
                        quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
                        qpos_values = np.concatenate([banana_position, quaternion])
                        set_joint(env, "banana", qpos_values)

                        save_current_view(env, os.path.join(image_dir, f"con_{1:04d}_loc_{1:04d}_frame_{i+1000:04d}.png"), os.path.join(depth_dir, f"con_{1:04d}_loc_{1:04d}_frame_{i+1000:04d}.npy"))
                        
                        banana_pos_file.write(f"{banana_x_coord:.6f}, {banana_y_coord:.6f}, {banana_z_coord:.6f}\n")
                        cam_pos_file.write(f"{-0.85:.6f}, {-0.85:.6f}, {2.8:.6f}\n")
                        cam_rot_file.write(f"{1.1:.6f}, {-0.3:.6f}, {-0.1:.6f}\n")
                        
                        print(f"Saved frame {i+1000} for condition {condition} and location {location}")

# obs = env.reset("test", 0)
# set_joint(env, "microwave", microwave_qpos)
# set_joint(env, "slide_cabinet", slide_cabinet_qpos)
# set_joint(env, "right_hinge_cabinet", right_hinge_cabinet_qpos)
# print(banana_x_coord, banana_y_coord)
# env._gym_env.set_body_position("banana", [banana_x_coord, banana_y_coord, banana_z_coord])
# print("INIT COMPLETE")



# save_current_view(env, os.path.join(image_dir, "frame0000.png"), os.path.join(depth_dir, "frame0000.npy"))
