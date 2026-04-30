import sys
import numpy as np
import pybullet as p
import time
import logging
import traceback
from typing import List, Tuple, Optional, Sequence, Collection, Dict, Any, cast, Set
import random
import json
from collections import defaultdict
from pathlib import Path
import dill as pkl
import os

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
import matplotlib.pyplot as plt
import PIL
from PIL import Image as PILImage
from PIL import ImageDraw
import shutil

# Try to import torch and torchvision for diffusion model sampling
try:
    import torch
    import importlib
    transforms = importlib.import_module("torchvision.transforms")
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logging.warning("torch or torchvision not available. results.txt generation will be skipped.")

try:
    import gymnasium as mujoco_kitchen_gym
    from gymnasium_robotics.utils import mujoco_utils
    import mujoco
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, \
        get_site_xmat, get_site_xpos
    from gymnasium_robotics.utils.rotations import mat2quat
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False
from predicators.envs import BaseEnv
from predicators.envs.kitchen import KitchenEnv
import gymnasium as mujoco_kitchen_gym

script_start = time.perf_counter()

# Global list to store all captured views (cam_view and depth) for multi-condition sampling
# Each element is a tuple: (cam_view_tensor, depth_tensor)
_captured_views_history = []

# Global counter to track plan attempts (for naming saved views)
_plan_counter = 0

#Defining test configuration, and overriding some default ones:
goal_name = "clean_mug"
args = {
    "env": "kitchen",
    # "approach": "oracle",
    "approach": "refinement_estimation",
    "seed": random.randint(0,10000),
    "use_gui": True,
    "num_test_tasks": 1,
    "kitchen_use_perfect_samplers": True,
    # Kitchen 任务类型:
    #   - "clean_mug"
    #   - "make_tea"
    #   - "make_milktea"（需要在 kitchen.py / nsrts.py 中已接好）
    # 这里只是设置默认值，真正的任务选择以 CFG.kitchen_goals 为准，
    # 因此你可以在外部 config / 命令行里覆盖它。
    "kitchen_goals": goal_name,
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
    "refinement_estimation_num_skeletons_generated": 20,
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

env = create_new_env("kitchen", use_gui=CFG.use_gui)

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

obs = env.reset("test", 0)
print("INIT COMPLETE")

# Save current view and generate results.txt
def save_current_view(env, rgb_path: str | Path, depth_path: str | Path) -> None:
    """Save RGB and depth images from current camera view."""
    gym_env = env._gym_env
    renderer = gym_env.robot_env.mujoco_renderer

    viewer = renderer.viewer
    cam = viewer.cam
    # print(cam.distance, cam.azimuth, cam.elevation, cam.lookat)

    # RGB viewer
    renderer._get_viewer("rgb_array").vopt.geomgroup[2] = 0
    # Depth viewer
    renderer._get_viewer("depth_array").vopt.geomgroup[2] = 0

    rgb = renderer.render(render_mode="rgb_array", camera_name="fourth_cap")
    depth = renderer.render(render_mode="depth_array", camera_name="fourth_cap")

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

def load_camera_info_from_files(test_datapoint_dir: Path):
    """Load camera position and rotation from files.
    
    Reads from camera_positions.txt and camera_rotations.txt in test_datapoint_dir.
    Returns default values if files don't exist.
    """
    cam_position_file = test_datapoint_dir / "camera_positions.txt"
    cam_rotation_file = test_datapoint_dir / "camera_rotations.txt"
    
    # Default values (from create_dataset.py)
    default_cam_pos = np.array([0.4, -0.6, 2.8])
    default_cam_rot = np.array([1.1, 0.4, 0.2])
    
    # Try to load from files
    cam_pos = default_cam_pos.copy()
    cam_rot = default_cam_rot.copy()
    
    if cam_position_file.exists():
        try:
            with open(cam_position_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 3:
                        cam_pos = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        except Exception as e:
            logging.warning(f"Could not read camera position from {cam_position_file}: {e}")
    
    if cam_rotation_file.exists():
        try:
            with open(cam_rotation_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 3:
                        cam_rot = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        except Exception as e:
            logging.warning(f"Could not read camera rotation from {cam_rotation_file}: {e}")
    
    return cam_pos, cam_rot

class CoordMinMaxNormalize(object):
    """Normalize 3D coordinates to [0, 1] range."""
    def __init__(self, min_val, max_val):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for CoordMinMaxNormalize")
        self.min_val = torch.tensor(min_val, dtype=torch.float32).view(3)
        self.max_val = torch.tensor(max_val, dtype=torch.float32).view(3)

    def __call__(self, coord):
        normalized_coord = (coord - self.min_val) / (self.max_val - self.min_val)
        return torch.clamp(normalized_coord, 0, 1).squeeze()


def build_transforms():
    """Build image transforms for RGB, Gray, and depth images."""
    if not _TORCH_AVAILABLE:
        return None, None, None
    
    transform_RGB = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_Gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform_RGB, transform_Gray, transform_depth


def load_camera_position_from_file(position_file: str):
    """Load camera position from file."""
    if os.path.isfile(position_file):
        if position_file.endswith('.npy'):
            pos = np.load(position_file)
        else:
            with open(position_file, 'r') as f:
                line = f.readline().strip()
                pos = np.array([float(x) for x in line.replace(',', ' ').split()])
    else:
        raise FileNotFoundError(f"Camera position file not found: {position_file}")
    
    if len(pos) != 3:
        raise ValueError(f"Camera position must have 3 values, got {len(pos)}")
    return pos


def load_camera_rotation_from_file(rotation_file: str):
    """Load camera rotation from file."""
    if os.path.isfile(rotation_file):
        if rotation_file.endswith('.npy'):
            rot = np.load(rotation_file)
        else:
            with open(rotation_file, 'r') as f:
                line = f.readline().strip()
                rot = np.array([float(x) for x in line.replace(',', ' ').split()])
    else:
        raise FileNotFoundError(f"Camera rotation file not found: {rotation_file}")
    
    if len(rot) != 3:
        raise ValueError(f"Camera rotation must have 3 values, got {len(rot)}")
    return rot


def generate_results_txt(test_datapoint_dir: Path, model_ckpt: str = None, n_sample: int = 100, seed: int = 42, 
                         captured_views_history: list = None):
    """Generate results.txt by sampling 3D coordinates using diffusion model.
    
    使用基于 RGB-D 的 diffusion 模型，从当前视角采样物体三维坐标。
    当前场景中有 5 个物体：mug, sponge, tea, milk, banana（banana 暂不用于任务），
    模型权重默认从 models/tea_single_objects_new/*/best_model.pth 中加载。
    当 captured_views_history 提供时，只使用最新视角做采样。
    
    Args:
        test_datapoint_dir: Directory containing RGB and depth images
        model_ckpt: Path to model checkpoint (默认为 models/tea_single_objects_new/*/best_model.pth)
        n_sample: Number of samples to generate
        seed: Random seed
        captured_views_history: List of (cam_view_tensor, depth_tensor); if set, use last view
    """
    if not _TORCH_AVAILABLE:
        logging.warning("torch/torchvision not available. Skipping results.txt generation.")
        return False
    
    # Import network_tea_single_object (RGB-D only, no floor_plan/cam_position/cam_rotation)
    try:
        possible_code_paths = [
            Path(__file__).parent.parent.parent / "code",
            Path(__file__).parent.parent.parent.parent / "code",
            Path("code"),
            Path.cwd() / "code",
        ]
        code_path = None
        for p in possible_code_paths:
            if (p / "network_tea_single_object.py").exists():
                code_path = p
                break
        if code_path is None:
            logging.warning("Could not find network_tea_single_object.py in code/ directory. Skipping results.txt generation.")
            return False
        code_path_str = str(code_path)
        if code_path_str not in sys.path:
            sys.path.insert(0, code_path_str)
        from network_tea_single_object import DDPMTeaSingle, MaskedDenseFusionTeaSingle
        logging.info(f"Successfully imported network_tea_single_object from {code_path}")
    except ImportError as e:
        logging.warning(f"Could not import network_tea_single_object: {e}. Skipping results.txt generation.")
        return False
    
    # Prepare file paths (only RGB-D required)
    rgb_image = str(test_datapoint_dir / "image.png")
    depth_image = str(test_datapoint_dir / "depth.png")
    output_file = str(test_datapoint_dir / "results.txt")
    
    # Check required files (RGB + depth only)
    if not Path(rgb_image).exists():
        logging.warning(f"RGB image not found: {rgb_image}")
        return False
    if not Path(depth_image).exists():
        logging.warning(f"Depth image not found: {depth_image}")
        return False
    
    try:

        # Set CUBLAS_WORKSPACE_CONFIG environment variable for deterministic CUDA operations
        # This is required when torch.use_deterministic_algorithms(True) is set globally
        # and CUDA >= 10.2 is used
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logging.info("Set CUBLAS_WORKSPACE_CONFIG environment variable for deterministic CUDA operations")
            
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"torch.nn路径: {torch.nn.__file__ if hasattr(torch.nn, '__file__') else 'N/A'}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Running diffusion model on {device.upper()}")
        
        # Model configuration (match network_tea_single_object / training_summary)
        n_feat = 128
        n_T = 500
        betas = (1e-4, 0.02)
        
        transform_RGB, _, transform_depth = build_transforms()
        
        # Get depth and cam_view: from latest captured view or from disk
        if captured_views_history is not None and len(captured_views_history) > 0:
            cam_view_tensor, depth_tensor = captured_views_history[-1]
            cam_view = cam_view_tensor.to(device).float()
            depth = depth_tensor.to(device).float()
            if cam_view.dim() == 3:
                cam_view = cam_view.unsqueeze(0)
            if depth.dim() == 3:
                depth = depth.unsqueeze(0)
            logging.info(f"Using latest of {len(captured_views_history)} captured views (RGB-D only).")
        else:
            logging.info(f"Loading RGB and depth from {test_datapoint_dir}...")
            rgb_img = PILImage.open(rgb_image).convert("RGB")
            cam_view = transform_RGB(rgb_img).unsqueeze(0).to(device).float()
            depth_img = PILImage.open(depth_image).convert("L")
            depth = transform_depth(depth_img).unsqueeze(0).to(device).float()
        
        # Helper: find checkpoint path for a specific object name.
        workspace_root = Path(__file__).parent.parent.parent

        def _find_model_for_object(obj_name: str) -> str | None:
            """Return checkpoint path for given object name, or None if not found."""
            candidates = [
                workspace_root / "models" / "tea_single_objects_new" / obj_name / "best_model.pth",
                Path("models") / "tea_single_objects_new" / obj_name / "best_model.pth",
                workspace_root / "models" / "tea_single_objects" / obj_name / "best_model.pth",
                Path("models") / "tea_single_objects" / obj_name / "best_model.pth",
            ]
            for p in candidates:
                if p.exists():
                    return str(p)
            return None

        # Decide which object models to use based on current kitchen task.
        # For clean_mug: only mug and sponge are relevant.
        goal = goal_name
        if goal == "clean_mug":
            object_names = ["mug", "sponge"]
        elif goal == "make_tea":
            # Assume mug and tea are relevant for locating tea-related objects.
            object_names = ["mug", "tea"]
        elif goal == "make_milktea":
            # Assume full pipeline needs mug, sponge, tea, milk.
            object_names = ["mug", "sponge", "tea", "milk"]
        else:
            # Default: use all four grippable objects except banana.
            object_names = ["mug", "sponge", "tea", "milk"]

        # If a single explicit model_ckpt is given, fall back to old behavior:
        # use that checkpoint once and ignore per-object mapping.
        single_checkpoint_mode = model_ckpt is not None

        # World coordinate conversion (shared for all objects).
        world_min = np.array([-1.2, -1.0, 0.0])
        world_max = np.array([1.0, 1.5, 3.0])
        world_range = world_max - world_min

        all_coords_list = []

        if single_checkpoint_mode:
            ckpt_path = Path(model_ckpt)
            if not ckpt_path.exists():
                logging.warning(f"Explicit model_ckpt {ckpt_path} does not exist. Skipping results.txt generation.")
                return False
            object_list_for_logging = ["(single_checkpoint)"]
            ckpt_map = {"(single_checkpoint)": str(ckpt_path)}
        else:
            object_list_for_logging = object_names
            ckpt_map = {}
            for name in object_names:
                ckpt_path = _find_model_for_object(name)
                if ckpt_path is None:
                    logging.warning(f"No checkpoint found for object '{name}', skipping this object.")
                else:
                    ckpt_map[name] = ckpt_path

        if not ckpt_map:
            logging.warning("No valid model checkpoints found for any object. Skipping results.txt generation.")
            return False

        # Loop over each object-specific model and append all sampled coordinates.
        per_object_coords: Dict[str, np.ndarray] = {}
        for obj_name in object_list_for_logging:
            if obj_name not in ckpt_map:
                continue
            ckpt_path = ckpt_map[obj_name]
            logging.info(f"Loading model for object '{obj_name}' from {ckpt_path}...")

            model = DDPMTeaSingle(
                nn_model=MaskedDenseFusionTeaSingle(n_feat=n_feat, out_dim=3),
                betas=betas,
                n_T=n_T,
                device=device,
            )
            ckpt = torch.load(ckpt_path, map_location=device)
            model_dict = model.state_dict()
            pretrained = {
                k: v for k, v in ckpt.items()
                if k.startswith("nn_model.") and k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(pretrained)
            model.load_state_dict(model_dict, strict=False)
            model.to(device)
            model.eval()
            logging.info(f"Model for '{obj_name}' loaded successfully.")

            logging.info(f"Sampling 3D coordinates for '{obj_name}' (RGB-D only)...")
            start_time = time.perf_counter()
            with torch.no_grad():
                generated_coords, _ = model.sample(
                    n_sample=n_sample,
                    depth=depth,
                    cam_view=cam_view,
                    device=device,
                    guide_w=1.0,
                )
            end_time = time.perf_counter()

            sampling_time = end_time - start_time
            avg_time_per_sample = sampling_time / n_sample
            logging.info(
                f"[{obj_name}] Sampling completed in {sampling_time:.2f} seconds "
                f"({sampling_time:.4f}s total, {avg_time_per_sample:.4f}s per sample)"
            )

            coords_np = generated_coords.cpu().numpy()
            coords_world = coords_np * world_range + world_min
            per_object_coords[obj_name] = coords_world
            all_coords_list.append(coords_world)

        if not all_coords_list:
            logging.warning("Sampling produced no coordinates. Skipping results.txt generation.")
            return False

        # Save per-object results first: one file per object type.
        # Example filenames:
        #   predicators/test_datapoint/results_mug.txt
        #   predicators/test_datapoint/results_sponge.txt
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        total_points = 0
        for obj_name, coords_world in per_object_coords.items():
            obj_suffix = obj_name if obj_name != "(single_checkpoint)" else "single"
            obj_output = Path(output_dir) / f"results_{obj_suffix}.txt"
            logging.info(f"Saving coordinates for '{obj_name}' to {obj_output}...")
            with open(obj_output, "w") as f_obj:
                for coord in coords_world:
                    f_obj.write(" ".join(map(str, coord)) + "\n")
            total_points += coords_world.shape[0]

        # For backward compatibility, also save a combined results.txt that
        # merges all sampled coordinates across objects.
        all_coords = np.vstack(all_coords_list)
        logging.info(f"Total sampled coordinates (all objects combined): {all_coords.shape[0]}")
        logging.info(f"Saving combined coordinates to {output_file}...")
        with open(output_file, "w") as f:
            for coord in all_coords:
                f.write(" ".join(map(str, coord)) + "\n")
        logging.info(
            f"Successfully saved {all_coords.shape[0]} coordinates to {output_file} "
            f"and {len(per_object_coords)} per-object files."
        )
        return True
        
    except Exception as e:
        logging.error(f"Error generating results.txt: {e}", exc_info=True)
        return False

def capture_and_sample_for_planning(env):
    """Capture current view and generate results.txt using neural network sampling.
    
    This function should be called before each planning attempt (both initial planning
    and replanning) to ensure the latest environment state is captured and sampled.
    All captured views are stored in global _captured_views_history for multi-condition sampling.
    Each captured view is also saved to disk with a plan counter-based filename.
    """
    global _captured_views_history, _plan_counter
    
    if not _TORCH_AVAILABLE:
        logging.warning("torch/torchvision not available. Skipping view capture and sampling.")
        return
    
    # Increment plan counter
    _plan_counter += 1
    
    # Save current view and generate results.txt
    test_datapoint_dir = Path("predicators/test_datapoint")
    test_datapoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a subdirectory for this plan attempt
    plan_views_dir = test_datapoint_dir / f"plan_{_plan_counter:03d}"
    plan_views_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Plan {_plan_counter}] Saving current camera view...")
    # Save views with plan counter in filename
    rgb_path = plan_views_dir / f"image_plan_{_plan_counter:03d}.png"
    depth_path = plan_views_dir / f"depth_plan_{_plan_counter:03d}.png"
    save_current_view(env, rgb_path, depth_path)
    logging.info(f"[Plan {_plan_counter}] Saved RGB image to {rgb_path}")
    logging.info(f"[Plan {_plan_counter}] Saved depth image to {depth_path}")
    
    # Also save to the default location for compatibility
    default_rgb_path = test_datapoint_dir / "image.png"
    default_depth_path = test_datapoint_dir / "depth.png"
    save_current_view(env, default_rgb_path, default_depth_path)

    # Load and preprocess current view for multi-condition sampling
    transform_RGB, transform_Gray, transform_depth = build_transforms()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load current RGB and depth images (use the plan-specific paths)
    rgb_img = PILImage.open(rgb_path).convert("RGB")
    cam_view_tensor = transform_RGB(rgb_img).unsqueeze(0).to(device).float()
    
    depth_img = PILImage.open(depth_path).convert("L")
    depth_tensor = transform_depth(depth_img).unsqueeze(0).to(device).float()
    
    # Add current view to history
    _captured_views_history.append((cam_view_tensor, depth_tensor))
    logging.info(f"[Plan {_plan_counter}] Added current view to history. Total captured views: {len(_captured_views_history)}")

    # Load camera information from files (fixed values)
    logging.info("Loading camera information from files...")
    cam_pos, cam_rot = load_camera_info_from_files(test_datapoint_dir)
    cam_position_file = test_datapoint_dir / "camera_positions.txt"
    cam_rotation_file = test_datapoint_dir / "camera_rotations.txt"

    # Ensure files exist with correct values
    cam_position_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cam_position_file, "w") as f:
        f.write(f"{cam_pos[0]:.6f}, {cam_pos[1]:.6f}, {cam_pos[2]:.6f}\n")

    with open(cam_rotation_file, "w") as f:
        f.write(f"{cam_rot[0]:.6f}, {cam_rot[1]:.6f}, {cam_rot[2]:.6f}\n")

    logging.info(f"Camera position: {cam_pos}")
    logging.info(f"Camera rotation: {cam_rot}")
    logging.info(f"Camera files ready at {cam_position_file} and {cam_rotation_file}")

    # Check if floor_0.png exists (copy from test_datapoint if available)
    floor_plan_src = test_datapoint_dir / "floor_0.png"
    if not floor_plan_src.exists():
        # Try to find floor plan in other locations
        possible_floor_plans = [
            Path("code/test_datapoint/floor_0.png"),
            Path("predicators/test_datapoint/floor_0.png"),
            Path(__file__).parent.parent.parent / "code" / "test_datapoint" / "floor_0.png",
        ]
        copied = False
        for fp in possible_floor_plans:
            if fp.exists():
                shutil.copy(fp, floor_plan_src)
                logging.info(f"Copied floor plan from {fp} to {floor_plan_src}")
                copied = True
                break
        
        if not copied:
            logging.warning(f"Floor plan image not found. Please ensure floor_0.png exists in {test_datapoint_dir}")

    # Generate results.txt using all captured views for multi-condition sampling
    logging.info(f"[Plan {_plan_counter}] Generating results.txt with {len(_captured_views_history)} conditions...")
    if generate_results_txt(test_datapoint_dir, n_sample=100, captured_views_history=_captured_views_history):
        logging.info(f"[Plan {_plan_counter}] Successfully generated results.txt")
    else:
        logging.warning(f"[Plan {_plan_counter}] Failed to generate results.txt. Continuing without it.")
    logging.info(f"[Plan {_plan_counter}] Finished saving images and generating results.txt")

# obs = gym_env.reset("test", 0)
# obs = gym_env.get_observation()
state = env.state_info_to_state(obs["state_info"])


# run oracle approach
def run_pipline(env, cogman, approach_train_tasks, offline_dataset):
    if cogman.is_learning_based:
        pass
    else:
        results = run_testing(env, cogman)
        results["num_offline_transitions"] = 0
        results["num_online_transitions"] = 0
        results["query_cost"] = 0.0
        results["learning_time"] = 0.0
        save_test_results(results, online_learning_cycle=None)

def run_testing(env, cogman):
    test_tasks = env.get_test_tasks()
    if CFG.approach != "oracle":
        test_tasks = [task.replace_goal_with_alt_goal() for task in test_tasks]
    num_found_policy = 0
    num_solved = 0
    cogman.reset_metrics()
    total_suc_time = 0.0
    total_low_level_action_cost = 0.0
    total_num_solve_timeouts = 0
    total_num_solve_failures = 0
    total_num_execution_timeouts = 0
    total_num_execution_failures = 0

    save_prefix = utils.get_config_path_str()
    metrics: Metrics = defaultdict(float)
    curr_num_nodes_created = 0.0
    curr_num_nodes_expanded = 0.0
    
    # Reset captured views history and plan counter at the start of testing
    global _captured_views_history, _plan_counter
    
    for test_task_idx, env_task in enumerate(test_tasks):
        # Reset captured views history and plan counter at the start of each task
        _captured_views_history = []
        _plan_counter = 0
        solve_start = time.perf_counter()
        try:
            # Capture current view and sample before planning
            logging.info(f"[Planning] Capturing view and sampling before initial planning for task {test_task_idx+1}...")
            capture_and_sample_for_planning(env)
            
            # We call reset here, outside of run_episode_and_get_observations,
            # so that we can log planning failures, timeouts, etc. This is
            # mostly for legacy reasons (before cogman existed separately
            # from approaches).
            cogman.reset(env_task)

            if hasattr(approach, '_last_nsrt_plan'):
                logging.info(f"Generated NSRT plan with {len(approach._last_nsrt_plan)} steps:") 
                for i, nsrt in enumerate(approach._last_nsrt_plan):
                    logging.info(f"  Step {i+1}: {nsrt}")


        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                         f"Approach failed to solve with error: {e}")
            if isinstance(e, ApproachTimeout):
                total_num_solve_timeouts += 1
            elif isinstance(e, ApproachFailure):
                total_num_solve_failures += 1
            if CFG.make_failure_videos and e.info.get("partial_refinements"):
                video = utils.create_video_from_partial_refinements(
                    e.info["partial_refinements"], env, "test", test_task_idx,
                    CFG.horizon)
                outfile = f"{save_prefix}__task{test_task_idx+1}_failure.mp4"
                utils.save_video(outfile, video)
            if CFG.crash_on_failure:
                raise e
            continue
        solve_time = time.perf_counter() - solve_start
        metrics[f"PER_TASK_task{test_task_idx}_solve_time"] = solve_time
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_created"] = cogman.metrics[
                "total_num_nodes_created"] - curr_num_nodes_created
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_expanded"] = cogman.metrics[
                "total_num_nodes_expanded"] - curr_num_nodes_expanded
        curr_num_nodes_created = cogman.metrics["total_num_nodes_created"]
        curr_num_nodes_expanded = cogman.metrics["total_num_nodes_expanded"]

        num_found_policy += 1
        make_video = False
        solved = False
        caught_exception = False
        traj = None
        exec_time = 0.0
        if CFG.make_test_videos or CFG.make_failure_videos:
            monitor = utils.VideoMonitor(env.render)
        else:
            monitor = None
        try:
            # Now, measure success by running the policy in the environment.
            traj, solved, execution_metrics = run_episode_and_get_observations(
                cogman,
                env,
                "test",
                test_task_idx,
                max_num_steps=CFG.horizon,
                monitor=monitor)
            num_opt = execution_metrics["num_options_executed"]
            metrics[f"PER_TASK_task{test_task_idx}_options_executed"] = num_opt
            exec_time = execution_metrics["policy_call_time"]
            metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
            if CFG.refinement_data_include_execution_cost:
                total_low_level_action_cost += (
                    len(traj[1]) *
                    CFG.refinement_data_low_level_execution_cost)
            if CFG.save_eval_trajs:
                # Save the successful trajectory, e.g., for playback on a
                # robot.
                traj_file = f"{save_prefix}__task{test_task_idx+1}.traj"
                traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
                # Include the original task too so we know the goal.
                traj_data = {
                    "task": env_task,
                    "trajectory": traj,
                    "pybullet_robot": CFG.pybullet_robot
                }
                with open(traj_file_path, "wb") as f:
                    pkl.dump(traj_data, f)
        except utils.EnvironmentFailure as e:
            log_message = f"Environment failed with error: {e}"
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            error_msg = str(e.args[0]) if e.args else ""
            
            # Recoverable execution failures: replan from current state.
            if isinstance(e, ApproachFailure) and (
                "NSRT plan exhausted" in error_msg
                or "plan exhausted" in error_msg
                or "failed to achieve the necessary atoms" in error_msg
                or "Observe extra discovery" in error_msg
            ):
                logging.info(
                    "[Replan] Recoverable failure (plan exhausted or necessary-atoms "
                    "mismatch). Attempting hot replanning..."
                )
                
                try:
                    # Capture current view and sample before replanning
                    logging.info(f"[Replan] Capturing view and sampling before replanning...")
                    capture_and_sample_for_planning(env)
                    
                    # Get current environment state
                    current_obs = env.get_observation()
                    # Convert observation to state using perceiver
                    current_state = cogman._perceiver.step(current_obs)
                    
                    # Get original goal from cogman._current_goal
                    # This avoids type checking issues with env_task.task.goal
                    # cogman._current_goal is set during cogman.reset() and is guaranteed to be Set[GroundAtom]
                    assert cogman._current_goal is not None, "Current goal should be set during reset"
                    original_goal = cogman._current_goal
                    
                    # Create new task with current state and original goal
                    new_task = Task(current_state, original_goal)
                    
                    # Replan: reset policy with new task
                    logging.info(f"[Replan] Creating new task from current state and replanning...")
                    cogman._reset_policy(new_task)
                    cogman._exec_monitor.reset(new_task)
                    cogman._exec_monitor.update_approach_info(
                        cogman._approach.get_execution_monitoring_info())
                    
                    # Continue execution without resetting environment
                    logging.info(f"[Replan] Replanning successful. Continuing execution from current state...")
                    traj, solved, execution_metrics = run_episode_and_get_observations(
                        cogman,
                        env,
                        "test",
                        test_task_idx,
                        max_num_steps=CFG.horizon,
                        do_env_reset=False,  # Important: don't reset environment
                        monitor=monitor)
                    
                    # Update metrics
                    num_opt = execution_metrics["num_options_executed"]
                    metrics[f"PER_TASK_task{test_task_idx}_options_executed"] = num_opt
                    exec_time = execution_metrics["policy_call_time"]
                    metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
                    if CFG.refinement_data_include_execution_cost:
                        total_low_level_action_cost += (
                            len(traj[1]) *
                            CFG.refinement_data_low_level_execution_cost)
                    if CFG.save_eval_trajs:
                        traj_file = f"{save_prefix}__task{test_task_idx+1}.traj"
                        traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
                        traj_data = {
                            "task": env_task,
                            "trajectory": traj,
                            "pybullet_robot": CFG.pybullet_robot
                        }
                        with open(traj_file_path, "wb") as f:
                            pkl.dump(traj_data, f)
                    
                    # Reset caught_exception flag since replanning succeeded
                    caught_exception = False
                    logging.info(f"[Replan] Replanning and continued execution completed.")
                    
                except Exception as replan_e:
                    # If replanning also failed, log and treat as original failure
                    replan_error_type = type(replan_e).__name__
                    replan_error_msg = str(replan_e)
                    replan_error_args = replan_e.args
                    logging.warning(f"[Replan] Replanning failed with error type: {replan_error_type}, "
                                   f"message: {replan_error_msg}, args: {replan_error_args}")
                    logging.warning(f"[Replan] Full traceback:\n{traceback.format_exc()}")
                    log_message = ("Approach failed at policy execution time with "
                                   f"error: {e}. Replanning also failed: {replan_error_type}: {replan_error_msg}")
                    if isinstance(e, ApproachTimeout):
                        total_num_execution_timeouts += 1
                    elif isinstance(e, ApproachFailure):
                        total_num_execution_failures += 1
                    caught_exception = True
            else:
                # Other types of failures, handle normally
                log_message = ("Approach failed at policy execution time with "
                               f"error: {e}")
                if isinstance(e, ApproachTimeout):
                    total_num_execution_timeouts += 1
                elif isinstance(e, ApproachFailure):
                    total_num_execution_failures += 1
                caught_exception = True
        if solved:
            log_message = "SOLVED"
            num_solved += 1
            total_suc_time += (solve_time + exec_time)
            make_video = CFG.make_test_videos
            video_file = f"{save_prefix}__task{test_task_idx+1}.mp4"
            metrics[f"PER_TASK_task{test_task_idx}_num_steps"] = len(traj[1])
        else:
            if not caught_exception:
                log_message = "Policy failed to reach goal"
            if CFG.crash_on_failure:
                raise RuntimeError(log_message)
            make_video = CFG.make_failure_videos
            video_file = f"{save_prefix}__task{test_task_idx+1}_failure.mp4"
        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: "
                     f"{log_message}")
        if make_video:
            assert monitor is not None
            video = monitor.get_video()
            utils.save_video(video_file, video)
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (total_suc_time /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["avg_ref_cost"] = ((total_low_level_action_cost +
                                cogman.metrics["total_refinement_time"]) /
                               num_solved if num_solved > 0 else float("inf"))
    metrics["min_num_samples"] = cogman.metrics[
        "min_num_samples"] if cogman.metrics["min_num_samples"] < float(
            "inf") else 0
    metrics["max_num_samples"] = cogman.metrics["max_num_samples"]
    metrics["min_skeletons_optimized"] = cogman.metrics[
        "min_num_skeletons_optimized"] if cogman.metrics[
            "min_num_skeletons_optimized"] < float("inf") else 0
    metrics["max_skeletons_optimized"] = cogman.metrics[
        "max_num_skeletons_optimized"]
    metrics["num_solve_timeouts"] = total_num_solve_timeouts
    metrics["num_solve_failures"] = total_num_solve_failures
    metrics["num_execution_timeouts"] = total_num_execution_timeouts
    metrics["num_execution_failures"] = total_num_execution_failures
    # Handle computing averages of total cogman metrics wrt the
    # number of found policies. Note: this is different from computing
    # an average wrt the number of solved tasks, which might be more
    # appropriate for some metrics, e.g. avg_suc_time above.
    for metric_name in [
            "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
            "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
            "num_failures_discovered"
    ]:
        total = cogman.metrics[f"total_{metric_name}"]
        metrics[f"avg_{metric_name}"] = (
            total / num_found_policy if num_found_policy > 0 else float("inf"))
    return metrics

def save_test_results(results: Metrics,
                       online_learning_cycle: Optional[int]) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    outfile = (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
               f"{online_learning_cycle}.pkl")
    # Save CFG alongside results.
    outdata = {
        "config": CFG,
        "results": results.copy(),
        # "git_commit_hash": utils.get_git_commit_hash()
    }
    # Dump the CFG, results, and git commit hash to a pickle file.
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    # Before printing the results, filter out keys that start with the
    # special prefix "PER_TASK_", to prevent an annoyingly long printout.
    del_keys = [k for k in results if k.startswith("PER_TASK_")]
    for k in del_keys:
        del results[k]
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")

run_pipline(env, cogman, approach_train_tasks, offline_dataset)
script_time = time.perf_counter() - script_start
logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")