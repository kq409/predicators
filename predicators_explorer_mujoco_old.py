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

#Defining test configuration, and overriding some default ones:
args = {
    "env": "kitchen",
    "approach": "oracle",
    "seed": random.randint(0,10000),
    "use_gui": True,
    "num_test_tasks": 1,
    "kitchen_use_perfect_samplers": True,
    "kitchen_goals": "clean_mug",
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
    for test_task_idx, env_task in enumerate(test_tasks):
        solve_start = time.perf_counter()
        try:
            # We call reset here, outside of run_episode_and_get_observations,
            # so that we can log planning failures, timeouts, etc. This is
            # mostly for legacy reasons (before cogman existed separately
            # from approaches).
            cogman.reset(env_task)
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