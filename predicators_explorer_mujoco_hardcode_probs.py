import sys
import time
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from predicators import utils
from predicators.structs import Metrics, Task
from predicators.settings import CFG
from predicators.envs import create_new_env, BaseEnv
from predicators.datasets import create_dataset
from predicators.perception import create_perceiver
from predicators.approaches import create_approach, ApproachTimeout, ApproachFailure
from predicators.execution_monitoring import create_execution_monitor
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.ground_truth_models import get_gt_options, parse_config_included_options

from predicators.refinement_estimators import oracle_refinement_estimator
from predicators.refinement_estimators.oracle_refinement_estimator import (
    kitchen_oracle_estimator_per_object,
)


###############################################################################
# Hard-coded region probabilities (per object, replacing NN-based statistics) #
###############################################################################

def get_hardcoded_region_probs_by_object(env: BaseEnv, initial_state) -> Dict[str, Dict[str, float]]:
    """Return hard-coded per-object region probabilities instead of NN analysis.

    For each object type (mug, sponge, tea, milk, banana), this function can
    provide a separate distribution over visible container regions:
        - 'microwave'           -> microhandle
        - 'right_hinge_cabinet' -> hinge2
        - 'slide_cabinet'       -> slide

    NOTE: Numeric values below are placeholders. You should overwrite them
    according to your empirical statistics later.
    """
    # Determine which containers are currently open based on initial_state.
    # We use the same Open_holds logic as KitchenEnv.
    from predicators.envs.kitchen import KitchenEnv

    slide = KitchenEnv.object_name_to_object("slide")
    hinge2 = KitchenEnv.object_name_to_object("hinge2")

    is_slide_open = KitchenEnv.Open_holds(initial_state, [slide])
    is_hinge2_open = KitchenEnv.Open_holds(initial_state, [hinge2])

    goal = CFG.kitchen_goals

    # Initialize result dict
    region_probs_by_obj: Dict[str, Dict[str, float]] = {}

    # Three phases, following your description:
    # 1) All doors closed (initial): both slide and hinge2 closed.
    # 2) slide_cabinet opened, hinge2 still closed.
    # 3) slide_cabinet and right_hinge_cabinet (hinge2) both open.

    # Probabilities are currently only specified for mug and sponge.
    # For tea/milk/banana we keep them uniform (or fallback) since they
    # are not involved in clean_mug right now.

    if not is_slide_open and not is_hinge2_open:
        # Phase 1: all closed
        # mug and sponge have the same distribution:
        #   microwave: 0.15, right_hinge_cabinet: 0.25, slide_cabinet: 0.6
        base = {
            "microwave": 0.15,
            "right_hinge_cabinet": 0.25,
            "slide_cabinet": 0.60,
        }
        region_probs_by_obj["mug"] = dict(base)
        region_probs_by_obj["sponge"] = dict(base)
    elif is_slide_open and not is_hinge2_open:
        # Phase 2: slide open, hinge2 closed
        # mug and sponge share:
        #   microwave: 0.375, right_hinge_cabinet: 0.625, slide_cabinet: 0
        base = {
            "microwave": 0.375,
            "right_hinge_cabinet": 0.625,
            "slide_cabinet": 0.0,
        }
        region_probs_by_obj["mug"] = dict(base)
        region_probs_by_obj["sponge"] = dict(base)
    elif is_slide_open and is_hinge2_open:
        # Phase 3: slide and hinge2 both open
        #   mug:    microwave: 1,   right_hinge_cabinet: 0, slide_cabinet: 0
        #   sponge: microwave: 0,   right_hinge_cabinet: 1, slide_cabinet: 0
        region_probs_by_obj["mug"] = {
            "microwave": 1.0,
            "right_hinge_cabinet": 0.0,
            "slide_cabinet": 0.0,
        }
        region_probs_by_obj["sponge"] = {
            "microwave": 0.0,
            "right_hinge_cabinet": 1.0,
            "slide_cabinet": 0.0,
        }
    else:
        # Any other unusual combination: fall back to a simple uniform prior
        for obj_name in ["banana", "mug", "sponge", "tea", "milk"]:
            region_probs_by_obj[obj_name] = {
                "microwave": 1.0 / 3.0,
                "right_hinge_cabinet": 1.0 / 3.0,
                "slide_cabinet": 1.0 / 3.0,
            }

    # For completeness, if user wants to support other tasks like make_tea /
    # make_milktea in the hard-coded script, they can extend the above logic
    # to also fill entries for "tea" and "milk". For now, clean_mug is the
    # primary focus, and mug/sponge are the only objects with detailed priors.

    return region_probs_by_obj


def get_hardcoded_default_region_probs() -> Dict[str, float]:
    """Return default (merged) region probabilities for fallback use.

    Used when we cannot infer which object is being searched, or when a
    particular object has no dedicated hard-coded distribution.

    NOTE: Values are placeholders; you can edit them later.
    """
    region_probs = {
        "microwave": 1.0 / 3.0,
        "right_hinge_cabinet": 1.0 / 3.0,
        "slide_cabinet": 1.0 / 3.0,
    }
    total = sum(region_probs.values())
    if total > 0:
        for k in region_probs:
            region_probs[k] /= total
    return region_probs


########################################################################
# Monkey-patch OracleRefinementEstimator to use hard-coded probabilities
########################################################################

def _patched_get_cost(self, initial_task: Task, skeleton, atoms_sequence) -> float:
    """Patched get_cost for OracleRefinementEstimator in kitchen env.

    This mirrors the new per-object estimator logic but replaces the
    diffusion-based region probabilities with hard-coded ones.
    """
    env_name = CFG.env
    if env_name == "narrow_passage":
        return oracle_refinement_estimator.narrow_passage_oracle_estimator(
            self._env, initial_task.init, skeleton, atoms_sequence
        )
    if env_name == "exit_garage":
        return oracle_refinement_estimator.exit_garage_oracle_estimator(
            self._env, initial_task.init, skeleton, atoms_sequence
        )
    if env_name == "kitchen":
        # Per-object priors (hard-coded), plus a default merged prior.
        region_probs_by_obj = get_hardcoded_region_probs_by_object(
            self._env, initial_task.init
        )
        region_probs_default = get_hardcoded_default_region_probs()
        return kitchen_oracle_estimator_per_object(
            self._env,
            initial_task.init,
            skeleton,
            atoms_sequence,
            region_probs_by_obj,
            region_probs_default,
        )
    raise NotImplementedError(
        f"No oracle refinement cost estimator for env {env_name}"
    )


def _apply_oracle_monkey_patch() -> None:
    """Install the patched get_cost into OracleRefinementEstimator."""
    OracleRefinementEstimator = oracle_refinement_estimator.OracleRefinementEstimator
    OracleRefinementEstimator.get_cost = _patched_get_cost  # type: ignore[assignment]


#############################
# Experiment setup and run. #
#############################

def _setup_logging() -> None:
    """Configure logging, similar to predicators_explorer_mujoco.py."""
    str_args = " ".join(sys.argv)
    Path(CFG.log_dir).mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if CFG.log_file:
        handlers.append(logging.FileHandler(CFG.log_file, mode="w"))
    logging.basicConfig(
        level=CFG.loglevel,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    if CFG.log_file:
        logging.info(f"Logging to {CFG.log_file}")
    logging.info(f"Running command: python {str_args}")
    logging.info("Full config:")
    logging.info(CFG)


def _create_env_and_approach():
    """Create kitchen env, predicates, options, perceiver, approach, etc."""
    env = create_new_env("kitchen", use_gui=CFG.use_gui)
    env.action_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)

    included_preds, excluded_preds = utils.parse_config_excluded_predicates(env)
    preds = (
        utils.replace_goals_with_agent_specific_goals(
            included_preds, excluded_preds, env
        )
        if CFG.approach != "oracle"
        else included_preds
    )

    env_train_tasks = env.get_train_tasks()
    perceiver = create_perceiver(CFG.perceiver)
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]
    stripped_train_tasks = [utils.strip_task(task, preds) for task in train_tasks]
    approach_train_tasks = [
        task.replace_goal_with_alt_goal() for task in stripped_train_tasks
    ]

    if CFG.option_learner == "no_learning":
        options = get_gt_options(env.get_name())
    else:
        options = parse_config_included_options(env)

    approach_name = CFG.approach
    if CFG.approach_wrapper:
        approach_name = f"{CFG.approach_wrapper}[{approach_name}]"
    approach = create_approach(
        approach_name, preds, options, env.types, env.action_space, approach_train_tasks
    )

    if approach.is_learning_based:
        offline_dataset = create_dataset(env, train_tasks, options, preds)
    else:
        offline_dataset = None

    execution_monitor = create_execution_monitor(CFG.execution_monitor)
    cogman = CogMan(approach, perceiver, execution_monitor)

    return env, cogman, approach_train_tasks, offline_dataset


def run_pipline(env, cogman, approach_train_tasks, offline_dataset):
    if cogman.is_learning_based:
        # Not used for this script, but kept for completeness.
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
            # Planning only (no diffusion, no camera capture, no results.txt).
            cogman.reset(env_task)
        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(
                f"Task {test_task_idx+1} / {len(test_tasks)}: "
                f"Approach failed to solve with error: {e}"
            )
            if isinstance(e, ApproachTimeout):
                total_num_solve_timeouts += 1
            elif isinstance(e, ApproachFailure):
                total_num_solve_failures += 1
            if CFG.crash_on_failure:
                raise e
            continue

        solve_time = time.perf_counter() - solve_start
        metrics[f"PER_TASK_task{test_task_idx}_solve_time"] = solve_time
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_created"
        ] = cogman.metrics["total_num_nodes_created"] - curr_num_nodes_created
        metrics[
            f"PER_TASK_task{test_task_idx}_nodes_expanded"
        ] = cogman.metrics["total_num_nodes_expanded"] - curr_num_nodes_expanded
        curr_num_nodes_created = cogman.metrics["total_num_nodes_created"]
        curr_num_nodes_expanded = cogman.metrics["total_num_nodes_expanded"]

        num_found_policy += 1
        solved = False
        caught_exception = False
        traj = None
        exec_time = 0.0
        if CFG.make_test_videos or CFG.make_failure_videos:
            monitor = utils.VideoMonitor(env.render)
        else:
            monitor = None
        try:
            traj, solved, execution_metrics = run_episode_and_get_observations(
                cogman,
                env,
                "test",
                test_task_idx,
                max_num_steps=CFG.horizon,
                monitor=monitor,
            )
            num_opt = execution_metrics["num_options_executed"]
            metrics[f"PER_TASK_task{test_task_idx}_options_executed"] = num_opt
            exec_time = execution_metrics["policy_call_time"]
            metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
            if CFG.refinement_data_include_execution_cost:
                total_low_level_action_cost += (
                    len(traj[1]) * CFG.refinement_data_low_level_execution_cost
                )
        except utils.EnvironmentFailure as e:
            log_message = f"Environment failed with error: {e}"
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            error_msg = str(e.args[0]) if e.args else ""

            # Hot replanning logic (copied from predicators_explorer_mujoco.py)
            if isinstance(e, ApproachFailure) and (
                "NSRT plan exhausted" in error_msg or "plan exhausted" in error_msg
            ):
                logging.info(
                    "[Replan] NSRT plan exhausted detected. Attempting hot replanning..."
                )

                try:
                    # Get current environment state
                    current_obs = env.get_observation()
                    # Convert observation to state using perceiver
                    current_state = cogman._perceiver.step(current_obs)

                    # Get original goal from cogman._current_goal
                    # cogman._current_goal is set during cogman.reset()
                    # and is guaranteed to be Set[GroundAtom]
                    assert (
                        cogman._current_goal is not None
                    ), "Current goal should be set during reset"
                    original_goal = cogman._current_goal

                    # Create new task with current state and original goal
                    new_task = Task(current_state, original_goal)

                    # Replan: reset policy with new task
                    logging.info(
                        "[Replan] Creating new task from current state and replanning..."
                    )
                    cogman._reset_policy(new_task)
                    cogman._exec_monitor.reset(new_task)
                    cogman._exec_monitor.update_approach_info(
                        cogman._approach.get_execution_monitoring_info()
                    )

                    # Continue execution without resetting environment
                    logging.info(
                        "[Replan] Replanning successful. Continuing execution from current state..."
                    )
                    traj, solved, execution_metrics = run_episode_and_get_observations(
                        cogman,
                        env,
                        "test",
                        test_task_idx,
                        max_num_steps=CFG.horizon,
                        monitor=monitor,
                        do_env_reset=False,
                    )

                    # Update metrics after replanning
                    num_opt = execution_metrics["num_options_executed"]
                    metrics[
                        f"PER_TASK_task{test_task_idx}_options_executed"
                    ] = num_opt
                    exec_time = execution_metrics["policy_call_time"]
                    metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
                    if CFG.refinement_data_include_execution_cost:
                        total_low_level_action_cost += (
                            len(traj[1]) * CFG.refinement_data_low_level_execution_cost
                        )

                    # Reset caught_exception flag since replanning succeeded
                    caught_exception = False
                    logging.info(
                        "[Replan] Replanning and continued execution completed."
                    )

                except Exception as replan_e:
                    # If replanning also failed, log and treat as original failure
                    replan_error_type = type(replan_e).__name__
                    replan_error_msg = str(replan_e)
                    replan_error_args = replan_e.args
                    logging.warning(
                        f"[Replan] Replanning failed with error type: {replan_error_type}, "
                        f"message: {replan_error_msg}, args: {replan_error_args}"
                    )
                    log_message = (
                        "Approach failed at policy execution time with "
                        f"error: {e}. Replanning also failed: {replan_error_type}: {replan_error_msg}"
                    )
                    if isinstance(e, ApproachTimeout):
                        total_num_execution_timeouts += 1
                    elif isinstance(e, ApproachFailure):
                        total_num_execution_failures += 1
                    caught_exception = True
            else:
                # Other types of failures, handle normally
                log_message = (
                    "Approach failed at policy execution time with "
                    f"error: {e}"
                )
                if isinstance(e, ApproachTimeout):
                    total_num_execution_timeouts += 1
                elif isinstance(e, ApproachFailure):
                    total_num_execution_failures += 1
                caught_exception = True

        if solved:
            log_message = "SOLVED"
            num_solved += 1
            total_suc_time += solve_time + exec_time
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

        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: {log_message}")
        if make_video:
            assert monitor is not None
            video = monitor.get_video()
            utils.save_video(video_file, video)

    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (
        total_suc_time / num_solved if num_solved > 0 else float("inf")
    )
    metrics["avg_ref_cost"] = (
        (total_low_level_action_cost + cogman.metrics["total_refinement_time"])
        / num_solved
        if num_solved > 0
        else float("inf")
    )
    metrics["min_num_samples"] = (
        cogman.metrics["min_num_samples"]
        if cogman.metrics["min_num_samples"] < float("inf")
        else 0
    )
    metrics["max_num_samples"] = cogman.metrics["max_num_samples"]
    metrics["min_skeletons_optimized"] = (
        cogman.metrics["min_num_skeletons_optimized"]
        if cogman.metrics["min_num_skeletons_optimized"] < float("inf")
        else 0
    )
    metrics["max_skeletons_optimized"] = cogman.metrics["max_num_skeletons_optimized"]
    metrics["num_solve_timeouts"] = total_num_solve_timeouts
    metrics["num_solve_failures"] = total_num_solve_failures
    metrics["num_execution_timeouts"] = total_num_execution_timeouts
    metrics["num_execution_failures"] = total_num_execution_failures

    for metric_name in [
        "num_samples",
        "num_skeletons_optimized",
        "num_nodes_expanded",
        "num_nodes_created",
        "num_nsrts",
        "num_preds",
        "plan_length",
        "num_failures_discovered",
    ]:
        total = cogman.metrics[f"total_{metric_name}"]
        metrics[f"avg_{metric_name}"] = (
            total / num_found_policy if num_found_policy > 0 else float("inf")
        )
    return metrics


def save_test_results(results: Metrics, online_learning_cycle: Optional[int]) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    outfile = f"{CFG.results_dir}/{utils.get_config_path_str()}__{online_learning_cycle}.pkl"
    outdata = {
        "config": CFG,
        "results": results.copy(),
    }
    with open(outfile, "wb") as f:
        import dill as pkl

        pkl.dump(outdata, f)
    del_keys = [k for k in results if k.startswith("PER_TASK_")]
    for k in del_keys:
        del results[k]
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")


def main() -> None:
    # Basic config for this script (similar to predicators_explorer_mujoco).
    args = {
        "env": "kitchen",
        "approach": "refinement_estimation",
        "seed": random.randint(0, 10000),
        "use_gui": True,
        "num_test_tasks": 1,
        "kitchen_use_perfect_samplers": True,
        # Set the kitchen task here: "clean_mug", "make_tea", "make_milktea"
        "kitchen_goals": "clean_mug",
        "pybullet_sim_steps_per_action": 20,
        "pybullet_camera_width": 1674,
        "pybullet_camera_height": 900,
        "render_state_dpi": 300,
        "video_fps": 10,
        "make_test_videos": False,
        "make_failure_videos": False,
        "log_file": "predicators/logs/predicators_explorer_mujoco_hardcode_probs.log",
        "log_dir": "predicators/logs",
        "loglevel": logging.DEBUG,
        "refinement_estimation_num_skeletons_generated": 10,
        # Increase high-level skeleton search timeout (in seconds).
        # This controls the `timeout` passed to `_skeleton_generator`.
        "timeout": 20.0,
    }
    utils.reset_config(args)

    Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)
    Path(CFG.eval_trajectories_dir).mkdir(parents=True, exist_ok=True)

    _setup_logging()

    # Apply oracle monkey-patch before creating approach / running planning.
    _apply_oracle_monkey_patch()

    env, cogman, approach_train_tasks, offline_dataset = _create_env_and_approach()

    script_start = time.perf_counter()
    run_pipline(env, cogman, approach_train_tasks, offline_dataset)
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


if __name__ == "__main__":
    main()

