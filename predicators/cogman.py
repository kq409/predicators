"""Cognitive manager (CogMan).

A wrapper around an approach that manages interaction with the environment.

Implements a perception module, which produces a State at each time step based
on the history of observations, and an execution monitor, which determines
whether to re-query the approach at each time step based on the states.

The name "CogMan" is due to Leslie Kaelbling.
"""
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.envs import BaseEnv
from predicators.execution_monitoring import BaseExecutionMonitor
from predicators.perception import BasePerceiver
from predicators.settings import CFG
from predicators.structs import Action, Dataset, EnvironmentTask, GroundAtom, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, Metrics, \
    Observation, State, Task, Video, _Option


def _option_params_to_json_list(params: np.ndarray) -> List[float]:
    """Flatten option params to a JSON-serializable list of floats."""
    if params is None or (hasattr(params, "size") and params.size == 0):
        return []
    arr = np.asarray(params, dtype=np.float64).flatten()
    return [float(x) for x in arr.tolist()]


def _option_to_execution_record(option: _Option, segment_index: int,
                                episode_index: int) -> Dict[str, Any]:
    """Serializable description of a grounded option (no timing yet)."""
    objects_payload: List[Dict[str, str]] = []
    object_names: List[str] = []
    for obj in option.objects:
        objects_payload.append({"name": obj.name, "type": obj.type.name})
        object_names.append(obj.name)
    return {
        "event": "option_finished",
        "episode_index": episode_index,
        "segment_index": segment_index,
        "option_name": option.name,
        "objects": objects_payload,
        "object_names": object_names,
        "params": _option_params_to_json_list(option.params),
    }


def _append_cogman_option_log(record: Dict[str, Any],
                              duration_wall_sec: float,
                              num_env_steps: int) -> None:
    """Append one JSON line if CFG.cogman_option_log_file is set."""
    path_str = getattr(CFG, "cogman_option_log_file", "") or ""
    if not path_str:
        return
    out = dict(record)
    out["duration_wall_sec"] = float(duration_wall_sec)
    out["num_env_steps"] = int(num_env_steps)
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(out, ensure_ascii=False) + "\n")


def _format_option_log_message(option: _Option, duration_wall_sec: float,
                               num_env_steps: int) -> str:
    rec = _option_to_execution_record(option, segment_index=-1,
                                      episode_index=-1)
    rec.pop("event", None)
    rec.pop("segment_index", None)
    rec.pop("episode_index", None)
    parts = [
        f"Finished option: {option.name}",
        f"duration={duration_wall_sec:.4f}s",
        f"steps={num_env_steps}",
    ]
    if rec["object_names"]:
        parts.append(f"objects={rec['object_names']}")
    if rec["params"]:
        parts.append(f"params={rec['params']}")
    return " | ".join(parts)


class CogMan:
    """Cognitive manager."""

    def __init__(self, approach: BaseApproach, perceiver: BasePerceiver,
                 execution_monitor: BaseExecutionMonitor) -> None:
        self._approach = approach
        self._perceiver = perceiver
        self._exec_monitor = execution_monitor
        self._current_policy: Optional[Callable[[State], Action]] = None
        self._current_goal: Optional[Set[GroundAtom]] = None
        self._override_policy: Optional[Callable[[State], Action]] = None
        self._termination_fn: Optional[Callable[[State], bool]] = None
        self._current_env_task: Optional[EnvironmentTask] = None
        self._episode_state_history: List[State] = []
        self._episode_action_history: List[Action] = []
        self._episode_option_switch_history: List[_Option] = []
        self._episode_images: Video = []
        self._episode_num = -1

    def reset(self, env_task: EnvironmentTask) -> None:
        """Start a new episode of environment interaction."""
        logging.info("[CogMan] Reset called.")
        self._episode_num += 1
        task = self._perceiver.reset(env_task)
        self._current_env_task = env_task
        self._current_goal = task.goal
        self._reset_policy(task)
        self._exec_monitor.reset(task)
        self._exec_monitor.update_approach_info(
            self._approach.get_execution_monitoring_info())
        self._episode_state_history = [task.init]
        self._episode_action_history = []
        self._episode_option_switch_history = []
        self._episode_images = []
        if CFG.make_cogman_videos:
            imgs = self._perceiver.render_mental_images(task.init, env_task)
            self._episode_images.extend(imgs)

    def step(self, observation: Observation) -> Optional[Action]:
        """Receive an observation and produce an action, or None for done."""
        state = self._perceiver.step(observation)
        if CFG.make_cogman_videos:
            assert self._current_env_task is not None
            imgs = self._perceiver.render_mental_images(
                state, self._current_env_task)
            self._episode_images.extend(imgs)
        # Replace the first step because the state was already added in reset().
        if not self._episode_action_history:
            self._episode_state_history[0] = state
        else:
            self._episode_state_history.append(state)
        if self._termination_fn is not None and self._termination_fn(state):
            logging.info("[CogMan] Termination triggered.")
            return None
        # Check if we should replan.
        if self._exec_monitor.step(state):
            logging.info("[CogMan] Replanning triggered.")
            assert self._current_goal is not None
            task = Task(state, self._current_goal)
            self._reset_policy(task)
            self._exec_monitor.reset(task)
            self._exec_monitor.update_approach_info(
                self._approach.get_execution_monitoring_info())
            # We only reset the approach if the override policy is
            # None, so this below assertion only works in this
            # case.
            if self._override_policy is None:
                assert not self._exec_monitor.step(state)
        assert self._current_policy is not None
        act = self._current_policy(state)
        self._exec_monitor.update_approach_info(
            self._approach.get_execution_monitoring_info())
        self._episode_action_history.append(act)
        return act

    def finish_episode(self, observation: Observation) -> None:
        """Called at the end of an episode."""
        logging.info("[CogMan] Finishing episode.")
        if len(self._episode_state_history) == len(
                self._episode_action_history):
            state = self._perceiver.step(observation)
            self._episode_state_history.append(state)
        if CFG.make_cogman_videos:
            save_prefix = utils.get_config_path_str()
            outfile = f"{save_prefix}__cogman__episode{self._episode_num}.mp4"
            utils.save_video(outfile, self._episode_images)

    # The methods below provide an interface to the approach. In the future,
    # we may want to move some of these methods into cogman properly, e.g.,
    # if we want the perceiver or execution monitor to learn from data.

    @property
    def is_learning_based(self) -> bool:
        """See BaseApproach docstring."""
        return self._approach.is_learning_based

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """See BaseApproach docstring."""
        return self._approach.learn_from_offline_dataset(dataset)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        """See BaseApproach docstring."""
        return self._approach.load(online_learning_cycle)

    def get_interaction_requests(self) -> List[InteractionRequest]:
        """See BaseApproach docstring."""
        return self._approach.get_interaction_requests()

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        """See BaseApproach docstring."""
        return self._approach.learn_from_interaction_results(results)

    @property
    def metrics(self) -> Metrics:
        """See BaseApproach docstring."""
        return self._approach.metrics

    def reset_metrics(self) -> None:
        """See BaseApproach docstring."""
        return self._approach.reset_metrics()

    def set_override_policy(self, policy: Callable[[State], Action]) -> None:
        """Used during online interaction."""
        self._override_policy = policy

    def unset_override_policy(self) -> None:
        """Give control back to the approach."""
        self._override_policy = None

    def set_termination_function(
            self, termination_fn: Callable[[State], bool]) -> None:
        """Used during online interaction."""
        self._termination_fn = termination_fn

    def unset_termination_function(self) -> None:
        """Reset to never willfully terminating."""
        self._termination_fn = None

    def get_current_history(self) -> LowLevelTrajectory:
        """Expose the most recent state, action history for learning."""
        return LowLevelTrajectory(self._episode_state_history,
                                  self._episode_action_history)

    def get_current_action_history(self) -> List[Action]:
        """Expose current episode action history without trajectory wrapper."""
        return list(self._episode_action_history)

    def get_current_option_switch_history(self) -> List[_Option]:
        """Expose option switch events from current episode."""
        return list(self._episode_option_switch_history)

    def _reset_policy(self, task: Task) -> None:
        """Call the approach or use the override policy."""
        if self._override_policy is not None:
            self._current_policy = self._override_policy
        else:
            self._current_policy = self._approach.solve(task,
                                                        timeout=CFG.timeout)


def run_episode_and_get_observations(
    cogman: CogMan,
    env: BaseEnv,
    train_or_test: str,
    task_idx: int,
    max_num_steps: int,
    do_env_reset: bool = True,
    terminate_on_goal_reached: bool = True,
    exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
    monitor: Optional[utils.LoggingMonitor] = None
) -> Tuple[Tuple[List[Observation], List[Action]], bool, Metrics]:
    """Execute cogman starting from the initial state of a train or test task
    in the environment.

    Note that the environment and cogman internal states are updated.
    Terminates when any of these conditions hold: (1) cogman.step
    returns None, indicating termination (2) max_num_steps is reached
    (3) cogman or env raise an exception of type in
    exceptions_to_break_on (4) terminate_on_goal_reached is True and the
    env goal is reached. Note that in the case where the exception is
    raised in step, we exclude the last action from the returned
    trajectory to maintain the invariant that the trajectory states are
    of length one greater than the actions. Ideally, this method would
    live in utils.py, but that results in import errors with this file.
    So we keep it here for now. It might be moved in the future.
    """
    if do_env_reset:
        env.reset(train_or_test, task_idx)
        if monitor is not None:
            monitor.reset(train_or_test, task_idx)
    obs = env.get_observation()
    observations = [obs]
    actions: List[Action] = []
    curr_option: Optional[_Option] = None
    curr_option_start_time: Optional[float] = None
    curr_option_start_step: int = 0
    metrics: Metrics = defaultdict(float)
    metrics["policy_call_time"] = 0.0
    metrics["num_options_executed"] = 0.0
    option_segment_seq = 0

    def _finalize_curr_option(end_time: float, end_step: int) -> None:
        nonlocal curr_option, curr_option_start_time, curr_option_start_step, \
            option_segment_seq
        if curr_option is None or curr_option_start_time is None:
            return
        option_name = curr_option.name
        duration_sec = max(0.0, end_time - curr_option_start_time)
        duration_steps = max(0, end_step - curr_option_start_step)
        metrics[f"option_{option_name}_total_time_sec"] += duration_sec
        metrics[f"option_{option_name}_total_steps"] += float(duration_steps)
        metrics[f"option_{option_name}_count"] += 1.0
        option_segment_seq += 1
        record = _option_to_execution_record(curr_option, option_segment_seq,
                                             cogman._episode_num)
        record["train_or_test"] = train_or_test
        record["env_task_idx"] = task_idx
        batch_tid = getattr(CFG, "batch_task_id", None)
        batch_eid = getattr(CFG, "batch_experiment_id", None)
        if batch_tid is not None:
            record["batch_task_id"] = batch_tid
        if batch_eid is not None:
            record["batch_experiment_id"] = batch_eid
        _append_cogman_option_log(record, duration_sec, duration_steps)
        logging.info(
            _format_option_log_message(curr_option, duration_sec,
                                       duration_steps))
    exception_raised_in_step = False
    if not (terminate_on_goal_reached and env.goal_reached()):
        for _ in range(max_num_steps):
            monitor_observed = False
            exception_raised_in_step = False
            try:
                start_time = time.perf_counter()
                act = cogman.step(obs)
                metrics["policy_call_time"] += time.perf_counter() - start_time
                if act is None:
                    _finalize_curr_option(time.perf_counter(), len(actions))
                    break
                if act.has_option() and act.get_option() != curr_option:
                    _finalize_curr_option(time.perf_counter(), len(actions))
                    curr_option = act.get_option()
                    curr_option_start_time = time.perf_counter()
                    curr_option_start_step = len(actions)
                    metrics["num_options_executed"] += 1
                    cogman._episode_option_switch_history.append(curr_option)
                    exec_parts = [f"Executing option: {curr_option.name}"]
                    if curr_option.objects:
                        exec_parts.append(
                            "objects="
                            + str([obj.name for obj in curr_option.objects]))
                    opt_params = _option_params_to_json_list(curr_option.params)
                    if opt_params:
                        exec_parts.append(f"params={opt_params}")
                    logging.info(" | ".join(exec_parts))
                # Note: it's important to call monitor.observe() before
                # env.step(), because the monitor may, for example, call
                # env.render(), which outputs images of the current env
                # state. If we instead called env.step() first, we would
                # mistakenly record images of the next time step instead of
                # the current one.
                if monitor is not None:
                    monitor.observe(obs, act)
                    monitor_observed = True
                # print(f"Executing action: {act.arr.tolist()}")
                obs = env.step(act)
                actions.append(act)
                observations.append(obs)
            except Exception as e:
                if exceptions_to_break_on is not None and \
                   any(issubclass(type(e), c) for c in exceptions_to_break_on):
                    if monitor_observed:
                        exception_raised_in_step = True
                    break
                if monitor is not None and not monitor_observed:
                    monitor.observe(obs, None)
                raise e
            if terminate_on_goal_reached and env.goal_reached():
                break
    _finalize_curr_option(time.perf_counter(), len(actions))
    if monitor is not None and not exception_raised_in_step:
        monitor.observe(obs, None)
    cogman.finish_episode(obs)
    traj = (observations, actions)
    solved = env.goal_reached()
    return traj, solved, metrics


def run_episode_and_get_states(
    cogman: CogMan,
    env: BaseEnv,
    train_or_test: str,
    task_idx: int,
    max_num_steps: int,
    do_env_reset: bool = True,
    terminate_on_goal_reached: bool = True,
    exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
    monitor: Optional[utils.LoggingMonitor] = None
) -> Tuple[LowLevelTrajectory, bool, Metrics]:
    """Execute cogman starting from the initial state of a train or test task
    in the environment.

    Return a trajectory involving States (which come from running a
    perceiver on observations). Having states instead of observations is
    useful for downstream learning (e.g. predicates, operators,
    samplers, etc.) Note that the only difference between this and the
    above run_episode_and_get_observations is that this method returns a
    trajectory of states instead of one of observations.
    """
    _, solved, metrics = run_episode_and_get_observations(
        cogman, env, train_or_test, task_idx, max_num_steps, do_env_reset,
        terminate_on_goal_reached, exceptions_to_break_on, monitor)
    ll_traj = cogman.get_current_history()
    return ll_traj, solved, metrics
