"""A bilevel planning approach that uses a refinement cost estimator.

Generates N proposed skeletons and then ranks them based on a given
refinement cost estimation function (e.g. a heuristic, learned model),
attempting to refine them in this order.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.refinement_estimators import BaseRefinementEstimator, \
    create_refinement_estimator
from predicators.settings import CFG
from predicators.structs import NSRT, Metrics, ParameterizedOption, \
    Predicate, Task, Type, _GroundNSRT, _Option, GroundAtom

from math import log, log10


def _lookup_ground_op_cost(
        ground_nsrt: _GroundNSRT,
        cost_by_name: dict,
        default_cost: float) -> float:
    """Look up cost for a ground NSRT. Supports:
    - Full key (name + objects in param order): "observecontainersponge robot0 hinge2 mug sponge tea"
    - Name only: "ObserveContainerSponge" (applies to all groundings)
    Lookup order: full key -> name -> name.lower() -> default_cost."""
    obj_names = [getattr(o, "name", str(o)).lower() for o in ground_nsrt.objects]
    full_key = f"{ground_nsrt.name.lower()} {' '.join(obj_names)}".strip()
    name_key = ground_nsrt.name
    name_key_lower = ground_nsrt.name.lower()
    # Try full key first, then name (case-insensitive)
    for key in (full_key, name_key, name_key_lower):
        if key in cost_by_name:
            return cost_by_name[key]
    return default_cost


def _compute_ground_op_cost_from_region_probs(
        ground_nsrt: _GroundNSRT,
        region_probs_by_obj: dict,
        region_probs_default: dict,
        default_cost: float) -> float:
    """Compute FD cost for a ground NSRT using oracle-style region probs.
    Only ObserveContainer* costs are adjusted; others use default_cost.
    ObserveContainer*: cost = 1.0 / success_prob (base=1, determinization)."""
    DEFAULT_SUCCESS_PROBABILITY = 1.0 / 3.0
    EPSILON = 1e-6
    BASE_COST_OBSERVE = 1.0
    container_to_region = {
        "microhandle": "microwave",
        "hinge2": "right_hinge_cabinet",
        "slide": "slide_cabinet",
    }
    nsrt_name = ground_nsrt.name
    # if nsrt_name in ["MoveToPreTurnOn", "PushOpenHingeDoor", "PushOpen"]:
    #     # return default_cost
    #     return 0
    if nsrt_name.startswith("ObserveContainer"):
        found_obj_names = sorted({
            a.objects[0].name for a in ground_nsrt.add_effects
            if getattr(a.predicate, "name", None) == "ObjectFound" and a.objects
        })
        num_effects = len(found_obj_names)
        return 0.1 + 0.1 * num_effects
    if nsrt_name.startswith("MoveToPrePickUp"):
        # if len(ground_nsrt.objects) < 2:
        #     return BASE_COST_OBSERVE
        container_name = getattr(ground_nsrt.objects[2], "name", str(ground_nsrt.objects[2]))
        region_name = container_to_region.get(container_name, "right_hinge_cabinet")
        found_obj_names = sorted({
            a.objects[0].name for a in ground_nsrt.preconditions
            if getattr(a.predicate, "name", None) == "ObjectFound" and a.objects
        })
        elem_probs = []
        for obj_name in found_obj_names:
            if obj_name in region_probs_by_obj:
                elem_probs.append(
                    region_probs_by_obj[obj_name].get(region_name, DEFAULT_SUCCESS_PROBABILITY))
            else:
                elem_probs.append(
                    region_probs_default.get(region_name, DEFAULT_SUCCESS_PROBABILITY))
        lambda_ = 1.0
        for p in elem_probs:
            return default_cost + lambda_ * (-log10(max(p, EPSILON)))
    return default_cost


class RefinementEstimationApproach(OracleApproach):
    """A bilevel planning approach that uses a refinement cost estimator."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        assert (CFG.refinement_estimation_num_skeletons_generated <=
                CFG.sesame_max_skeletons_optimized), \
               "refinement_estimation_num_skeletons_generated should not be" \
               "greater than sesame_max_skeletons_optimized"
        estimator_name = CFG.refinement_estimator
        self._refinement_estimator = create_refinement_estimator(
            estimator_name)
        # If the refinement estimator is learning based, try to load
        # trained model if the file exists.
        if self._refinement_estimator.is_learning_based:
            config_path_str = utils.get_config_path_str()
            model_file = f"{estimator_name}_{config_path_str}.estimator"
            model_file_path = Path(CFG.approach_dir) / model_file
            try:
                self._refinement_estimator.load_model(model_file_path)
                logging.info(f"Loaded trained estimator model "
                             f"from {model_file_path}")
            except FileNotFoundError:
                logging.info(f"Could not find estimator model file "
                             f"at {model_file_path}")
        
        # Initialize skeleton log file path
        self._skeleton_log_counter = 0
        self._skeleton_log_dir = Path(CFG.log_dir) / "skeleton_logs"
        os.makedirs(self._skeleton_log_dir, exist_ok=True)

    @classmethod
    def get_name(cls) -> str:
        return "refinement_estimation"

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int,
            **kwargs: Any) -> Tuple[List[_Option], List[_GroundNSRT], Metrics]:
        """Generates a plan choosing the best skeletons based on a given
        refinement cost estimator."""
        result = super()._run_sesame_plan(
            task,
            nsrts,
            preds,
            timeout,
            seed,
            refinement_estimator=self._refinement_estimator,
            **kwargs)
        return result

    def _run_task_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int, **kwargs: Any
    ) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]], Metrics]:
        """Generates a plan choosing the best skeletons based on a given
        refinement cost estimator when using task planning only (without sim)."""
        from predicators.planning import (
            PlanningFailure,
            PlanningTimeout,
            run_task_plan_once,
            task_plan_grounding,
            task_plan,
            _SkeletonSearchTimeout,
        )
        from predicators import utils as pred_utils
        from predicators.settings import CFG
        from itertools import islice

        # FD mode: use run_task_plan_once for a single skeleton
        # Supports fdopt, fdsat, fdopt-costs, fdsat-costs
        fd_planners = ("fdopt", "fdsat", "fdopt-costs", "fdsat-costs")
        if CFG.sesame_task_planner in fd_planners:
            ground_op_costs = None
            default_cost = 1.0
            cost_precision = 3
            if CFG.sesame_task_planner.endswith("-costs"):
                init_atoms = pred_utils.abstract(task.init, preds)
                objects = set(task.init)
                ground_nsrts, _ = task_plan_grounding(
                    init_atoms, objects, nsrts, allow_noops=False)
                region_probs_by_obj = getattr(
                    CFG, "fd_region_probs_by_obj", None) or {}
                region_probs_default = getattr(
                    CFG, "fd_region_probs_default", None) or {}
                cost_by_name = getattr(
                    CFG, "fd_ground_op_cost_by_name", None) or {}
                if region_probs_by_obj or region_probs_default:
                    ground_op_costs = {
                        gn.op: _compute_ground_op_cost_from_region_probs(
                            gn, region_probs_by_obj, region_probs_default,
                            default_cost)
                        for gn in ground_nsrts
                    }
                else:
                    ground_op_costs = {
                        gn.op: _lookup_ground_op_cost(
                            gn, cost_by_name, default_cost)
                        for gn in ground_nsrts
                    }
            try:
                plan, atoms_seq, metrics = run_task_plan_once(
                    task,
                    nsrts,
                    preds,
                    self._types,
                    timeout,
                    seed,
                    task_planning_heuristic=self._task_planning_heuristic,
                    max_horizon=float(CFG.horizon),
                    ground_op_costs=ground_op_costs,
                    default_cost=default_cost,
                    cost_precision=cost_precision,
                    **kwargs)
            except PlanningFailure as e:
                raise ApproachFailure(e.args[0], e.info)
            except PlanningTimeout as e:
                raise ApproachTimeout(e.args[0], e.info)
            skeleton_data = (plan, atoms_seq, metrics)
            cost = self._refinement_estimator.get_cost(task, plan, atoms_seq)
            proposed_skeletons = [skeleton_data]
            sorted_skeletons = [(skeleton_data, cost)]
            self._log_skeletons(
                task, proposed_skeletons, sorted_skeletons,
                self._refinement_estimator)
            logging.info("FD skeleton (single): %d steps, refinement cost=%.4f",
                         len(plan), cost)
            return plan, atoms_seq, metrics

        # A* mode: generate multiple skeletons and rank by refinement_estimator
        init_atoms = pred_utils.abstract(task.init, preds)
        goal = task.goal
        objects = set(task.init)
        
        ground_nsrts, reachable_atoms = task_plan_grounding(
            init_atoms, objects, nsrts)
        heuristic = pred_utils.create_task_planning_heuristic(
            self._task_planning_heuristic, init_atoms, goal, ground_nsrts, preds,
            objects)
        
        # Generate multiple skeletons
        gen = task_plan(
            init_atoms,
            goal,
            ground_nsrts,
            reachable_atoms,
            heuristic,
            seed,
            timeout,
            max_skeletons_optimized=CFG.refinement_estimation_num_skeletons_generated,
            use_visited_state_set=True)
        
        # Collect proposed skeletons. If skeleton search times out AFTER at
        # least one skeleton has been yielded, treat it as "no more skeletons"
        # instead of a hard failure.
        proposed_skeletons = []
        try:
            for skeleton_data in islice(
                gen, CFG.refinement_estimation_num_skeletons_generated
            ):
                proposed_skeletons.append(skeleton_data)
            print("Skeletons generated successfully")
        except _SkeletonSearchTimeout:
            if not proposed_skeletons:
                # No skeletons at all: propagate failure to keep old behavior.
                raise
            # Otherwise, we already have some candidate skeletons; just stop
            # collecting more and proceed with cost-based ranking.
            print("Skeletons generation timed out")
        print(f"Skeletons generated: {len(proposed_skeletons)}")
        if not proposed_skeletons:
            # If no skeletons generated, fall back to default behavior
            return super()._run_task_plan(task, nsrts, preds, timeout, seed, **kwargs)
        
        # Rank skeletons by refinement cost
        estimator: BaseRefinementEstimator = self._refinement_estimator
        skeleton_costs = []
        for skeleton_data in proposed_skeletons:
            cost = estimator.get_cost(task, skeleton_data[0], skeleton_data[1])
            skeleton_costs.append((skeleton_data, cost))
        
        # Sort by cost
        sorted_skeletons = sorted(skeleton_costs, key=lambda x: x[1])
        
        # Write skeleton information to separate log file
        self._log_skeletons(task, proposed_skeletons, sorted_skeletons, estimator)
        
        # Return the best skeleton
        plan, atoms_seq, metrics = sorted_skeletons[0][0]
        return plan, atoms_seq, metrics
    
    def _log_skeletons(self, task: Task, proposed_skeletons: List[Tuple],
                       sorted_skeletons: List[Tuple],
                       estimator: BaseRefinementEstimator) -> None:
        """Write skeleton information to a separate log file."""
        self._skeleton_log_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._skeleton_log_dir / f"skeletons_{timestamp}_{self._skeleton_log_counter}.json"
        
        # Prepare skeleton data for logging
        skeleton_log_data = {
            "timestamp": timestamp,
            "task_goal": [str(atom) for atom in task.goal],
            "num_skeletons_generated": len(proposed_skeletons),
            "skeletons": []
        }
        
        # Log all skeletons with their costs
        for idx, (skeleton_data, cost) in enumerate(sorted_skeletons):
            plan, atoms_seq, _ = skeleton_data
            skeleton_info = {
                "rank": idx + 1,
                "cost": float(cost),
                "plan_length": len(plan),
                "steps": []
            }
            
            # Add detailed step information
            for step_idx, nsrt in enumerate(plan):
                step_info = {
                    "step": step_idx + 1,
                    "nsrt_name": nsrt.name,
                    "objects": [obj.name for obj in nsrt.objects],
                    "object_types": [str(obj.type) for obj in nsrt.objects]
                }
                skeleton_info["steps"].append(step_info)
            
            # Add atoms sequence information
            skeleton_info["atoms_sequence"] = [
                [str(atom) for atom in atoms] for atoms in atoms_seq
            ]
            
            skeleton_log_data["skeletons"].append(skeleton_info)
        
        # Write to JSON file
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(skeleton_log_data, f, indent=2, ensure_ascii=False)
        
        # Also write a human-readable text log
        text_log_file = self._skeleton_log_dir / f"skeletons_{timestamp}_{self._skeleton_log_counter}.txt"
        with open(text_log_file, "w", encoding="utf-8") as f:
            f.write(f"Skeleton Log - {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Task Goal: {[str(atom) for atom in task.goal]}\n")
            f.write(f"Number of Skeletons Generated: {len(proposed_skeletons)}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Skeletons Sorted by Cost (Best to Worst):\n")
            f.write("-" * 80 + "\n")
            for rank, (skeleton_data, cost) in enumerate(sorted_skeletons, 1):
                plan, atoms_seq, _ = skeleton_data
                f.write(f"\nRank {rank}: Cost = {cost:.4f}\n")
                f.write(f"Plan Length: {len(plan)} steps\n")
                f.write("Plan Steps:\n")
                for step_idx, nsrt in enumerate(plan, 1):
                    objects_str = ", ".join([f"{obj.name}({obj.type})" for obj in nsrt.objects])
                    f.write(f"  Step {step_idx}: {nsrt.name} with objects [{objects_str}]\n")
                f.write("\n")

    @property
    def refinement_estimator(self) -> BaseRefinementEstimator:
        """Getter for _refinement_estimator."""
        return self._refinement_estimator
