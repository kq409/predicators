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
from predicators.approaches.oracle_approach import OracleApproach
from predicators.refinement_estimators import BaseRefinementEstimator, \
    create_refinement_estimator
from predicators.settings import CFG
from predicators.structs import NSRT, Metrics, ParameterizedOption, \
    Predicate, Task, Type, _GroundNSRT, _Option, GroundAtom


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
        from predicators.planning import task_plan_grounding, task_plan, \
            _SkeletonSearchTimeout
        from predicators import utils as pred_utils
        from predicators.settings import CFG
        from itertools import islice
        
        # Generate multiple skeletons and rank them using refinement_estimator
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
