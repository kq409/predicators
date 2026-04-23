"""A bilevel planning approach that uses a refinement cost estimator.

Generates N proposed skeletons and then ranks them based on a given
refinement cost estimation function (e.g. a heuristic, learned model),
attempting to refine them in this order.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _load_option_execution_costs(summary_path: Path) -> Dict[Tuple[str, str], float]:
    """Load option mean env-step costs from option execution summary JSON."""
    if not summary_path.exists():
        logging.warning("Physical cost summary not found: %s", summary_path)
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as err:
        logging.warning("Failed reading physical cost summary %s: %s", summary_path, err)
        return {}
    if not isinstance(payload, list):
        logging.warning(
            "Physical cost summary format invalid (expected list): %s",
            summary_path,
        )
        return {}
    costs: Dict[Tuple[str, str], float] = {}
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        option_name = str(rec.get("option_name", "")).strip().lower()
        if not option_name:
            continue
        key = str(rec.get("object_names_key", "")).strip().lower()
        if not key:
            obj_names = rec.get("object_names", [])
            if isinstance(obj_names, list):
                key = "|".join(str(x).strip().lower() for x in obj_names)
        try:
            mean_steps = float(rec.get("num_env_steps_mean", 0.0))
        except (TypeError, ValueError):
            continue
        if mean_steps <= 0.0:
            continue
        costs[(option_name, key)] = mean_steps
    return costs


def _lookup_physical_cost_from_option_summary(
    ground_nsrt: _GroundNSRT,
    option_costs: Dict[Tuple[str, str], float],
) -> Optional[Tuple[float, str]]:
    """Lookup physical cost using fully-matched option name + objects.

    Returns (cost, matched_object_names_key) if matched, else None.
    """
    option_name = str(ground_nsrt.option.name).strip().lower()
    option_obj_names = [getattr(o, "name", str(o)).lower() for o in ground_nsrt.option_objs]
    exact_key = "|".join(option_obj_names)
    sorted_key = "|".join(sorted(option_obj_names))
    for key in (exact_key, sorted_key):
        val = option_costs.get((option_name, key))
        if val is not None:
            return float(val), key
    return None


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
    EPSILON = 1e-2
    BASE_COST_OBSERVE = 40
    container_to_region = {
        "microhandle": "microwave",
        "hinge2": "right_hinge_cabinet",
        "slide": "slide_cabinet",
    }
    nsrt_name = ground_nsrt.name
    # if nsrt_name in ["MoveToPreTurnOn", "PushOpenHingeDoor", "PushOpen"]:
    #     # return default_cost
    #     return 0
    # if nsrt_name.startswith("ObserveContainer"):
    #     found_obj_names = sorted({
    #         a.objects[0].name for a in ground_nsrt.add_effects
    #         if getattr(a.predicate, "name", None) == "ObjectFound" and a.objects
    #     })
    #     num_effects = len(found_obj_names)
    #     return 0.1 + 0.1 * num_effects
    if nsrt_name.startswith("ObserveContainer"):
        cost = BASE_COST_OBSERVE
        # if len(ground_nsrt.objects) < 2:
        #     return BASE_COST_OBSERVE
        container_name = getattr(ground_nsrt.objects[1], "name", str(ground_nsrt.objects[1]))
        region_name = container_to_region.get(container_name, "right_hinge_cabinet")
        found_obj_names = sorted({
            a.objects[0].name for a in ground_nsrt.add_effects
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
        for p in elem_probs:
            cost = cost * (1 / max(p, EPSILON))
        print(f"NSRT name: {nsrt_name}, objects: {found_obj_names}, cost: {cost}")
        if cost > 100000:
            cost = 100000
            print(f"Clipped cost: {cost}")
        return cost
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
        fd_planners = ("fdopt", "fdsat", "fdopt-costs", "fdsat-costs", "kstar", "kstar-costs")
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
                use_real_physical_costs = bool(
                    getattr(CFG, "use_real_physical_option_costs", False)
                )
                summary_path_raw = str(
                    getattr(
                        CFG,
                        "real_physical_cost_summary_json",
                        "experiment_results/option_execution_summary.json",
                    )
                ).strip()
                option_costs: Dict[Tuple[str, str], float] = {}
                if use_real_physical_costs and summary_path_raw:
                    option_costs = _load_option_execution_costs(Path(summary_path_raw))
                    logging.info(
                        "[FD costs] using physical option costs from %s (%d entries)",
                        summary_path_raw,
                        len(option_costs),
                    )

                def _fallback_cost(gn: _GroundNSRT) -> float:
                    if region_probs_by_obj or region_probs_default:
                        return _compute_ground_op_cost_from_region_probs(
                            gn, region_probs_by_obj, region_probs_default,
                            default_cost)
                    return _lookup_ground_op_cost(gn, cost_by_name, default_cost)

                ground_op_costs = {}
                for gn in ground_nsrts:
                    # Keep ObserveContainer-related actions on the original cost path.
                    if gn.name.startswith("ObserveContainer"):
                        ground_op_costs[gn.op] = _fallback_cost(gn)
                        continue
                    if option_costs:
                        matched = _lookup_physical_cost_from_option_summary(
                            gn, option_costs
                        )
                        if matched is not None:
                            physical_cost, matched_key = matched
                            old_cost = _fallback_cost(gn)
                            ground_op_costs[gn.op] = physical_cost
                            logging.info(
                                "[FD costs replaced] nsrt=%s option=%s option_objs=%s "
                                "matched_key=%s old_cost=%.6f new_cost=%.6f",
                                gn.name,
                                gn.option.name,
                                [getattr(o, "name", str(o)) for o in gn.option_objs],
                                matched_key,
                                old_cost,
                                physical_cost,
                            )
                            continue
                        # Penalize unmatched MoveToTargetObject to avoid shortcut plans.
                        if gn.name == "MoveToTargetObject" or gn.name == "MoveToolTo":
                            ground_op_costs[gn.op] = 1000.0
                            # logging.info(
                            #     "[FD costs unmatched] nsrt=%s option=%s option_objs=%s "
                            #     "assigned_cost=%.1f",
                            #     gn.name,
                            #     gn.option.name,
                            #     [getattr(o, "name", str(o)) for o in gn.option_objs],
                            #     ground_op_costs[gn.op],
                            # )
                            continue
                    ground_op_costs[gn.op] = _fallback_cost(gn)
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
        """Write skeleton information to a log file under current plan dir."""
        del estimator  # only used for interface consistency
        task_id = int(getattr(CFG, "batch_task_id", 0))
        exp_id = int(getattr(CFG, "batch_experiment_id", 0))
        plan_dir = self._infer_latest_plan_dir()
        plan_id = self._infer_plan_id_from_dir(plan_dir)
        prefix = f"task_{task_id}_experiment_{exp_id:03d}_plan_{plan_id:03d}"
        log_file = plan_dir / f"{prefix}.json"
        
        # Prepare skeleton data for logging
        skeleton_log_data = {
            "timestamp": prefix,
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
        text_log_file = plan_dir / f"log_{prefix}.log"
        with open(text_log_file, "w", encoding="utf-8") as f:
            f.write(f"Skeleton Log - {prefix}\n")
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

    def _infer_latest_plan_dir(self) -> Path:
        """Get latest plan directory from current experiment directory."""
        exp_dir = Path(CFG.log_dir)
        candidates = [
            p for p in exp_dir.iterdir()
            if p.is_dir() and p.name.startswith("task") and "_plan_" in p.name
        ]
        if not candidates:
            return exp_dir
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]

    @staticmethod
    def _infer_plan_id_from_dir(plan_dir: Path) -> int:
        """Extract plan id from a directory named like task1_plan_003."""
        if "_plan_" not in plan_dir.name:
            return 0
        suffix = plan_dir.name.split("_plan_", maxsplit=1)[1]
        return int(suffix) if suffix.isdigit() else 0

    @property
    def refinement_estimator(self) -> BaseRefinementEstimator:
        """Getter for _refinement_estimator."""
        return self._refinement_estimator
