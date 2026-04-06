"""A hand-written refinement cost estimator."""

from typing import List, Set, Dict
from pathlib import Path

from predicators.envs import BaseEnv
from predicators.envs.kitchen import KitchenEnv
from predicators.envs.exit_garage import ExitGarageEnv
from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task, _GroundNSRT

# Repository root (diffusion_behavior) and experiment results directory
_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_RESULTS_DIR = _REPO_ROOT / "experiment_results"

class OracleRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that returns a hand-designed cost
    estimation."""

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def get_cost(self, initial_task: Task, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        env_name = CFG.env
        if env_name == "narrow_passage":
            return narrow_passage_oracle_estimator(self._env,
                                                   initial_task.init, skeleton,
                                                   atoms_sequence)
        if env_name == "exit_garage":
            return exit_garage_oracle_estimator(self._env, initial_task.init,
                                                skeleton, atoms_sequence)

        if env_name == "kitchen":
            # Per-object region probabilities, computed from per-object results_*.txt
            region_probs_by_obj = calculate_kitchen_region_probs_by_object(
                self._env, initial_task.init)
            # Fallback combined probabilities (from merged results.txt) for
            # objects that do not have individual distributions.
            region_probs_default = calculate_kitchen_region_probs(
                self._env, initial_task.init)
            return kitchen_oracle_estimator_per_object(
                self._env, initial_task.init,
                skeleton, atoms_sequence,
                region_probs_by_obj, region_probs_default)

        # Given environment doesn't have an implemented oracle estimator
        raise NotImplementedError(
            f"No oracle refinement cost estimator for env {env_name}")


def narrow_passage_oracle_estimator(
    env: BaseEnv,
    initial_state: State,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
) -> float:
    """Oracle refinement estimation function for narrow_passage env."""
    del atoms_sequence  # unused

    # Extract door and passage widths from the state
    door_type, _, _, _, wall_type = sorted(env.types)
    door, = initial_state.get_objects(door_type)
    door_width = initial_state.get(door, "width")
    _, middle_wall, right_wall = sorted(initial_state.get_objects(wall_type))
    passage_x = initial_state.get(middle_wall, "x") + \
                initial_state.get(middle_wall, "width")
    passage_width = initial_state.get(right_wall, "x") - passage_x

    # If the door is wider than the passage, then opening the door is
    # beneficial. Otherwise, opening the door should be costly.
    cost_of_open_door = -1 if door_width > passage_width else 1

    # Sum metric of difficulty over skeleton
    cost = 0
    for ground_nsrt in skeleton:
        if ground_nsrt.name == "MoveAndOpenDoor":
            cost += cost_of_open_door
        elif ground_nsrt.name == "MoveToTarget":
            cost += 1
    return cost


def exit_garage_oracle_estimator(
    env: BaseEnv,
    initial_state: State,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
) -> float:
    """Oracle refinement estimation function for exit_garage env."""
    del atoms_sequence  # unused

    assert isinstance(env, ExitGarageEnv)
    obstacle_radius = env.obstacle_radius
    obstruction_ub = env.exit_top + 2 * obstacle_radius
    obstruction_lb = env.exit_top - env.exit_height - 2 * obstacle_radius

    # Each picked-up obstacle decreases the refinement cost of DriveCarToExit
    # if it is in the direct path of the car to the exit, otherwise it has a
    # positive cost and should be avoided
    cost: float = 0
    for ground_nsrt in skeleton:
        if ground_nsrt.name == "ClearObstacle":
            obstacle = ground_nsrt.objects[1]
            obstacle_y = initial_state.get(obstacle, "y")
            if obstruction_lb < obstacle_y < obstruction_ub:
                cost -= 1
            else:
                cost += 0.5
    return cost


def define_kitchen_regions() -> Dict[str, Dict]:
    """
    Define regions based on analyze_sample_regions.py conditions for visible locations.
    Each region is defined by (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    regions = {
        'microwave': {
            'x_range': (-0.4, -0.1),
            'y_range': (0.8, 0.95),
            'z_range': (1.65, 1.65),  # Fixed z coordinate
            'xy_tolerance': 0.05,  # Tolerance for x and y coordinates
            'z_tolerance': 0.15  # Tolerance for z coordinate
        },
        'right_hinge_cabinet': {
            'x_range': (-0.5, -0.3),
            'y_range': (0.7, 0.8),
            'z_range': (2.45, 2.45),
            'xy_tolerance': 0.05,
            'z_tolerance': 0.15
        },
        'slide_cabinet': {
            'x_range': (-0.05, 0.23),
            'y_range': (0.7, 0.8),
            'z_range': (2.45, 2.45),
            'xy_tolerance': 0.05,
            'z_tolerance': 0.15
        },
    }
    return regions


def point_in_region(point: tuple, region: Dict) -> bool:
    """
    Check if a point (x, y, z) is within a region.
    Uses different tolerances for x/y and z.
    """
    x, y, z = point
    x_min, x_max = region['x_range']
    y_min, y_max = region['y_range']
    z_min, z_max = region['z_range']
    xy_tolerance = region['xy_tolerance']
    z_tolerance = region['z_tolerance']
    
    # Check x range with tolerance
    in_x = (x_min - xy_tolerance) <= x <= (x_max + xy_tolerance)
    
    # Check y range with tolerance
    in_y = (y_min - xy_tolerance) <= y <= (y_max + xy_tolerance)
    
    # Check z range with tolerance (since z is usually fixed)
    if z_min == z_max:
        in_z = abs(z - z_min) <= z_tolerance
    else:
        in_z = (z_min - z_tolerance) <= z <= (z_max + z_tolerance)
    
    return in_x and in_y and in_z


def load_object_coordinates(file_path: str = None) -> List[tuple]:
    """
    Load object coordinates from results.txt file.
    
    Args:
        file_path: Path to results.txt file. If None, uses default path.
        
    Returns:
        List of (x, y, z) tuples representing object positions
    """
    if file_path is None:
        # 默认：从 experiment_results/ 下最新的 results.txt 读取
        if not EXPERIMENT_RESULTS_DIR.exists():
            return []

        candidate_files = list(EXPERIMENT_RESULTS_DIR.rglob("results.txt"))
        if not candidate_files:
            return []

        # 选修改时间最新的一个结果文件
        file_path = max(candidate_files, key=lambda p: p.stat().st_mtime)
    
    coordinates = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        coordinates.append((x, y, z))
                except ValueError:
                    # Skip invalid lines
                    continue
    except FileNotFoundError:
        # If file not found, return empty list
        return []
    except Exception:
        # Handle any other errors gracefully
        return []
    
    return coordinates


def load_object_coordinates_for_object(obj_name: str) -> List[tuple]:
    """Load object coordinates from a per-object results file.

    Expected filenames (searched under experiment_results/), e.g.:
        experiment_results/.../results_mug.txt
        experiment_results/.../results_sponge.txt
    """
    suffix = f"results_{obj_name}.txt"
    if not EXPERIMENT_RESULTS_DIR.exists():
        return []

    candidate_files = list(EXPERIMENT_RESULTS_DIR.rglob(suffix))
    if not candidate_files:
        return []

    file_path = max(candidate_files, key=lambda p: p.stat().st_mtime)
    return load_object_coordinates(str(file_path))


def calculate_kitchen_region_probs(env: BaseEnv, initial_state: State) -> Dict[str, float]:
    """
    Calculate probabilities for four regions based on object positions from results.txt.
    Used for clean_mug and make_tea tasks (objects: mug, tea, sponge).
    Returns a dictionary mapping region names to probabilities.
    
    The probabilities are calculated by analyzing all object positions in results.txt
    (from models/tea_single_objects) and computing the percentage of points in each region.
    
    Args:
        env: Kitchen environment instance (unused, kept for compatibility)
        initial_state: Initial state (unused, kept for compatibility)
        
    Returns:
        Dictionary with keys: 'microwave', 'right_hinge_cabinet', 'slide_cabinet'
        Values are probabilities (0.0 to 1.0) representing the percentage of object
        positions in each region
    """
    # Load object coordinates from results.txt (from diffusion models in tea_single_objects)
    coordinates = load_object_coordinates()
    
    if not coordinates:
        # If no coordinates found, return uniform probabilities
        return {
            'microwave': 1/3,
            'right_hinge_cabinet': 1/3,
            'slide_cabinet': 1/3,
        }
    
    # Define regions
    regions = define_kitchen_regions()
    
    # Initialize region counts
    region_counts = {
        'microwave': 0,
        'right_hinge_cabinet': 0,
        'slide_cabinet': 0,
    }
    
    total_points = len(coordinates)
    unassigned_count = 0
    
    # Count points in each region
    for point in coordinates:
        assigned = False
        for region_name, region in regions.items():
            if point_in_region(point, region):
                region_counts[region_name] += 1
                assigned = True
                break
        
        if not assigned:
            unassigned_count += 1
    
    # Calculate probabilities (percentage of points in each region)
    region_probs = {}
    for region_name in region_counts.keys():
        count = region_counts[region_name]
        # Probability is the percentage of points in this region
        region_probs[region_name] = count / total_points if total_points > 0 else 0.0
    
    # Normalize probabilities to sum to 1.0 (distribute unassigned points proportionally)
    # or keep as is if we want to reflect actual distribution
    total_prob = sum(region_probs.values())
    if total_prob > 0:
        # Normalize to ensure probabilities sum to 1.0
        for region_name in region_probs.keys():
            region_probs[region_name] = region_probs[region_name] / total_prob
    
    return region_probs


def calculate_kitchen_region_probs_for_object(obj_name: str) -> Dict[str, float]:
    """Calculate region probabilities for a single object type.

    Uses per-object coordinates loaded from results_<obj_name>.txt.
    """
    coordinates = load_object_coordinates_for_object(obj_name)
    if not coordinates:
        # If no coordinates for this object, return empty dict to signal missing.
        return {}

    regions = define_kitchen_regions()
    region_counts = {
        "microwave": 0,
        "right_hinge_cabinet": 0,
        "slide_cabinet": 0,
    }

    total_points = len(coordinates)
    for point in coordinates:
        for region_name, region in regions.items():
            if point_in_region(point, region):
                region_counts[region_name] += 1
                break

    region_probs: Dict[str, float] = {}
    for region_name, count in region_counts.items():
        region_probs[region_name] = count / total_points if total_points > 0 else 0.0

    total_prob = sum(region_probs.values())
    if total_prob > 0:
        for k in region_probs:
            region_probs[k] /= total_prob
    return region_probs


def calculate_kitchen_region_probs_by_object(
    env: BaseEnv, initial_state: State
) -> Dict[str, Dict[str, float]]:
    """Compute per-object region probabilities from per-object results files.

    Returns:
        {
          "mug": {"microwave": p1, "right_hinge_cabinet": p2, "slide_cabinet": p3},
          "sponge": {...},
          ...
        }
    """
    del env, initial_state  # unused (kept for interface compatibility)

    # Objects we might care about in kitchen tasks.
    candidate_objects = ["banana", "mug", "sponge", "tea", "milk"]
    region_probs_by_obj: Dict[str, Dict[str, float]] = {}
    for obj_name in candidate_objects:
        probs = calculate_kitchen_region_probs_for_object(obj_name)
        if probs:
            region_probs_by_obj[obj_name] = probs

    return region_probs_by_obj


def kitchen_oracle_estimator_per_object(
    env: BaseEnv,
    initial_state: State,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
    region_probs_by_obj: Dict[str, Dict[str, float]],
    region_probs_default: Dict[str, float],
) -> float:
    """Oracle refinement estimation for kitchen with per-object region probs.

    For each observe sequence
        MoveToPreTurnOn(container) -> PushOpen/PushOpenHingeDoor(container) -> ObserveContainer(...)
    we try to infer which object (mug/sponge/tea/milk/banana) is being searched
    next in the NSRT plan, and use that object's region probabilities when
    applying the self-loop determinization formula:

        ĉ = C / p(obj, container)

    If we cannot infer the object or there is no per-object distribution for it,
    we fall back to region_probs_default (computed from merged results.txt).

    Additionally, if an object has already been found (according to the current
    high-level state), we treat its success probability as 1.0 instead of using
    the diffusion-based prior. This avoids incorrectly assigning infinite cost
    when a skeleton revisits containers after an object has already been found.

    """
    # Container names that can be observed
    container_names = {"hinge2", "slide", "microhandle"}
    
    # Map container names to region names
    container_to_region = {
        "microhandle": "microwave",
        "hinge2": "right_hinge_cabinet",
        "slide": "slide_cabinet",
    }
    
    # Base costs for different action types
    BASE_COST_MOVE = 1.0
    BASE_COST_OPEN = 1.0
    BASE_COST_OBSERVE = 0.0  # Observe cost is set to 0
    BASE_COST_OTHER = 1.0
    
    # Default success probability if container or region missing
    DEFAULT_SUCCESS_PROBABILITY = 1.0 / 3.0  # Assuming 3 containers initially
    
    # Small epsilon to avoid division by zero
    EPSILON = 1e-6

    def _is_object_found(obj_name: str) -> bool:
        """Check whether a given object has already been found using state.

        We follow the same logic as KitchenEnv._ObjectFound_holds: first look
        at the State feature "<obj>.found", then fall back to the environment-
        level found-status dictionary and, finally, real-state checks.

        This avoids relying on atoms_sequence alignment and directly queries
        the current high-level state / environment belief.
        """
        # Currently only implemented for KitchenEnv.
        if not isinstance(env, KitchenEnv):
            return False

        try:
            obj = KitchenEnv.object_name_to_object(obj_name)
        except KeyError:
            return False

        # First: try the state "found" feature, which ObserveContainer sets.
        try:
            found_in_state = initial_state.get(obj, "found")
            if found_in_state:
                return True
        except (ValueError, KeyError):
            pass

        # Second: use the KitchenEnv-level predicate over (obj, container).
        # Even though this estimator only needs "found anywhere", we still
        # query per-container to match the new predicate signature.
        for container_name in container_names:
            container = KitchenEnv.object_name_to_object(container_name)
            if KitchenEnv._ObjectFound_holds(initial_state, [obj, container]):
                return True
        return False
    
    total_cost = 0.0
    
    # Track which indices have been processed as part of observe sequences
    processed_indices = set()
    
    # A search sequence is: MoveToPreTurnOn(container) -> PushOpen/PushOpenHingeDoor(container) -> ObserveContainer(...)
    i = 0
    while i < len(skeleton):
        if i in processed_indices:
            i += 1
            continue
            
        ground_nsrt = skeleton[i]
        nsrt_name = ground_nsrt.name
        # Record object names for logging (important for generic NSRTs like *Object)
        obj_names = [
            getattr(o, "name", str(o)) for o in getattr(ground_nsrt, "objects", [])
        ]
        obj_names_str = ",".join(obj_names) if obj_names else "None"
        
        # Check if this is part of an observe sequence
        if nsrt_name == "MoveToPreTurnOn":
            # Check if the object is a container
            if len(ground_nsrt.objects) >= 2:
                obj = ground_nsrt.objects[1]
                container_name = obj.name if hasattr(obj, 'name') else str(obj)
                
                # Check if this is a container and if there's an observe sequence following
                if container_name in container_names:
                    # Look ahead to see if this is followed by PushOpen and Observe
                    is_observe_sequence = False
                    move_cost = BASE_COST_MOVE
                    open_cost = BASE_COST_OPEN
                    observe_cost = BASE_COST_OBSERVE
                    sequence_length = 1  # At least the MoveToPreTurnOn
                    
                    # Check next NSRT for PushOpen (can be PushOpenHingeDoor or PushOpen)
                    if i + 1 < len(skeleton):
                        next_nsrt = skeleton[i + 1]
                        if next_nsrt.name in ["PushOpenHingeDoor", "PushOpen"]:
                            # Check if it's the same container
                            if len(next_nsrt.objects) >= 2:
                                next_obj = next_nsrt.objects[1]
                                next_container_name = next_obj.name if hasattr(next_obj, 'name') else str(next_obj)
                                if next_container_name == container_name:
                                    open_cost = BASE_COST_OPEN
                                    # Check for observe action (including ObserveContainer* variants).
                                    if i + 2 < len(skeleton):
                                        observe_nsrt = skeleton[i + 2]
                                        if observe_nsrt.name.startswith("ObserveContainer"):
                                            is_observe_sequence = True
                                            observe_cost = BASE_COST_OBSERVE
                                            sequence_length = 3
                    
                    if is_observe_sequence:
                        # Apply self-loop determinization formula
                        # c_a = c'_a = C (success and failure costs are equal)
                        # ĉ = C / p_a
                        base_cost = move_cost + open_cost + observe_cost

                        # New logic:
                        # Determine which objects are optimistic-found by this specific
                        # ObserveContainer* NSRT variant, then compute success
                        # probability from those objects' per-region extraction priors.
                        #
                        # Example: ObserveContainerSpongeMug ->
                        # found_objs=[sponge,mug], success_prob = p(sponge|region) * p(mug|region)
                        observe_nsrt = skeleton[i + 2]
                        found_obj_names: List[str] = []
                        for atom in observe_nsrt.add_effects:
                            # We only care about ObjectFound(...) add-effects.
                            if getattr(atom.predicate, "name", None) == "ObjectFound":
                                if atom.objects:
                                    found_obj_names.append(atom.objects[0].name)
                        # De-duplicate while preserving deterministic order.
                        found_obj_names = sorted(set(found_obj_names))

                        if container_name in container_to_region:
                            region_name = container_to_region[container_name]
                        else:
                            region_name = "UNKNOWN"

                        elem_probs: List[float] = []
                        for obj_name in found_obj_names:
                            if _is_object_found(obj_name):
                                # If already found in the high-level state, use p=1
                                # for that element to avoid inflating cost incorrectly.
                                elem_prob = 1.0
                            else:
                                # Use per-object extraction probabilities when available.
                                if obj_name in region_probs_by_obj:
                                    elem_prob = region_probs_by_obj[obj_name].get(
                                        region_name, DEFAULT_SUCCESS_PROBABILITY)
                                else:
                                    # Fallback to merged/default extraction distribution.
                                    elem_prob = region_probs_default.get(
                                        region_name, DEFAULT_SUCCESS_PROBABILITY)
                            elem_probs.append(elem_prob)

                        # Combine multiple elements into a single success probability.
                        # (Assume independence across the optimistic-found elements.)
                        if not elem_probs:
                            success_prob = DEFAULT_SUCCESS_PROBABILITY
                        else:
                            success_prob = 1.0
                            for p in elem_probs:
                                success_prob *= p

                        # Calculate determinized cost
                        if success_prob < EPSILON:
                            determinized_cost = float('inf')
                        else:
                            determinized_cost = base_cost / max(success_prob, EPSILON)
                        
                        total_cost += determinized_cost
                        # Mark all NSRTs in this sequence as processed
                        for j in range(i, min(i + sequence_length, len(skeleton))):
                            processed_indices.add(j)
                    else:
                        # Not part of observe sequence, use base cost
                        total_cost += BASE_COST_MOVE
                else:
                    # Not a container, use base cost
                    total_cost += BASE_COST_MOVE
            else:
                total_cost += BASE_COST_MOVE
        
        elif nsrt_name in ["PushOpenHingeDoor", "PushOpen"]:
            # Only add cost if not already processed as part of observe sequence
            # PushOpen is used for slide container, PushOpenHingeDoor for other containers
            if i not in processed_indices:
                total_cost += BASE_COST_OPEN
        
        elif nsrt_name.startswith("ObserveContainer"):
            # Only add cost if not already processed as part of observe sequence
            # Observe cost is 0 as per user's design
            if i not in processed_indices:
                total_cost += BASE_COST_OBSERVE
        
        else:
            # Other actions use base cost
            total_cost += BASE_COST_OTHER
        
        i += 1

    print(f"Total cost: {total_cost}")
    
    return total_cost