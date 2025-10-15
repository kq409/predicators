"""A Kitchen environment wrapping kitchen from https://github.com/google-
research/relay-policy-learning."""
import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import matplotlib
import numpy as np
import PIL
from gym.spaces import Box
from PIL import ImageDraw

try:
    import gymnasium as mujoco_kitchen_gym
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, \
        get_site_xmat, get_site_xpos
    from gymnasium_robotics.utils.rotations import mat2quat
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False
from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, \
    Observation, Predicate, State, Type, Video

_TRACKED_SITES = [
    "hinge_site1", "hinge_site2", "kettle_site", "microhandle_site",
    "knob1_site", "knob2_site", "knob3_site", "knob4_site", "light_site",
    "slide_site", "banana_site", "EEF"  # Added banana_site
]

_TRACKED_SITE_TO_JOINT = {
    "knob1_site": "knob_Joint_1",
    "knob2_site": "knob_Joint_2",
    "knob3_site": "knob_Joint_3",
    "knob4_site": "knob_Joint_4",
    "slide_site": "slide_cabinet",
    "hinge_site1": "right_hinge_cabinet",
    "hinge_site2": "left_hinge_cabinet",
}

_TRACKED_BODIES = ["Burner 1", "Burner 2", "Burner 3", "Burner 4"]
KETTLE_ON_BURNER1_POS = [0.169, 0.35, 1.626]
KETTLE_ON_BURNER2_POS = [-0.269, 0.35, 1.626]
KETTLE_ON_BURNER3_POS = [0.169, 0.65, 1.626]
KETTLE_ON_BURNER4_POS = [-0.269, 0.65, 1.626]


class KitchenEnv(BaseEnv):
    """Kitchen environment wrapping dm_control Kitchen."""

    # Types
    object_type = Type("object", ["x", "y", "z"])
    gripper_type = Type("gripper", ["x", "y", "z", "qw", "qx", "qy", "qz"],
                        parent=object_type)
    on_off_type = Type("on_off", ["x", "y", "z", "angle"], parent=object_type)
    hinge_door_type = Type("hinge_door", ["x", "y", "z", "angle", "observed"],
                           parent=on_off_type)
    knob_type = Type("knob", ["x", "y", "z", "angle"], parent=on_off_type)
    switch_type = Type("switch", ["x", "y", "z", "angle"], parent=on_off_type)
    surface_type = Type("surface", ["x", "y", "z"], parent=object_type)
    kettle_type = Type("kettle", ["x", "y", "z"], parent=object_type)
    banana_type = Type("banana", ["x", "y", "z"], parent=object_type)

    obj_name_to_type = {
        "gripper": gripper_type,
        "hinge2": hinge_door_type,
        "kettle": kettle_type,
        "microhandle": hinge_door_type,
        "knob1": knob_type,
        "knob2": knob_type,
        "knob3": knob_type,
        "knob4": knob_type,
        "light": switch_type,
        "slide": hinge_door_type,
        "hinge1": hinge_door_type,
        "burner1": surface_type,
        "burner2": surface_type,
        "burner3": surface_type,
        "burner4": surface_type,
        "banana": banana_type,
    }

    at_pre_turn_atol = 0.1  # tolerance for AtPreTurnOn/Off
    ontop_atol = 0.18  # tolerance for OnTop
    on_angle_thresh = -0.28  # -0.4  # dial is On if less than this threshold
    light_on_thresh = -0.39  # light is On if less than this threshold
    microhandle_open_thresh = -0.65
    hinge_open_thresh = 0.084
    cabinet_open_thresh = 0.02
    at_pre_pushontop_yz_atol = 0.1  # tolerance for AtPrePushOnTop
    at_pre_pullontop_yz_atol = 0.04  # tolerance for AtPrePullOnTop
    at_pre_pushontop_x_atol = 1.0  # other tolerance for AtPrePushOnTop
    observe_tol = 0.15  # tolerance for observation position
    banana_detection_thresh = 0.5  # threshold for banana detection

    obj_name_to_pre_push_dpos = {
        ("kettle", "on"): (-0.05, -0.2, 0.00),
        ("kettle", "off"): (0.0, 0.0, 0.08),
        ("knob4", "on"): (-0.1, -0.12, 0.05),
        ("knob4", "off"): (0.05, -0.12, -0.05),
        ("knob3", "on"): (0.0, -0.12, 0.05),
        ("knob3", "off"): (0.12, -0.12, -0.05),
        ("light", "on"): (0.1, -0.05, -0.05),
        ("light", "off"): (-0.1, -0.05, -0.05),
        ("microhandle", "on"): (0.0, -0.1, 0.13),
        ("microhandle", "off"): (0.0, -0.1, 0.2),
        ("hinge1", "on"): (0.08, -0.02, 0.05),
        ("hinge1", "off"): (-0.3, 0.0, 0.0),
        # ("hinge2", "on"): (0.1, -0.15, 0.0),
        ("hinge2", "on"): (0.02, -0.05, -0.13),    # Changed for opening hinge2
        ("hinge2", "off"): (-0.1, -0.1, 0.0),
        ("slide", "on"): (-0.2, -0.12, 0.0),
        ("slide", "off"): (0.15, -0.1, 0.0),
    }

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        assert _MJKITCHEN_IMPORTED, "Failed to import kitchen gym env. \
Install from https://github.com/NishanthJKumar/Gymnasium-Robotics. \
BE SURE TO INSTALL FROM GITHUB SOURCE THOUGH; do not blindly install as the \
README of that repo suggests!"

        if use_gui:
            assert not CFG.make_test_videos or CFG.make_failure_videos, \
                "Turn off --use_gui to make videos in kitchen env"

        self._pred_name_to_pred = self.create_predicates()

        render_mode = "human" if self._using_gui else "rgb_array"
        self._gym_env = mujoco_kitchen_gym.make("FrankaKitchen-v1",
                                                render_mode=render_mode,
                                                ik_controller=True)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, train_or_test="test")

    @classmethod
    def get_name(cls) -> str:
        return "kitchen"

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def get_object_centric_state_info(self) -> Dict[str, Any]:
        """Parse State into Object Centric State."""
        mujoco_model = self._gym_env.model  # type: ignore
        mujoco_data = self._gym_env.data  # type: ignore
        mujoco_model_names = self._gym_env.robot_env.model_names  # type: ignore
        state_info = {}
        for site in _TRACKED_SITES:
            state_info[site] = get_site_xpos(mujoco_model, mujoco_data,
                                             site).copy()
            # Include rotation for gripper.
            if site == "EEF":
                xmat = get_site_xmat(mujoco_model, mujoco_data, site).copy()
                quat = mat2quat(xmat)
                state_info[site] = np.concatenate([state_info[site], quat])
        for joint in _TRACKED_SITE_TO_JOINT.values():
            state_info[joint] = get_joint_qpos(mujoco_model, mujoco_data,
                                               joint).copy()
        for body in _TRACKED_BODIES:
            body_id = mujoco_model_names.body_name2id[body]
            state_info[body] = mujoco_data.xpos[body_id].copy()

        # Add new objects to state info
        # self._add_new_objects_to_state_info(state_info, mujoco_model, mujoco_data)
        return state_info

    def _add_new_objects_to_state_info(self, state_info: Dict[str, Any], 
                                        mujoco_model, mujoco_data):
        """Add new objects to state info"""
        object_names = ["banana"]
        
        for object_name in object_names:
            try:
                # Find the body of the object
                object_id = None
                for i in range(mujoco_model.nbody):
                    body_name = mujoco_model.names[mujoco_model.name_bodyadr[i]:].decode('utf-8').split('\x00', 1)[0]
                    if body_name == object_name:
                        object_id = i
                        break
                
                if object_id is not None:
                    # Get the position and orientation
                    position = mujoco_data.xpos[object_id].copy()
                    quat = mujoco_data.xquat[object_id].copy()
                    
                    # Combine the position and orientation
                    state_info[f"banana_{object_name}"] = np.concatenate([position, quat])
                    
                    # Also store the position and orientation separately
                    state_info[f"banana_{object_name}_pos"] = position
                    state_info[f"banana_{object_name}_quat"] = quat
            except Exception as e:
                print(f"Error adding {object_name} to state info: {e}")

    @classmethod
    def get_pre_push_delta_pos(cls, obj: Object,
                               on_or_off: str) -> Tuple[float, float, float]:
        """Get dx, dy, dz offset for pushing."""
        try:
            dx, dy, dz = cls.obj_name_to_pre_push_dpos[(obj.name, on_or_off)]
        except KeyError:
            dx, dy, dz = (0.0, 0.0, 0.0)
        return (dx, dy, dz)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A gym environment cannot render "
                                  "arbitrary states.")

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        assert caption is None
        curr_img_arr: Image = self._gym_env.render()  # type: ignore
        if CFG.kitchen_render_set_of_marks:
            # Add text labels for the burners to the image. Useful for VLM-based
            # predicate invention.
            curr_img_pil = PIL.Image.fromarray(curr_img_arr)  # type: ignore
            draw = ImageDraw.Draw(curr_img_pil)
            # Specify the font size and type (default font is used here)
            font = utils.get_scaled_default_font(draw, 3)
            # Define the text and position
            burner1_text = "burner1"
            burner1_position = (300, 285)
            burner2_text = "burner2"
            burner2_position = (210, 305)
            burner3_text = "burner3"
            burner3_position = (260, 225)
            burner4_text = "burner4"
            burner4_position = (185, 240)
            knob1_text = "knob1"
            knob1_position = (260, 155)
            knob2_text = "knob2"
            knob2_position = (160, 170)
            knob3_text = "knob3"
            knob3_position = (260, 125)
            knob4_text = "knob4"
            knob4_position = (160, 125)
            burner1_img = utils.add_text_to_draw_img(draw, burner1_position,
                                                     burner1_text, font)
            burner2_img = utils.add_text_to_draw_img(burner1_img,
                                                     burner2_position,
                                                     burner2_text, font)
            burner3_img = utils.add_text_to_draw_img(burner2_img,
                                                     burner3_position,
                                                     burner3_text, font)
            burner4_img = utils.add_text_to_draw_img(burner3_img,
                                                     burner4_position,
                                                     burner4_text, font)
            knob1_img = utils.add_text_to_draw_img(burner4_img, knob1_position,
                                                   knob1_text, font)
            knob2_img = utils.add_text_to_draw_img(knob1_img, knob2_position,
                                                   knob2_text, font)
            knob3_img = utils.add_text_to_draw_img(knob2_img, knob3_position,
                                                   knob3_text, font)
            _ = utils.add_text_to_draw_img(knob3_img, knob4_position,
                                           knob4_text, font)
            curr_img_arr = np.array(curr_img_pil)
        return [curr_img_arr]

    @property
    def predicates(self) -> Set[Predicate]:
        return set(self._pred_name_to_pred.values())

    @property
    def goal_predicates(self) -> Set[Predicate]:
        OnTop = self._pred_name_to_pred["OnTop"]
        TurnedOn = self._pred_name_to_pred["TurnedOn"]
        KettleBoiling = self._pred_name_to_pred["KettleBoiling"]
        KnobAndBurnerLinked = self._pred_name_to_pred["KnobAndBurnerLinked"]
        goal_preds = set()
        if CFG.kitchen_goals in ["all", "kettle_only"]:
            goal_preds.add(OnTop)
        if CFG.kitchen_goals in ["all", "knob_only"]:
            goal_preds.add(TurnedOn)
            goal_preds.add(KnobAndBurnerLinked)
        if CFG.kitchen_goals in ["all", "light_only"]:
            goal_preds.add(TurnedOn)
        if CFG.kitchen_goals in ["all", "boil_kettle"]:
            goal_preds.add(KettleBoiling)
            goal_preds.add(KnobAndBurnerLinked)
        return goal_preds

    @classmethod
    def create_predicates(cls) -> Dict[str, Predicate]:
        """Exposed for perceiver."""
        preds = {
            Predicate("AtPreTurnOff", [cls.gripper_type, cls.on_off_type],
                      cls._AtPreTurnOff_holds),
            Predicate("AtPreTurnOn", [cls.gripper_type, cls.on_off_type],
                      cls._AtPreTurnOn_holds),
            Predicate("AtPrePushOnTop", [cls.gripper_type, cls.kettle_type],
                      cls._AtPrePushOnTop_holds),
            Predicate("AtPrePullKettle", [cls.gripper_type, cls.kettle_type],
                      cls._AtPrePullKettle_holds),
            Predicate("OnTop", [cls.kettle_type, cls.surface_type],
                      cls._OnTop_holds),
            Predicate("NotOnTop", [cls.kettle_type, cls.surface_type],
                      cls._NotOnTop_holds),
            Predicate("TurnedOn", [cls.on_off_type], cls.On_holds),
            Predicate("TurnedOff", [cls.on_off_type], cls.Off_holds),
            Predicate("Open", [cls.on_off_type], cls.Open_holds),
            Predicate("Closed", [cls.on_off_type], cls.Closed_holds),
            Predicate("BurnerAhead", [cls.surface_type, cls.surface_type],
                      cls._BurnerAhead_holds),
            Predicate("BurnerBehind", [cls.surface_type, cls.surface_type],
                      cls._BurnerBehind_holds),
            Predicate("KettleBoiling",
                      [cls.kettle_type, cls.surface_type, cls.knob_type],
                      cls._KettleBoiling_holds),
            Predicate("KnobAndBurnerLinked", [cls.knob_type, cls.surface_type],
                      cls._KnobAndBurnerLinkedHolds),
            # New predicates for banana search
            Predicate("AtPreObserve", [cls.gripper_type, cls.hinge_door_type],
                      cls._AtPreObserve_holds),
            Predicate("Observed", [cls.hinge_door_type], cls._Observed_holds),
            Predicate("NotObserved", [cls.hinge_door_type], cls._NotObserved_holds),
            Predicate("ContainsBanana", [cls.hinge_door_type], cls._ContainsBanana_holds),
            Predicate("NotContainsBanana", [cls.hinge_door_type], cls._NotContainsBanana_holds),
            # Predicate("BananaIn", [cls.banana_type, cls.hinge_door_type], cls._BananaIn_holds),
            Predicate("BananaFound", [cls.banana_type], cls._BananaFound_holds),
            # Predicate("BananaVisible", [cls.banana_type], cls._BananaVisible_holds),
            Predicate("CanObserve", [cls.hinge_door_type], cls._CanObserve_holds),
            # Predicate("NeedsToOpen", [cls.hinge_door_type], cls._NeedsToOpen_holds),
        }

        return {p.name: p for p in preds}

    @property
    def types(self) -> Set[Type]:
        return {
            self.gripper_type, self.object_type, self.on_off_type,
            self.knob_type, self.kettle_type, self.switch_type,
            self.hinge_door_type, self.surface_type, self.banana_type
        }

    @property
    def action_space(self) -> Box:
        return cast(Box, self._gym_env.action_space)

    @classmethod
    def object_name_to_object(cls, obj_name: str) -> Object:
        """Made public for perceiver."""
        return Object(obj_name, cls.obj_name_to_type[obj_name])

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed = utils.get_task_seed(train_or_test, task_idx)
        self._current_observation = self._reset_initial_state_from_seed(
            seed, train_or_test)
        return self._copy_observation(self._current_observation)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError(
            "Simulate not implemented for kitchen env. " +
            "Try using --bilevel_plan_without_sim True")

    def step(self, action: Action) -> Observation:
        self._gym_env.step(action.arr)
        if self._using_gui:
            self._gym_env.render()
        self._current_observation = {
            "state_info": self.get_object_centric_state_info(),
            "obs_images": self.render()
        }
        return self._copy_observation(self._current_observation)

    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any]) -> State:
        """Get state from state info dictionary."""
        assert "EEF" in state_info  # sanity check
        state_dict = {}
        for key, val in state_info.items():
            if key == "EEF":
                obj = cls.object_name_to_object("gripper")
                state_dict[obj] = {
                    "x": val[0],
                    "y": val[1],
                    "z": val[2],
                    "qw": val[3],
                    "qx": val[4],
                    "qy": val[5],
                    "qz": val[6],
                }
            elif key in _TRACKED_SITE_TO_JOINT.values():
                continue  # used below
            else:
                obj_name = key.replace("_site", "").replace(" ", "").lower()
                obj = cls.object_name_to_object(obj_name)
                if key in _TRACKED_SITE_TO_JOINT:
                    joint = _TRACKED_SITE_TO_JOINT[key]
                    angle = state_info[joint][0]
                else:
                    angle = 0
                if obj.is_instance(cls.hinge_door_type):
                    # For containers, include observed status
                    state_dict[obj] = {
                        "x": val[0],
                        "y": val[1],
                        "z": val[2],
                        "angle": angle,
                        "observed": False  # Initialize as not observed
                    }
                else:
                    state_dict[obj] = {
                        "x": val[0],
                        "y": val[1],
                        "z": val[2],
                        "angle": angle
                    }
        
        # Ensure all containers are in state_dict with observed status
        containers = ["hinge1", "hinge2", "slide", "microhandle"]
        for container_name in containers:
            container = cls.object_name_to_object(container_name)
            if container not in state_dict:
                # If container not in state_dict, create it with default values
                state_dict[container] = {
                    "x": 0.0, "y": 0.0, "z": 0.0, "angle": 0.0, "observed": False
                }
        
        state = utils.create_state_from_dict(state_dict)
        state.simulator_state = {}
        return state

    def goal_reached(self) -> bool:
        state = self.state_info_to_state(
            self._current_observation["state_info"])
        kettle = self.object_name_to_object("kettle")
        burner4 = self.object_name_to_object("burner4")
        burner3 = self.object_name_to_object("burner3")
        knob4 = self.object_name_to_object("knob4")
        knob3 = self.object_name_to_object("knob3")
        light = self.object_name_to_object("light")
        banana = self.object_name_to_object("banana")
        goal_desc = self._current_task.goal_description
        kettle_on_burner4 = self._OnTop_holds(state, [kettle, burner4])
        kettle_on_burner3 = self._OnTop_holds(state, [kettle, burner3])
        knob4_turned_on = self.On_holds(state, [knob4])
        knob3_turned_on = self.On_holds(state, [knob3])
        light_turned_on = self.On_holds(state, [light])
        kettle_boiling4 = self._KettleBoiling_holds(state,
                                                    [kettle, burner4, knob4])
        kettle_boiling3 = self._KettleBoiling_holds(state,
                                                    [kettle, burner3, knob3])
        banana_found = self._BananaFound_holds(state, [banana])
        if goal_desc == ("Move the kettle to the back burner and turn it on; "
                         "also turn on the light"):
            return kettle_on_burner4 and knob4_turned_on and light_turned_on
        if goal_desc == "Move the kettle to the back left burner":
            return kettle_on_burner4
        if goal_desc == "Move the kettle to the back right burner":
            return kettle_on_burner3
        if goal_desc == "Turn on the back left burner":
            return knob4_turned_on
        if goal_desc == "Turn on the back right burner":
            return knob3_turned_on
        if goal_desc == "Turn on the light":
            return light_turned_on
        if goal_desc == ("Move the kettle to the back left burner "
                         "and turn it on"):
            return kettle_boiling4
        if goal_desc == ("Move the kettle to the back right burner "
                         "and turn it on"):
            return kettle_boiling3
        if goal_desc == ("Find the banana"):
            return banana_found
        raise NotImplementedError(f"Unrecognized goal: {goal_desc}")

    def _get_tasks(self, num: int,
                   train_or_test: str) -> List[EnvironmentTask]:
        tasks = []

        assert CFG.kitchen_goals in [
            "all", "kettle_only", "knob_only", "light_only", "boil_kettle", "find_banana"
        ]
        goal_descriptions: List[str] = []
        if CFG.kitchen_goals in ["all", "kettle_only"]:
            if train_or_test == "train":
                goal_descriptions.append(
                    "Move the kettle to the back left burner")
            else:
                goal_descriptions.append(
                    "Move the kettle to the back right burner")
        if CFG.kitchen_goals in ["all", "knob_only"]:
            if train_or_test == "train":
                goal_descriptions.append("Turn on the back left burner")
            else:
                goal_descriptions.append("Turn on the back right burner")
        if CFG.kitchen_goals in ["all", "light_only"]:
            goal_descriptions.append("Turn on the light")
        if CFG.kitchen_goals in ["all", "boil_kettle"]:
            if train_or_test == "train":
                goal_descriptions.append(
                    "Move the kettle to the back left burner and turn it on")
            else:
                goal_descriptions.append(
                    "Move the kettle to the back right burner and turn it on")
        if CFG.kitchen_goals in ["all", "find_banana"]:
            goal_descriptions.append("Find the banana")
        if CFG.kitchen_goals == "all":
            desc = (
                "Move the kettle to the back left burner and turn it on; also "
                "turn on the light")
            goal_descriptions.append(desc)

        for task_idx in range(num):
            seed = utils.get_task_seed(train_or_test, task_idx)
            init_obs = self._reset_initial_state_from_seed(seed, train_or_test)
            goal_idx = task_idx % len(goal_descriptions)
            goal_description = goal_descriptions[goal_idx]
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int,
                                       train_or_test: str) -> Observation:
        self._gym_env.reset(seed=seed)
        print("RESET START")
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
        self._gym_env.set_body_position(  # type: ignore
            "kettle", (kettle_x_coord, kettle_y_coord, 1.626))

        self._setup_new_objects(seed, train_or_test)
        self.get_object_centric_state_info()

        print("RESET COMPLETE")

        return {
            "state_info": self.get_object_centric_state_info(),
            "obs_images": self.render()
        }

    def _get_new_objects(self, object_name: str) -> List[Tuple[int, str, int]]:
        """Get all new objects."""
        try:
            model = self._gym_env.model
            object_id = []
            
            print(f"Total bodies in model: {model.nbody}")
            print(f"Total mocap bodies: {model.nmocap}")
            
            for i in range(model.nbody):
                body_name = model.names[model.name_bodyadr[i]:].decode('utf-8').split('\x00', 1)[0]
                if body_name == object_name:
                    object_id = i
                    break
            
            if object_id is None:
                return None
            
            # Get the position
            return data.xpos[object_id].copy()
        except Exception as e:
            print(f"Error getting {object_name} position: {e}")
            return None

    def _setup_new_objects(self, seed: int, train_or_test: str) -> None:
        """Set up new objects."""
        rng = np.random.default_rng(seed)
        
        # New objects position setting
        if train_or_test == "train":
            object_positions = [
                [0.0, 0.5, 1.6],
            ]
        else:
            object_positions = [
                # [-0.8, 0.7, 1.7], # Microwave
                [-0.45, 0.85, 2.5], # Upper right cabinet
                # [0.25, 0.9, 2.5], # Upper slide door
                # [-0.224, 0.71, 2.6], # Hinge2 center, for test only!
            ]
        
        # Set the position of each object
        for i, pos in enumerate(object_positions):
            object_name = "banana" if i == 0 else f"banana_{i+1}"
            
            # Add randomness
            if CFG.kitchen_randomize_init_state:
                offset_x = rng.uniform(-0.1, 0.1)
                offset_y = rng.uniform(-0.1, 0.1)
                offset_z = rng.uniform(-0.05, 0.05)
                final_pos = [
                    pos[0] + offset_x,
                    pos[1] + offset_y,
                    pos[2] + offset_z
                ]
            else:
                final_pos = pos
            
            try:
                # Now we can use set_body_position
                self._gym_env.set_body_position(object_name, final_pos)
                print(f"Successfully set {object_name} position: {final_pos}")
            except Exception as e:
                print(f"Failed to set {object_name} position: {e}")

    @classmethod
    def _AtPreTurn_holds(cls, state: State, objects: Sequence[Object],
                         on_or_off: str) -> bool:
        """Helper for _AtPreTurnOn_holds() and _AtPreTurnOff_holds()."""
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        # On refers to Open and Off to Close
        dpos = cls.get_pre_push_delta_pos(obj, on_or_off)
        gripper_xyz = np.array([
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ])
        return np.allclose(obj_xyz + dpos,
                           gripper_xyz,
                           atol=cls.at_pre_turn_atol)

    @classmethod
    def _AtPreTurnOn_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        return cls._AtPreTurn_holds(state, objects, "on")

    @classmethod
    def _AtPreTurnOff_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        return cls._AtPreTurn_holds(state, objects, "off")

    @classmethod
    def _AtPrePushOnTop_holds(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        # The main thing that's different from _AtPreTurnOn_holds is that the
        # x position has a much higher range of allowed values, since it can
        # be anywhere behind the object.
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        dpos = cls.get_pre_push_delta_pos(obj, "on")
        target_x, target_y, target_z = obj_xyz + dpos
        gripper_x, gripper_y, gripper_z = [
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ]
        if not np.allclose([target_y, target_z], [gripper_y, gripper_z],
                           atol=cls.at_pre_pushontop_yz_atol):
            return False
        return np.isclose(target_x,
                          gripper_x,
                          atol=cls.at_pre_pushontop_x_atol)

    @classmethod
    def _AtPrePullKettle_holds(cls, state: State,
                               objects: Sequence[Object]) -> bool:
        gripper, obj = objects
        obj_xyz = np.array(
            [state.get(obj, "x"),
             state.get(obj, "y"),
             state.get(obj, "z")])
        dpos = cls.get_pre_push_delta_pos(obj, "off")
        target_x, target_y, target_z = obj_xyz + dpos
        gripper_x, gripper_y, gripper_z = [
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ]
        if not np.allclose([target_y, target_z], [gripper_y, gripper_z],
                           atol=cls.at_pre_pullontop_yz_atol):
            return False
        return np.isclose(target_x,
                          gripper_x,
                          atol=cls.at_pre_pushontop_x_atol)

    @classmethod
    def _OnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        obj1_xy = [state.get(obj1, "x"), state.get(obj1, "y")]
        obj2_xy = [
            state.get(obj2, "x"),
            state.get(obj2, "y"),
        ]
        return np.allclose(obj1_xy,
                           obj2_xy, atol=cls.ontop_atol) and state.get(
                               obj1, "z") > state.get(obj2, "z")

    @classmethod
    def _NotOnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return not cls._OnTop_holds(state, objects)

    @classmethod
    def On_holds(cls,
                 state: State,
                 objects: Sequence[Object],
                 thresh_pad: float = -0.06) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.knob_type):
            return state.get(obj, "angle") < cls.on_angle_thresh - thresh_pad
        if obj.is_instance(cls.switch_type):
            return state.get(obj, "x") < cls.light_on_thresh - thresh_pad
        return False

    @classmethod
    def Off_holds(cls,
                  state: State,
                  objects: Sequence[Object],
                  thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        # Can't do not On_holds() because of thresh_pad logic.
        obj = objects[0]
        if obj.is_instance(cls.knob_type):
            return state.get(obj, "angle") >= cls.on_angle_thresh + thresh_pad
        if obj.is_instance(cls.switch_type):
            return state.get(obj, "x") >= cls.light_on_thresh + thresh_pad
        return False

    @classmethod
    def Open_holds(cls,
                   state: State,
                   objects: Sequence[Object],
                   thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        obj = objects[0]
        if obj.is_instance(cls.hinge_door_type):
            # if obj.name in ("hinge1", "hinge2"):
            #     return state.get(obj, "angle") > cls.hinge_open_thresh    # Changed for opening hinge2
            if obj.name in ("hinge2"):
                return state.get(obj, "x") > -0.25
            if obj.name == "microhandle":
                return state.get(
                    obj, "x") < cls.microhandle_open_thresh - thresh_pad
            return state.get(obj, "x") > cls.cabinet_open_thresh + thresh_pad
        return False

    @classmethod
    def Closed_holds(cls,
                     state: State,
                     objects: Sequence[Object],
                     thresh_pad: float = 0.0) -> bool:
        """Made public for use in ground-truth options."""
        # Can't do not Open_holds() because of thresh_pad logic.
        obj = objects[0]
        if obj.is_instance(cls.hinge_door_type):
            if obj.name in ("hinge1", "hinge2"):
                return state.get(obj, "angle") <= cls.hinge_open_thresh
            if obj.name == "microhandle":
                return state.get(
                    obj, "x") >= cls.microhandle_open_thresh + thresh_pad
            return state.get(obj, "x") <= cls.cabinet_open_thresh - thresh_pad
        return False

    @classmethod
    def _BurnerAhead_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Static predicate useful for deciding between pushing or pulling the
        kettle."""
        burner1, burner2 = objects
        if burner1 == burner2:
            return False
        return state.get(burner1, "y") > state.get(burner2, "y")

    @classmethod
    def _BurnerBehind_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Static predicate useful for deciding between pushing or pulling the
        kettle."""
        burner1, burner2 = objects
        if burner1 == burner2:
            return False
        return not cls._BurnerAhead_holds(state, objects)

    @classmethod
    def _KettleBoiling_holds(cls, state: State,
                             objects: Sequence[Object]) -> bool:
        """Predicate that's necessary for goal specification."""
        kettle, burner, knob = objects
        return cls.On_holds(state, [knob]) and cls._OnTop_holds(
            state, [kettle, burner]) and cls._KnobAndBurnerLinkedHolds(
                state, [knob, burner])

    @classmethod
    def _KnobAndBurnerLinkedHolds(cls, state: State,
                                  objects: Sequence[Object]) -> bool:
        """Predicate that's necessary for goal specification."""
        del state  # unused
        knob, burner = objects
        # NOTE: we assume the knobs and burners are
        # all named "knob1", "burner1", .... And that "knob1" corresponds
        # to "burner1"
        return knob.name[-1] == burner.name[-1]

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

    # New predicate implementations for banana search
    @classmethod
    def _AtPreObserve_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is in observation position."""
        gripper, container = objects
        gripper_xyz = np.array([
            state.get(gripper, "x"),
            state.get(gripper, "y"), 
            state.get(gripper, "z")
        ])
        container_xyz = np.array([
            state.get(container, "x"),
            state.get(container, "y"),
            state.get(container, "z")
        ])
        # Calculate observation position based on container type
        if container.name in ["slide"]:
            # For slide cabinet, observe from the side
            observe_pos = container_xyz + np.array([0.0, -0.2, 0.0])
        elif container.name in ["hinge1", "hinge2"]:
            # For hinge cabinets, observe from the front
            observe_pos = container_xyz + np.array([0.0, -0.2, 0.0])
        elif container.name in ["microhandle"]:
            # For microwave, observe from the front
            observe_pos = container_xyz + np.array([0.0, -0.2, 0.0])
        else:
            # Default observation position
            observe_pos = container_xyz + np.array([0.0, -0.2, 0.0])
        return np.allclose(gripper_xyz, observe_pos, atol=cls.observe_tol)

    @classmethod
    def _Observed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container has been observed."""
        container = objects[0]
        # Check if the container's observed status is stored in the state
        return state.get(container, "observed")

    @classmethod
    def _NotObserved_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container has not been observed."""
        return not cls._Observed_holds(state, objects)

    @classmethod
    def _ContainsBanana_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container contains banana."""
        container = objects[0]
        banana = cls.object_name_to_object("banana")
        
        # Get banana position
        banana_xyz = np.array([
            state.get(banana, "x"),
            state.get(banana, "y"),
            state.get(banana, "z")
        ])
        
        # Get all container positions and calculate distances
        all_containers = ["hinge1", "hinge2", "slide", "microhandle"]
        container_distances = {}
        
        for container_name in all_containers:
            container_obj = cls.object_name_to_object(container_name)
            container_xyz = np.array([
                state.get(container_obj, "x"),
                state.get(container_obj, "y"),
                state.get(container_obj, "z")
            ])
            distance = np.linalg.norm(banana_xyz - container_xyz)
            container_distances[container_name] = distance
        
        # Find the closest container
        closest_container = min(container_distances, key=container_distances.get)
        closest_distance = container_distances[closest_container]
        
        # Check if current container is the closest one
        if container.name != closest_container:
            return False
        
        # If current container is the closest, check if it's within detection threshold
        return closest_distance < cls.banana_detection_thresh

    @classmethod
    def _NotContainsBanana_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container does not contain banana."""
        return not cls._ContainsBanana_holds(state, objects)

    # @classmethod
    # def _BananaIn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if banana is in specific container."""
    #     banana, container = objects
    #     # Check if banana position is within container bounds
    #     banana_xyz = np.array([
    #         state.get(banana, "x"),
    #         state.get(banana, "y"),
    #         state.get(banana, "z")
    #     ])
    #     container_xyz = np.array([
    #         state.get(container, "x"),
    #         state.get(container, "y"),
    #         state.get(container, "z")
    #     ])
    #     # Simple distance check (would need more sophisticated bounds checking)
    #     distance = np.linalg.norm(banana_xyz - container_xyz)
    #     return distance < cls.banana_detection_thresh

    @classmethod
    def _BananaFound_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if banana has been found in any observed container."""
        banana = objects[0]
        # Check if any container contains the banana AND has been observed
        containers = ["hinge1", "hinge2", "slide", "microhandle"]
        for container_name in containers:
            container = cls.object_name_to_object(container_name)
            if (cls._ContainsBanana_holds(state, [container]) and 
                cls._Observed_holds(state, [container])):
                return True
        return False

    @classmethod
    def _BananaVisible_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if banana is visible (on countertop)."""
        banana = objects[0]
        # Check if banana is on a surface (not in container)
        banana_z = state.get(banana, "z")
        # Assume countertop height is around 1.6
        return banana_z > 1.5

    @classmethod
    def _CanObserve_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container can be observed (open and not observed)."""
        container = objects[0]
        # Check if container is open (using the appropriate method based on container type)
        is_open = False
        if container.name in ["hinge1", "hinge2", "slide", "microhandle"]:
            is_open = cls.Open_holds(state, [container])
        else:
            # For other container types, assume they can be observed if not observed
            is_open = True
        return (is_open and cls._NotObserved_holds(state, [container]))

    @classmethod
    def _NeedsToOpen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container needs to be opened for observation."""
        container = objects[0]
        # Check if container is closed (using the appropriate method based on container type)
        is_closed = False
        if container.name in ["hinge1", "hinge2", "slide", "microhandle"]:
            is_closed = cls.Closed_holds(state, [container])
        else:
            # For other container types, assume they don't need to be opened
            is_closed = False
        return (is_closed and cls._NotObserved_holds(state, [container]))
