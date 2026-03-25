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
    from gymnasium_robotics.utils import mujoco_utils
    import mujoco
    from gymnasium_robotics.utils.mujoco_utils import get_joint_qpos, \
        get_site_xmat, get_site_xpos
    from gymnasium_robotics.utils.rotations import mat2quat
    from gymnasium_robotics.utils.rotations import euler2quat, quat2euler, \
        subtract_euler
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
    "slide_site", "mug_site", "milk_site", "sponge_site", "tea_site", "EEF"
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

_CONTAINER_SITE_TO_NAME = {
    "hinge_site1": "hinge1",
    "hinge_site2": "hinge2",
    "slide_site": "slide",
    "microhandle_site": "microhandle",
}

_TRACKED_BODIES = ["Burner 1", "Burner 2", "Burner 3", "Burner 4"]
KETTLE_ON_BURNER1_POS = [0.169, 0.35, 1.626]
KETTLE_ON_BURNER2_POS = [-0.269, 0.35, 1.626]
KETTLE_ON_BURNER3_POS = [0.169, 0.65, 1.626]
KETTLE_ON_BURNER4_POS = [-0.269, 0.65, 1.626]


class KitchenEnv(BaseEnv):
    """Kitchen environment wrapping dm_control Kitchen."""

    # Current env instance for options that need to set MuJoCo state (e.g. hinge2 qpos).
    _current_env: Optional["KitchenEnv"] = None

    # Types
    object_type = Type("object", ["x", "y", "z"])
    gripper_type = Type("gripper", ["x", "y", "z", "qw", "qx", "qy", "qz", "finger1_pos", "finger2_pos"],
                        parent=object_type)
    on_off_type = Type("on_off", ["x", "y", "z", "angle"], parent=object_type)
    hinge_door_type = Type("hinge_door", ["x", "y", "z", "angle", "observed"],
                           parent=on_off_type)
    knob_type = Type("knob", ["x", "y", "z", "angle"], parent=on_off_type)
    switch_type = Type("switch", ["x", "y", "z", "angle"], parent=on_off_type)
    surface_type = Type("surface", ["x", "y", "z"], parent=object_type)
    kettle_type = Type("kettle", ["x", "y", "z"], parent=object_type)
    grippable_object_type = Type("grippable_object", ["x", "y", "z", "found", "grasped"], parent=object_type)
    # banana_type = Type("banana", ["x", "y", "z", "found", "grasped"], parent=grippable_object_type)
    mug_type = Type("mug", ["x", "y", "z", "found", "grasped"], parent=grippable_object_type)
    milk_type = Type("milk", ["x", "y", "z", "found", "grasped"], parent=grippable_object_type)
    sponge_type = Type("sponge", ["x", "y", "z", "found", "grasped"], parent=grippable_object_type)
    tea_type = Type("tea", ["x", "y", "z", "found", "grasped"], parent=grippable_object_type)


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
        # "banana": banana_type,
        "mug": mug_type,
        "milk": milk_type,
        "sponge": sponge_type,
        "tea": tea_type,
        "countertop": object_type,
        "sink": object_type,
    }

    # Class level dictionary to store container observed status
    _container_observed_status: Dict[str, bool] = {}
    
    # Class level dictionary to store grippable object found status
    _grippable_object_found_status: Dict[str, bool] = {}
    # Class level dictionary to store grippable object grasped status
    _grippable_object_grasped_status: Dict[str, bool] = {}
    # Class level dictionary to store relative position of grasped object in gripper frame
    # Format: {object_name: (pos_in_gripper_frame, quat_in_gripper_frame)}
    _grasped_object_relative_pose: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # Instance-level dictionary to store body IDs for grasped objects (for applying forces)
    _grasped_object_body_ids: Dict[str, int] = {}
    # Store original gravity to restore later
    _original_gravity: Optional[np.ndarray] = None

    at_pre_turn_atol = 0.1  # tolerance for AtPreTurnOn/Off
    ontop_atol = 0.18  # tolerance for OnTop
    ontop_z_atol = 0.1  # tolerance for OnTop z position
    on_angle_thresh = -0.28  # -0.4  # dial is On if less than this threshold
    light_on_thresh = -0.39  # light is On if less than this threshold
    # microhandle_open_thresh = -0.65
    microhandle_open_thresh = -0.55
    hinge_open_thresh = 0.084
    cabinet_open_thresh = 0.02
    # slide_open_thresh = 0.2
    slide_open_thresh = 0.27
    at_pre_pushontop_yz_atol = 0.1  # tolerance for AtPrePushOnTop
    at_pre_pullontop_yz_atol = 0.04  # tolerance for AtPrePullOnTop
    at_pre_pushontop_x_atol = 1.0  # other tolerance for AtPrePushOnTop
    observe_tol = 0.15  # tolerance for observation position
    detection_thresh = 0.75  # threshold for detection
    at_pre_pick_up_tol = 0.1  # tolerance for AtPrePickUp
    pick_up_tol = 0.05  # tolerance for picking up banana
    gripper_closed_threshold = 0.05  # threshold for gripper closed
    in_sink_tol = 0.3  # threshold for mug in sink

    obj_name_to_pre_push_dpos = {
        ("kettle", "on"): (-0.05, -0.2, 0.00),
        ("kettle", "off"): (0.0, 0.0, 0.08),
        ("knob4", "on"): (-0.1, -0.12, 0.05),
        ("knob4", "off"): (0.05, -0.12, -0.05),
        ("knob3", "on"): (0.0, -0.12, 0.05),
        ("knob3", "off"): (0.12, -0.12, -0.05),
        ("light", "on"): (0.1, -0.05, -0.05),
        ("light", "off"): (-0.1, -0.05, -0.05),
        # ("microhandle", "on"): (0.0, -0.1, 0.13),
        ("microhandle", "on"): (0.02, -0.08, 0.08),
        ("microhandle", "off"): (0.0, -0.1, 0.2),
        ("hinge1", "on"): (0.08, -0.02, 0.05),
        ("hinge1", "off"): (-0.3, 0.0, 0.0),
        # ("hinge2", "on"): (0.1, -0.15, 0.0),
        ("hinge2", "on"): (0.02, -0.08, -0.05),    # Changed for opening hinge2
        # ("hinge2", "on"): (0.02, -0.04, -0.14),    # Changed for opening hinge2
        ("hinge2", "off"): (-0.1, -0.1, 0.0),
        # ("slide", "on"): (-0.2, -0.12, 0.0),
        ("slide", "on"): (-0.07, -0.05, 0.0),
        ("slide", "off"): (0.15, -0.1, 0.0),
    }

    obj_name_to_pre_pick_dpos = {
        # ("banana", "hinge2"): (0.0, -0.05, 0.05),
        # ("banana", "slide"): (0.0, 0.05, 0.0),
        # ("banana", "microhandle"): (0.0, -0.05, 0.05),
        ("mug", "hinge2"): (0.0, -0.1, 0.2),
        ("mug", "slide"): (0.0, -0.1, 0.2),
        ("mug", "microhandle"): (0.0, -0.1, 0.1),
        ("mug", "sink"): (0.0, 0.0, 0.12),
        ("sponge", "hinge2"): (0.0, -0.1, 0.1),
        ("sponge", "slide"): (0.0, -0.1, 0.1),
        ("sponge", "microhandle"): (0.0, -0.1, 0.1),
        ("sponge", "sink"): (0.0, 0.0, 0.12),
        ("tea", "hinge2"): (0.0, -0.1, 0.1),
        ("tea", "slide"): (0.0, -0.1, 0.1),
        ("tea", "microhandle"): (0.0, -0.1, 0.1),
        ("tea", "sink"): (0.0, 0.0, 0.12),
        ("milk", "hinge2"): (0.0, -0.1, 0.1),
        ("milk", "slide"): (0.0, -0.1, 0.1),
        ("milk", "microhandle"): (0.0, -0.1, 0.1),
        ("milk", "sink"): (0.0, 0.0, 0.12),
    }

    obj_name_to_xyz = {
        "hinge1": np.array([-0.682, 0.582, 2.6]),
        # "hinge2": np.array([-0.526, 0.582, 2.6]),
        "hinge2": np.array([-0.4, 0.582, 2.6]),
        "slide": np.array([0.15, 0.507, 2.6]),
        # "microhandle": np.array([-0.64187852, 0.49210206, 1.792]),
        "microhandle": np.array([-0.3187852, 0.74210206, 1.792]),
        "countertop": np.array([0.0, 0.5, 1.626]),
        "sink": np.array([0.2, 0.3, 1.68]),
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

        # Initialize all container observed status to False
        for container_name in _CONTAINER_SITE_TO_NAME.values():
            if container_name not in self._container_observed_status:
                self._container_observed_status[container_name] = False
        
        # Initialize grippable object found and grasped status to False
        grippable_objects = ["mug", "milk", "sponge", "tea"]
        for object_name in grippable_objects:
            if object_name not in self._grippable_object_found_status:
                self._grippable_object_found_status[object_name] = False
            if object_name not in self._grippable_object_grasped_status:
                self._grippable_object_grasped_status[object_name] = False
        
        # Initialize instance-level dictionary for body IDs
        self._grasped_object_body_ids: Dict[str, int] = {}
        # Store original gravity to restore later
        self._original_gravity: Optional[np.ndarray] = None
                
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

        # Get gripper finger joint positions
        # Common Franka gripper joint names
        finger_joint_names = ["robot:finger_joint1", "robot:finger_joint2"]
        for finger_joint_name in finger_joint_names:
            try:
                finger_pos = get_joint_qpos(mujoco_model, mujoco_data, finger_joint_name)
                state_info[finger_joint_name] = finger_pos.copy()
            except (KeyError, AttributeError):
                # If joint name doesn't exist, try alternative names
                pass

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
        # BananaFound = self._pred_name_to_pred["BananaFound"]
        # BananaOnTop = self._pred_name_to_pred["BananaOnTop"]
        MugInSink = self._pred_name_to_pred["MugInSink"]
        SpongeInSink = self._pred_name_to_pred["SpongeInSink"]
        MugWashed = self._pred_name_to_pred["MugWashed"]
        TeaMade = self._pred_name_to_pred["TeaMade"]
        TeaInSink = self._pred_name_to_pred["TeaInSink"]
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
        # if CFG.kitchen_goals in ["all", "find_banana"]:
        #     goal_preds.add(BananaFound)
        # if CFG.kitchen_goals in ["all", "take_out_banana"]:
        #     goal_preds.add(BananaOnTop)
        if CFG.kitchen_goals in ["all", "put_mug_in_sink"]:
            goal_preds.add(MugInSink)
        if CFG.kitchen_goals in ["all", "clean_mug"]:
            goal_preds.add(MugWashed)
        if CFG.kitchen_goals in ["all", "make_tea"]:
            goal_preds.add(TeaMade)
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
            Predicate("AtPrePickUp", [cls.gripper_type, cls.grippable_object_type, cls.object_type],
                      cls._AtPrePickUp_holds),
            Predicate("Observed", [cls.hinge_door_type], cls._Observed_holds),
            Predicate("NotObserved", [cls.hinge_door_type], cls._NotObserved_holds),
            # Predicate("BananaFound", [cls.banana_type], cls._BananaFound_holds),
            # For the grippable objects, we additionally parameterize by the
            # hinge door/container where the object was observed to be found.
            Predicate("ObjectFound", [cls.grippable_object_type, cls.hinge_door_type],
                      cls._ObjectFound_holds),
            Predicate("ObjectNotFound", [cls.grippable_object_type, cls.hinge_door_type],
                      cls._ObjectNotFound_holds),
            Predicate("CanObserve", [cls.hinge_door_type], cls._CanObserve_holds),
            # Predicate("BananaOnTop", [cls.banana_type, cls.object_type], cls._BananaOnTop_holds),
            # Predicate("BananaPickedUp", [cls.gripper_type, cls.banana_type], cls._BananaPickedUp_holds),
            Predicate("ObjectPickedUp", [cls.gripper_type, cls.grippable_object_type], cls._ObjectPickedUp_holds),
            Predicate("GripperFree", [cls.gripper_type], cls._GripperFree_holds),
            Predicate("MugInSink", [cls.mug_type, cls.object_type], cls._OnTop_holds),
            Predicate("SpongeInSink", [cls.sponge_type, cls.object_type], cls._OnTop_holds),
            Predicate("MugWashed", [cls.sponge_type, cls.object_type], cls._OnTop_holds),
            Predicate("TeaInSink", [cls.tea_type, cls.object_type], cls._OnTop_holds),
            Predicate("TeaMade", [cls.tea_type, cls.object_type], cls._OnTop_holds),
        }

        return {p.name: p for p in preds}

    @property
    def types(self) -> Set[Type]:
        return {
            self.gripper_type, self.object_type, self.on_off_type,
            self.knob_type, self.kettle_type, self.switch_type,
            self.hinge_door_type, self.surface_type,
            self.mug_type, self.milk_type, self.sponge_type, self.tea_type,
            self.grippable_object_type,
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
        KitchenEnv._current_env = self
        # Restore gravity if it was modified
        if self._original_gravity is not None:
            self._gym_env.model.opt.gravity[:] = self._original_gravity
            self._original_gravity = None
        
        self._current_task = self.get_task(train_or_test, task_idx)
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed = utils.get_task_seed(train_or_test, task_idx)
        self._current_observation = self._reset_initial_state_from_seed(
            seed, train_or_test)
        print("Reset complete")
        return self._copy_observation(self._current_observation)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError(
            "Simulate not implemented for kitchen env. " +
            "Try using --bilevel_plan_without_sim True")

    def step(self, action: Action) -> Observation:
        # Before step: Set grasped objects to their target positions and zero velocities
        # This prevents physics simulation from affecting them during step()
        self._pre_step_update_grasped_objects()
        
        self._gym_env.step(action.arr)
        
        # After step: Update positions of grasped objects again (kinematic attachment)
        # This ensures they stay attached even if step() moved them
        self._update_grasped_objects()
        
        if self._using_gui:
            self._gym_env.render()
        self._current_observation = {
            "state_info": self.get_object_centric_state_info(),
            "obs_images": self.render()
        }
        return self._copy_observation(self._current_observation)
    
    def _pre_step_update_grasped_objects(self) -> None:
        """Set grasped objects to target positions and disable gravity before step().
        This prevents physics simulation from affecting them during step().
        """
        model = self._gym_env.model
        data = self._gym_env.data
        
        # Store original gravity if not already stored
        if self._original_gravity is None:
            self._original_gravity = model.opt.gravity.copy()
        
        # Check if any objects are grasped
        has_grasped_objects = any(
            self._grippable_object_grasped_status.get(obj_name, False)
            for obj_name in self._grasped_object_relative_pose.keys()
        )
        
        # Disable gravity if any object is grasped, restore if none are grasped
        if has_grasped_objects:
            model.opt.gravity = [0.0, 0.0, -0.001]   # Disable gravity globally
            # print(f"Gravity: {model.opt.gravity}")
        else:
            # Restore original gravity when no objects are grasped
            if self._original_gravity is not None:
                model.opt.gravity[:] = self._original_gravity
        
        if not self._grasped_object_relative_pose:
            return
        
        # Get current gripper pose
        state_info = self.get_object_centric_state_info()
        gripper_pos = state_info["EEF"][:3]  # x, y, z
        gripper_quat = state_info["EEF"][3:7]  # qw, qx, qy, qz
        
        # Update each grasped object BEFORE step
        for obj_name, (rel_pos, rel_quat) in self._grasped_object_relative_pose.items():
            if not self._grippable_object_grasped_status.get(obj_name, False):
                continue
            
            # Get or cache body ID
            if obj_name not in self._grasped_object_body_ids:
                try:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                    self._grasped_object_body_ids[obj_name] = body_id
                except Exception:
                    continue
            
            # Transform relative position from gripper frame to world frame
            R_gripper = self._quat_to_rot_matrix(gripper_quat)
            obj_world_pos = gripper_pos + R_gripper @ rel_pos
            
            # For quaternion, multiply gripper quaternion with relative quaternion
            obj_world_quat = self._quat_multiply(gripper_quat, rel_quat)
            
            # Set object position BEFORE step
            try:
                self.set_joint(obj_name, np.concatenate([obj_world_pos, obj_world_quat]))
                # Set velocity to zero
                self._set_object_velocity_zero(obj_name)
            except Exception as e:
                print(f"Warning: Failed to pre-update grasped object {obj_name} position: {e}")
    
    def _update_grasped_objects(self) -> None:
        """Update positions of grasped objects to follow gripper after step().
        This ensures objects stay attached even if step() moved them.
        """
        if not self._grasped_object_relative_pose:
            return
        
        # Get current gripper pose
        state_info = self.get_object_centric_state_info()
        gripper_pos = state_info["EEF"][:3]  # x, y, z
        gripper_quat = state_info["EEF"][3:7]  # qw, qx, qy, qz
        
        # Update each grasped object AFTER step
        for obj_name, (rel_pos, rel_quat) in self._grasped_object_relative_pose.items():
            if not self._grippable_object_grasped_status.get(obj_name, False):
                continue
            
            # Transform relative position from gripper frame to world frame
            # obj_world_pos = gripper_pos + R_gripper * rel_pos
            R_gripper = self._quat_to_rot_matrix(gripper_quat)
            obj_world_pos = gripper_pos + R_gripper @ rel_pos
            
            # For quaternion, multiply gripper quaternion with relative quaternion
            # quat_multiply(gripper_quat, rel_quat) gives object quaternion in world frame
            obj_world_quat = self._quat_multiply(gripper_quat, rel_quat)
            
            # Set object position using set_joint
            try:
                self.set_joint(obj_name, np.concatenate([obj_world_pos, obj_world_quat]))
                # Set velocity to zero to prevent physics simulation
                self._set_object_velocity_zero(obj_name)
            except Exception as e:
                print(f"Warning: Failed to update grasped object {obj_name} position: {e}")
    
    def _set_object_velocity_zero(self, obj_name: str) -> None:
        """Set object velocity to zero to disable physics simulation."""
        try:
            model = self._gym_env.model
            data = self._gym_env.data
            
            # Find the joint (freejoint) for this object
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj_name)
            if joint_id >= 0:
                joint_type = model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    # Free joint has 6 DOF for velocity (3 linear + 3 angular)
                    qvel_start = model.jnt_dofadr[joint_id]
                    data.qvel[qvel_start:qvel_start + 6] = 0.0
                    mujoco.mj_forward(model, data)
        except Exception as e:
            # Silently fail if object doesn't have a joint or other error
            pass
    
    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 * q2.
        
        Args:
            q1: [qw, qx, qy, qz]
            q2: [qw, qx, qy, qz]
        
        Returns:
            [qw, qx, qy, qz]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])
    
    @staticmethod
    def _quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.
        
        Args:
            q: [qw, qx, qy, qz]
        
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    @classmethod
    def state_info_to_state(cls, state_info: Dict[str, Any]) -> State:
        """Get state from state info dictionary."""
        assert "EEF" in state_info  # sanity check
        state_dict = {}

        finger1_value = state_info["robot:finger_joint1"][0]    # Value the larger, the more open
        finger2_value = state_info["robot:finger_joint2"][0]    # Value the larger, the more open

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
                    "finger1_pos": finger1_value,
                    "finger2_pos": finger2_value,
                }
            elif key in _TRACKED_SITE_TO_JOINT.values():
                continue  # used below
            elif "finger" in key.lower() and "joint" in key.lower():
                continue  # already processed above
            else:
                obj_name = key.replace("_site", "").replace(" ", "").lower()
                obj = cls.object_name_to_object(obj_name)
                if key in _TRACKED_SITE_TO_JOINT:
                    joint = _TRACKED_SITE_TO_JOINT[key]
                    angle = state_info[joint][0]
                else:
                    angle = 0
                if key in _CONTAINER_SITE_TO_NAME:
                    container_name = _CONTAINER_SITE_TO_NAME[key]
                    observed = cls.get_container_observed(container_name)
                else:
                    observed = False
                if obj.is_instance(cls.hinge_door_type):
                    # For containers, include observed status
                    state_dict[obj] = {
                        "x": val[0],
                        "y": val[1],
                        "z": val[2],
                        "angle": angle,
                        "observed": observed  # Initialize as not observed
                    }
                elif obj.is_instance(cls.grippable_object_type):
                    # For banana, include found status
                    object_name = obj.name if hasattr(obj, 'name') else "grippable_object"
                    found = cls._grippable_object_found_status.get(object_name, False)
                    grasped = cls._grippable_object_grasped_status.get(object_name, False)
                    state_dict[obj] = {
                        "x": val[0],
                        "y": val[1],
                        "z": val[2],
                        "found": found,  # Initialize found status
                        "grasped": grasped  # Initialize grasped status
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
                observed = cls._container_observed_status.get(container_name, False)
                state_dict[container] = {
                    "x": 0.0, "y": 0.0, "z": 0.0, "angle": 0.0, "observed": observed
                }
        
        # Ensure banana is in state_dict with found status
        # banana = cls.object_name_to_object("banana")
        # if banana not in state_dict:
        #     banana_name = banana.name if hasattr(banana, 'name') else "banana"
        #     found = cls._banana_found_status.get(banana_name, False)
        #     state_dict[banana] = {
        #         "x": 0.0, "y": 0.0, "z": 0.0, "found": found
        #     }
        
        # Ensure sink is in state_dict (sink may not be tracked in state_info)
        sink = cls.object_name_to_object("sink")
        if sink not in state_dict:
            # Get sink position from obj_name_to_xyz if available
            sink_pos = cls.obj_name_to_xyz.get("sink", np.array([1.3, 0.5, 1.3]))
            state_dict[sink] = {
                "x": float(sink_pos[0]),
                "y": float(sink_pos[1]),
                "z": float(sink_pos[2])
            }
        
        state = utils.create_state_from_dict(state_dict)
        state.simulator_state = {}
        return state

    @classmethod
    def set_container_observed(cls, container_name: str, observed: bool = True) -> None:
        """Set the observed status of a container."""
        if container_name in _CONTAINER_SITE_TO_NAME.values():
            cls._container_observed_status[container_name] = observed

    @classmethod
    def get_container_observed(cls, container_name: str) -> bool:
        """Get the observed status of a container."""
        return cls._container_observed_status.get(container_name, False)

    # @classmethod
    # def set_banana_found(cls, banana_name: str = "banana", found: bool = True) -> None:
    #     """Set the found status of banana."""
    #     cls._grippable_object_found_status[banana_name] = found

    @classmethod
    def set_mug_found(cls, mug_name: str = "mug", found: bool = True) -> None:
        """Set the found status of mug."""
        cls._grippable_object_found_status[mug_name] = found

    @classmethod
    def set_sponge_found(cls, sponge_name: str = "sponge", found: bool = True) -> None:
        """Set the found status of sponge."""
        cls._grippable_object_found_status[sponge_name] = found

    @classmethod
    def set_tea_found(cls, tea_name: str = "tea", found: bool = True) -> None:
        """Set the found status of tea."""
        cls._grippable_object_found_status[tea_name] = found

    # @classmethod
    # def get_banana_found(cls, banana_name: str = "banana") -> bool:
    #     """Get the found status of banana."""
    #     return cls._grippable_object_found_status.get(banana_name, False)
    
    @classmethod
    def set_grippable_object_grasped(cls, obj_name: str, grasped: bool = True,
                                     gripper_pos: Optional[np.ndarray] = None,
                                     gripper_quat: Optional[np.ndarray] = None,
                                     obj_pos: Optional[np.ndarray] = None,
                                     obj_quat: Optional[np.ndarray] = None) -> None:
        """Set the grasped status of a grippable object and record relative pose.
        
        Args:
            obj_name: Name of the object
            grasped: Whether the object is grasped
            gripper_pos: Gripper position [x, y, z] (required if grasped=True)
            gripper_quat: Gripper quaternion [qw, qx, qy, qz] (required if grasped=True)
            obj_pos: Object position [x, y, z] (required if grasped=True)
            obj_quat: Object quaternion [qw, qx, qy, qz] (required if grasped=True)
        """
        cls._grippable_object_grasped_status[obj_name] = grasped
        
        if grasped:
            if gripper_pos is None or gripper_quat is None or obj_pos is None or obj_quat is None:
                raise ValueError("gripper_pos, gripper_quat, obj_pos, and obj_quat must be provided when grasping")
            
            # Calculate relative position and orientation in gripper frame
            rel_pos, rel_quat = cls._compute_relative_pose(
                gripper_pos, gripper_quat, obj_pos, obj_quat)
            cls._grasped_object_relative_pose[obj_name] = (rel_pos, rel_quat)
        else:
            # Clear relative pose when releasing
            if obj_name in cls._grasped_object_relative_pose:
                del cls._grasped_object_relative_pose[obj_name]
    
    @classmethod
    def _compute_relative_pose(cls, gripper_pos: np.ndarray, gripper_quat: np.ndarray,
                               obj_pos: np.ndarray, obj_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute object pose relative to gripper frame.
        
        Args:
            gripper_pos: [x, y, z]
            gripper_quat: [qw, qx, qy, qz]
            obj_pos: [x, y, z]
            obj_quat: [qw, qx, qy, qz]
        
        Returns:
            (rel_pos, rel_quat): Relative position [x, y, z] and quaternion [qw, qx, qy, qz] in gripper frame
        """
        # Compute relative position: transform obj_pos to gripper frame
        # rel_pos = R_gripper^T * (obj_pos - gripper_pos)
        # where R_gripper is rotation matrix from gripper quaternion
        
        R_gripper = cls._quat_to_rot_matrix(gripper_quat)
        rel_pos_vec = obj_pos - gripper_pos
        rel_pos = R_gripper.T @ rel_pos_vec
        
        # Compute relative quaternion: rel_quat = gripper_quat^-1 * obj_quat
        # Quaternion inverse: [w, -x, -y, -z] for unit quaternion
        gripper_quat_inv = np.array([gripper_quat[0], -gripper_quat[1], 
                                     -gripper_quat[2], -gripper_quat[3]])
        rel_quat = cls._quat_multiply(gripper_quat_inv, obj_quat)
        
        return rel_pos, rel_quat

    def goal_reached(self) -> bool:
        state = self.state_info_to_state(
            self._current_observation["state_info"])
        kettle = self.object_name_to_object("kettle")
        gripper = self.object_name_to_object("gripper")
        burner2 = self.object_name_to_object("burner2")
        burner4 = self.object_name_to_object("burner4")
        burner3 = self.object_name_to_object("burner3")
        knob4 = self.object_name_to_object("knob4")
        knob3 = self.object_name_to_object("knob3")
        light = self.object_name_to_object("light")
        # banana = self.object_name_to_object("banana")
        mug = self.object_name_to_object("mug")
        sponge = self.object_name_to_object("sponge")
        tea = self.object_name_to_object("tea")
        sink = self.object_name_to_object("sink")
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
        # banana_found = self._BananaFound_holds(state, [banana])
        # take_out_banana = self._BananaOnTop_holds(state, [banana, burner2])
        mug_in_sink = self._OnTop_holds(state, [mug, sink])
        sponge_in_sink = self._OnTop_holds(state, [sponge, sink])
        tea_in_sink = self._OnTop_holds(state, [tea, sink])
        mug_washed = self._OnTop_holds(state, [sponge, sink])
        tea_made = self._OnTop_holds(state, [tea, sink])

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
        # if goal_desc == ("Find the banana"):
        #     return banana_found
        # if goal_desc == ("Take out the banana"):
        #     return take_out_banana
        if goal_desc == ("Put the mug in the sink"):
            return mug_in_sink
        if goal_desc == ("Clean the mug"):
            return mug_washed
        if goal_desc == ("Make a cup of tea"):
            return tea_made
        raise NotImplementedError(f"Unrecognized goal: {goal_desc}")

    def _get_tasks(self, num: int,
                   train_or_test: str) -> List[EnvironmentTask]:
        tasks = []

        assert CFG.kitchen_goals in [
            "all", "kettle_only", "knob_only", "light_only", "boil_kettle", "put_mug_in_sink", "clean_mug", "make_tea"
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
        # if CFG.kitchen_goals in ["all", "find_banana"]:
        #     goal_descriptions.append("Find the banana")
        # if CFG.kitchen_goals in ["all", "take_out_banana"]:
        #     goal_descriptions.append("Take out the banana")
        if CFG.kitchen_goals in ["all", "put_mug_in_sink"]:
            goal_descriptions.append("Put the mug in the sink")
        if CFG.kitchen_goals in ["all", "clean_mug"]:
            goal_descriptions.append("Clean the mug")
        if CFG.kitchen_goals in ["all", "make_tea"]:
            goal_descriptions.append("Make a cup of tea")
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
            # "kettle", (kettle_x_coord, kettle_y_coord, 1.626))
            "kettle", (kettle_x_coord, kettle_y_coord, 0.0))

        self._setup_new_objects(seed, train_or_test)
        self.get_object_centric_state_info()

        print("RESET COMPLETE")

        return {
            "state_info": self.get_object_centric_state_info(),
            "obs_images": self.render()
        }
    def get_screenshot(self) -> Image:
        return self.render()

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

    def set_joint(self, joint_name: str, value: float):
        model = self._gym_env.model          # MuJoCo mjModel
        data = self._gym_env.data            # MuJoCo mjData
        mujoco_utils.set_joint_qpos(model, data, joint_name, value)
        mujoco.mj_forward(model, data) 

    def set_object_positions_override(self, object_positions):
        """Optionally override default new-object positions.

        object_positions should be a list of 4 [x, y, z] lists in the order:
        [\"mug\", \"milk\", \"sponge\", \"tea\"]. If None is passed,
        the override is cleared and the default hardcoded positions are used.
        """
        self._object_positions_override = object_positions

    def _setup_new_objects(self, seed: int, train_or_test: str) -> None:
        """Set up new objects."""
        rng = np.random.default_rng(seed)

        # If an override is set (e.g., from an experiment script), prefer it.
        object_positions = getattr(self, "_object_positions_override", None)
        if object_positions is None:
            # Default hardcoded positions (backward compatible behavior).
            if train_or_test == "train":
                object_positions = [
                    [-0.3, 0.82, 1.7],
                    [-0.15, 0.85, 1.7],
                    [-0.4, 0.7, 2.45],
                    [-0.1, 0.85, 1.7],
                ]
            else:
                object_positions = [
                    [-0.3, 0.82, 1.7],
                    [-0.15, 0.85, 1.7],
                    [-0.4, 0.7, 2.45],
                    [-0.1, 0.85, 1.7],
                ]

        # if train_or_test == "train":
        #     object_positions = [
        #         [0.1, 0.9, 2.45],
                # [-0.15, 0.7, 1.7],
                # [-0.3, 0.85, 2.45],
                # [-0.4, 0.7, 1.7],
                # [-0.45, 0.85, 2.45],
        #     ]
        # else:
        #     object_positions = [
        #         [0.1, 0.9, 2.45],
                # [-0.15, 0.7, 1.7],
                # [-0.3, 0.85, 2.45],
                # [-0.4, 0.7, 1.7],
                # [-0.45, 0.85, 2.45],
        #     ]
            # object_positions = [
            #     # [-0.8, 0.7, 1.7], # Microwave
            #     # [-0.45, 0.8, 2.4], # Upper right cabinet
            #     [-0.025, 0.77, 2.4], # Upper slide door
            #     # [-0.224, 0.71, 2.6], # Hinge2 center, for test only!
            #     # [-0.2, 0.5, 2.0], # Lookat marker position
            # ]
        # quaternion = [0.70710678, 0.70710678, 0.0, 0.0]
        # quaternion = [0.0, 0.0, 0.0, 1.0]   # No rotation
        # quaternion = [0.70710678, 0.0, 0.70710678, 0.0] # 90 degree rotation around y axis
        # quaternion = [0.70710678, 0.0, 0.0, 0.70710678] # 90 degree rotation around z axis
        # euler = (0.0, 0.0, 1 * np.pi / 64)
        euler = (0.0, 0.0, 0.0)
        quaternion = euler2quat(euler)
        # Set the position of each object
        for i, pos in enumerate(object_positions):
            objects = ["mug", "milk", "sponge", "tea"]
            object_name = objects[i]

            
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
                self.set_joint(object_name, np.concatenate([final_pos, quaternion]))
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
                               obj1, "z") > state.get(obj2, "z") and np.isclose(
                                state.get(obj1, "z"), state.get(obj2, "z"), atol=cls.ontop_z_atol)

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
                return state.get(obj, "x") < -0.65
            if obj.name == "microhandle":
                return state.get(obj, "x") < cls.microhandle_open_thresh - thresh_pad
            return state.get(obj, "x") > cls.slide_open_thresh + thresh_pad
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
        if container.name in ["hinge2"]:
            # For hinge2 cabinet (where objects are found), observe from the side
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
        # return np.allclose(gripper_xyz, observe_pos, atol=cls.observe_tol)
        # TODO: Assume always in observation position for now
        return True
        # return np.allclose(gripper_xyz, observe_pos, atol=0.5)

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
    def _ContainsObject_holds(cls, state: State, objects: Sequence[Object], obj_name: str) -> bool:
        """Check if container contains object (banana or mug).
        
        Args:
            state: Current state
            objects: Sequence containing [gripper, container]
            obj_name: Name of the object to check ("banana" or "mug")
        """
        gripper, container = objects
        obj = cls.object_name_to_object(obj_name)
        
        # Check if object has been picked up
        if cls._grippable_object_grasped_status.get(obj_name, False):
            return False
        
        # Get object position
        obj_xyz = np.array([
            state.get(obj, "x"),
            state.get(obj, "y"),
            state.get(obj, "z")
        ])
        
        # Get all container positions and calculate distances
        all_containers = ["hinge2", "slide", "microhandle"]
        container_distances = {}
        
        for container_name in all_containers:
            container_obj = cls.object_name_to_object(container_name)
            container_xyz = cls.obj_name_to_xyz[container_name]
            distance = np.linalg.norm(obj_xyz - container_xyz)
            container_distances[container_name] = distance
        
        # Find the closest container
        closest_container = min(container_distances, key=container_distances.get)
        closest_distance = container_distances[closest_container]
        
        # Check if current container is the closest one
        if container.name != closest_container:
            return False
        
        # If current container is the closest, check if it's within detection threshold
        return closest_distance < cls.detection_thresh

    # @classmethod
    # def _ContainsBanana_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if container contains banana. Delegates to _ContainsObject_holds."""
    #     return cls._ContainsObject_holds(state, objects, "banana")

    # @classmethod
    # def _NotContainsBanana_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if container does not contain banana."""
    #     gripper, container = objects
    #     return not cls._ContainsBanana_holds(state, [gripper, container])

    @classmethod
    def _ContainsMug_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container contains mug. Delegates to _ContainsObject_holds."""
        return cls._ContainsObject_holds(state, objects, "mug")

    @classmethod
    def _NotContainsMug_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container does not contain mug."""
        gripper, container = objects
        return not cls._ContainsMug_holds(state, [gripper, container])

    @classmethod
    def _ContainsSponge_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container contains sponge. Delegates to _ContainsObject_holds."""
        return cls._ContainsObject_holds(state, objects, "sponge")

    @classmethod
    def _ContainsTea_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container contains tea. Delegates to _ContainsObject_holds."""
        return cls._ContainsObject_holds(state, objects, "tea")

    @classmethod
    def _NotContainsSponge_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container does not contain sponge."""
        gripper, container = objects
        return not cls._ContainsSponge_holds(state, [gripper, container])

    @classmethod
    def _ObjectFound_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if object has been found in a specific container.

        This predicate is parameterized by both the object and the container.
        We require:
        1) the container has been observed, and
        2) in the real state, the container contains the object (within
           detection threshold), and
        3) the object is not currently grasped (handled in _ContainsObject_holds).
        """
        obj = objects[0]
        container = objects[1]
        obj_name = obj.name if hasattr(obj, "name") else ""

        if not cls._Observed_holds(state, [container]):
            return False

        # _Contains* predicates expect [gripper_placeholder, container]; the
        # placeholder gripper is not used in their internal distance logic.
        gripper = cls.object_name_to_object("gripper")
        if obj_name == "mug":
            return cls._ContainsMug_holds(state, [gripper, container])
        if obj_name == "sponge":
            return cls._ContainsSponge_holds(state, [gripper, container])
        if obj_name == "tea":
            return cls._ContainsTea_holds(state, [gripper, container])
        # Unsupported object types (e.g., banana, if re-enabled later).
        return False

    @classmethod
    def _ObjectNotFound_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Negation of _ObjectFound_holds over (obj, container)."""
        return not cls._ObjectFound_holds(state, objects)


    # @classmethod
    # def _BananaFound_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if banana has been found. Delegates to _ObjectFound_holds."""
    #     return cls._ObjectFound_holds(state, objects)

    # @classmethod
    # def _CanObserve_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if container can be observed (open and not observed)."""
    #     container = objects[0]
    #     # Check if container is open (using the appropriate method based on container type)
    #     is_open = False
    #     if container.name in ["hinge2", "slide", "microhandle"]:
    #         is_open = cls.Open_holds(state, [container])
    #     elif container.name in ["hinge1"]:
    #         is_open = False
    #     else:
    #         # For other container types, assume they can be observed if not observed
    #         is_open = True
    #     return (is_open and cls._NotObserved_holds(state, [container]))

    @classmethod
    def _CanObserve_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if container can be observed (open and not observed)."""
        container = objects[0]
        if container.name in ["hinge2", "slide", "microhandle"]:
            return True
        elif container.name in ["hinge1"]:
            return False
        else:
            return True


    @classmethod
    def get_pre_pick_delta_pos(cls, objects: Sequence[Object]) -> Tuple[float, float, float]:
        """Get dx, dy, dz offset for pushing."""
        obj, container = objects
        try:
            if container is None or not hasattr(container, 'name'):
                return (0.0, 0.0, 0.0)
            dx, dy, dz = cls.obj_name_to_pre_pick_dpos[(obj.name, container.name)]
        except KeyError:
            dx, dy, dz = (0.0, 0.0, 0.0)
        return (dx, dy, dz)

    @classmethod
    def _AtPrePickUp_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is in pre-pick up position."""
        gripper, object, obj_place = objects
        gripper_xyz = np.array([
            state.get(gripper, "x"),
            state.get(gripper, "y"),
            state.get(gripper, "z")
        ])
        dpos = cls.get_pre_pick_delta_pos([object, obj_place])
        object_xyz = np.array([
            state.get(object, "x"),
            state.get(object, "y"),
            state.get(object, "z")
        ])
        obj_place_xyz = np.array([
            state.get(obj_place, "x"),
            state.get(obj_place, "y"),
            state.get(obj_place, "z")
        ])
        return np.allclose(gripper_xyz, object_xyz + dpos, atol=cls.at_pre_pick_up_tol)

    # @classmethod
    # def _BananaOnTop_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if banana is on top of the surface."""
    #     banana, obj_place = objects
    #     banana_xy = [state.get(banana, "x"), state.get(banana, "y")]
    #     obj_place_xy = [state.get(obj_place, "x"), state.get(obj_place, "y")]
    #     z_delta = state.get(banana, "z") - state.get(obj_place, "z")
    #     return np.allclose(banana_xy, obj_place_xy, atol=cls.ontop_atol) and state.get(banana, "z") > state.get(obj_place, "z") and z_delta < 0.02

    # @classmethod
    # def _BananaPickedUp_holds(cls, state: State, objects: Sequence[Object]) -> bool:
    #     """Check if banana has been picked up."""
    #     return cls._grippable_object_grasped_status.get("banana", False)
    
    @classmethod
    def _ObjectPickedUp_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if a grippable object has been picked up."""
        # objects: [gripper, obj]
        obj = objects[1]
        obj_name = obj.name if hasattr(obj, "name") else ""
        return cls._grippable_object_grasped_status.get(obj_name, False)

    @classmethod
    def _GripperFree_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if gripper is free (not holding any object)."""
        return not any(cls._grippable_object_grasped_status.values())

