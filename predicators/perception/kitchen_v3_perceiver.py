"""A Kitchen v3-specific perceiver."""

from predicators.envs.kitchen_v3 import KitchenV3Env
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Observation, \
    State, Task, Video


class KitchenV3Perceiver(BasePerceiver):
    """A Kitchen v3-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "kitchen_v3"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)
        pred_name_to_pred = KitchenV3Env.create_predicates()
        OnTop = pred_name_to_pred["OnTop"]
        TurnedOn = pred_name_to_pred["TurnedOn"]
        KettleBoiling = pred_name_to_pred["KettleBoiling"]
        kettle = KitchenV3Env.object_name_to_object("kettle")
        knob4 = KitchenV3Env.object_name_to_object("knob4")
        knob3 = KitchenV3Env.object_name_to_object("knob3")
        burner4 = KitchenV3Env.object_name_to_object("burner4")
        burner3 = KitchenV3Env.object_name_to_object("burner3")
        burner2 = KitchenV3Env.object_name_to_object("burner2")
        light = KitchenV3Env.object_name_to_object("light")
        # banana = KitchenV3Env.object_name_to_object("banana")
        # BananaFound = pred_name_to_pred["BananaFound"]
        # BananaOnTop = pred_name_to_pred["BananaOnTop"]
        MugOnCountertop = pred_name_to_pred["MugOnCountertop"]
        TeaOnCountertop = pred_name_to_pred["TeaOnCountertop"]
        TeaPoured = pred_name_to_pred["TeaPoured"]
        MilkTeaMade = pred_name_to_pred["MilkTeaMade"]
        mug = KitchenV3Env.object_name_to_object("mug")
        tea = KitchenV3Env.object_name_to_object("tea")
        countertop = KitchenV3Env.object_name_to_object("countertop")
        milk = KitchenV3Env.object_name_to_object("milk")
        goal_desc = env_task.goal_description
        if goal_desc == (
                "Move the kettle to the back left burner and turn it on; "
                "also turn on the light"):
            goal = {
                GroundAtom(TurnedOn, [knob4]),
                GroundAtom(OnTop, [kettle, burner4]),
                GroundAtom(TurnedOn, [light]),
            }
        elif goal_desc == "Move the kettle to the back left burner":
            goal = {GroundAtom(OnTop, [kettle, burner4])}
        elif goal_desc == "Move the kettle to the back right burner":
            goal = {GroundAtom(OnTop, [kettle, burner3])}
        elif goal_desc == "Turn on the back left burner":
            goal = {
                GroundAtom(TurnedOn, [knob4]),
            }
        elif goal_desc == "Turn on the back right burner":
            goal = {
                GroundAtom(TurnedOn, [knob3]),
            }
        elif goal_desc == "Turn on the light":
            goal = {
                GroundAtom(TurnedOn, [light]),
            }
        elif goal_desc == ("Move the kettle to the back left burner "
                           "and turn it on"):
            goal = {GroundAtom(KettleBoiling, [kettle, burner4, knob4])}
        elif goal_desc == ("Move the kettle to the back right burner "
                           "and turn it on"):
            goal = {GroundAtom(KettleBoiling, [kettle, burner3, knob3])}
        # elif goal_desc == "Find the banana":
        #     goal = {GroundAtom(BananaFound, [banana])}
        # elif goal_desc == "Take out the banana":
        #     goal = {
        #         GroundAtom(BananaOnTop, [banana, burner2])
        #     }
        elif goal_desc == "Put the mug on the countertop":
            goal = {GroundAtom(MugOnCountertop, [mug, countertop])}
        elif goal_desc == "Put the tea on the countertop":
            goal = {GroundAtom(TeaOnCountertop, [tea, countertop])}
        elif goal_desc == "Put the tea and mug on the countertop":
            goal = {
                GroundAtom(TeaOnCountertop, [tea, countertop]),
                GroundAtom(MugOnCountertop, [mug, countertop])
            }
        elif goal_desc == "Make a cup of tea":
            goal = {
                GroundAtom(TeaPoured, [tea]),
                GroundAtom(TeaOnCountertop, [tea, countertop]),
                GroundAtom(MugOnCountertop, [mug, countertop]),
            }
        elif goal_desc == "Make a cup of milk tea":
            goal = {
                GroundAtom(MilkTeaMade, [milk, tea, countertop])
            }
        else:
            raise NotImplementedError(f"Unrecognized goal: {goal_desc}")
        return Task(state, goal)

    def step(self, observation: Observation) -> State:
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        state = KitchenV3Env.state_info_to_state(obs["state_info"])
        assert state.simulator_state is not None
        state.simulator_state["images"] = obs["obs_images"]
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for kitchen_v3")

