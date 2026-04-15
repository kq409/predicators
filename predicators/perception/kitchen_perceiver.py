"""A Kitchen-specific perceiver."""

from predicators.envs.kitchen import KitchenEnv
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, GroundAtom, Observation, \
    State, Task, Video


class KitchenPerceiver(BasePerceiver):
    """A Kitchen-specific perceiver."""

    @classmethod
    def get_name(cls) -> str:
        return "kitchen"

    def reset(self, env_task: EnvironmentTask) -> Task:
        state = self._observation_to_state(env_task.init_obs)
        pred_name_to_pred = KitchenEnv.create_predicates()
        OnTop = pred_name_to_pred["OnTop"]
        TurnedOn = pred_name_to_pred["TurnedOn"]
        KettleBoiling = pred_name_to_pred["KettleBoiling"]
        kettle = KitchenEnv.object_name_to_object("kettle")
        knob4 = KitchenEnv.object_name_to_object("knob4")
        knob3 = KitchenEnv.object_name_to_object("knob3")
        burner4 = KitchenEnv.object_name_to_object("burner4")
        burner3 = KitchenEnv.object_name_to_object("burner3")
        burner2 = KitchenEnv.object_name_to_object("burner2")
        light = KitchenEnv.object_name_to_object("light")
        # banana = KitchenEnv.object_name_to_object("banana")
        # BananaFound = pred_name_to_pred["BananaFound"]
        # BananaOnTop = pred_name_to_pred["BananaOnTop"]
        MugOnCountertop = pred_name_to_pred["MugOnCountertop"]
        SpongeOnCountertop = pred_name_to_pred["SpongeOnCountertop"]
        MugWashed = pred_name_to_pred["MugWashed"]
        TeaMade = pred_name_to_pred["TeaMade"]
        MilkTeaMade = pred_name_to_pred["MilkTeaMade"]
        mug = KitchenEnv.object_name_to_object("mug")
        sponge = KitchenEnv.object_name_to_object("sponge")
        tea = KitchenEnv.object_name_to_object("tea")
        countertop = KitchenEnv.object_name_to_object("countertop")
        milk = KitchenEnv.object_name_to_object("milk")
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
        elif goal_desc == "CleanMug" or goal_desc == "Clean the mug":
            goal = {
                GroundAtom(MugWashed, [sponge, countertop])
            }
        elif goal_desc == "Make a cup of tea":
            goal = {
                GroundAtom(TeaMade, [tea, countertop])
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
        state = KitchenEnv.state_info_to_state(obs["state_info"])
        assert state.simulator_state is not None
        state.simulator_state["images"] = obs["obs_images"]
        return state

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        raise NotImplementedError("Mental images not implemented for kitchen")
