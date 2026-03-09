"""Ground-truth NSRTs for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class KitchenGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Kitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        gripper_type = types["gripper"]
        on_off_type = types["on_off"]
        kettle_type = types["kettle"]
        surface_type = types["surface"]
        switch_type = types["switch"]
        knob_type = types["knob"]
        hinge_door_type = types["hinge_door"]
        banana_type = types["banana"]
        mug_type = types["mug"]
        sponge_type = types["sponge"]
        object_type = types["object"]
        tea_type = types["tea"]
        # Objects
        gripper = Variable("?gripper", gripper_type)
        on_off_obj = Variable("?on_off_obj", on_off_type)
        kettle = Variable("?kettle", kettle_type)
        surface_from = Variable("?surface_from", surface_type)
        surface_to = Variable("?surface_to", surface_type)
        switch = Variable("?switch", switch_type)
        knob = Variable("?knob", knob_type)
        hinge_door = Variable("?hinge_door", hinge_door_type)
        container = Variable("?container", hinge_door_type)  # Use hinge_door_type instead
        banana = Variable("?banana", banana_type)
        mug = Variable("?mug", mug_type)
        tea = Variable("?tea", tea_type)
        sponge = Variable("?sponge", sponge_type)
        origin = Variable("?origin", object_type)
        destination = Variable("?destination", object_type)


        # Options
        MoveToPrePushOnTop = options["MoveToPrePushOnTop"]
        MoveToPreTurnOff = options["MoveToPreTurnOff"]
        MoveToPreTurnOn = options["MoveToPreTurnOn"]
        MoveToPrePullKettle = options["MoveToPrePullKettle"]
        PullKettle = options["PullKettle"]
        PushObjOnObjForward = options["PushObjOnObjForward"]
        TurnOffSwitch = options["TurnOffSwitch"]
        TurnOnSwitch = options["TurnOnSwitch"]
        TurnOffKnob = options["TurnOffKnob"]
        TurnOnKnob = options["TurnOnKnob"]
        PushOpen = options["PushOpen"]
        PushClose = options["PushClose"]
        PushKettleOntoBurner = options["PushKettleOntoBurner"]
        MoveAndTurnOnKnob = options["MoveAndTurnOnKnob"]
        MoveToObservePosition = options["MoveToObservePosition"]
        MoveToPrePickUp = options["MoveToPrePickUp"]
        Pick = options["Pick"]
        MoveToTarget = options["MoveToTarget"]
        Place = options["Place"]
        WashMug = options["WashMug"]
        MakeTea = options["MakeTea"]
        
        ObserveContainer = options["ObserveContainer"]


        # Predicates
        AtPreTurnOn = predicates["AtPreTurnOn"]
        AtPreTurnOff = predicates["AtPreTurnOff"]
        AtPrePushOnTop = predicates["AtPrePushOnTop"]
        AtPrePullKettle = predicates["AtPrePullKettle"]
        Closed = predicates["Closed"]
        TurnedOn = predicates["TurnedOn"]
        TurnedOff = predicates["TurnedOff"]
        OnTop = predicates["OnTop"]
        Open = predicates["Open"]
        NotOnTop = predicates["NotOnTop"]
        BurnerAhead = predicates["BurnerAhead"]
        BurnerBehdind = predicates["BurnerBehind"]
        KettleBoiling = predicates["KettleBoiling"]
        KnobAndBurnerLinked = predicates["KnobAndBurnerLinked"]
        AtPreObserve = predicates["AtPreObserve"]
        Observed = predicates["Observed"]
        NotObserved = predicates["NotObserved"]
        ContainsBanana = predicates["ContainsBanana"]
        NotContainsBanana = predicates["NotContainsBanana"]
        # IsSlide = predicates["IsSlide"]  # TEMPORARY HARDCODE: Predicate to identify microhandle (container where objects are found)
        # BananaIn = predicates["BananaIn"]  # Removed - not needed
        BananaFound = predicates["BananaFound"]
        MugFound = predicates["MugFound"]
        SpongeFound = predicates["SpongeFound"]
        TeaFound = predicates["TeaFound"]
        ContainsSponge = predicates["ContainsSponge"]
        NotContainsSponge = predicates["NotContainsSponge"]
        # BananaVisible = predicates["BananaVisible"]  # Removed - not needed
        CanObserve = predicates["CanObserve"]
        # NeedsToOpen = predicates["NeedsToOpen"]  # Removed - not needed
        AtPrePickUp = predicates["AtPrePickUp"]
        BananaPickedUp = predicates["BananaPickedUp"]
        MugPickedUp = predicates["MugPickedUp"]
        SpongePickedUp = predicates["SpongePickedUp"]
        TeaPickedUp = predicates["TeaPickedUp"]
        BananaOnTop = predicates["BananaOnTop"]
        MugInSink = predicates["MugInSink"]
        TeaInSink = predicates["TeaInSink"]
        SpongeInSink = predicates["SpongeInSink"]
        MugWashed = predicates["MugWashed"]
        TeaMade = predicates["TeaMade"]

        nsrts = set()

        # MoveToPreTurnOff
        parameters = [gripper, on_off_obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(AtPreTurnOff, [gripper, on_off_obj])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPreTurnOff
        option_vars = [gripper, on_off_obj]

        def moveto_preturnoff_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "off"),
                              dtype=np.float32)
            return params

        move_to_pre_turn_off_nsrt = NSRT("MoveToPreTurnOff", parameters,
                                         preconditions, add_effects,
                                         delete_effects, ignore_effects,
                                         option, option_vars,
                                         moveto_preturnoff_sampler)

        # MoveToPreTurnOn
        parameters = [gripper, on_off_obj]
        preconditions = set()
        add_effects = {LiftedAtom(AtPreTurnOn, [gripper, on_off_obj])}
        delete_effects = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPreTurnOn
        option_vars = [gripper, on_off_obj]

        def moveto_preturnon_sampler(state: State, goal: Set[GroundAtom],
                                     rng: np.random.Generator,
                                     objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "on"),
                              dtype=np.float32)
            print(f"MoveToPreTurnOn Target position: {obj.name} + {params}")
            return params

        move_to_pre_turn_on_nsrt = NSRT("MoveToPreTurnOn", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, moveto_preturnon_sampler)

        # MoveToPrePushOnTop
        parameters = [gripper, kettle]
        preconditions = set()
        add_effects = {LiftedAtom(AtPrePushOnTop, [gripper, kettle])}
        delete_effects = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePushOnTop
        option_vars = [gripper, kettle]

        def moveto_prepushontop_sampler(state: State, goal: Set[GroundAtom],
                                        rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "on"),
                              dtype=np.float32)
            if not CFG.kitchen_use_perfect_samplers:
                # Truncated on the right to avoid robot self collisions.
                params[0] += rng.uniform(-0.25, 0.05)
            return params

        move_to_pre_push_on_top_nsrt = NSRT("MoveToPrePushOnTop", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            moveto_prepushontop_sampler)

        # MoveToPrePullKettle
        parameters = [gripper, kettle]
        preconditions = set()
        add_effects = {LiftedAtom(AtPrePullKettle, [gripper, kettle])}
        delete_effects = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePullKettle
        option_vars = [gripper, kettle]

        def moveto_prepullkettle_sampler(state: State, goal: Set[GroundAtom],
                                         rng: np.random.Generator,
                                         objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "off"),
                              dtype=np.float32)
            if not CFG.kitchen_use_perfect_samplers:
                params[0] += rng.uniform(-0.05, 0.05)
            return params

        move_to_pre_pull_kettle_nsrt = NSRT("MoveToPrePullKettle", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            moveto_prepullkettle_sampler)

        # PushObjOnObjForward
        parameters = [gripper, kettle, surface_from, surface_to]
        preconditions = {
            LiftedAtom(AtPrePushOnTop, [gripper, kettle]),
            LiftedAtom(NotOnTop, [kettle, surface_to]),
            LiftedAtom(BurnerAhead, [surface_to, surface_from]),
            LiftedAtom(OnTop, [kettle, surface_from]),
        }
        add_effects = {LiftedAtom(OnTop, [kettle, surface_to])}
        delete_effects = {LiftedAtom(NotOnTop, [kettle, surface_to])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PushObjOnObjForward
        option_vars = [gripper, kettle, surface_to]

        def push_obj_on_obj_forward_sampler(state: State,
                                            goal: Set[GroundAtom],
                                            rng: np.random.Generator,
                                            objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the y axis.
            if CFG.kitchen_use_perfect_samplers:
                push_angle = 0.0
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        push_obj_on_obj_forward_nsrt = NSRT("PushObjOnObjForward", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            push_obj_on_obj_forward_sampler)

        # PushKettleOntoBurner
        def move_and_push_kettle_sampler(state: State, goal: Set[GroundAtom],
                                         rng: np.random.Generator,
                                         objs: Sequence[Object]) -> Array:
            move_sample = moveto_prepushontop_sampler(state, goal, rng,
                                                      objs[:2])
            push_sample = push_obj_on_obj_forward_sampler(
                state, goal, rng, objs)
            return np.concatenate([move_sample, push_sample], axis=0)

        parameters = [gripper, kettle, surface_from, surface_to]
        preconditions = {
            LiftedAtom(NotOnTop, [kettle, surface_to]),
            LiftedAtom(BurnerAhead, [surface_to, surface_from]),
        }
        add_effects = {LiftedAtom(OnTop, [kettle, surface_to])}
        delete_effects = {LiftedAtom(NotOnTop, [kettle, surface_to])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PushKettleOntoBurner
        option_vars = [gripper, kettle, surface_to]
        push_kettle_onto_burner_nsrt = NSRT("PushKettleOntoBurner", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            move_and_push_kettle_sampler)

        # PushObjOnObjForwardToBoilKettle
        parameters = [gripper, kettle, surface_from, surface_to, knob]
        preconditions = {
            LiftedAtom(AtPrePushOnTop, [gripper, kettle]),
            LiftedAtom(NotOnTop, [kettle, surface_to]),
            LiftedAtom(BurnerAhead, [surface_to, surface_from]),
            LiftedAtom(OnTop, [kettle, surface_from]),
            LiftedAtom(TurnedOn, [knob]),
            LiftedAtom(KnobAndBurnerLinked, [knob, surface_to])
        }
        add_effects = {
            LiftedAtom(OnTop, [kettle, surface_to]),
            LiftedAtom(KettleBoiling, [kettle, surface_to, knob])
        }
        delete_effects = {LiftedAtom(NotOnTop, [kettle, surface_to])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PushObjOnObjForward
        option_vars = [gripper, kettle, surface_to]
        push_obj_on_obj_forward_and_boil_kettle_nsrt = NSRT(
            "PushObjOnObjForwardAndBoilKettle", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            push_obj_on_obj_forward_sampler)

        # PushKettleOntoBurnerAndBoil
        parameters = [gripper, kettle, surface_from, surface_to, knob]
        preconditions = {
            LiftedAtom(NotOnTop, [kettle, surface_to]),
            LiftedAtom(BurnerAhead, [surface_to, surface_from]),
            LiftedAtom(TurnedOn, [knob]),
            LiftedAtom(KnobAndBurnerLinked, [knob, surface_to])
        }
        add_effects = {
            LiftedAtom(OnTop, [kettle, surface_to]),
            LiftedAtom(KettleBoiling, [kettle, surface_to, knob])
        }
        delete_effects = {LiftedAtom(NotOnTop, [kettle, surface_to])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle, OnTop
        }
        option = PushKettleOntoBurner
        option_vars = [gripper, kettle, surface_to]
        push_kettle_onto_burner_and_boil_nsrt = NSRT(
            "PushKettleOntoBurnerAndBoil", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            move_and_push_kettle_sampler)

        # PullKettle
        parameters = [gripper, kettle, surface_from, surface_to]
        preconditions = {
            LiftedAtom(AtPrePullKettle, [gripper, kettle]),
            LiftedAtom(NotOnTop, [kettle, surface_to]),
            LiftedAtom(BurnerBehdind, [surface_to, surface_from]),
            LiftedAtom(OnTop, [kettle, surface_from]),
        }
        add_effects = {LiftedAtom(OnTop, [kettle, surface_to])}
        delete_effects = {LiftedAtom(AtPrePullKettle, [gripper, kettle])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PullKettle
        option_vars = [gripper, kettle, surface_to]

        def pull_kettle_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to pull w.r.t. the negative y axis.
            if CFG.kitchen_use_perfect_samplers:
                pull_angle = np.pi + (np.pi / 16.0)
            else:
                pull_angle = rng.uniform(7 * np.pi / 8, 9 * np.pi / 8)
            return np.array([pull_angle], dtype=np.float32)

        pull_kettle_nsrt = NSRT("PullKettle", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars, pull_kettle_sampler)

        # TurnOffSwitch
        parameters = [gripper, switch]
        preconditions = {
            LiftedAtom(AtPreTurnOff, [gripper, switch]),
            LiftedAtom(TurnedOn, [switch])
        }
        add_effects = {LiftedAtom(TurnedOff, [switch])}
        delete_effects = {LiftedAtom(TurnedOn, [switch])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOffSwitch
        option_vars = [gripper, switch]

        # The same sampler is used for both on and off, since the option
        # internally takes care of the direction change.
        def switch_turn_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                # Push slightly inward.
                push_angle = np.pi / 4
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        turn_off_switch_nsrt = NSRT("TurnOffSwitch", parameters, preconditions,
                                    add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    switch_turn_sampler)

        # TurnOnSwitch
        parameters = [gripper, switch]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, switch]),
            LiftedAtom(TurnedOff, [switch])
        }
        add_effects = {LiftedAtom(TurnedOn, [switch])}
        delete_effects = {LiftedAtom(TurnedOff, [switch])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOnSwitch
        option_vars = [gripper, switch]

        turn_on_switch_nsrt = NSRT("TurnOnSwitch", parameters, preconditions,
                                   add_effects, delete_effects, ignore_effects,
                                   option, option_vars, switch_turn_sampler)

        # TurnOnKnob
        parameters = [gripper, knob]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, knob]),
            LiftedAtom(TurnedOff, [knob])
        }
        add_effects = {LiftedAtom(TurnedOn, [knob])}
        delete_effects = {LiftedAtom(TurnedOff, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOnKnob
        option_vars = [gripper, knob]

        def knob_turn_on_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                push_angle = np.pi / 9
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        turn_on_knob_nsrt = NSRT("TurnOnKnob", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, knob_turn_on_sampler)

        # MoveAndTurnOnKnob
        parameters = [gripper, knob]
        preconditions = {LiftedAtom(TurnedOff, [knob])}
        add_effects = {LiftedAtom(TurnedOn, [knob])}
        delete_effects = {LiftedAtom(TurnedOff, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveAndTurnOnKnob
        option_vars = [gripper, knob]

        def move_and_knob_turn_on_sampler(state: State, goal: Set[GroundAtom],
                                          rng: np.random.Generator,
                                          objs: Sequence[Object]) -> Array:
            turn_on_sample = knob_turn_on_sampler(state, goal, rng, objs)
            return turn_on_sample

        move_and_turn_on_knob_nsrt = NSRT("MoveAndTurnOnKnob", parameters,
                                          preconditions, add_effects,
                                          delete_effects, ignore_effects,
                                          option, option_vars,
                                          move_and_knob_turn_on_sampler)

        # TurnOnKnobAndBoilKettle
        parameters = [gripper, knob, surface_to, kettle]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, knob]),
            LiftedAtom(TurnedOff, [knob]),
            LiftedAtom(OnTop, [kettle, surface_to]),
            LiftedAtom(KnobAndBurnerLinked, [knob, surface_to])
        }
        add_effects = {
            LiftedAtom(TurnedOn, [knob]),
            LiftedAtom(KettleBoiling, [kettle, surface_to, knob])
        }
        delete_effects = {LiftedAtom(TurnedOff, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOnKnob
        option_vars = [gripper, knob]
        turn_on_knob_and_boil_kettle_nsrt = NSRT("TurnOnKnobAndBoilKettle",
                                                 parameters, preconditions,
                                                 add_effects, delete_effects,
                                                 ignore_effects, option,
                                                 option_vars,
                                                 knob_turn_on_sampler)

        # TurnOnKnobAndBoilKettle
        parameters = [gripper, knob, surface_to, kettle]
        preconditions = {
            LiftedAtom(TurnedOff, [knob]),
            LiftedAtom(OnTop, [kettle, surface_to]),
            LiftedAtom(KnobAndBurnerLinked, [knob, surface_to])
        }
        add_effects = {
            LiftedAtom(TurnedOn, [knob]),
            LiftedAtom(KettleBoiling, [kettle, surface_to, knob])
        }
        delete_effects = {LiftedAtom(TurnedOff, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveAndTurnOnKnob
        option_vars = [gripper, knob]
        # NOTE: commenting out this NSRT to make demonstrations for VLM
        # predicate invention in kitchen unimodal to make learning visual
        # predicates easier (if we move kettle before turning on knob,
        # it's hard to see that the burner is actually on...)
        # move_and_turn_on_knob_and_boil_kettle_nsrt = NSRT(
        #     "MoveAndTurnOnKnobAndBoilKettle", parameters, preconditions,
        #     add_effects, delete_effects, ignore_effects, option, option_vars,
        #     move_and_knob_turn_on_sampler)
        _ = NSRT("MoveAndTurnOnKnobAndBoilKettle", parameters, preconditions,
                 add_effects, delete_effects, ignore_effects, option,
                 option_vars, move_and_knob_turn_on_sampler)

        # TurnOffKnob
        parameters = [gripper, knob]
        preconditions = {
            LiftedAtom(AtPreTurnOff, [gripper, knob]),
            LiftedAtom(TurnedOn, [knob])
        }
        add_effects = {LiftedAtom(TurnedOff, [knob])}
        delete_effects = {LiftedAtom(TurnedOn, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOffKnob
        option_vars = [gripper, knob]

        def knob_turn_off_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the y-z plane.
            if CFG.kitchen_use_perfect_samplers:
                push_angle = -np.pi / 16
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        turn_off_knob_nsrt = NSRT("TurnOffKnob", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, knob_turn_off_sampler)

        # PushOpenHingeDoor
        parameters = [gripper, hinge_door]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, hinge_door]),
            LiftedAtom(Closed, [hinge_door])
        }
        add_effects = {LiftedAtom(Open, [hinge_door])}
        delete_effects = {LiftedAtom(Closed, [hinge_door])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PushOpen
        option_vars = [gripper, hinge_door]

        def push_open_hinge_door_sampler(state: State, goal: Set[GroundAtom],
                                         rng: np.random.Generator,
                                         objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                # Push slightly inward.
                if objs[1].name == "slide":
                    push_angle = 1 * np.pi / 8
                elif objs[1].name == "microhandle":
                    # push_angle = -6 * np.pi / 8
                    push_angle = -5 * np.pi / 8
                    # push_angle = 17 * np.pi / 16
                # elif objs[1].name == "hinge2":
                #     push_angle = -9 * np.pi / 32
                elif objs[1].name == "hinge2":
                    # push_angle = -4 * np.pi / 16
                    push_angle = -7 * np.pi / 8
                else:
                    push_angle = -np.pi / 2
            else:
                if objs[1].name == "slide":
                    push_angle = rng.uniform(0, np.pi / 6)
                # elif objs[1].name == "hinge2":
                #     push_angle = rng.uniform(0, np.pi / 2)
                else:
                    push_angle = rng.uniform(np.pi, 5 * np.pi / 4)

            return np.array([push_angle], dtype=np.float32)

        push_open_hinge_door_nsrt = NSRT("PushOpenHingeDoor", parameters,
                                         preconditions, add_effects,
                                         delete_effects, ignore_effects,
                                         option, option_vars,
                                         push_open_hinge_door_sampler)

        # PushCloseHingeDoor
        parameters = [gripper, hinge_door]
        preconditions = {
            LiftedAtom(AtPreTurnOff, [gripper, hinge_door]),
            LiftedAtom(Open, [hinge_door])
        }
        add_effects = {LiftedAtom(Closed, [hinge_door])}
        delete_effects = {LiftedAtom(Open, [hinge_door])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PushClose
        option_vars = [gripper, hinge_door]

        def push_close_hinge_door_sampler(state: State, goal: Set[GroundAtom],
                                          rng: np.random.Generator,
                                          objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                # Push slightly inward.
                if objs[1].name == "slide":
                    push_angle = np.pi
                else:
                    push_angle = np.pi / 2
            else:
                if objs[1].name == "slide":
                    push_angle = rng.uniform(2 * np.pi / 3, 4 * np.pi / 3)
                else:
                    push_angle = rng.uniform(np.pi / 3, 2 * np.pi / 3)

            return np.array([push_angle], dtype=np.float32)

        push_close_hinge_door_nsrt = NSRT("PushCloseHingeDoor", parameters,
                                          preconditions, add_effects,
                                          delete_effects, ignore_effects,
                                          option, option_vars,
                                          push_close_hinge_door_sampler)

        # New NSRTs for banana search
        # MoveToObservePosition
        parameters = [gripper, container]
        preconditions = {LiftedAtom(Open, [container])}
        add_effects = {LiftedAtom(AtPreObserve, [gripper, container])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToObservePosition
        option_vars = [gripper, container]

        def moveto_observe_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            _, container = objs
            # Calculate observation position (in front of container)
            params = np.array([0.0, -0.2, 0.0], dtype=np.float32)
            return params

        move_to_observe_nsrt = NSRT("MoveToObservePosition", parameters,
                                   preconditions, add_effects,
                                   delete_effects, ignore_effects,
                                   option, option_vars,
                                   moveto_observe_sampler)

        # ObserveContainer - unified observe NSRT, always assume every object can be found during planning
        # But actually decide whether a certain object is found based on actual situation during execution
        parameters = [gripper, container, banana, mug, sponge, tea]
        preconditions = {
            LiftedAtom(AtPreObserve, [gripper, container]),
            LiftedAtom(Open, [container]),
            LiftedAtom(CanObserve, [container]),
            LiftedAtom(NotObserved, [container])
        }
        # Assume always find all objects during planning (optimistic planning);
        # execution-time ObserveContainer option will set found flags based on reality.
        add_effects = {
            LiftedAtom(BananaFound, [banana]),
            LiftedAtom(SpongeFound, [sponge]),
            LiftedAtom(MugFound, [mug]),
            LiftedAtom(TeaFound, [tea]),
            LiftedAtom(Observed, [container])
        }
        delete_effects = {LiftedAtom(NotObserved, [container])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = ObserveContainer
        option_vars = [gripper, container, banana, mug, sponge, tea]

        def observe_container_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            return np.array([0.0], dtype=np.float32)

        observe_container_nsrt = NSRT("ObserveContainer", parameters,
                                     preconditions, add_effects,
                                     delete_effects, ignore_effects,
                                     option, option_vars,
                                     observe_container_sampler)

        # MoveToPrePickUpBanana
        parameters = [gripper, banana, container]
        preconditions = {
            LiftedAtom(Open, [container]),
            # Remove ContainsBanana precondition - only depend on BananaFound
            LiftedAtom(Observed, [container]),
            LiftedAtom(BananaFound, [banana])
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, banana, container])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePickUp
        option_vars = [gripper, banana, container]

        def move_to_pre_pick_up_sampler(state: State, goal: Set[GroundAtom],
                                        rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            gripper, object, container = objs
            params = np.array(KitchenEnv.get_pre_pick_delta_pos((object, container)),
                              dtype=np.float32)
            return params

        move_to_pre_pick_up_banana_nsrt = NSRT("MoveToPrePickUpBanana", parameters, preconditions,
                                        add_effects, delete_effects, ignore_effects,
                                        option, option_vars, move_to_pre_pick_up_sampler)

        # PickBanana
        parameters = [gripper, banana, container]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, banana, container]),
            LiftedAtom(Open, [container]),
            # Remove ContainsBanana precondition - only depend on BananaFound
            LiftedAtom(Observed, [container]),
            LiftedAtom(BananaFound, [banana])
        }
        add_effects = {LiftedAtom(BananaPickedUp, [gripper, banana])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = Pick
        option_vars = [gripper, banana, container]
        
        def pick_sampler(state: State, goal: Set[GroundAtom],
                        rng: np.random.Generator,
                        objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            gripper, banana, container = objs
            params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return params

        pick_banana_nsrt = NSRT("PickBanana", parameters, preconditions,
                        add_effects, delete_effects, ignore_effects,
                        option, option_vars, pick_sampler)

        # MoveToPrePickUpMug
        parameters = [gripper, mug, container]
        preconditions = {
            LiftedAtom(Open, [container]),
            LiftedAtom(Observed, [container]),
            LiftedAtom(MugFound, [mug]),
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, mug, container])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePickUp
        option_vars = [gripper, mug, container]

        move_to_pre_pick_up_mug_nsrt = NSRT("MoveToPrePickUpMug", parameters, preconditions,
                                            add_effects, delete_effects, ignore_effects,
                                            option, option_vars, move_to_pre_pick_up_sampler)

        # PickMug
        parameters = [gripper, mug, container]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, mug, container]),
            LiftedAtom(Open, [container]),
            LiftedAtom(Observed, [container]),
            LiftedAtom(MugFound, [mug]),
        }
        add_effects = {LiftedAtom(MugPickedUp, [gripper, mug])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = Pick
        option_vars = [gripper, mug, container]
        
        def pick_mug_sampler(state: State, goal: Set[GroundAtom],
                            rng: np.random.Generator,
                            objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return params

        pick_mug_nsrt = NSRT("PickMug", parameters, preconditions,
                            add_effects, delete_effects, ignore_effects,
                            option, option_vars, pick_mug_sampler)

        # PickTea
        parameters = [gripper, tea, container]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, tea, container]),
            LiftedAtom(Open, [container]),
            LiftedAtom(Observed, [container]),
            LiftedAtom(TeaFound, [tea]),
        }
        add_effects = {LiftedAtom(TeaPickedUp, [gripper, tea])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = Pick
        option_vars = [gripper, tea, container]
        
        def pick_tea_sampler(state: State, goal: Set[GroundAtom],
                            rng: np.random.Generator,
                            objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return params

        pick_tea_nsrt = NSRT("PickTea", parameters, preconditions,
                            add_effects, delete_effects, ignore_effects,
                            option, option_vars, pick_tea_sampler)

        # MoveToTargetBanana
        parameters = [gripper, banana, origin, destination]
        preconditions = {
            LiftedAtom(BananaPickedUp, [gripper, banana]),
            LiftedAtom(AtPrePickUp, [gripper, banana, origin]),
        }
        # TODO: Need to add the TABLETOP case
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, banana, destination])}
        delete_effects: {LiftedAtom(AtPrePickUp, [gripper, banana, origin])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToTarget
        option_vars = [gripper, banana, origin, destination]

        def move_to_target_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            gripper, object, origin, destination = objs
            params = np.array(KitchenEnv.get_pre_pick_delta_pos([object, destination]),
                            dtype=np.float32)
            return params

        move_to_target_banana_nsrt = NSRT("MoveToTargetBanana", parameters, preconditions,
                                   add_effects, delete_effects, ignore_effects,
                                   option, option_vars, move_to_target_sampler)

        # MoveToTargetMug
        parameters = [gripper, mug, origin, destination]
        preconditions = {
            LiftedAtom(MugPickedUp, [gripper, mug]),
            LiftedAtom(AtPrePickUp, [gripper, mug, origin]),
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, mug, destination])}
        delete_effects = {LiftedAtom(AtPrePickUp, [gripper, mug, origin])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToTarget
        option_vars = [gripper, mug, origin, destination]

        def move_to_target_mug_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            gripper, mug, origin, destination = objs
            params = np.array(KitchenEnv.get_pre_pick_delta_pos([mug, destination]),
                            dtype=np.float32)
            return params

        move_to_target_mug_nsrt = NSRT("MoveToTargetMug", parameters, preconditions,
                                      add_effects, delete_effects, ignore_effects,
                                      option, option_vars, move_to_target_mug_sampler)

        # MoveToTargetTea
        parameters = [gripper, tea, origin, destination]
        preconditions = {
            LiftedAtom(TeaPickedUp, [gripper, tea]),
            LiftedAtom(AtPrePickUp, [gripper, tea, origin]),
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, tea, destination])}
        delete_effects = {LiftedAtom(AtPrePickUp, [gripper, tea, origin])}
        ignore_effects = {AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle}
        option = MoveToTarget
        option_vars = [gripper, tea, origin, destination]

        def move_to_target_tea_sampler(state: State, goal: Set[GroundAtom],
                                       rng: np.random.Generator,
                                       objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            gripper, tea, origin, destination = objs
            params = np.array(KitchenEnv.get_pre_pick_delta_pos([tea, destination]),
                            dtype=np.float32)
            return params

        move_to_target_tea_nsrt = NSRT("MoveToTargetTea", parameters, preconditions,
                                       add_effects, delete_effects, ignore_effects,
                                       option, option_vars, move_to_target_tea_sampler)

        # MoveToPrePickUpSponge
        parameters = [gripper, sponge, container]
        preconditions = {
            LiftedAtom(Open, [container]),
            LiftedAtom(Observed, [container]),
            LiftedAtom(SpongeFound, [sponge]),
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, sponge, container])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePickUp
        option_vars = [gripper, sponge, container]

        move_to_pre_pick_up_sponge_nsrt = NSRT("MoveToPrePickUpSponge", parameters, preconditions,
                                               add_effects, delete_effects, ignore_effects,
                                               option, option_vars, move_to_pre_pick_up_sampler)

        # MoveToPrePickUpTea
        parameters = [gripper, tea, container]
        preconditions = {
            LiftedAtom(Open, [container]),
            LiftedAtom(Observed, [container]),
            LiftedAtom(TeaFound, [tea]),
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, tea, container])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePickUp
        option_vars = [gripper, tea, container]
        
        move_to_pre_pick_up_tea_nsrt = NSRT("MoveToPrePickUpTea", parameters, preconditions,
                                            add_effects, delete_effects, ignore_effects,
                                            option, option_vars, move_to_pre_pick_up_sampler)

        # PickSponge
        parameters = [gripper, sponge, container]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, sponge, container]),
            LiftedAtom(Open, [container]),
            LiftedAtom(Observed, [container]),
            LiftedAtom(SpongeFound, [sponge]),
        }
        add_effects = {LiftedAtom(SpongePickedUp, [gripper, sponge])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = Pick
        option_vars = [gripper, sponge, container]
        
        def pick_sponge_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return params

        pick_sponge_nsrt = NSRT("PickSponge", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars, pick_sponge_sampler)

        # MoveToTargetSponge
        parameters = [gripper, sponge, origin, destination]
        preconditions = {
            LiftedAtom(SpongePickedUp, [gripper, sponge]),
            LiftedAtom(AtPrePickUp, [gripper, sponge, origin]),
        }
        add_effects = {LiftedAtom(AtPrePickUp, [gripper, sponge, destination])}
        delete_effects = {LiftedAtom(AtPrePickUp, [gripper, sponge, origin])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToTarget
        option_vars = [gripper, sponge, origin, destination]

        def move_to_target_sponge_sampler(state: State, goal: Set[GroundAtom],
                                         rng: np.random.Generator,
                                         objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            gripper, sponge, origin, destination = objs
            params = np.array(KitchenEnv.get_pre_pick_delta_pos([sponge, destination]),
                            dtype=np.float32)
            return params

        move_to_target_sponge_nsrt = NSRT("MoveToTargetSponge", parameters, preconditions,
                                          add_effects, delete_effects, ignore_effects,
                                          option, option_vars, move_to_target_sponge_sampler)

        # PlaceBanana
        parameters = [gripper, banana, destination]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, banana, destination]),
            LiftedAtom(BananaPickedUp, [gripper, banana]),
            # LiftedAtom(Open, [obj_place]),
            # LiftedAtom(ContainsBanana, [obj_place]),
            # LiftedAtom(Observed, [obj_place]),
            # LiftedAtom(BananaFound, [banana])
        }
        add_effects = {LiftedAtom(BananaOnTop, [banana, destination])}
        delete_effects = {LiftedAtom(BananaPickedUp, [gripper, banana])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = Place
        option_vars = [gripper, banana, destination]

        def place_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            params = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return params

        place_nsrt = NSRT("Place", parameters, preconditions,
                          add_effects, delete_effects, ignore_effects,
                          option, option_vars, place_sampler)

        # PlaceMugInSink
        parameters = [gripper, mug, destination]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, mug, destination]),
            LiftedAtom(MugPickedUp, [gripper, mug]),
        }
        add_effects = {LiftedAtom(MugInSink, [mug, destination])}
        delete_effects = {
            LiftedAtom(MugPickedUp, [gripper, mug]),
            LiftedAtom(AtPrePickUp, [gripper, mug, destination])
        }
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle, AtPrePickUp
        }
        option = Place
        option_vars = [gripper, mug, destination]

        place_mug_in_sink_nsrt = NSRT("PlaceMugInSink", parameters, preconditions,
                                      add_effects, delete_effects, ignore_effects,
                                      option, option_vars, place_sampler)

        # WashMug
        parameters = [gripper, sponge, mug, destination]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, sponge, destination]),
            LiftedAtom(SpongePickedUp, [gripper, sponge]),
            LiftedAtom(MugInSink, [mug, destination]),
        }
        add_effects = {
            LiftedAtom(SpongeInSink, [sponge, destination]),
            LiftedAtom(MugWashed, [sponge, destination]),
        }
        delete_effects = {
            LiftedAtom(SpongePickedUp, [gripper, sponge]),
            LiftedAtom(AtPrePickUp, [gripper, sponge, destination])
        }
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle, AtPrePickUp
        }
        option = WashMug
        option_vars = [gripper, sponge, mug, destination]

        wash_mug_nsrt = NSRT("WashMug", parameters, preconditions,
                                         add_effects, delete_effects, ignore_effects,
                                         option, option_vars, place_sampler)

        # MakeTea
        parameters = [gripper, tea, mug, destination]
        preconditions = {
            LiftedAtom(AtPrePickUp, [gripper, tea, destination]),
            LiftedAtom(TeaPickedUp, [gripper, tea]),
            LiftedAtom(MugInSink, [mug, destination]),
        }
        add_effects = {
            LiftedAtom(TeaInSink, [tea, destination]),
            LiftedAtom(TeaMade, [tea, destination])
            }
        delete_effects = {
            LiftedAtom(TeaPickedUp, [gripper, tea]),
            LiftedAtom(AtPrePickUp, [gripper, tea, destination])
            }
        ignore_effects = {AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle, AtPrePickUp}
        option = MakeTea
        option_vars = [gripper, tea, mug, destination]

        make_tea_nsrt = NSRT("MakeTea", parameters, preconditions,
                             add_effects, delete_effects, ignore_effects,
                             option, option_vars, place_sampler)

        # OpenContainer
        # parameters = [gripper, container]
        # preconditions = {
        #     LiftedAtom(AtPreTurnOn, [gripper, container]),
        #     LiftedAtom(Closed, [container])
        # }
        # add_effects = {LiftedAtom(Open, [container])}
        # delete_effects = {LiftedAtom(Closed, [container])}
        # ignore_effects = {
        #     AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        # }
        # option = OpenContainer
        # option_vars = [gripper, container]

        # def open_container_sampler(state: State, goal: Set[GroundAtom],
        #                                  rng: np.random.Generator,
        #                                  objs: Sequence[Object]) -> Array:
        #     del state, goal  # unused
        #     print(f"open_container_sampler called with container: {objs[1].name}")
        #     # Sample a direction to push w.r.t. the x axis.
        #     if CFG.kitchen_use_perfect_samplers:
        #         # Push slightly inward.
        #         if objs[1].name == "slide":
        #             push_angle = 1 * np.pi / 8
        #         elif objs[1].name == "microhandle":
        #             push_angle = np.pi / 2
        #         elif objs[1].name == "hinge1":
        #             push_angle = np.pi / 2
        #         else:
        #             push_angle = -np.pi / 2
        #     else:
        #         if objs[1].name == "slide":
        #             push_angle = rng.uniform(0, np.pi / 6)
        #         elif objs[1].name == "microhandle":
        #             push_angle = rng.uniform(np.pi / 2, np.pi)
        #         elif objs[1].name == "hinge1":
        #             push_angle = rng.uniform(np.pi / 2, np.pi)
        #         else:
        #             push_angle = rng.uniform(np.pi, 5 * np.pi / 4)

        #     print(f"open_container_sampler returning: {push_angle}")
        #     return np.array([push_angle], dtype=np.float32)

        # open_container_nsrt = NSRT("OpenContainer", parameters,
        #                           preconditions, add_effects,
        #                           delete_effects, ignore_effects,
        #                           option, option_vars,
        #                           open_container_sampler)



        # Add the relevant NSRTs to the set to be returned.
        # NOTE: if kitchen_use_combo_move_nsrts is set to true, we use NSRTs
        # that couple moving with other actions implicitly (i.e., move NSRTs
        # aren't separate in any way). This is useful for e.g. in VLM predicate
        # invention since moving places doesn't really turn on any predicates
        # that are easily-classified.
        if not CFG.kitchen_use_combo_move_nsrts:
            nsrts.add(move_to_pre_push_on_top_nsrt)
            nsrts.add(push_obj_on_obj_forward_nsrt)
            nsrts.add(push_obj_on_obj_forward_and_boil_kettle_nsrt)
            nsrts.add(turn_on_knob_nsrt)
            nsrts.add(turn_on_knob_and_boil_kettle_nsrt)
        else:
            nsrts.add(push_kettle_onto_burner_nsrt)
            nsrts.add(push_kettle_onto_burner_and_boil_nsrt)
            # nsrts.add(move_and_turn_on_knob_and_boil_kettle_nsrt)
            nsrts.add(move_and_turn_on_knob_nsrt)
        nsrts.add(move_to_pre_pull_kettle_nsrt)
        nsrts.add(pull_kettle_nsrt)
        nsrts.add(turn_off_switch_nsrt)
        nsrts.add(turn_on_switch_nsrt)
        nsrts.add(turn_off_knob_nsrt)
        nsrts.add(push_open_hinge_door_nsrt)
        nsrts.add(move_to_pre_turn_on_nsrt)
        nsrts.add(move_to_pre_turn_off_nsrt)
        nsrts.add(push_close_hinge_door_nsrt)
        
        # Add new banana search NSRTs
        nsrts.add(move_to_observe_nsrt)
        nsrts.add(observe_container_nsrt)

        nsrts.add(move_to_pre_pick_up_banana_nsrt)
        nsrts.add(pick_banana_nsrt)
        nsrts.add(move_to_pre_pick_up_mug_nsrt)
        nsrts.add(pick_mug_nsrt)
        nsrts.add(move_to_pre_pick_up_tea_nsrt)
        nsrts.add(pick_tea_nsrt)
        nsrts.add(move_to_pre_pick_up_sponge_nsrt)
        nsrts.add(pick_sponge_nsrt)
        nsrts.add(move_to_target_banana_nsrt)
        nsrts.add(move_to_target_mug_nsrt)
        nsrts.add(move_to_target_sponge_nsrt)
        nsrts.add(move_to_target_tea_nsrt)
        nsrts.add(place_nsrt)
        nsrts.add(place_mug_in_sink_nsrt)
        nsrts.add(wash_mug_nsrt)
        nsrts.add(make_tea_nsrt)
        return nsrts
