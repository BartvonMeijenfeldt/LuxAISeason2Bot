from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING, Dict, List, Any, Sequence

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import is_my_turn_to_place_factory
from objects.game_state import GameState
from objects.actors.factory import Factory
from objects.actors.unit import Unit
from objects.actions.unit_action_plan import ActionPlan, UnitActionPlan
from logic.early_setup import get_factory_spawn_loc
from logic.goals.goal import Goal
from logic.goals.unit_goal import ActionQueueGoal, UnitGoal
from logic.constraints import Constraints

# from logic.action_plan_resolution import ConflichtBasedPlanResolver
import datetime

if TYPE_CHECKING:
    from objects.actors.actor import Actor


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.prev_steps_goals: dict[str, UnitGoal] = {}

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player)

        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            if is_my_turn_to_place_factory(game_state, step):
                spawn_loc = get_factory_spawn_loc(obs)
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        # start = datetime.datetime.now()

        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player)

        # factory_goal_collections = self.get_factory_goal_collections(game_state)
        # unit_goal_collections = self.get_unit_goal_collections(game_state)
        # actor_goal_collections = {**factory_goal_collections, **unit_goal_collections}

        # actor_goals, actor_action_plans = self.resolve_goals(actor_goal_collections, game_state)
        actor_goals = self.resolve_goals(game_state)

        self._update_prev_step_goals(actor_goals)
        actions = self.get_actions(actor_goals)

        # end = datetime.datetime.now()

        # time_taken = (end - start) / datetime.timedelta(seconds=1)
        # if time_taken > 3:
        #     print(f"{game_state.real_env_steps}: {self.player} {time_taken: .1f}")

        return actions

    # def get_factory_goal_collections(self, game_state: GameState) -> Dict[Factory, GoalCollection]:
    #     return {factory: factory.generate_goals(game_state) for factory in game_state.player_factories}

    # def get_unit_goal_collections(self, game_state: GameState) -> Dict[Unit, GoalCollection]:
    #     unit_goal_collections: Dict[Unit, GoalCollection] = {}

    #     for unit in game_state.player_units:
    #         unit_action_queue_goal = self._get_action_queue_goal(unit=unit)
    #         goal_collection = unit.generate_goals(game_state, unit_action_queue_goal)
    #         unit_goal_collections[unit] = goal_collection

    #     return unit_goal_collections

    def resolve_goals(self, game_state: GameState) -> Dict[Actor, Goal]:
        constraints = Constraints()
        goals: Dict[Actor, Goal] = {}
        reserved_goals = set()
        importance_sorted_actors = self.get_sorted_actors(game_state)
        for actor in importance_sorted_actors:
            if isinstance(actor, Unit) and actor.has_actions_in_queue:
                action_queue_goal = self._get_action_queue_goal(unit=actor)
                goal = actor.get_best_goal(game_state, constraints, reserved_goals, action_queue_goal)
            else:
                goal = actor.get_best_goal(game_state, constraints, reserved_goals)

            constraints = constraints.add_negative_constraints(goal.action_plan.time_coordinates)
            reserved_goals.add(goal.key)
            goals[actor] = goal

        # TODO POWER GOALS

        return goals

        # actor_goals, actor_action_plans = ConflichtBasedPlanResolver(
        #     actor_goal_collections=actor_goal_collections, game_state=game_state
        # ).resolve()

        # return actor_goals

    def get_sorted_actors(self, game_state: GameState) -> List[Actor]:
        actors = game_state.player_factories + game_state.player_units
        return self._get_sorted_actors(actors)

    def _get_sorted_actors(self, actors: Sequence[Actor]) -> List[Actor]:
        return sorted(actors, key=self._actor_importance_key)

    @staticmethod
    def _actor_importance_key(actor: Actor) -> int:
        if isinstance(actor, Factory):
            return 1
        elif isinstance(actor, Unit):
            if actor.unit_type == "HEAVY":
                if actor.has_actions_in_queue:
                    return 2
                else:
                    return 3
            else:
                if actor.has_actions_in_queue:
                    return 4
                else:
                    return 5

        else:
            raise ValueError(f"{actor} is not of type Unit or Factory")

    # def resolve_goals(
    #     self, actor_goal_collections: Dict[Actor, GoalCollection], game_state: GameState
    # ) -> Tuple[Dict[Actor, Goal], Dict[Actor, ActionPlan]]:

    #     # actor_goals, actor_action_plans = ConflichtBasedPlanResolver(
    #     #     actor_goal_collections=actor_goal_collections, game_state=game_state
    #     # ).resolve()

    #     return actor_goals, actor_action_plans

    def get_actions(self, actor_goal_collections: Dict[Actor, Goal]) -> Dict[str, Any]:
        return {
            actor.unit_id: goal.action_plan.to_lux_output()
            for actor, goal in actor_goal_collections.items()
            if self._is_new_action_plan(actor, goal.action_plan)
        }

    def _get_action_queue_goal(self, unit: Unit) -> ActionQueueGoal:
        last_step_goal = self.prev_steps_goals[unit.unit_id]
        action_plan = UnitActionPlan(original_actions=unit.action_queue, actor=unit, is_set=True)
        action_queue_goal = ActionQueueGoal(unit=unit, action_plan=action_plan, goal=last_step_goal)
        return action_queue_goal

    def _is_new_action_plan(self, actor: Actor, plan: ActionPlan) -> bool:
        if isinstance(actor, Factory):
            return len(plan.actions) > 0
        elif isinstance(actor, Unit):
            return plan.actions != actor.action_queue
        else:
            raise ValueError("Actor is not Factory nor Unit!")

    def _update_prev_step_goals(self, actor_goal_collections: Dict[Actor, Goal]) -> None:
        self.prev_steps_goals = {
            unit.unit_id: goal
            for unit, goal in actor_goal_collections.items()
            if isinstance(unit, Unit) and isinstance(goal, UnitGoal)
        }
