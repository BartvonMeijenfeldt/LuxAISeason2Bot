from __future__ import annotations
import numpy as np
import logging
import time

from typing import TYPE_CHECKING, Dict, List, Any, Sequence

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import is_my_turn_to_place_factory
from objects.game_state import GameState
from objects.actors.factory import Factory
from objects.actors.unit import Unit
from objects.actions.action_plan import ActionPlan
from objects.actions.unit_action_plan import UnitActionPlan
from logic.early_setup import get_factory_spawn_loc
from logic.goals.goal import Goal
from logic.goals.unit_goal import ActionQueueGoal, UnitGoal
from logic.goals.factory_goal import BuildLightGoal
from logic.constraints import Constraints
from logic.goal_resolution.power_availabilty_tracker import PowerAvailabilityTracker

logging.basicConfig(level=logging.INFO)


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
        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player, self.prev_steps_goals)

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
        start_time = time.time()

        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player, self.prev_steps_goals)
        actor_goals = self.resolve_goals(game_state)

        self._update_prev_step_goals(actor_goals)
        actions = self.get_actions(actor_goals)

        end_time = time.time()
        time_taken = end_time - start_time
        if time_taken < 1:
            logging.info(f"{game_state.real_env_steps}: player {game_state.player_team.team_id} {time_taken: 0.1f}")
        else:
            logging.warning(f"{game_state.real_env_steps}: player {game_state.player_team.team_id} {time_taken: 0.1f}")

        return actions

    def resolve_goals(self, game_state: GameState) -> Dict[Actor, Goal]:
        constraints = Constraints()
        goals: Dict[Actor, Goal] = {}
        reserved_goals = set()
        importance_sorted_actors = self.get_sorted_actors(game_state)

        power_tracker = self.get_power_tracker(importance_sorted_actors)
        for actor in importance_sorted_actors:
            goal = actor.get_best_goal(game_state, constraints, power_tracker, reserved_goals)
            constraints = constraints.add_negative_constraints(goal.action_plan.time_coordinates)
            power_tracker.update_power_available(
                power_requests=goal.action_plan.get_power_requests(game_state)
            )

            reserved_goals.add(goal.key)
            goals[actor] = goal

            # TODO, make units add future plans as well
            if game_state.real_env_steps < 6 and isinstance(actor, Factory):
                for t in range(game_state.real_env_steps + 1, 7):
                    goal = BuildLightGoal(actor)
                    goal.generate_and_evaluate_action_plan(game_state, constraints, power_tracker)
                    power_requests = goal.action_plan.get_power_requests(game_state)
                    for power_request in power_requests:
                        power_request.t = t
                    power_tracker.update_power_available(
                        power_requests=power_requests
                    )

        return goals

    def get_sorted_actors(self, game_state: GameState) -> List[Actor]:
        actors = game_state.player_factories + game_state.player_units
        return self._get_sorted_actors(actors)

    def _get_sorted_actors(self, actors: Sequence[Actor]) -> List[Actor]:
        return sorted(actors, key=self._actor_importance_key)

    @staticmethod
    def get_power_tracker(actors: List[Actor]) -> PowerAvailabilityTracker:
        factories = [factory for factory in actors if isinstance(factory, Factory)]
        return PowerAvailabilityTracker(factories)

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
