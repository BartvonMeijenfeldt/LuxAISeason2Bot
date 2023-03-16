from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import is_my_turn_to_place_factory
from objects.game_state import GameState
from objects.actors.factory import Factory
from objects.actors.unit import Unit
from objects.actions.unit_action_plan import ActionPlan, UnitActionPlan
from logic.early_setup import get_factory_spawn_loc
from logic.goals.goal import Goal, GoalCollection
from logic.goals.unit_goal import ActionQueueGoal, UnitGoal
from logic.action_plan_resolution import ActionPlanResolver

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

        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player)

        factory_goal_collections = self.get_factory_goal_collections(game_state)
        unit_goal_collections = self.get_unit_goal_collections(game_state)
        actor_goal_collections = {**factory_goal_collections, **unit_goal_collections}

        actor_goals, actor_action_plans = self.resolve_goals(actor_goal_collections, game_state)
        self._update_prev_step_goals(actor_goals)
        actions = self.get_actions(actor_action_plans)

        return actions

    def get_factory_goal_collections(self, game_state: GameState) -> Dict[Factory, GoalCollection]:
        return {factory: factory.generate_goals(game_state) for factory in game_state.player_factories}

    def get_unit_goal_collections(self, game_state: GameState) -> Dict[Unit, GoalCollection]:
        unit_goal_collections: Dict[Unit, GoalCollection] = {}

        for unit in game_state.player_units:
            unit_action_queue_goal = self._get_action_queue_goal(unit=unit)
            goal_collection = unit.generate_goals(game_state, unit_action_queue_goal)
            unit_goal_collections[unit] = goal_collection

        return unit_goal_collections

    def resolve_goals(
        self, actor_goal_collections: Dict[Actor, GoalCollection], game_state: GameState
    ) -> Tuple[Dict[Actor, Goal], Dict[Actor, ActionPlan]]:

        actor_goals, actor_action_plans = ActionPlanResolver(
            actor_goal_collections=actor_goal_collections, game_state=game_state
        ).resolve()

        return actor_goals, actor_action_plans

    def get_actions(self, actor_action_plans: Dict[Actor, ActionPlan]) -> Dict[str, Any]:
        return {
            actor.unit_id: plan.to_lux_output()
            for actor, plan in actor_action_plans.items()
            if self._is_new_action_plan(actor, plan)
        }

    def _get_action_queue_goal(self, unit: Unit) -> Optional[ActionQueueGoal]:
        if not unit.has_actions_in_queue:
            return None

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
            raise ValueError("Actor is not Factor nor Unit!")

    def _update_prev_step_goals(self, actor_goal_collections: Dict[Actor, Goal]) -> None:
        self.prev_steps_goals = {
            unit.unit_id: goal
            for unit, goal in actor_goal_collections.items()
            if isinstance(unit, Unit) and isinstance(goal, UnitGoal)
        }
