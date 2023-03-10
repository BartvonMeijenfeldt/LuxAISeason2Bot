import numpy as np

from typing import List, Dict, Optional

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import is_my_turn_to_place_factory
from objects.game_state import GameState
from objects.unit import Unit
from objects.action_plan import ActionPlan
from logic.early_setup import get_factory_spawn_loc
from logic.goals.unit_goal import ActionQueueGoal, UnitGoal
from logic.action_plan_resolution import ActionPlanResolver


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
        factory_actions = self.get_factory_actions(game_state)
        unit_actions = self.get_unit_actions(game_state)

        actions = {**factory_actions, **unit_actions}
        return actions

    def get_factory_actions(self, game_state: GameState) -> Dict[str, List[np.ndarray]]:
        actions = dict()
        for factory in game_state.player_factories:
            action = factory.act(game_state=game_state)
            if isinstance(action, int):
                actions[factory.unit_id] = action

        return actions

    def get_unit_actions(self, game_state: GameState) -> Dict[str, List[np.ndarray]]:
        unit_goal_collections = {}

        for unit in game_state.player_units:
            unit_action_queue_goal = self._get_action_queue_goal(unit=unit)
            goal_collection = unit.generate_goals(game_state, unit_action_queue_goal)
            unit_goal_collections[unit] = goal_collection

        unit_goals, unit_action_plans = ActionPlanResolver(
            actor_goal_collections=unit_goal_collections, game_state=game_state
        ).resolve()

        unit_actions = {
            unit.unit_id: plan.to_action_arrays()
            for unit, plan in unit_action_plans.items()
            if self._is_new_action_plan(unit, plan)
        }

        self._update_prev_step_goals(unit_goals)

        return unit_actions

    def _get_action_queue_goal(self, unit: Unit) -> Optional[ActionQueueGoal]:
        if not unit.has_actions_in_queue:
            return None

        last_step_goal = self.prev_steps_goals[unit.unit_id]
        action_plan = ActionPlan(original_actions=unit.action_queue, unit=unit, is_set=True)
        action_queue_goal = ActionQueueGoal(unit=unit, action_plan=action_plan, goal=last_step_goal)
        return action_queue_goal

    def _is_new_action_plan(self, unit: Unit, plan: ActionPlan) -> bool:
        return plan.actions != unit.action_queue

    def _update_prev_step_goals(self, unit_goal_collections: Dict[Unit, UnitGoal]) -> None:
        self.prev_steps_goals = {unit.unit_id: goal for unit, goal in unit_goal_collections.items()}
