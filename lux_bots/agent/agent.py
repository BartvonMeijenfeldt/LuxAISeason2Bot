import numpy as np

from lux.kit import obs_to_game_state
from objects.game_state import GameState
from objects.unit import Unit
from objects.action import ActionPlan
from lux.config import EnvConfig
from lux.utils import is_my_turn_to_place_factory
from logic.early_setup import get_factory_spawn_loc


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

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
        factory_actions = get_factory_actions(game_state)
        unit_actions = get_unit_actions(game_state)

        actions = factory_actions | unit_actions
        return actions


def get_factory_actions(game_state: GameState) -> dict:
    actions = dict()
    for factory in game_state.player_factories:
        action = factory.act(game_state=game_state)
        if action:
            actions[factory.unit_id] = action

    return actions


def get_unit_actions(game_state: GameState) -> dict[str, list[np.array]]:
    action_plans_all = dict()

    for unit in game_state.player_units:
        if not unit.has_actions_in_queue:
            goals = unit.generate_goals(game_state=game_state)
            actions_plans = [
                action_plan
                for goal in goals
                for action_plan in goal.generate_action_plans(game_state)
                if action_plan.is_valid
            ]
            action_plans_all[unit] = actions_plans

    best_action_plans = pick_best_collective_action_plan(action_plans_all)
    unit_actions = {unit_id: plan.to_action_arrays() for unit_id, plan in best_action_plans.items()}

    return unit_actions


def pick_best_collective_action_plan(action_plans: dict[Unit, list[ActionPlan]]) -> dict[str, ActionPlan]:
    # TODO
    best_action_plan = dict()
    for unit, plans in action_plans.items():
        if plans:
            best_action_plan[unit.unit_id] = plans[0]

    return best_action_plan
