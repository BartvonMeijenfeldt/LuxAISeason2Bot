import numpy as np

from lux.kit import obs_to_game_state
from objects.game_state import GameState
from objects.unit import Unit
from objects.action import Action
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
        if factory.can_build_heavy(game_state):
            actions[factory.unit_id] = factory.build_heavy()
        elif game_state.env_steps > 800 and factory.can_water(game_state):
            actions[factory.unit_id] = factory.water()

    return actions


# def get_unit_actions(game_state: GameState) -> dict:
#     unit_actions = dict()

#     for unit in game_state.player_units:
#         closest_factory_tile = game_state.get_closest_factory_tile(c=unit.pos)
#         adjacent_to_factory = closest_factory_tile.distance_to(c=unit.pos) == 1

#         ICE_MINNIG_CUTOFF = 40

#         # previous ice mining code
#         if unit.cargo.ice < ICE_MINNIG_CUTOFF:
#             closest_ice_tile = game_state.ice_coordinates.get_closest_tile(c=unit.pos)
#             if closest_ice_tile == unit.pos:
#                 if unit.power >= unit.dig_cost + unit.action_queue_cost:
#                     unit_actions[unit.unit_id] = [unit.dig(repeat=0, n=1)]
#             else:
#                 unit_actions[unit.unit_id] = CollectIceGoal(unit_pos=unit.pos, ice_pos=closest_ice_tile).generate_plan(
#                     game_state=game_state
#                 )
#         # else if we have enough ice, we go back to the factory and dump it.
#         elif unit.cargo.ice >= ICE_MINNIG_CUTOFF:
#             direction = unit.pos.direction_to(target=closest_factory_tile)
#             if adjacent_to_factory:
#                 if unit.power >= unit.action_queue_cost:
#                     unit_actions[unit.unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
#             else:
#                 move_cost = unit.move_cost(game_state, direction)
#                 if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost:
#                     unit_actions[unit.unit_id] = [unit.move(direction, repeat=0, n=1)]

#     return unit_actions


def get_unit_actions(game_state: GameState) -> dict:
    action_plans_all = dict()

    for unit in game_state.player_units:
        if not unit.has_actions_in_queue:
            goals = unit.generate_goals(game_state=game_state)
            actions_plans = [action_plan for goal in goals for action_plan in goal.generate_action_plans(game_state)]
            action_plans_all[unit] = actions_plans

    best_action_plans = pick_best_collective_action_plan(action_plans_all)

    return best_action_plans


def pick_best_collective_action_plan(action_plans: dict[Unit, list[Action]]):
    # TODO
    best_action_plan = dict()
    for unit, action_plans in action_plans.items():
        best_action_plan[unit.unit_id] = action_plans[0]

    return best_action_plan
