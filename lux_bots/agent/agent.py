import numpy as np

from typing import Sequence

from agent.lux.kit import obs_to_game_state
from agent.lux.config import EnvConfig
from agent.lux.factory import Factory
from agent.lux.utils import is_my_turn_to_place_factory
from agent.logic.early_setup import get_factory_spawn_loc


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
        unit_actions = dict()
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player)
        player_factories = game_state.player_factories
        # player_factory_tiles = get_factory_tiles(player_factories)
        factory_actions = get_factory_actions(player_factories, game_state)

        player_units = game_state.player_units
        ice_coordinates = game_state.ice_coordinates

        for unit in player_units:
            closest_factory_tile = game_state.get_closest_factory_tile(c=unit.pos)
            adjacent_to_factory = closest_factory_tile.distance_to(c=unit.pos) == 1

            # previous ice mining code
            if unit.cargo.ice < 40:
                closest_ice_tile = ice_coordinates.get_closest_tile(c=unit.pos)
                if closest_ice_tile.distance_to(c=unit.pos) == 0:
                    if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                        unit_actions[unit.unit_id] = [unit.dig(repeat=0, n=1)]
                else:
                    direction = unit.pos.direction_to(target=closest_ice_tile)
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        unit_actions[unit.unit_id] = [unit.move(direction, repeat=0, n=1)]
            # else if we have enough ice, we go back to the factory and dump it.
            elif unit.cargo.ice >= 40:
                direction = unit.pos.direction_to(target=closest_factory_tile)
                if adjacent_to_factory:
                    if unit.power >= unit.action_queue_cost(game_state):
                        unit_actions[unit.unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                else:
                    move_cost = unit.move_cost(game_state, direction)
                    if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                        unit_actions[unit.unit_id] = [unit.move(direction, repeat=0, n=1)]

        actions = factory_actions | unit_actions
        return actions


# def get_factory_tiles(factories: Sequence[Factory]):
#     return np.array([factory.pos for factory in factories])


def get_factory_actions(factories: Sequence[Factory], game_state) -> dict:
    actions = dict()
    for factory in factories:
        if factory.can_build_heavy(game_state):
            actions[factory.unit_id] = factory.build_heavy()
        if factory.can_water(game_state):
            actions[factory.unit_id] = factory.water()

    return actions
