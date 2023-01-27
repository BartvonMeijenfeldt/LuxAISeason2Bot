from agent.lux.kit import obs_to_game_state
from agent.lux.config import EnvConfig
from agent.lux.utils import direction_to, is_my_turn_to_place_factory
import numpy as np
from scipy.signal import convolve2d

class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            if is_my_turn_to_place_factory(self.player, game_state, step):
                spawn_loc = get_factory_spawn_loc(obs)
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = factory.build_heavy()
            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

                # previous ice mining code
                if unit.cargo.ice < 40:
                    ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= 40:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        return actions


def get_factory_spawn_loc(obs: dict) -> tuple:
    neighbouring_ice = sum_closest_numbers(obs["board"]['ice'], r=4)
    neighbouring_ice = zero_invalid_spawns(neighbouring_ice, valid_spawns=obs["board"]["valid_spawns_mask"])
    spawn_loc = get_coordinate_biggest(neighbouring_ice)
    return spawn_loc


def sum_closest_numbers(x: np.ndarray, r: int) -> np.ndarray:
    conv_array = _get_conv_filter(r=r)
    sum_closest_numbers = convolve2d(x, conv_array, mode='same')
    return sum_closest_numbers


def _get_conv_filter(r: int) -> np.ndarray:
    array_size = 2 * r + 1

    list_filter = []

    for i in range(array_size):
        v = []
        distance_i = abs(r - i)
        for j in range(array_size):
            distance_j = abs(r - j)
            if distance_i + distance_j <= r:
                v.append(1)
            else:
                v.append(0)

        list_filter.append(v)

    array = np.array(list_filter)

    return array


def zero_invalid_spawns(x: np.ndarray, valid_spawns: list) -> np.ndarray:
    x = x.copy()
    valid_spawns = np.array(valid_spawns)
    x[~valid_spawns] = 0
    return x


def get_coordinate_biggest(x: np.ndarray):
    biggest_loc_int = np.argmax(x)
    x, y = np.unravel_index(biggest_loc_int, x.shape)
    return (x, y)
