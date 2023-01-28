import numpy as np

from dataclasses import dataclass
from collections import defaultdict
from agent.lux.cargo import UnitCargo

from agent.lux.config import EnvConfig
from agent.lux.team import Team
from agent.lux.unit import Unit
from agent.objects.coordinate import Coordinate, CoordinateList
from agent.lux.factory import Factory


def process_action(action):
    return to_json(action)


def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj


def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state


def process_obs(player, game_state, step, obs):
    if step == 0:
        # at step 0 we get the entire map information
        game_state = from_json(obs)
    else:
        # use delta changes to board to update game state
        obs = from_json(obs)
        for k in obs:
            if k != "board":
                game_state[k] = obs[k]
            else:
                if "valid_spawns_mask" in obs[k]:
                    game_state["board"]["valid_spawns_mask"] = obs[k]["valid_spawns_mask"]
        for item in ["rubble", "lichen", "lichen_strains"]:
            for k, v in obs["board"][item].items():
                k = k.split(",")
                x, y = int(k[0]), int(k[1])
                game_state["board"][item][x, y] = v

    return game_state


def obs_to_game_state(step, env_cfg: EnvConfig, obs, player: str, opp: str):
    units = create_units(obs=obs, env_cfg=env_cfg)
    factories = create_factories(obs=obs, env_cfg=env_cfg)
    factory_occupancy_map = create_factory_occupancy_map(factories, obs["board"]["rubble"].shape)

    player_team = Team(**obs["teams"][player], agent=player) if player in obs["teams"] else None
    opp_team = Team(**obs["teams"][opp], agent=opp) if opp in obs["teams"] else None

    board = Board(
        rubble=obs["board"]["rubble"],
        ice=obs["board"]["ice"],
        ore=obs["board"]["ore"],
        lichen=obs["board"]["lichen"],
        lichen_strains=obs["board"]["lichen_strains"],
        factory_occupancy_map=factory_occupancy_map,
        factories_per_team=obs["board"]["factories_per_team"],
        valid_spawns_mask=obs["board"]["valid_spawns_mask"],
    )

    return GameState(
        env_cfg=env_cfg,
        env_steps=step,
        board=board,
        player_units=units[player],
        opp_units=units[opp],
        player_factories=factories[player],
        opp_factories=factories[opp],
        player_team=player_team,
        opp_team=opp_team,
    )


def create_units(obs, env_cfg) -> dict[str, list[Unit]]:
    units = defaultdict(list)

    for agent in obs["units"]:
        for unit_data in obs["units"][agent].values():
            unit_data = unit_data.copy()
            unit_data["pos"] = Coordinate(*unit_data["pos"])
            unit_data["cargo"] = UnitCargo(**unit_data["cargo"])
            unit = Unit(**unit_data, unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]], env_cfg=env_cfg)
            units[agent].append(unit)

    return units


def create_factories(obs, env_cfg) -> dict[str, list[Factory]]:
    factories = defaultdict(list)

    for agent in obs["factories"]:
        for factory_data in obs["factories"][agent].values():
            factory_data = factory_data.copy()
            factory_data["center"] = Coordinate(*factory_data["pos"])
            del factory_data["pos"]
            factory_data["cargo"] = UnitCargo(**factory_data["cargo"])
            factory = Factory(**factory_data, env_cfg=env_cfg)
            factories[agent].append(factory)

    return factories


def create_factory_occupancy_map(factories: dict[str, list[Factory]], board_shape):
    factory_occupancy_map = np.ones(board_shape, dtype=int) * -1

    for agent_factories in factories.values():
        for factory in agent_factories:
            factory_occupancy_map[factory.pos_slice] = factory.strain_id

    return factory_occupancy_map


@dataclass
class Board:
    rubble: np.ndarray
    ice: np.ndarray
    ore: np.ndarray
    lichen: np.ndarray
    lichen_strains: np.ndarray
    factory_occupancy_map: np.ndarray
    factories_per_team: int
    valid_spawns_mask: np.ndarray

    @property
    def length(self):
        return self.rubble.shape[0]

    @property
    def width(self):
        return self.rubble.shape[1]

    @property
    def ice_coordinates(self) -> CoordinateList:
        ice_locations = np.argwhere(self.ice == 1)
        return CoordinateList([Coordinate(*xy) for xy in ice_locations])


@dataclass
class GameState:
    """
    A GameState object at step env_steps. Copied from luxai_s2/state/state.py
    """

    env_steps: int
    env_cfg: dict
    board: Board
    player_units: list[Unit]
    opp_units: list[Unit]
    player_factories: list[Factory]
    opp_factories: list[Factory]
    player_team: Team
    opp_team: Team

    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for extra factory placement and + 1 for bidding step
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        else:
            return self.env_steps

    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH

    @property
    def ice_coordinates(self) -> CoordinateList:
        return self.board.ice_coordinates

    @property
    def player_factory_tiles(self) -> CoordinateList:
        return CoordinateList([c for factory in self.player_factories for c in factory.coordinates])

    def get_all_closest_factory_tiles(self, c: Coordinate) -> CoordinateList:
        return self.player_factory_tiles.get_all_closest_tiles(c)

    def get_closest_factory_tile(self, c: Coordinate) -> Coordinate:
        return self.player_factory_tiles.get_closest_tile(c)
