import numpy as np

from typing import Dict, Sequence, Optional, List
from dataclasses import dataclass, field


from objects.game_state import GameState
from lux.kit import obs_to_game_state
from lux.config import EnvConfig


PLAYER_TEAM_ID = 0
OPP_TEAM_ID = 1


@dataclass
class UnitPos:
    x: int
    y: int
    id: int = field(default=0)


@dataclass
class FactoryPos(UnitPos):
    p: int = field(default=500)


@dataclass
class UnitPositions:
    player_lights: Sequence[UnitPos] = field(default_factory=list)
    player_heavies: Sequence[UnitPos] = field(default_factory=list)
    opp_lights: Sequence[UnitPos] = field(default_factory=list)
    opp_heavies: Sequence[UnitPos] = field(default_factory=list)


@dataclass
class FactoryPositions:
    player: Sequence[FactoryPos] = field(default_factory=list)
    opp: Sequence[FactoryPos] = field(default_factory=list)


def _get_units(unit_positions: UnitPositions) -> Dict[str, dict]:

    player_light_units = [
        _generate_unit(team_id=PLAYER_TEAM_ID, unit_type="LIGHT", pos=pos)
        for i, pos in enumerate(unit_positions.player_lights)
    ]

    player_heavy_units = [
        _generate_unit(team_id=PLAYER_TEAM_ID, unit_type="HEAVY", pos=pos)
        for i, pos in enumerate(unit_positions.player_heavies, start=len(player_light_units))
    ]

    player_units = {unit["unit_id"]: unit for unit in player_light_units + player_heavy_units}

    opp_light_units = [
        _generate_unit(team_id=OPP_TEAM_ID, unit_type="LIGHT", pos=pos)
        for i, pos in enumerate(unit_positions.opp_lights, start=len(player_units))
    ]

    opp_heavy_units = [
        _generate_unit(team_id=OPP_TEAM_ID, unit_type="HEAVY", pos=pos)
        for i, pos in enumerate(unit_positions.opp_heavies, start=len(player_units) + len(opp_light_units))
    ]

    opp_units = {unit["unit_id"]: unit for unit in opp_light_units + opp_heavy_units}

    return {f"player_{PLAYER_TEAM_ID}": player_units, f"player_{OPP_TEAM_ID}": opp_units}


def _generate_unit(
    team_id: int,
    unit_type: str,
    pos: UnitPos,
    power: int = 500,
    cargo: Optional[dict] = None,
    action_queue: Optional[List[np.ndarray]] = None,
) -> dict:
    if cargo is None:
        cargo = dict(ice=0, ore=0, water=0, metal=0)

    if action_queue is None:
        action_queue = []

    unit_id_str = f"unit_{pos.id}"

    np_pos = np.array([pos.x, pos.y])

    return dict(
        team_id=team_id,
        unit_id=unit_id_str,
        power=power,
        unit_type=unit_type,
        pos=np_pos,
        cargo=cargo,
        action_queue=[],
        prev_step_goal=None,
    )


def _get_teams(player_factory_strains: List[int], opp_factory_strains: List[int]) -> Dict[str, dict]:
    player_0 = dict(
        team_id=PLAYER_TEAM_ID,
        faction=" AlphaStrike",
        water=0,
        metal=0,
        factories_to_place=0,
        factory_strains=player_factory_strains,
        place_first=True,
        bid=0,
    )
    player_1 = dict(
        team_id=OPP_TEAM_ID,
        faction=" AlphaStrike",
        water=0,
        metal=0,
        factories_to_place=0,
        factory_strains=opp_factory_strains,
        place_first=False,
        bid=0,
    )

    return dict(player_0=player_0, player_1=player_1)


@dataclass
class RubbleTile:
    x: int
    y: int
    rubble: int


@dataclass
class ResourceTile:
    x: int
    y: int


@dataclass
class LichenTile:
    x: int
    y: int
    lichen: int
    strain: int


@dataclass
class Tiles:
    rubble: Sequence[RubbleTile] = field(default_factory=list)
    ore: Sequence[ResourceTile] = field(default_factory=list)
    ice: Sequence[ResourceTile] = field(default_factory=list)
    lichen: Sequence[LichenTile] = field(default_factory=list)


def _get_board(tiles: Tiles) -> dict:
    board_shape = (48, 48)
    rubble = np.zeros(board_shape)
    ore = np.zeros(board_shape)
    ice = np.zeros(board_shape)
    lichen = np.zeros(board_shape)
    lichen_strains = np.full(board_shape, -1)

    for rubble_tile in tiles.rubble:
        rubble[rubble_tile.x, rubble_tile.y] = rubble_tile.rubble

    for ore_tile in tiles.ore:
        ore[ore_tile.x, ore_tile.y] = 1

    for ice_tile in tiles.ice:
        ice[ice_tile.x, ice_tile.y] = 1

    for lichen_tile in tiles.lichen:
        lichen[lichen_tile.x, lichen_tile.y] = lichen_tile.lichen
        lichen_strains[lichen_tile.x, lichen_tile.y] = lichen_tile.strain

    # TODO, make valid spawn correct
    valid_spawns_mask = np.full(board_shape, False)
    # TODO make factories per team correct
    factories_per_team = 1

    return dict(
        rubble=rubble,
        ore=ore,
        ice=ice,
        lichen=lichen,
        lichen_strains=lichen_strains,
        valid_spawns_mask=valid_spawns_mask,
        factories_per_team=factories_per_team,
    )


def get_factories(factory_positions: FactoryPositions) -> Dict[str, dict]:

    player_factories = {
        f"factory_{pos.id}": _generate_factory(PLAYER_TEAM_ID, pos=pos) for pos in factory_positions.player
    }

    opp_factories = {f"factory_{pos.id}": _generate_factory(OPP_TEAM_ID, pos=pos) for pos in factory_positions.opp}

    return {f"player_{PLAYER_TEAM_ID}": player_factories, f"player_{OPP_TEAM_ID}": opp_factories}


def _generate_factory(team_id: int, pos: FactoryPos, cargo: Optional[dict] = None) -> dict:

    if cargo is None:
        cargo = dict(ice=0, ore=0, water=0, metal=0)

    np_pos = np.array([pos.x, pos.y])

    return dict(
        pos=np_pos,
        power=pos.p,
        cargo=cargo,
        unit_id=f"factory_{pos.id}",
        strain_id=pos.id,
        team_id=team_id,
    )


def _get_obs(
    unit_positions: UnitPositions,
    factory_positions: FactoryPositions,
    tiles: Tiles,
    real_env_steps: int,
) -> dict:

    units = _get_units(unit_positions)
    factories = get_factories(factory_positions)

    player_factory_strains = [factory["strain_id"] for factory in factories[f"player_{PLAYER_TEAM_ID}"].values()]
    opp_factory_strains = [factory["strain_id"] for factory in factories[f"player_{OPP_TEAM_ID}"].values()]
    teams = _get_teams(player_factory_strains=player_factory_strains, opp_factory_strains=opp_factory_strains)

    board = _get_board(tiles)

    global_id = 0

    return dict(
        units=units,
        teams=teams,
        factories=factories,
        board=board,
        real_env_steps=real_env_steps,
        global_id=global_id,
    )


def get_state(
    unit_positions: Optional[UnitPositions] = None,
    factory_positions: Optional[FactoryPositions] = None,
    tiles: Optional[Tiles] = None,
    real_env_steps: int = 0,
) -> GameState:

    if unit_positions is None:
        unit_positions = UnitPositions()

    if factory_positions is None:
        factory_positions = FactoryPositions()

    if tiles is None:
        tiles = Tiles()

    obs = _get_obs(
        unit_positions=unit_positions,
        factory_positions=factory_positions,
        tiles=tiles,
        real_env_steps=real_env_steps,
    )
    env_cfg = EnvConfig()
    player = f"player_{PLAYER_TEAM_ID}"
    opp = f"player_{OPP_TEAM_ID}"
    return obs_to_game_state(obs["real_env_steps"], env_cfg, obs, player, opp, {})
