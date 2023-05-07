from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Iterable

import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from objects.coordinate import Coordinate, CoordinateList
from logic.goals.unit_goal import DigGoal
from utils.distances import (
    init_empty_positions,
    get_min_distances_between_positions,
)
from utils.positions import append_positions, positions_to_set
from config import CONFIG

if TYPE_CHECKING:
    from objects.actors.unit import Unit
    from objects.actors.factory import Factory


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
    player_units: list[Unit]
    opp_units: list[Unit]
    player_factories: list[Factory]
    opp_factories: list[Factory]

    def __post_init__(self) -> None:
        self.size = self.rubble.shape[0]  # Board is square

        self.ice_positions = np.transpose(np.where(self.ice))
        self.ore_positions = np.transpose(np.where(self.ore))
        self.resource_positions = append_positions(self.ice_positions, self.ore_positions)
        self.resource_positions_set = positions_to_set(self.resource_positions)
        self._is_rubble_no_resource = self._get_is_rubble_no_resource()
        self.rubble_positions = self._get_rubble_positions()

        self.ice_positions_set = {tuple(pos) for pos in self.ice_positions}
        self.ore_positions_set = {tuple(pos) for pos in self.ore_positions}

        self.is_empty_array = self._get_is_empty_array()

        self.player_factory_tiles_set = {tuple(pos) for factory in self.player_factories for pos in factory.positions}
        self.opp_factory_tiles_set = {tuple(pos) for factory in self.opp_factories for pos in factory.positions}
        self.player_lights = [light for light in self.player_units if light.is_light]
        self.player_heavies = [heavy for heavy in self.player_units if heavy.is_heavy]
        self.opp_lights = [light for light in self.opp_units if light.is_light]
        self.opp_heavies = [heavy for heavy in self.opp_units if heavy.is_heavy]

        self._pos_tuple_to_player_unit = defaultdict(lambda: None, {unit.tc.xy: unit for unit in self.player_units})
        self._pos_tuple_to_opp_unit = defaultdict(lambda: None, {unit.tc.xy: unit for unit in self.opp_units})

        valid_tiles_set = {(x, y) for x in range(self.size) for y in range(self.size)}
        self.valid_tiles_set = valid_tiles_set - self.opp_factory_tiles_set

        for factory in self.player_factories + self.opp_factories:
            factory.set_positions(self)

        assigned_units = set().union(*[factory.units for factory in self.player_factories])

        for unit in self.player_units:
            if unit not in assigned_units:
                unit.remove_goal_and_private_action_plan()
                closest_factory = min(self.player_factories, key=lambda f: unit.tc.distance_to(f.center_tc))
                closest_factory.add_unit(unit)

        self.player_factory_tiles = self._get_factory_tiles(self.player_factories)
        self.opp_factory_tiles = self._get_factory_tiles(self.opp_factories)
        self.player_lichen_tiles = self._get_lichen_coordinates_from_factories(factories=self.player_factories)
        self.opp_lichen_tiles = self._get_lichen_coordinates_from_factories(factories=self.opp_factories)
        self.player_factories_or_lichen_tiles = self.player_factory_tiles + self.player_lichen_tiles
        self._strain_id_to_index = {factory.strain_id: i for i, factory in enumerate(self.player_factories)}

        if self.player_factory_tiles:
            distance_to_player_factory_tiles = self._get_dis_to_player_factory_tiles_array()
            self._min_distance_to_all_player_factories = np.min(distance_to_player_factory_tiles, axis=2)
            self._min_distance_to_player_factory = np.min(self._min_distance_to_all_player_factories, axis=2)

            self._closest_player_factory = np.argmin(self._min_distance_to_all_player_factories, axis=2)
            self._closest_player_factory_tile = np.argmin(
                distance_to_player_factory_tiles.reshape(self.size, self.size, -1, order="F"), axis=2
            )

        if self.opp_factory_tiles:
            distance_to_opp_factory_tiles = self._get_dis_to_opp_factory_tiles_array()
            min_distance_to_opp_player_factories = np.min(distance_to_opp_factory_tiles, axis=2)
            self._min_distance_to_opp_factory = np.min(min_distance_to_opp_player_factories, axis=2)

        self._min_distance_to_opp_heavies = self._get_min_dis_tiles_to_opponent_heavies()
        self._min_distance_to_player_factory_or_lichen = self._get_min_dis_tiles_to_positions(
            self.player_factories_or_lichen_tiles
        )

        if self.player_factory_tiles and self.opp_factory_tiles:
            self.resource_ownership = self._get_resource_ownership()
            self.minable_ice_positions_set = self._get_minable_positions(self.ice_positions_set)
            self.minable_ore_positions_set = self._get_minable_positions(self.ore_positions_set)
            self.minable_positions_set = self.minable_ice_positions_set | self.minable_ore_positions_set

    def _get_minable_positions(self, resource_positoins: Iterable[tuple]) -> set[tuple]:
        min_ownership_required = CONFIG.MIN_OWNERSHIP_REQUIRED_FOR_MINING
        return {tuple(pos) for pos in resource_positoins if self.resource_ownership[pos] > min_ownership_required}

    def _get_resource_ownership(self) -> dict[tuple, float]:
        resource_positions = self.resource_positions
        distances_to_player = get_min_distances_between_positions(resource_positions, self.player_factory_positions)
        distances_to_opp = get_min_distances_between_positions(resource_positions, self.opp_factory_positions)
        percent_owner_ship = distances_to_opp / (distances_to_opp + distances_to_player)
        return {tuple(pos): percent for pos, percent in zip(resource_positions, percent_owner_ship)}

    @property
    def unspreadable_positions(self) -> np.ndarray:
        return append_positions(self.resource_positions, self.factory_positions)

    @property
    def factory_positions(self) -> np.ndarray:
        return append_positions(self.player_factory_positions, self.opp_factory_positions)

    @property
    def player_factory_positions(self) -> np.ndarray:
        if not self.player_factory_tiles:
            return init_empty_positions()

        return self.player_factory_tiles.to_positions()

    @property
    def opp_factory_positions(self) -> np.ndarray:
        if not self.opp_factory_tiles:
            return init_empty_positions()

        return self.opp_factory_tiles.to_positions()

    def are_empty_postions(self, positions: np.ndarray) -> np.ndarray:
        return self.is_empty_array[positions[:, 0], positions[:, 1]]

    def are_rubble_positions(self, positions: np.ndarray) -> np.ndarray:
        return self._is_rubble_no_resource[positions[:, 0], positions[:, 1]]

    def _get_min_dis_tiles_to_opponent_heavies(self) -> np.ndarray:
        heavy_positions = np.array([[heavy.x, heavy.y] for heavy in self.opp_heavies])
        return self._get_min_distance_tiles_to_positions(heavy_positions)

    def _get_min_dis_tiles_to_positions(self, positions: CoordinateList) -> np.ndarray:
        tiles_positions = positions.to_positions()
        return self._get_min_distance_tiles_to_positions(tiles_positions)

    def _get_min_distance_tiles_to_positions(self, tiles_positions: np.ndarray) -> np.ndarray:
        if not tiles_positions.shape[0]:
            return np.full((self.size, self.size), np.inf)

        tiles_positions = tiles_positions.transpose()
        tiles_xy = self._get_tiles_xy_array()

        diff = tiles_xy[..., None] - tiles_positions[None, ...]
        abs_dis_xy = np.abs(diff)
        abs_dis = np.sum(abs_dis_xy, axis=2)
        return np.min(abs_dis, axis=2)

    def _get_dis_to_player_factory_tiles_array(self) -> np.ndarray:
        tiles_xy = self._get_tiles_xy_array()
        player_factory_tiles_xy = self._get_player_factory_tiles_array()
        return self._get_distance_tiles_factories(tiles_xy, player_factory_tiles_xy)

    def _get_dis_to_opp_factory_tiles_array(self) -> np.ndarray:
        tiles_xy = self._get_tiles_xy_array()
        opp_factory_tiles_xy = self._get_opp_factory_tiles_array()
        return self._get_distance_tiles_factories(tiles_xy, opp_factory_tiles_xy)

    def _get_tiles_xy_array(self) -> np.ndarray:
        """dimensions of (x: size, y: size, xy: 2)"""
        tiles_x = np.arange(self.size)
        tiles_y = np.arange(self.size)
        xx, yy = np.meshgrid(tiles_x, tiles_y, indexing="ij")
        return np.stack([xx, yy], axis=2)

    def _get_player_factory_tiles_array(self) -> np.ndarray:
        """dimensions (xy: 2, factory_tile: 9, factory: nr_factories)"""
        factory_tiles_pos = np.array([[pos for pos in factory.positions] for factory in self.player_factories])
        return factory_tiles_pos.transpose()

    def _get_opp_factory_tiles_array(self) -> np.ndarray:
        """dimensions (xy: 2, factory_tile: 9, factory: nr_factories)"""
        factory_tiles_pos = np.array([[pos for pos in factory.positions] for factory in self.opp_factories])
        return factory_tiles_pos.transpose()

    def _get_distance_tiles_factories(self, tiles_xy: np.ndarray, player_factories_xy: np.ndarray) -> np.ndarray:
        diff = tiles_xy[..., None, None] - player_factories_xy[None, None, ...]
        abs_dis = np.abs(diff)
        return np.sum(abs_dis, axis=2)

    def _get_factory_tiles(self, factories: Iterable[Factory]) -> CoordinateList:
        return CoordinateList([c for factory in factories for c in factory.coordinates])

    def _get_lichen_coordinates_from_factories(self, factories: Iterable[Factory]) -> CoordinateList:
        strain_ids = [f.strain_id for f in factories]
        lichen_coordinates = np.argwhere(np.isin(self.lichen_strains, strain_ids) & (self.lichen > 0))
        return CoordinateList([Coordinate(*xy) for xy in lichen_coordinates])

    def _get_rubble_positions(self) -> np.ndarray:
        is_rubble_no_resource = self._get_is_rubble_no_resource()

        rubble_positions = np.argwhere(is_rubble_no_resource)
        return rubble_positions

    def _get_is_rubble_no_resource(self) -> np.ndarray:
        is_rubble = self.rubble > 0
        is_no_ice = self.ice == 0
        is_no_ore = self.ore == 0

        return is_rubble & is_no_ice & is_no_ore

    def _get_is_empty_array(self) -> np.ndarray:
        is_no_rubble = self.rubble == 0
        is_no_ice = self.ice == 0
        is_no_ore = self.ore == 0
        is_no_lichen = self.lichen == 0
        is_no_factory = self.factory_occupancy_map == -1

        is_empty = is_no_rubble & is_no_ice & is_no_ore & is_no_lichen & is_no_factory
        return is_empty

    def get_min_distance_to_player_factory_or_lichen(self, c: Coordinate) -> int:
        return self._min_distance_to_player_factory_or_lichen[c.x, c.y]

    def is_valid_c_for_player(self, c: Coordinate) -> bool:
        return c.xy in self.valid_tiles_set

    def is_player_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.player_factory_tiles_set

    def is_opponent_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.opp_factory_tiles_set

    def is_off_the_board(self, c: Coordinate) -> bool:
        return not self.is_off_the_board(c=c)

    def get_player_unit_on_c(self, c: Coordinate) -> Optional[Unit]:
        return self._pos_tuple_to_player_unit[c.xy]

    def get_opp_unit_on_c(self, c: Coordinate) -> Optional[Unit]:
        return self._pos_tuple_to_opp_unit[c.xy]

    def get_closest_player_factory(self, c: Coordinate) -> Factory:
        closest_factory_index = self._closest_player_factory[c.x, c.y]
        return self.player_factories[closest_factory_index]

    def get_min_distance_to_any_opp_factory(self, c: Coordinate) -> int:
        return self._min_distance_to_opp_factory[c.x, c.y]

    def get_min_distance_to_any_player_factory(self, c: Coordinate) -> int:
        return self._min_distance_to_player_factory[c.x, c.y]

    def get_min_distance_to_player_factory(self, c: Coordinate, strain_id: int) -> int:
        factory_index = self._strain_id_to_index[strain_id]
        return self._min_distance_to_all_player_factories[c.x, c.y, factory_index]

    def get_closest_player_factory_tile(self, c: Coordinate) -> Coordinate:
        closest_player_factory_tile_index = self._closest_player_factory_tile[c.x, c.y]
        return self.player_factory_tiles[closest_player_factory_tile_index]

    def is_rubble_tile(self, c: Coordinate) -> bool:
        return self._is_rubble_no_resource[c.x, c.y]

    def is_opponent_lichen_tile(self, c: Coordinate) -> bool:
        if self.lichen[c.xy] == 0:
            return False

        lichen_strain = self.lichen_strains[c.xy]
        return lichen_strain in {f.strain_id for f in self.opp_factories}

    def get_min_dis_to_opp_heavy(self, c: Coordinate) -> float:
        return self._min_distance_to_opp_heavies[c.x, c.y]

    def is_opponent_heavy_on_tile(self, c: Coordinate) -> bool:
        return self.get_min_dis_to_opp_heavy(c) == 0

    def get_neighboring_opponents(self, c: Coordinate) -> list[Unit]:
        neighboring_opponents = []

        for tc in c.neighbors:
            if tc.xy == c.xy:
                continue

            opponent_on_c = self.get_opp_unit_on_c(tc)
            if opponent_on_c:
                neighboring_opponents.append(opponent_on_c)

        return neighboring_opponents

    @property
    def positions_in_dig_goals(self) -> set[tuple]:
        return {unit.goal.dig_c.xy for unit in self.player_units if isinstance(unit.goal, DigGoal)}

    @property
    def positions_in_heavy_dig_goals(self) -> set[tuple]:
        return {unit.goal.dig_c.xy for unit in self.player_units if unit.is_heavy and isinstance(unit.goal, DigGoal)}

    def is_resource_c(self, c: Coordinate) -> bool:
        return c.xy in self.resource_positions_set

    def get_lichen_at_pos(self, pos: tuple) -> int:
        return self.lichen[pos]
