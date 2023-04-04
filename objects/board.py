from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Iterable

import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from objects.coordinate import Coordinate, CoordinateList
from image_processing import get_islands
from positions import init_empty_positions
from distances import (
    get_distances_between_pos_and_positions,
    get_n_closests_positions_between_positions,
)

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
        self._is_rubble_no_resource = self._get_is_rubble_no_resource()

        self.ice_coordinates = self._get_ice_coordinates()
        self.ore_coordinates = self._get_ore_coordinates()
        self.rubble_coordinates = self._get_rubble_coordinates()
        self.is_empty_array = self._get_is_empty_array()
        self.empty_islands = get_islands(self.is_empty_array)

        self.player_factory_tiles_set = {c.xy for factory in self.player_factories for c in factory.coordinates}
        self.opp_factory_tiles_set = {c.xy for factory in self.opp_factories for c in factory.coordinates}
        self.player_lights = [light for light in self.player_units if light.is_light]
        self.player_heavies = [heavy for heavy in self.player_units if heavy.is_heavy]
        self.opp_lights = [light for light in self.opp_units if light.is_light]
        self.opp_heavies = [heavy for heavy in self.opp_units if heavy.is_heavy]

        self._pos_tuple_to_opponent = defaultdict(lambda: None, {opp.tc.xy: opp for opp in self.opp_units})

        self.player_nr_lights = len(self.player_lights)
        self.player_nr_heavies = len(self.player_heavies)
        with np.errstate(invalid="ignore"):
            self.player_light_heavy_ratio = np.divide(self.player_nr_lights, self.player_nr_heavies)

        valid_tiles_set = {(x, y) for x in range(self.size) for y in range(self.size)}
        self.valid_tiles_set = valid_tiles_set - self.opp_factory_tiles_set

        for factory in self.player_factories:
            factory.set_positions(self)

        self.rubble_to_remove_positions = self._get_rubble_to_remove_positions()
        self.rubble_to_remove_positions_set = set(map(tuple, self.rubble_to_remove_positions))

        self.player_factory_tiles = self._get_factory_tiles(self.player_factories)
        self.opp_factory_tiles = self._get_factory_tiles(self.opp_factories)
        self.player_lichen_tiles = self._get_lichen_coordinates_from_factories(factories=self.player_factories)
        self.opp_lichen_tiles = self._get_lichen_coordinates_from_factories(factories=self.opp_factories)
        self.player_factories_or_lichen_tiles = self.player_factory_tiles + self.player_lichen_tiles
        self.opp_factories_or_lichen_tiles = self.opp_factory_tiles + self.opp_lichen_tiles

        self._strain_id_to_index = {factory.strain_id: i for i, factory in enumerate(self.player_factories)}

        if self.player_factory_tiles:
            distance_to_player_factory_tiles = self._get_dis_to_player_factory_tiles_array()
            self._min_distance_to_all_player_factories = np.min(distance_to_player_factory_tiles, axis=2)
            self._min_distance_to_player_factory = np.min(self._min_distance_to_all_player_factories, axis=2)
            self._closest_player_factory = np.argmin(self._min_distance_to_all_player_factories, axis=2)
            self._closest_player_factory_tile = np.argmin(
                distance_to_player_factory_tiles.reshape(self.size, self.size, -1, order="F"), axis=2
            )

        self._min_distance_to_opp_heavies = self._get_min_dis_to_opponent_heavies()
        self._min_distance_to_player_factory_or_lichen = self._get_dis_to_coordinates_array(
            self.player_factories_or_lichen_tiles
        )
        self._min_distance_to_opp_factory_or_lichen = self._get_dis_to_coordinates_array(
            self.opp_factories_or_lichen_tiles
        )

    def _get_rubble_to_remove_positions(self) -> np.ndarray:
        rubble_to_remove_positions = init_empty_positions()
        for factory in self.player_factories:
            positions = factory.rubble_positions_to_clear
            rubble_to_remove_positions = np.append(rubble_to_remove_positions, positions, axis=0)

        return rubble_to_remove_positions

    def get_rubble_to_remove_positions(self, c: Coordinate, max_distance: int) -> np.ndarray:
        pos = np.array(c.xy)
        distances = get_distances_between_pos_and_positions(pos, self.rubble_to_remove_positions)
        return self.rubble_to_remove_positions[distances <= max_distance]

    def is_rubble_to_remove_c(self, c: Coordinate) -> bool:
        return c.xy in self.rubble_to_remove_positions_set

    def are_positions_empty(self, positions: np.ndarray) -> np.ndarray:
        return self.is_empty_array[positions[:, 0], positions[:, 1]]

    def _get_min_dis_to_opponent_heavies(self) -> np.ndarray:
        tiles_heavy = np.array([[heavy.x, heavy.y] for heavy in self.opp_heavies]).transpose()
        return self._get_min_distance_tiles_to_coordinates(tiles_heavy)

    def _get_dis_to_coordinates_array(self, coordinates: CoordinateList) -> np.ndarray:
        tiles_coordinates = coordinates.to_array()
        return self._get_min_distance_tiles_to_coordinates(tiles_coordinates)

    def _get_min_distance_tiles_to_coordinates(self, tiles_coordinates: np.ndarray) -> np.ndarray:
        if not tiles_coordinates.shape[0]:
            return np.full((self.size, self.size), np.inf)

        tiles_xy = self._get_tiles_xy_array()

        diff = tiles_xy[..., None] - tiles_coordinates[None, ...]
        abs_dis_xy = np.abs(diff)
        abs_dis = np.sum(abs_dis_xy, axis=2)
        return np.min(abs_dis, axis=2)

    def _get_dis_to_player_factory_tiles_array(self) -> np.ndarray:
        tiles_xy = self._get_tiles_xy_array()
        player_factory_tiles_xy = self._get_player_factory_tiles_array()
        return self._get_distance_tiles_factories(tiles_xy, player_factory_tiles_xy)

    def _get_tiles_xy_array(self) -> np.ndarray:
        """dimensions of (x: size, y: size, xy: 2)"""
        tiles_x = np.arange(self.size)
        tiles_y = np.arange(self.size)
        xx, yy = np.meshgrid(tiles_x, tiles_y, indexing="ij")
        return np.stack([xx, yy], axis=2)

    def _get_player_factory_tiles_array(self) -> np.ndarray:
        """dimensions (xy: 2, factory_tile: 9, factory: nr_factories)"""
        factory_tiles_pos = np.array([[[c.x, c.y] for c in factory.coordinates] for factory in self.player_factories])
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

    def _get_ice_coordinates(self) -> CoordinateList:
        ice_locations = np.argwhere(self.ice == 1)
        return CoordinateList([Coordinate(*xy) for xy in ice_locations])

    def _get_ore_coordinates(self) -> CoordinateList:
        ore_locations = np.argwhere(self.ore == 1)
        return CoordinateList([Coordinate(*xy) for xy in ore_locations])

    def _get_rubble_coordinates(self) -> CoordinateList:
        is_rubble_no_resource = self._get_is_rubble_no_resource()

        rubble_positions = np.argwhere(is_rubble_no_resource)
        return CoordinateList([Coordinate(*xy) for xy in rubble_positions])

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

    def get_min_distance_to_opp_factory_or_lichen(self, c: Coordinate) -> int:
        return self._min_distance_to_opp_factory_or_lichen[c.x, c.y]

    def is_valid_c_for_player(self, c: Coordinate) -> bool:
        return c.xy in self.valid_tiles_set

    def is_player_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.player_factory_tiles_set

    def is_opponent_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.opp_factory_tiles_set

    def is_on_the_board(self, c: Coordinate) -> bool:
        return self.is_x_on_the_board(c.x) and self.is_y_on_the_board(c.y)

    def is_x_on_the_board(self, x: int) -> bool:
        return 0 <= x < self.size

    def is_y_on_the_board(self, y: int) -> bool:
        return 0 <= y < self.size

    def is_off_the_board(self, c: Coordinate) -> bool:
        return not self.is_off_the_board(c=c)

    def get_opponent_on_c(self, c: Coordinate) -> Optional[Unit]:
        return self._pos_tuple_to_opponent[c.xy]

    def get_n_closest_ore_positions_to_factory(self, factory: Factory, n: int) -> np.ndarray:
        return get_n_closests_positions_between_positions(self.ore_positions, factory.positions, n)

    def get_n_closest_ice_positions_to_factory(self, factory: Factory, n: int) -> np.ndarray:
        return get_n_closests_positions_between_positions(self.ice_positions, factory.positions, n)

    def get_closest_player_factory(self, c: Coordinate) -> Factory:
        closest_factory_index = self._closest_player_factory[c.x, c.y]
        return self.player_factories[closest_factory_index]

    def get_min_distance_to_any_player_factory(self, c: Coordinate) -> int:
        return self._min_distance_to_player_factory[c.x, c.y]

    def get_min_distance_to_player_factory(self, c: Coordinate, strain_id: int) -> int:
        factory_index = self._strain_id_to_index[strain_id]
        return self._min_distance_to_all_player_factories[c.x, c.y, factory_index]

    def get_closest_player_factory_tile(self, c: Coordinate) -> Coordinate:
        closest_player_factory_tile_index = self._closest_player_factory_tile[c.x, c.y]
        return self.player_factory_tiles[closest_player_factory_tile_index]

    def is_ice_tile(self, c: Coordinate) -> bool:
        return self.ice[c.x, c.y] == 1

    def is_ore_tile(self, c: Coordinate) -> bool:
        return self.ore[c.x, c.y] == 1

    def is_rubble_tile(self, c: Coordinate) -> bool:
        return self._is_rubble_no_resource[c.x, c.y]

    def is_resource_tile(self, c: Coordinate) -> bool:
        return self.is_ice_tile(c) or self.is_ore_tile(c)

    def get_closest_ice_tile(self, c: Coordinate) -> Coordinate:
        return self.ice_coordinates.get_closest_tile(c=c)

    def get_closest_ore_tile(self, c: Coordinate) -> Coordinate:
        return self.ore_coordinates.get_closest_tile(c=c)

    def get_closest_rubble_tile(self, c: Coordinate, exclude_c: Optional[CoordinateList] = None) -> Coordinate:
        return self.rubble_coordinates.get_closest_tile(c=c, exclude_c=exclude_c)

    def get_all_closest_rubble_tiles(self, c: Coordinate, exclude_c: Optional[CoordinateList] = None) -> CoordinateList:
        return self.rubble_coordinates.get_all_closest_tiles(c=c, exclude_c=exclude_c)

    def get_n_closest_rubble_tiles(self, c: Coordinate, n: int) -> CoordinateList:
        return self.rubble_coordinates.get_n_closest_tiles(c=c, n=n)

    def get_n_closest_opp_lichen_tiles(self, c: Coordinate, n: int) -> CoordinateList:
        return self.opp_lichen_tiles.get_n_closest_tiles(c=c, n=n)

    def get_n_closest_ice_tiles(self, c: Coordinate, n: int) -> CoordinateList:
        return self.ice_coordinates.get_n_closest_tiles(c=c, n=n)

    def get_n_closest_ore_tiles(self, c: Coordinate, n: int) -> CoordinateList:
        return self.ore_coordinates.get_n_closest_tiles(c=c, n=n)

    def get_min_dis_to_opp_heavy(self, c: Coordinate) -> float:
        return self._min_distance_to_opp_heavies[c.x, c.y]

    def is_opponent_heavy_on_tile(self, c: Coordinate) -> bool:
        return self.get_min_dis_to_opp_heavy(c) == 0

    def get_neighboring_opponents(self, c: Coordinate) -> list[Unit]:
        neighboring_opponents = []

        for tc in c.neighbors:
            if tc.xy == c.xy:
                continue

            opponent_on_c = self.get_opponent_on_c(tc)
            if opponent_on_c:
                neighboring_opponents.append(opponent_on_c)

        return neighboring_opponents
