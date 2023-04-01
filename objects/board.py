from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Iterable, Generator, Sequence

import numpy as np

from dataclasses import dataclass

from objects.coordinate import Coordinate, CoordinateList
from lux.config import EnvConfig

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
        self.player_factory_tiles_set = {c.xy for factory in self.player_factories for c in factory.coordinates}
        self.opp_factory_tiles_set = {c.xy for factory in self.opp_factories for c in factory.coordinates}
        self.opp_lights = [light for light in self.opp_units if light.unit_type == "LIGHT"]
        self.opp_heavies = [light for light in self.opp_units if light.unit_type == "HEAVY"]

        self.player_factory_tiles = self._get_factory_tiles(self.player_factories)
        self.opp_factory_tiles = self._get_factory_tiles(self.opp_factories)
        self.player_lichen_tiles = self._get_lichen_coordinates_from_factories(factories=self.player_factories)
        self.opp_lichen_tiles = self._get_lichen_coordinates_from_factories(factories=self.opp_factories)
        self.player_factories_or_lichen_tiles = self.player_factory_tiles + self.player_lichen_tiles
        self.opp_factories_or_lichen_tiles = self.opp_factory_tiles + self.opp_lichen_tiles

        self.ice_coordinates = self._get_ice_coordinates()
        self.ore_coordinates = self._get_ore_coordinates()
        self.rubble_coordinates = self._get_rubble_coordinates()
        self._is_empty_array = self._get_is_empty_array()
        self.width = self.rubble.shape[0]
        self.length = self.rubble.shape[1]

        self._strain_id_to_index = {factory.strain_id: i for i, factory in enumerate(self.player_factories)}

        if self.player_factory_tiles:
            distance_to_player_factory_tiles = self._get_dis_to_player_factory_tiles_array()
            self._min_distance_to_all_player_factories = np.min(distance_to_player_factory_tiles, axis=2)
            self._min_distance_to_player_factory = np.min(self._min_distance_to_all_player_factories, axis=2)
            self._closest_player_factory = np.argmin(self._min_distance_to_all_player_factories, axis=2)
            self._closest_player_factory_tile = np.argmin(
                distance_to_player_factory_tiles.reshape(self.width, self.length, -1, order="F"), axis=2
            )

        self._min_distance_to_opp_heavies = self._get_min_dis_to_opponent_heavies()
        self._min_distance_to_player_factory_or_lichen = self._get_dis_to_coordinates_array(
            self.player_factories_or_lichen_tiles
        )
        self._min_distance_to_opp_factory_or_lichen = self._get_dis_to_coordinates_array(
            self.opp_factories_or_lichen_tiles
        )

    def _get_min_dis_to_opponent_heavies(self) -> np.ndarray:
        tiles_heavy = np.array([[heavy.x, heavy.y] for heavy in self.opp_heavies]).transpose()
        return self._get_min_manhattan_distance_tiles_to_coordinates(tiles_heavy)

    def _get_dis_to_coordinates_array(self, coordinates: CoordinateList) -> np.ndarray:
        tiles_coordinates = coordinates.to_array()
        return self._get_min_manhattan_distance_tiles_to_coordinates(tiles_coordinates)

    def _get_min_manhattan_distance_tiles_to_coordinates(self, tiles_coordinates: np.ndarray) -> np.ndarray:
        if not tiles_coordinates.shape[0]:
            return np.full((self.width, self.length), np.inf)

        tiles_xy = self._get_tiles_xy_array()

        diff = tiles_xy[..., None] - tiles_coordinates[None, ...]
        abs_dis_xy = np.abs(diff)
        abs_dis = np.sum(abs_dis_xy, axis=2)
        return np.min(abs_dis, axis=2)

    def _get_dis_to_player_factory_tiles_array(self) -> np.ndarray:
        tiles_xy = self._get_tiles_xy_array()
        player_factory_tiles_xy = self._get_player_factory_tiles_array()
        return self._get_manhattan_distance_tiles_factories(tiles_xy, player_factory_tiles_xy)

    def _get_tiles_xy_array(self) -> np.ndarray:
        """dimensions of (x: 48, y: 48, xy: 2)"""
        tiles_x = np.arange(self.width)
        tiles_y = np.arange(self.length)
        xx, yy = np.meshgrid(tiles_x, tiles_y, indexing="ij")
        return np.stack([xx, yy], axis=2)

    def _get_player_factory_tiles_array(self) -> np.ndarray:
        """dimensions (xy: 2, factory_tile: 9, factory: nr_factories)"""
        factory_tiles_pos = np.array([[[c.x, c.y] for c in factory.coordinates] for factory in self.player_factories])
        return factory_tiles_pos.transpose()

    def _get_manhattan_distance_tiles_factories(
        self, tiles_xy: np.ndarray, player_factories_xy: np.ndarray
    ) -> np.ndarray:

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
        is_rubble = self.rubble > 0
        is_no_ice = self.ice == 0
        is_no_ore = self.ore == 0

        rubble_locations = np.argwhere(is_rubble & is_no_ice & is_no_ore)
        return CoordinateList([Coordinate(*xy) for xy in rubble_locations])

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
        return not self.is_opponent_factory_tile(c=c) and self.is_on_the_board(c=c)

    def is_player_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.player_factory_tiles_set

    def is_opponent_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.opp_factory_tiles_set

    def is_on_the_board(self, c: Coordinate) -> bool:
        return self.is_x_on_the_board(c.x) and self.is_y_on_the_board(c.y)

    def is_x_on_the_board(self, x: int) -> bool:
        return 0 <= x < self.width

    def is_y_on_the_board(self, y: int) -> bool:
        return 0 <= y < self.length

    def is_off_the_board(self, c: Coordinate) -> bool:
        return not self.is_off_the_board(c=c)

    def get_opponent_on_c(self, c: Coordinate) -> Optional[Unit]:
        for unit in self.opp_units:
            if unit.tc.xy == c.xy:
                return unit

        return None

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

    def get_valid_neighbor_coordinates(self, c: Coordinate) -> CoordinateList:
        coordinates = [
            neighbor_c
            for neighbor_c in c.neighbors
            if self.is_on_the_board(neighbor_c) and not self.is_opponent_factory_tile(c=neighbor_c)
        ]
        return CoordinateList(coordinates)

    def get_min_dis_to_opp_heavy(self, c: Coordinate) -> float:
        return self._min_distance_to_opp_heavies[c.x, c.y]

    def get_max_nr_tiles_to_water(self, strain_id: int) -> int:
        nr_positions_can_be_spread_to = self._get_nr_positions_can_be_spread_to(strain_id)
        nr_connected_lichen = len(self._get_connected_lichen_positions(strain_id))
        nr_tiles_to_water = nr_positions_can_be_spread_to + nr_connected_lichen
        return nr_tiles_to_water

    def _get_nr_positions_can_be_spread_to(self, strain_id: int) -> int:
        positions_can_spread = self._get_positions_can_spread(strain_id)
        neighbor_positions = self._get_neighboring_positions_to_array(positions_can_spread)
        nr_positions_can_be_spread_to = self._get_nr_empty_tiles(neighbor_positions)
        return nr_positions_can_be_spread_to

    def _get_positions_can_spread(self, strain_id: int) -> np.ndarray:
        factory_positions = self._get_factory_positions(strain_id)
        lichen_positions_can_spread = self._get_lichen_positions_can_spread(strain_id)
        positions_can_spread = np.append(factory_positions, lichen_positions_can_spread, axis=0)
        return positions_can_spread

    def _get_factory_positions(self, strain_id: int) -> np.ndarray:
        return np.argwhere(self.factory_occupancy_map == strain_id)

    def _get_connected_lichen_positions(self, strain_id: int) -> np.ndarray:
        factory_positions = self._get_factory_positions(strain_id)
        queue: list[tuple] = []
        seen: set[tuple] = {tuple(pos) for pos in factory_positions}

        for pos in factory_positions:
            tuple_pos = tuple(pos)
            for new_pos in self._get_neighboring_positions_to_sequence(tuple_pos):
                if new_pos not in seen:
                    seen.add(new_pos)
                    queue.append(new_pos)

        connected = []

        while queue:
            tuple_pos = queue.pop()
            if self.lichen_strains[tuple_pos] != strain_id:
                continue

            connected.append(tuple_pos)

            for new_pos in self._get_neighboring_positions_to_sequence(tuple_pos):
                if new_pos not in seen:
                    seen.add(new_pos)
                    queue.append(new_pos)

        if not connected:
            return np.empty((0, 2), dtype=int)

        return np.array(connected)

    def _get_lichen_positions_can_spread(self, strain_id: int) -> np.ndarray:
        connected_lichen_positions = self._get_connected_lichen_positions(strain_id)

        if not connected_lichen_positions.shape[0]:
            return connected_lichen_positions

        can_spread_mask = (
            self.lichen[connected_lichen_positions[:, 0], connected_lichen_positions[:, 1]]
            >= EnvConfig.MIN_LICHEN_TO_SPREAD
        )
        return connected_lichen_positions[can_spread_mask]

    def _get_neighboring_positions_to_array(self, positions: np.ndarray) -> np.ndarray:
        neighbor_positions = positions[..., None] + NEIGHBORING_DIRECTIONS_ARRAY.transpose()[None, ...]
        neighbor_positions = np.swapaxes(neighbor_positions, 1, 2).reshape(-1, 2)
        is_valid_mask = np.logical_and(neighbor_positions >= 0, neighbor_positions < self.width).all(axis=1)
        neighbor_positions = neighbor_positions[is_valid_mask]
        unique_neighbors = np.unique(neighbor_positions, axis=0)
        return unique_neighbors

    def _get_nr_empty_tiles(self, positions) -> int:
        empty_mask = self._is_empty_array[positions[:, 0], positions[:, 1]]
        nr_empty_tiles = empty_mask.sum()
        return nr_empty_tiles

    def _get_neighboring_positions_to_sequence(self, pos: Sequence[int]) -> Generator[tuple, None, None]:
        for dir_pos in NEIGHBORING_DIRECTIONS_ARRAY:
            x, y = pos[0] + dir_pos[0], pos[1] + dir_pos[1]
            if self.is_x_on_the_board(x) and self.is_y_on_the_board(y):
                yield ((x, y))


NEIGHBORING_DIRECTIONS_ARRAY = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
