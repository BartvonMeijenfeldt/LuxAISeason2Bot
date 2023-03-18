from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from dataclasses import dataclass

from objects.coordinate import Coordinate, CoordinateList

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
        self.player_factories_tiles_set = {(c.xy) for factory in self.player_factories for c in factory.coordinates}
        self.opponent_factories_tiles_set = {(c.xy) for factory in self.opp_factories for c in factory.coordinates}

        self.player_factory_tiles = CoordinateList(
            [c for factory in self.player_factories for c in factory.coordinates]
        )
        self.opponent_factory_tiles = CoordinateList([c for factory in self.opp_factories for c in factory.coordinates])
        self.ice_coordinates = self.get_ice_coordinates()
        self.ore_coordinates = self.get_ore_coordinates()
        self.rubble_coordinates = self.get_rubble_coordinates()

    @property
    def length(self):
        return self.rubble.shape[0]

    @property
    def width(self):
        return self.rubble.shape[1]

    def get_ice_coordinates(self) -> CoordinateList:
        ice_locations = np.argwhere(self.ice == 1)
        return CoordinateList([Coordinate(*xy) for xy in ice_locations])

    def get_ore_coordinates(self) -> CoordinateList:
        ore_locations = np.argwhere(self.ore == 1)
        return CoordinateList([Coordinate(*xy) for xy in ore_locations])

    def get_rubble_coordinates(self) -> CoordinateList:
        is_rubble = self.rubble > 0
        is_no_ice = self.ice == 0
        is_no_ore = self.ore == 0

        rubble_locations = np.argwhere(is_rubble & is_no_ice & is_no_ore)
        return CoordinateList([Coordinate(*xy) for xy in rubble_locations])

    def is_valid_c_for_player(self, c: Coordinate) -> bool:
        return not self.is_opponent_factory_tile(c=c) and self.is_on_the_board(c=c)

    def is_player_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.player_factories_tiles_set

    def is_opponent_factory_tile(self, c: Coordinate) -> bool:
        return c.xy in self.opponent_factories_tiles_set

    def is_on_the_board(self, c: Coordinate) -> bool:
        return 0 <= c.x < self.width and 0 <= c.y < self.length

    def is_off_the_board(self, c: Coordinate) -> bool:
        return not self.is_off_the_board(c=c)

    def get_opponent_on_c(self, c: Coordinate) -> Optional[Unit]:
        for unit in self.opp_units:
            if unit.tc.xy == c.xy:
                return unit

        return None

    def get_closest_factory(self, c: Coordinate) -> Factory:
        return min(self.player_factories, key=lambda x: x.min_dis_to(c))

    def get_closest_factory_tile(self, c: Coordinate) -> Coordinate:
        return self.player_factory_tiles.get_closest_tile(c)

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

    def get_n_closest_ice_tiles(self, c: Coordinate, n: int) -> CoordinateList:
        return self.ice_coordinates.get_n_closest_tiles(c=c, n=n)

    def get_valid_neighbor_coordinates(self, c: Coordinate) -> CoordinateList:
        coordinates = [
            neighbor_c
            for neighbor_c in c.neighbors
            if self.is_on_the_board(neighbor_c) and not self.is_opponent_factory_tile(c=neighbor_c)
        ]
        return CoordinateList(coordinates)
