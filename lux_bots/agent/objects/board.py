from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from dataclasses import dataclass

from objects.coordinate import Coordinate, CoordinateList, Direction

if TYPE_CHECKING:
    from objects.unit import Unit
    from objects.factory import Factory


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

    @property
    def rubble_coordinates(self) -> CoordinateList:
        is_rubble = self.rubble > 0
        is_no_ice = self.ice == 0
        is_no_ore = self.ore == 0

        rubble_locations = np.argwhere(is_rubble & is_no_ice & is_no_ore)
        return CoordinateList([Coordinate(*xy) for xy in rubble_locations])

    @property
    def player_factory_tiles(self) -> CoordinateList:
        return CoordinateList([c for factory in self.player_factories for c in factory.coordinates])   

    @property
    def opponent_factory_tiles(self) -> CoordinateList:
        return CoordinateList([c for factory in self.opp_factories for c in factory.coordinates])

    def is_player_factory_tile(self, c: Coordinate) -> bool:
        return c in self.player_factory_tiles

    def is_opponent_factory_tile(self, c: Coordinate) -> bool:
        return c in self.opponent_factory_tiles

    def is_on_the_board(self, c: Coordinate) -> bool:
        return 0 <= c.x < self.width and 0 <= c.y < self.length

    def is_off_the_board(self, c: Coordinate) -> bool:
        return not self.is_off_the_board(c=c)

    def get_closest_factory_tile(self, c: Coordinate) -> Coordinate:
        return self.player_factory_tiles.get_closest_tile(c)

    def get_closest_ice_tile(self, c: Coordinate) -> Coordinate:
        return self.ice_coordinates.get_closest_tile(c=c)

    def get_closest_rubble_tile(self, c: Coordinate) -> Coordinate:
        return self.rubble_coordinates.get_closest_tile(c=c)

    def get_neighbors_coordinate(self, c: Coordinate) -> CoordinateList:
        coordinates = [c + direction.value for direction in Direction if self.is_on_the_board(c + direction.value)]
        return CoordinateList(coordinates)
