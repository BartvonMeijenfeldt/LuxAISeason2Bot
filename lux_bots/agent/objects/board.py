import numpy as np

from dataclasses import dataclass

from agent.objects.coordinate import Coordinate, CoordinateList


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
