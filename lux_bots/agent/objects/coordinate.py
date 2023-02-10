from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int

    def __add__(self, other) -> Coordinate:
        if isinstance(other, Direction):
            other = other.value

        new_x = self.x + other.x
        new_y = self.y + other.y
        return Coordinate(new_x, new_y)

    def __sub__(self, other: Coordinate) -> Coordinate:
        new_x = self.x - other.x
        new_y = self.y - other.y
        return Coordinate(new_x, new_y)

    def __eq__(self, other: Coordinate) -> bool:
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: Coordinate) -> bool:
        if self.x != other.x:
            return self.x < other.x
        else:
            return self.y < other.y

    def __iter__(self):
        return iter((self.x, self.y))

    @property
    def neighbors(self) -> CoordinateList:
        neighbors = [self + direction.value for direction in Direction]
        return CoordinateList(neighbors)

    def distance_to(self, c: Coordinate) -> int:
        """Manhattan distance to point

        Args:
            coordinate: Other coordinate to get the distance to

        Returns:
            Distance
        """
        dis_x = abs(self.x - c.x)
        dis_y = abs(self.y - c.y)
        return dis_x + dis_y


@dataclass(eq=True, frozen=True)
class TimeCoordinate(Coordinate):
    t: int

    def __iter__(self):
        return iter((self.x, self.y, self.t))


@dataclass
class CoordinateList:
    coordinates: list[Coordinate]

    def dis_to_tiles(self, c: Coordinate, exclude_c: Optional[CoordinateList] = None) -> list[int]:
        if exclude_c is None:
            return [c.distance_to(factory_c) for factory_c in self.coordinates]

        return [c.distance_to(factory_c) for factory_c in self.coordinates if factory_c not in exclude_c]

    def min_dis_to(self, c: Coordinate, exclude_c: Optional[CoordinateList] = None) -> int:
        return min(self.dis_to_tiles(c, exclude_c=exclude_c))

    def get_all_closest_tiles(self, c: Coordinate, exclude_c: Optional[CoordinateList] = None) -> CoordinateList:
        min_dis = self.min_dis_to(c, exclude_c=exclude_c)

        if exclude_c is None:
            coordinates = [c_l for c_l in self.coordinates if min_dis == c_l.distance_to(c)]
        else:
            coordinates = [c_l for c_l in self.coordinates if min_dis == c_l.distance_to(c) and c_l not in exclude_c]

        return CoordinateList(coordinates)

    def get_closest_tile(self, c: Coordinate, exclude_c: Optional[CoordinateList] = None) -> Coordinate:
        return self.get_all_closest_tiles(c=c, exclude_c=exclude_c)[0]

    def get_n_closest_tiles(self, c: Coordinate, n: int) -> CoordinateList:
        coordinates_sorted = sorted(self.coordinates, key=c.distance_to)
        n_closest_coordinates = coordinates_sorted[:n]
        return CoordinateList(n_closest_coordinates)

    def append(self, c: Coordinate) -> None:
        self.coordinates.append(c)

    def __add__(self, other: CoordinateList) -> CoordinateList:
        return CoordinateList(self.coordinates + other.coordinates)

    def __iter__(self):
        return iter(self.coordinates)

    def __getitem__(self, key):
        return self.coordinates[key]

    def __contains__(self, c):
        return c in self.coordinates

    def __len__(self):
        return len(self.coordinates)


class Direction(Enum):
    CENTER = Coordinate(0, 0)
    UP = Coordinate(0, -1)
    RIGHT = Coordinate(1, 0)
    DOWN = Coordinate(0, 1)
    LEFT = Coordinate(-1, 0)

    @property
    def number(self) -> int:
        return DIRECTION_NUMBER[self]


DIRECTION_NUMBER: dict = {
    Direction.CENTER: 0,
    Direction.UP: 1,
    Direction.RIGHT: 2,
    Direction.DOWN: 3,
    Direction.LEFT: 4,
}
