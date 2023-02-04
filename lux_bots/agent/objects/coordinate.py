from dataclasses import dataclass
from enum import Enum


@dataclass
class Coordinate:
    x: int
    y: int

    def __add__(self, other) -> "Coordinate":
        if isinstance(other, Direction):
            other = other.value

        new_x = self.x + other.x
        new_y = self.y + other.y
        return Coordinate(new_x, new_y)

    def __sub__(self, other: "Coordinate") -> "Coordinate":
        new_x = self.x - other.x
        new_y = self.y - other.y
        return Coordinate(new_x, new_y)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __eq__(self, other: "Coordinate") -> "Coordinate":
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: "Coordinate") -> "Coordinate":
        if self.x != other.x:
            return self.x < other.x
        else:
            return self.y < other.y

    def __iter__(self):
        return iter((self.x, self.y))

    def distance_to(self, c: "Coordinate") -> int:
        """Manhatten distance to point

        Args:
            coordinate: Other coordinate to get the distance to

        Returns:
            Distance
        """
        dis_x = abs(self.x - c.x)
        dis_y = abs(self.y - c.y)
        return dis_x + dis_y

    def direction_to(self, target: "Coordinate") -> int:
        dx, dy = target - self

        if dx == 0 and dy == 0:
            return Direction.CENTER
        if abs(dx) > abs(dy):
            if dx > 0:
                return Direction.RIGHT
            else:
                return Direction.LEFT
        else:
            if dy > 0:
                return Direction.DOWN
            else:
                return Direction.UP


@dataclass
class CoordinateList:
    coordinates: list[Coordinate]

    def dis_to_tiles(self, c: Coordinate, exclude_c: "CoordinateList" = None) -> list[int]:
        if exclude_c is None:
            exclude_c = []

        return [c.distance_to(factory_c) for factory_c in self.coordinates if factory_c not in exclude_c]

    def min_dis_to(self, c: Coordinate, exclude_c: "CoordinateList" = None) -> int:
        return min(self.dis_to_tiles(c, exclude_c=exclude_c))

    def get_all_closest_tiles(self, c: Coordinate, exclude_c: "CoordinateList" = None) -> "CoordinateList":
        if exclude_c is None:
            exclude_c = []

        min_dis = self.min_dis_to(c, exclude_c=exclude_c)

        coordinates = [c_l for c_l in self.coordinates if min_dis == c_l.distance_to(c) and c_l not in exclude_c]

        return CoordinateList(coordinates)

    def get_closest_tile(self, c: Coordinate, exclude_c: "CoordinateList" = None) -> Coordinate:
        return self.get_all_closest_tiles(c=c, exclude_c=exclude_c)[0]

    def get_n_closest_tiles(self, c: Coordinate, n: int) -> "CoordinateList":
        coordinates_sorted = sorted(self.coordinates, key=c.distance_to)
        n_closest_coordinates = coordinates_sorted[:n]
        return CoordinateList(n_closest_coordinates)

    def append(self, c: Coordinate) -> None:
        self.coordinates.append(c)

    def __add__(self, other: "CoordinateList") -> "CoordinateList":
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
    CENTER: Coordinate = Coordinate(0, 0)
    UP: Coordinate = Coordinate(0, -1)
    RIGHT: Coordinate = Coordinate(1, 0)
    DOWN: Coordinate = Coordinate(0, 1)
    LEFT: Coordinate = Coordinate(-1, 0)

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
