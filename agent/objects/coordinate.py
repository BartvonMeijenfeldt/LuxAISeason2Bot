from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from objects.action import Action, DigAction, PickupAction
from objects.resource import Resource
from objects.direction import Direction


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int

    def __eq__(self, other: Coordinate) -> bool:
        return self.x == other.x and self.y == other.y

    def __add__(self, other) -> Coordinate:
        if isinstance(other, Action):
            return self._add_action(other)

        if isinstance(other, Direction):
            return self._add_direction(other)

        if isinstance(other, Coordinate):
            return self._add_coordinate(other)

        raise TypeError(f"Unexpected type of other: {type(other)}")

    def _add_action(self, action: Action) -> Coordinate:
        c = self
        for _ in range(action.n):
            direction = action.unit_direction
            c = self._add_direction(direction)

        return c

    def _add_direction(self, direction: Direction) -> Coordinate:
        c = Coordinate(*direction.value)
        return self._add_coordinate(c=c)

    def _add_coordinate(self, c: Coordinate) -> Coordinate:
        new_x = self.x + c.x
        new_y = self.y + c.y
        return Coordinate(new_x, new_y)

    def __sub__(self, other: Coordinate) -> Coordinate:
        new_x = self.x - other.x
        new_y = self.y - other.y
        return Coordinate(new_x, new_y)

    def __lt__(self, other: Coordinate) -> bool:
        if self.x != other.x:
            return self.x < other.x
        else:
            return self.y < other.y

    def __iter__(self):
        return iter((self.x, self.y))

    @property
    def xy(self) -> tuple[int, int]:
        return self.x, self.y

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

    def __lt__(self, other: TimeCoordinate) -> bool:
        return self.t < other.t

    def __add__(self, other) -> TimeCoordinate:
        c = super().__add__(other)

        nr_time_steps = other.n if isinstance(other, Action) else 1
        new_t = self.t + nr_time_steps
        return TimeCoordinate(c.x, c.y, new_t)


@dataclass(eq=True, frozen=True)
class DigCoordinate(Coordinate):
    nr_digs: int

    def __eq__(self, other: DigCoordinate) -> bool:
        return self.x == other.x and self.y == other.y and self.nr_digs == other.nr_digs

    def __iter__(self):
        return iter((self.x, self.y, self.nr_digs))

    def __add__(self, other) -> DigCoordinate:
        tc = super().__add__(other)

        nr_digs = other.n if isinstance(other, DigAction) else 0
        new_d = self.nr_digs + nr_digs

        return DigCoordinate(tc.x, tc.y, new_d)


@dataclass(eq=True, frozen=True)
class DigTimeCoordinate(TimeCoordinate):
    nr_digs: int

    def __iter__(self):
        return iter((self.x, self.y, self.t, self.nr_digs))

    def __add__(self, other) -> DigTimeCoordinate:
        tc = super().__add__(other)

        nr_digs = other.n if isinstance(other, DigAction) else 0
        new_d = self.nr_digs + nr_digs

        return DigTimeCoordinate(tc.x, tc.y, tc.t, new_d)


@dataclass(eq=True, frozen=True)
class PowerTimeCoordinate(TimeCoordinate):
    power_recharged: int

    def __iter__(self):
        return iter((self.x, self.y, self.t, self.power_recharged))

    def __add__(self, other) -> PowerTimeCoordinate:
        tc = super().__add__(other)

        if isinstance(other, PickupAction) and other.resource == Resource.Power:
            recharged_amount = other.n * other.amount
        else:
            recharged_amount = 0
        new_recharged_amount = self.power_recharged + recharged_amount

        return PowerTimeCoordinate(tc.x, tc.y, tc.t, new_recharged_amount)


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
