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
        x, y = self._add_get_new_xy(other)
        return Coordinate(x, y)

    def _add_get_new_xy(self, other) -> tuple[int, int]:
        if isinstance(other, Action):
            return self._add_get_new_xy_action(other)

        if isinstance(other, Direction):
            return self._add_get_new_xy_direction(other)

        if isinstance(other, Coordinate):
            return self._add_get_new_xy_coordinate(other)

        raise TypeError(f"Unexpected type of other: {type(other)}")

    def _add_get_new_xy_action(self, action: Action) -> tuple[int, int]:
        direction_tuple = action.unit_direction.value
        x = self.x + direction_tuple[0] * action.n
        y = self.y + direction_tuple[1] * action.n
        return x, y

    def _add_get_new_xy_direction(self, direction: Direction) -> tuple[int, int]:
        direction_tuple = direction.value
        x = self.x + direction_tuple[0]
        y = self.y + direction_tuple[1]
        return x, y

    def _add_get_new_xy_coordinate(self, c: Coordinate) -> tuple[int, int]:
        x = self.x + c.x
        y = self.y + c.y
        return x, y

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
        x, y = super()._add_get_new_xy(other)
        t = self._add_get_new_t(other)
        return TimeCoordinate(x, y, t)

    def _add_get_new_t(self, other) -> int:
        nr_time_steps = other.n if isinstance(other, Action) else 1
        new_t = self.t + nr_time_steps
        return new_t


@dataclass(eq=True, frozen=True)
class DigCoordinate(Coordinate):
    nr_digs: int

    def __eq__(self, other: DigCoordinate) -> bool:
        return self.x == other.x and self.y == other.y and self.nr_digs == other.nr_digs

    def __iter__(self):
        return iter((self.x, self.y, self.nr_digs))

    def __add__(self, other) -> DigCoordinate:
        x, y = super()._add_get_new_xy(other)
        nr_digs = self._add_get_new_nr_digs(other)
        return DigCoordinate(x, y, nr_digs)

    def _add_get_new_nr_digs(self, other) -> int:
        added_digs = other.n if isinstance(other, DigAction) else 0
        return self.nr_digs + added_digs


@dataclass(eq=True, frozen=True)
class DigTimeCoordinate(DigCoordinate, TimeCoordinate):
    def __iter__(self):
        return iter((self.x, self.y, self.t, self.nr_digs))

    def __add__(self, other) -> DigTimeCoordinate:
        x, y = super()._add_get_new_xy(other)
        t = super()._add_get_new_t(other)
        nr_digs = super()._add_get_new_nr_digs(other)

        return DigTimeCoordinate(x, y, t, nr_digs)


@dataclass(eq=True, frozen=True)
class PowerTimeCoordinate(TimeCoordinate):
    power_recharged: int

    def __iter__(self):
        return iter((self.x, self.y, self.t, self.power_recharged))

    def __add__(self, other) -> PowerTimeCoordinate:
        x, y = super()._add_get_new_xy(other)
        t = super()._add_get_new_t(other)

        if isinstance(other, PickupAction) and other.resource == Resource.Power:
            recharged_amount = other.n * other.amount
        else:
            recharged_amount = 0
        new_recharged_amount = self.power_recharged + recharged_amount

        return PowerTimeCoordinate(x, y, t, new_recharged_amount)

    def _add_get_power_recharged(self, other) -> int:
        if isinstance(other, PickupAction) and other.resource == Resource.Power:
            added_power_recharged = other.n * other.amount
        else:
            added_power_recharged = 0
        return self.power_recharged + added_power_recharged


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
