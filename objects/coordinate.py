from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from objects.actions.unit_action import (
    DigAction,
    PickupAction,
    TransferAction,
    UnitAction,
)
from objects.direction import Direction
from objects.resource import Resource
from utils.utils import is_day

if TYPE_CHECKING:
    from lux.config import UnitConfig
    from lux.kit import GameState


@dataclass(frozen=True)
class Coordinate:
    x: int
    y: int

    def __eq__(self, other: Coordinate) -> bool:
        return self.x == other.x and self.y == other.y

    def __add__(self, other) -> Coordinate:
        x, y = self._add_get_new_xy(other)
        return Coordinate(x, y)

    def add_action(self, action: UnitAction) -> Coordinate:
        """Next coordinate of unit if a unit standing on this coordinate would carry out the action.

        Args:
            action: Action of unit.

        Returns:
            Next coordinate.
        """
        x, y = self._add_get_new_xy_action(action)
        return Coordinate(x, y)

    def _add_get_new_xy(self, other) -> tuple[int, int]:
        if isinstance(other, Direction):
            return self._add_get_new_xy_direction(other)

        if isinstance(other, Coordinate):
            return self._add_get_new_xy_coordinate(other)

        raise TypeError(f"Unexpected type of other: {type(other)}")

    def _add_get_new_xy_action(self, action: UnitAction) -> tuple[int, int]:
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
    def neighbors(self) -> list[Coordinate]:
        """Neighboring coordinates."""
        neighbors = [self + direction for direction in Direction]
        return neighbors

    @property
    def non_stationary_neighbors(self) -> list[Coordinate]:
        """Neighboring coordinates, excluding the coordinate achieved by being stationary."""
        neighbors = [self + direction for direction in Direction if direction != direction.CENTER]
        return neighbors

    def distance_to(self, c: Coordinate) -> int:
        """Manhattan distance to coordinate

        Args:
            coordinate: Other coordinate to get the distance to

        Returns:
            Distance
        """
        dis_x = abs(self.x - c.x)
        dis_y = abs(self.y - c.y)
        return dis_x + dis_y

    def direction_to(self, c: Coordinate) -> Direction:
        if self.x < c.x:
            return Direction.RIGHT
        elif self.x > c.x:
            return Direction.LEFT
        elif self.y < c.y:
            return Direction.DOWN
        elif self.y > c.y:
            return Direction.UP
        else:
            return Direction.CENTER


@dataclass(eq=True, frozen=True)
class TimeCoordinate(Coordinate):
    t: int

    def __repr__(self) -> str:
        return f"TC[x={self.x} y={self.y} t={self.t}]"

    def __lt__(self, other: TimeCoordinate) -> bool:
        return self.t < other.t

    def __add__(self, other) -> TimeCoordinate:
        x, y = self._add_get_new_xy(other)
        t = self._add_get_new_t()
        return TimeCoordinate(x, y, t)

    def _add_get_new_t(self) -> int:
        return self.t + 1

    def add_action(self, action: UnitAction) -> TimeCoordinate:
        x, y = self._add_get_new_xy_action(action)
        t = self._add_get_new_t_action(action)
        return TimeCoordinate(x, y, t)

    def _add_get_new_t_action(self, action: UnitAction) -> int:
        return self.t + action.n

    @property
    def neighbors(self) -> list[TimeCoordinate]:
        neighbors = [self + direction for direction in Direction]
        return neighbors

    @property
    def non_stationary_neighbors(self) -> list[TimeCoordinate]:
        neighbors = [self + direction for direction in Direction if direction != Direction.CENTER]
        return neighbors

    @property
    def xyt(self) -> tuple[int, int, int]:
        return self.x, self.y, self.t


@dataclass(eq=True, frozen=True)
class DigCoordinate(Coordinate):
    d: int

    def __eq__(self, other: DigCoordinate) -> bool:
        return self.x == other.x and self.y == other.y and self.d == other.d

    def __add__(self, other) -> DigCoordinate:
        x, y = self._add_get_new_xy(other)
        return DigCoordinate(x, y, self.d)

    def add_action(self, action: UnitAction) -> DigCoordinate:
        x, y = self._add_get_new_xy_action(action)
        d = self._add_get_new_d_action(action)
        return DigCoordinate(x, y, d)

    def _add_get_new_d_action(self, action: UnitAction) -> int:
        return self.d + action.n if isinstance(action, DigAction) else self.d


@dataclass(eq=True, frozen=True)
class DigTimeCoordinate(DigCoordinate, TimeCoordinate):
    def __add__(self, other) -> DigTimeCoordinate:
        x, y = super()._add_get_new_xy(other)
        t = super()._add_get_new_t()
        return DigTimeCoordinate(x, y, t, self.d)

    def add_action(self, action: UnitAction) -> DigTimeCoordinate:
        x, y = self._add_get_new_xy_action(action)
        t = self._add_get_new_t_action(action)
        d = self._add_get_new_d_action(action)
        return DigTimeCoordinate(x, y, t, d)


@dataclass(eq=True, frozen=True)
class PowerTimeCoordinate(TimeCoordinate):
    p: int
    unit_cfg: UnitConfig
    game_state: GameState

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.t, self.p))

    def __eq__(self, other: PowerTimeCoordinate) -> bool:
        return self.x == other.x and self.y == other.y and self.t == other.t and self.p == other.p

    def __add__(self, other) -> PowerTimeCoordinate:
        x, y = super()._add_get_new_xy(other)
        t = super()._add_get_new_t()
        p = self._add_get_p()
        return PowerTimeCoordinate(x, y, t, p, self.unit_cfg, self.game_state)

    def _add_get_p(self) -> int:
        return self.p

    def add_action(self, action: UnitAction) -> PowerTimeCoordinate:
        x, y = self._add_get_new_xy_action(action)
        t = self._add_get_new_t_action(action)
        p = self._add_get_new_p_action(action)

        return PowerTimeCoordinate(x, y, t, p, self.unit_cfg, self.game_state)

    def _add_get_new_p_action(self, action: UnitAction) -> int:
        try:
            p = self.p + action.get_power_change(self.unit_cfg, self, self.game_state.board)
        except IndexError:
            return -1

        if is_day(self.t):
            p += self.unit_cfg.CHARGE

        return p


@dataclass(eq=True, frozen=True)
class ResourceCoordinate(Coordinate):
    q: int
    resource: Resource

    def __eq__(self, other: ResourceCoordinate) -> bool:
        return self.x == other.x and self.y == other.y and self.q == other.q and self.resource == other.resource

    def __add__(self, other) -> ResourceCoordinate:
        x, y = super()._add_get_new_xy(other)
        q = self._add_get_q()
        return ResourceCoordinate(x, y, q, self.resource)

    def _add_get_q(self) -> int:
        return self.q

    def add_action(self, action: UnitAction) -> ResourceCoordinate:
        x, y = self._add_get_new_xy_action(action)
        q = self._add_get_new_q_action(action)

        return ResourceCoordinate(x, y, q, self.resource)

    def _add_get_new_q_action(self, action: UnitAction) -> int:
        if isinstance(action, PickupAction) and action.resource == self.resource:
            return self.q + action.n  # * action.amount
        elif isinstance(action, TransferAction) and action.resource == self.resource:
            return self.q - action.n  # * action.amount
        else:
            return self.q


@dataclass(eq=True, frozen=True)
class ResourcePowerTimeCoordinate(ResourceCoordinate, PowerTimeCoordinate):
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.t, self.p, self.q, self.resource))

    def __add__(self, other) -> ResourcePowerTimeCoordinate:
        x, y = super()._add_get_new_xy(other)
        t = super()._add_get_new_t()
        p = self._add_get_p()
        return ResourcePowerTimeCoordinate(x, y, t, p, self.unit_cfg, self.game_state, self.q, self.resource)

    def add_action(self, action: UnitAction) -> ResourcePowerTimeCoordinate:
        x, y = self._add_get_new_xy_action(action)
        t = self._add_get_new_t_action(action)
        p = self._add_get_new_p_action(action)
        q = self._add_get_new_q_action(action)

        return ResourcePowerTimeCoordinate(x, y, t, p, self.unit_cfg, self.game_state, q, self.resource)

    def __repr__(self) -> str:
        return f"RPTC[x={self.x} y={self.y}, t={self.t}, p={self.p}, q={self.q}"


@dataclass(eq=True, frozen=True)
class ResourceTimeCoordinate(ResourceCoordinate, TimeCoordinate):
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.t, self.q, self.resource))

    def __add__(self, other) -> ResourceTimeCoordinate:
        x, y = super()._add_get_new_xy(other)
        t = super()._add_get_new_t()
        return ResourceTimeCoordinate(x, y, t, self.q, self.resource)

    def add_action(self, action: UnitAction) -> ResourceTimeCoordinate:
        x, y = self._add_get_new_xy_action(action)
        t = self._add_get_new_t_action(action)
        q = self._add_get_new_q_action(action)

        return ResourceTimeCoordinate(x, y, t, q, self.resource)

    def __repr__(self) -> str:
        return f"PPPTC[x={self.x} y={self.y}, t={self.t}, q={self.q}"


@dataclass
class CoordinateList:
    coordinates: list[Coordinate]

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

    def to_positions(self) -> np.ndarray:
        return np.array([[c.x, c.y] for c in self])
