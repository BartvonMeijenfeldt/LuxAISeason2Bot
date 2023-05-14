from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Iterable

from objects.coordinate import TimeCoordinate


@dataclass
class Constraints:
    negative: set[tuple[int, int, int]] = field(default_factory=set)
    stationary_danger_coordinates: dict[tuple[int, int, int], float] = field(default_factory=dict)
    moving_danger_coordinates: dict[tuple[int, int, int], float] = field(default_factory=dict)

    def __copy__(self) -> Constraints:
        constraints_negative = copy(self.negative)
        danger_coordinates = copy(self.stationary_danger_coordinates)
        moving_danger_coordinates = copy(self.moving_danger_coordinates)
        return Constraints(constraints_negative, danger_coordinates, moving_danger_coordinates)

    @property
    def key(self) -> str:
        return str(self.negative)

    def __bool__(self) -> bool:
        if self.negative:
            return True

        return False

    def remove_negative_constraints(self, tcs: Iterable[TimeCoordinate]) -> None:
        xyt_set = {tc.xyt for tc in tcs}
        self.negative.difference_update(xyt_set)

    def add_negative_constraints(self, tcs: Iterable[TimeCoordinate]) -> None:
        xyt_set = {tc.xyt for tc in tcs}
        self.negative.update(xyt_set)

    def add_negative_constraint(self, tc: TimeCoordinate) -> None:
        self.negative.add(tc.xyt)

    def add_moving_danger_coordinates(self, danger_coordinates: dict[TimeCoordinate, float]):
        for tc, value in danger_coordinates.items():
            self.moving_danger_coordinates[tc.xyt] = value

    def add_stationary_danger_coordinates(self, danger_coordinates: dict[TimeCoordinate, float]):
        for tc, value in danger_coordinates.items():
            self.stationary_danger_coordinates[tc.xyt] = value

    def get_danger_cost(self, tc: TimeCoordinate, is_stationary_action: bool) -> float:
        if is_stationary_action:
            return self._get_stationary_danger_cost(tc)
        else:
            return self._get_moving_danger_cost(tc)

    def _get_stationary_danger_cost(self, tc: TimeCoordinate) -> float:
        return self.stationary_danger_coordinates.get(tc.xyt, 0)

    def _get_moving_danger_cost(self, tc: TimeCoordinate) -> float:
        return self.moving_danger_coordinates.get(tc.xyt, 0)

    def any_tc_violates_constraint(self, tcs: Iterable[TimeCoordinate]) -> bool:
        if not self:
            return False

        return any(self.tc_not_allowed(tc) for tc in tcs)

    def tc_violates_constraint(self, tc: TimeCoordinate) -> bool:
        if not self:
            return False

        return self.tc_not_allowed(tc=tc)

    def tc_not_allowed(self, tc: TimeCoordinate) -> bool:
        return tc.xyt in self.negative

    def any_tc_not_allowed(self, tcs: Iterable[TimeCoordinate]) -> bool:
        tcs_set = {tc.xyt for tc in tcs}
        if self.negative & tcs_set:
            return True

        return False

    def __repr__(self) -> str:
        neg_str = f"neg={self.negative}, " if self.negative else ""

        return f"Constraints: {neg_str}"
