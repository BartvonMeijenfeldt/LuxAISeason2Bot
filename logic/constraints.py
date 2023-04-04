from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterable
from collections import defaultdict
from copy import copy

from objects.coordinate import TimeCoordinate


@dataclass
class Constraints:
    negative: set[tuple[int, int, int]] = field(default_factory=set)
    negative_t: set[int] = field(default_factory=set)
    danger_coordinates: defaultdict[tuple[int, int, int], float] = field(default_factory=lambda: defaultdict(lambda: 0))

    def __copy__(self) -> Constraints:
        constraints_negative = copy(self.negative)
        constraints_negative_t = copy(self.negative_t)
        danger_coordinates = copy(self.danger_coordinates)
        return Constraints(constraints_negative, constraints_negative_t, danger_coordinates)

    @property
    def key(self) -> str:
        return str(self.negative)

    def __bool__(self) -> bool:
        if self.negative:
            return True

        return False

    def add_negative_constraints(self, tcs: Iterable[TimeCoordinate]) -> None:
        for tc in tcs:
            self.add_negative_constraint(tc)

    def add_negative_constraint(self, tc: TimeCoordinate) -> None:
        self.negative.add(tc.xyt)
        self.negative_t.add(tc.t)

    def add_danger_coordinates(self, danger_coordinates: dict[TimeCoordinate, float]):
        for tc, value in danger_coordinates.items():
            self.danger_coordinates[tc.xyt] = value

    def get_danger_cost(self, tc: TimeCoordinate) -> float:
        return self.danger_coordinates[tc.xyt]

    @property
    def max_t(self) -> Optional[int]:
        if not self:
            return None

        return max(self.negative_t)

    def any_tc_violates_constraint(self, tcs: Iterable[TimeCoordinate]) -> bool:
        if not self:
            return False

        return any(self.tc_in_negative_constraints(tc) for tc in tcs)

    def tc_violates_constraint(self, tc: TimeCoordinate) -> bool:
        if not self:
            return False

        return self.tc_in_negative_constraints(tc=tc)

    def tc_in_negative_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.xyt in self.negative

    def can_not_add_negative_constraint(self, tc: TimeCoordinate) -> bool:
        return self.tc_in_negative_constraints(tc)

    def __repr__(self) -> str:
        neg_str = f"neg={self.negative}, " if self.negative else ""

        return f"Constraints: {neg_str}"
