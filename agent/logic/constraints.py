from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from objects.coordinate import TimeCoordinate, PowerTimeCoordinate


@dataclass
class Constraints:
    positive: set[TimeCoordinate] = field(default_factory=set)
    positive_t: set[int] = field(default_factory=set)
    negative: set[TimeCoordinate] = field(default_factory=set)
    max_power_request: Optional[int] = field(default=None)
    parent: str = field(default_factory=str)

    @property
    def key(self) -> str:
        return str(self.positive) + str(self.negative) + str(self.max_power_request)

    def __bool__(self) -> bool:
        if self.positive or self.negative or self.max_power_request is not None:
            return True

        return False

    def add_positive_constraint(self, tc: TimeCoordinate) -> None:
        self.positive.add(tc)
        self.positive_t.add(tc.t)

    def add_negative_constraint(self, tc: TimeCoordinate) -> None:
        self.negative.add(tc)

    def set_max_power_request(self, power: int) -> None:
        self.max_power_request = power

    @property
    def has_time_constraints(self) -> bool:
        if self.positive or self.negative:
            return True
        else:
            return False

    def tc_in_constraints(self, tc: TimeCoordinate) -> bool:
        return tc in self.positive or tc in self.negative

    def tc_violates_constraint(self, tc: TimeCoordinate) -> bool:
        # To convert DigTimeCoordinates or PowerTimeCoordinates to pure TimeCoordinates
        tc = TimeCoordinate(tc.x, tc.y, tc.t)
        return (
            self._is_positive_constraint_violated(tc=tc)
            or self._is_negative_constraint_violated(tc=tc)
            or self._is_power_constraint_violated(tc)
        )

    def _is_positive_constraint_violated(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.positive_t and tc not in self.positive

    def _is_negative_constraint_violated(self, tc: TimeCoordinate) -> bool:
        return tc in self.negative

    def _is_power_constraint_violated(self, tc: TimeCoordinate) -> bool:
        if not self.max_power_request or not isinstance(tc, PowerTimeCoordinate):
            return False

        return tc.p > self.max_power_request
