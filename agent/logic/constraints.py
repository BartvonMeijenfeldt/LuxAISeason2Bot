from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from copy import copy

from objects.coordinate import TimeCoordinate, PowerTimeCoordinate


@dataclass
class Constraints:
    parent: str = field(default_factory=str)
    max_power_request: Optional[int] = field(init=False, default=None)
    positive: set[tuple[int, int, int]] = field(init=False, default_factory=set)
    positive_t: set[int] = field(init=False, default_factory=set)
    negative: set[tuple[int, int, int]] = field(init=False, default_factory=set)
    negative_t: set[int] = field(init=False, default_factory=set)

    @property
    def key(self) -> str:
        return str(self.positive) + str(self.negative) + str(self.max_power_request)

    def __bool__(self) -> bool:
        if self.positive or self.negative or self.max_power_request is not None:
            return True

        return False

    def add_positive_constraint(self, tc: TimeCoordinate) -> None:
        self.positive.add(tc.xyt)
        self.positive_t.add(tc.t)

    def add_negative_constraint(self, tc: TimeCoordinate) -> None:
        self.negative.add(tc.xyt)
        self.negative_t.add(tc.t)

    def set_max_power_request(self, power: int) -> None:
        self.max_power_request = power

    @property
    def max_t(self) -> Optional[int]:
        if not self.has_time_constraints:
            return None

        all_t = self.positive_t | self.negative_t
        return max(all_t)

    @property
    def has_time_constraints(self) -> bool:
        if self.positive or self.negative:
            return True
        else:
            return False

    def tc_in_constraints(self, tc: TimeCoordinate) -> bool:
        return self.tc_in_negative_constraints(tc) or self.tc_in_positive_constraints(tc)

    def tc_violates_constraint(self, tc: TimeCoordinate) -> bool:
        return (
            self._is_positive_constraint_violated(tc=tc)
            or self._is_negative_constraint_violated(tc=tc)
            or self._is_power_constraint_violated(tc)
        )

    def _is_positive_constraint_violated(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.positive_t and tc.xyt not in self.positive

    def _is_negative_constraint_violated(self, tc: TimeCoordinate) -> bool:
        return self.tc_in_negative_constraints(tc)

    def tc_in_positive_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.xyt in self.positive

    def tc_in_negative_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.xyt in self.negative

    def t_in_constraints(self, t: int) -> bool:
        return self.t_in_positive_constraints(t) or self.t_in_negative_constraints(t)

    def t_in_positive_constraints(self, t: int) -> bool:
        return t in self.positive_t

    def t_in_negative_constraints(self, t: int) -> bool:
        return t in self.negative_t

    def _is_power_constraint_violated(self, tc: TimeCoordinate) -> bool:
        # TODO, this does not check properly how much power gets asked and at what timestep
        if not self.max_power_request or not isinstance(tc, PowerTimeCoordinate):
            return False

        return tc.p > self.max_power_request

    def __copy__(self) -> Constraints:
        copy_constraints = Constraints(self.parent)
        copy_constraints.max_power_request = self.max_power_request

        copy_constraints.positive = copy(self.positive)
        copy_constraints.positive_t = copy(self.positive_t)
        copy_constraints.negative = copy(self.negative)
        copy_constraints.negative_t = copy(self.negative_t)
        return copy_constraints

    def __repr__(self) -> str:
        pos_str = f"pos={self.positive}, " if self.positive else ""
        neg_str = f"neg={self.negative}, " if self.negative else ""
        pow_str = f"pow={self.max_power_request}" if self.max_power_request else ""

        return f"Constraints: {pos_str}{neg_str}{pow_str}"
