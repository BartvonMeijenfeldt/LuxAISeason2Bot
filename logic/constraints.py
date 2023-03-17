from __future__ import annotations

import bisect

from dataclasses import dataclass, field
from typing import Optional
from copy import copy

from objects.coordinate import TimeCoordinate, PowerCoordinate


@dataclass
class Constraints:
    parent: str = field(default_factory=str)
    max_power_request: Optional[int] = field(init=False, default=None)
    positive: set[tuple[int, int, int]] = field(init=False, default_factory=set)
    negative: set[tuple[int, int, int]] = field(init=False, default_factory=set)

    positive_t: set[int] = field(init=False, default_factory=set)
    positive_sorted: list[TimeCoordinate] = field(init=False, default_factory=list)
    negative_t: set[int] = field(init=False, default_factory=set)

    def _copy_with_parent_key(self) -> Constraints:
        # Like a copy, but stores the key of the object that created this copy

        copy_constraints = Constraints(parent=self.key)
        copy_constraints.max_power_request = self.max_power_request

        copy_constraints.positive = copy(self.positive)
        copy_constraints.positive_t = copy(self.positive_t)
        copy_constraints.positive_sorted = copy(self.positive_sorted)
        copy_constraints.negative = copy(self.negative)
        copy_constraints.negative_t = copy(self.negative_t)
        return copy_constraints

    @property
    def key(self) -> str:
        return str(self.positive) + str(self.negative) + str(self.max_power_request)

    def __bool__(self) -> bool:
        if self.positive or self.negative or self.max_power_request is not None:
            return True

        return False

    def add_positive_constraint(self, tc: TimeCoordinate) -> Constraints:
        constraints = self._copy_with_parent_key()

        constraints.positive.add(tc.xyt)
        bisect.insort_right(constraints.positive_sorted, tc)
        constraints.positive_t.add(tc.t)

        return constraints

    def add_negative_constraint(self, tc: TimeCoordinate) -> Constraints:
        constraints = self._copy_with_parent_key()

        constraints.negative.add(tc.xyt)
        constraints.negative_t.add(tc.t)

        return constraints

    def add_power_constraint(self, max_power_request: int) -> Constraints:
        constraints = self._copy_with_parent_key()

        constraints.max_power_request = max_power_request

        return constraints

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
            or self.tc_in_negative_constraints(tc=tc)
            or self._is_power_constraint_violated(tc)
        )

    def _is_positive_constraint_violated(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.positive_t and tc.xyt not in self.positive

    def tc_in_positive_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.positive_t and tc.xyt in self.positive

    def tc_in_negative_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.negative_t and tc.xyt in self.negative

    def t_in_positive_constraints(self, t: int) -> bool:
        return t in self.positive_t

    def t_in_negative_constraints(self, t: int) -> bool:
        return t in self.negative_t

    def can_not_add_positive_constraint(self, tc: TimeCoordinate) -> bool:
        return self.t_in_positive_constraints(tc.t) or self.tc_in_negative_constraints(tc)

    def can_not_add_negative_constraint(self, tc: TimeCoordinate) -> bool:
        return self.tc_in_constraints(tc)

    def can_not_add_max_power_constraint(self) -> bool:
        return self.max_power_request == 0

    def can_fullfill_next_positive_constraint(self, cur_tc: TimeCoordinate) -> bool:
        next_postive_constraint = self.get_next_positive_constraint(cur_tc)
        if not next_postive_constraint:
            return True

        nr_steps_available = next_postive_constraint.t - cur_tc.t
        min_nr_steps_required = next_postive_constraint.distance_to(cur_tc)
        return nr_steps_available >= min_nr_steps_required

    def get_next_positive_constraint(self, tc: TimeCoordinate) -> Optional[TimeCoordinate]:
        index = bisect.bisect_right(self.positive_sorted, tc)

        try:
            return self.positive_sorted[index]
        except IndexError:
            return None

    def _is_power_constraint_violated(self, tc: TimeCoordinate) -> bool:
        # TODO, this does not check properly how much power gets asked and at what timestep
        if not self.max_power_request or not isinstance(tc, PowerCoordinate):
            return False

        return tc.p > self.max_power_request

    def __repr__(self) -> str:
        pos_str = f"pos={self.positive}, " if self.positive else ""
        neg_str = f"neg={self.negative}, " if self.negative else ""
        pow_str = f"pow={self.max_power_request}" if self.max_power_request else ""

        return f"Constraints: {pos_str}{neg_str}{pow_str}"
