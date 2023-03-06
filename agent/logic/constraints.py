from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from objects.coordinate import TimeCoordinate, PowerTimeCoordinate


@dataclass
class Constraints:
    max_power_request: Optional[int] = field(init=False, default=None)
    parent: str = field(default_factory=str)

    def __post_init__(self) -> None:
        self.positive: dict[int, set[tuple[int, int]]] = defaultdict(set)
        self.negative: dict[int, set[tuple[int, int]]] = defaultdict(set)

    @property
    def key(self) -> str:
        return str(self.positive) + str(self.negative) + str(self.max_power_request)

    def __bool__(self) -> bool:
        if self.positive or self.negative or self.max_power_request is not None:
            return True

        return False

    def add_positive_constraint(self, tc: TimeCoordinate) -> None:
        self.positive[tc.t].add(tc.xy)

    def add_negative_constraint(self, tc: TimeCoordinate) -> None:
        self.negative[tc.t].add(tc.xy)

    def set_max_power_request(self, power: int) -> None:
        self.max_power_request = power

    @property
    def max_t(self) -> Optional[int]:
        if not self.has_time_constraints:
            return None

        all_t = self.positive.keys() | self.negative.keys()
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
        return tc.t in self.positive and tc.xy not in self.positive[tc.t]

    def _is_negative_constraint_violated(self, tc: TimeCoordinate) -> bool:
        return self.tc_in_negative_constraints(tc)

    def tc_in_positive_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.positive and tc.xy in self.positive[tc.t]

    def tc_in_negative_constraints(self, tc: TimeCoordinate) -> bool:
        return tc.t in self.negative and tc.xy in self.negative[tc.t]

    def t_in_positive_constraints(self, t: int) -> bool:
        return t in self.positive

    def _is_power_constraint_violated(self, tc: TimeCoordinate) -> bool:
        # TODO, this does not check properly how much power gets asked and at what timestep
        if not self.max_power_request or not isinstance(tc, PowerTimeCoordinate):
            return False

        return tc.p > self.max_power_request
