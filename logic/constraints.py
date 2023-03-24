from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterable
from copy import copy

from objects.coordinate import TimeCoordinate


@dataclass
class Constraints:
    parent: str = field(default_factory=str)
    negative: set[tuple[int, int, int]] = field(init=False, default_factory=set)
    negative_t: set[int] = field(init=False, default_factory=set)

    def _copy_with_parent_key(self) -> Constraints:
        # Like a copy, but stores the key of the object that created this copy

        copy_constraints = Constraints(parent=self.key)
        copy_constraints.negative = copy(self.negative)
        copy_constraints.negative_t = copy(self.negative_t)
        return copy_constraints

    @property
    def key(self) -> str:
        return str(self.negative)

    def __bool__(self) -> bool:
        if self.negative:
            return True

        return False

    def add_negative_constraint(self, tc: TimeCoordinate) -> Constraints:
        constraints = self._copy_with_parent_key()
        constraints._add_negative_constraint(tc)

        return constraints

    def add_negative_constraints(self, tcs: Iterable[TimeCoordinate]) -> Constraints:
        constraints = self._copy_with_parent_key()
        for tc in tcs:
            constraints._add_negative_constraint(tc)

        return constraints

    def _add_negative_constraint(self, tc: TimeCoordinate) -> None:
        self.negative.add(tc.xyt)
        self.negative_t.add(tc.t)

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
