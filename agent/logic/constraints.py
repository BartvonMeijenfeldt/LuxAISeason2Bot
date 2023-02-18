from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from objects.coordinate import TimeCoordinate


@dataclass
class Constraints:
    positive: dict[int, TimeCoordinate] = field(default_factory=dict)
    negative: dict[int, list[TimeCoordinate]] = field(default_factory=lambda: defaultdict(list))

    def __bool__(self):
        if self.positive or self.negative:
            return True

        return False
