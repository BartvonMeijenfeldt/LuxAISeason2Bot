from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from objects.coordinate import TimeCoordinate


@dataclass
class Constraints:
    positive: dict[int, TimeCoordinate] = field(default_factory=dict)
    negative: dict[int, list[TimeCoordinate]] = field(default_factory=lambda: defaultdict(list))
    max_power_request: Optional[int] = field(default=None)

    def __bool__(self):
        if self.positive or self.negative or self.max_power_request:
            return True

        return False
