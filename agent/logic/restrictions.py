from __future__ import annotations

from dataclasses import dataclass

from objects.coordinate import TimeCoordinate


@dataclass
class Restrictions:
    positive: dict[int, TimeCoordinate]
    negative: dict[int, list[TimeCoordinate]]
