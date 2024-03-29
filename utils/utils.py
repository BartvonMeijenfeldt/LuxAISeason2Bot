import heapq
import itertools
from typing import Any

from lux.config import EnvConfig


class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, int, Any]] = []
        self._counter = itertools.count()

    def is_empty(self) -> bool:
        return not self.elements

    def put(self, item: Any, priority: float):
        item = (priority, -next(self._counter), item)
        heapq.heappush(self.elements, item)

    def __getitem__(self, index: int) -> Any:
        return self.elements[index][2]

    def pop(self) -> Any:
        return heapq.heappop(self.elements)[2]

    def __len__(self) -> int:
        return len(self.elements)


def is_day(t: int) -> bool:
    return t % EnvConfig.CYCLE_LENGTH < EnvConfig.DAY_LENGTH
