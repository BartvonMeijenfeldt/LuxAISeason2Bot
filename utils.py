import heapq
from typing import Any


class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, Any]] = []

    def is_empty(self) -> bool:
        return not self.elements

    def put(self, item: Any, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def __getitem__(self, index: int) -> Any:
        return self.elements[index][1]

    def pop(self) -> Any:
        return heapq.heappop(self.elements)[1]

    def __len__(self) -> int:
        return len(self.elements)
