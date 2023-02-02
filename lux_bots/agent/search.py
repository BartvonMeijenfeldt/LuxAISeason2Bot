import heapq

from typing import TypeVar
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from agent.objects.coordinate import Coordinate, CoordinateList, Direction
from agent.objects.board import Board
from agent.objects.action import MoveAction


T = TypeVar("T")


class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, T]] = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


class Graph(metaclass=ABCMeta):
    @abstractmethod
    def valid_neighbors(self, id: Coordinate) -> list[Coordinate]:
        pass

    @abstractmethod
    def cost(self, from_c: Coordinate, to_c: Coordinate) -> float:
        pass

    @abstractmethod
    def heuristic(self, a: Coordinate, b: Coordinate) -> float:
        pass


@dataclass
class TestGraph(Graph):
    time_to_power_cost: float

    def valid_neighbors(self, c: Coordinate) -> CoordinateList:
        valid_neighbors = [c + d for d in [Coordinate(1, 0), Coordinate(-1, 0), Coordinate(0, 1), Coordinate(0, -1)]]
        return CoordinateList(valid_neighbors)

    def cost(self, from_c: Coordinate, to_c: Coordinate) -> float:
        power_cost = 1
        return power_cost + self.time_to_power_cost

    def heuristic(self, a: Coordinate, b: Coordinate) -> float:
        dis_ab = a.distance_to(b)
        return dis_ab * self.time_to_power_cost


@dataclass
class PowerCostGraph(Graph):
    board: Board
    time_to_power_cost: float

    def valid_neighbors(self, c: Coordinate) -> CoordinateList:
        valid_neighbors = [
            c for c in self.board.get_neighbors_coordinate(c=c) if not self.board.is_opponent_factory_tile(c=c)
        ]
        return CoordinateList(valid_neighbors)

    def cost(self, from_c: Coordinate, to_c: Coordinate) -> float:
        """From_c and to_c must be neighboring points"""
        rubble_to = self.board.rubble[tuple(to_c)]
        power_cost = 1 + 0.05 * rubble_to
        return power_cost + self.time_to_power_cost

    def heuristic(self, a: Coordinate, b: Coordinate) -> float:
        dis_ab = a.distance_to(b)
        return dis_ab * self.time_to_power_cost


def get_plan_a_to_b(graph: Graph, start: Coordinate, end: Coordinate) -> list[MoveAction]:
    frontier = PriorityQueue()
    came_from: dict[Coordinate, Coordinate] = {}
    cost_so_far: dict[Coordinate, float] = {}

    cost_so_far[start] = 0
    frontier.put(start, 0)

    while not frontier.empty():
        current: Coordinate = frontier.get()

        if current == end:
            break

        for next in graph.valid_neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + graph.heuristic(next, end)
                frontier.put(next, priority)
                came_from[next] = current

    solution = []
    cur_c = end

    while cur_c in came_from:
        next_c = came_from[cur_c]
        delta = cur_c - next_c
        direction = Direction(delta)
        action = MoveAction(direction=direction, repeat=0, n=1)
        solution.insert(0, action)
        cur_c = next_c

    return solution
