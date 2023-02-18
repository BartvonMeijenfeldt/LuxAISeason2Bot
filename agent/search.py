import heapq
import math

from typing import TypeVar
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from objects.coordinate import PowerTimeCoordinate, DigTimeCoordinate, TimeCoordinate, DigCoordinate, Coordinate
from objects.direction import Direction
from objects.board import Board
from objects.action import Action, MoveAction, DigAction, PickupAction
from objects.resource import Resource
from objects.unit import UnitConfig
from logic.constraints import Constraints


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


@dataclass(kw_only=True)
class Graph(metaclass=ABCMeta):
    board: Board
    time_to_power_cost: float
    unit_cfg: UnitConfig
    constraints: Constraints = field(default_factory=Constraints)

    @abstractmethod
    def potential_actions(self, c: Coordinate) -> list[Action]:
        ...

    def get_valid_action_nodes(self, c: TimeCoordinate) -> list[tuple[Action, TimeCoordinate]]:
        action_nodes = []

        for action in self.potential_actions(c=c):
            new_c = c + action
            if self.board.is_on_the_board(c=new_c):
                action_nodes.append((action, new_c))

        return action_nodes

    def cost(self, action: Action, from_c: TimeCoordinate, to_c: TimeCoordinate) -> float:
        if self._is_constraint_violated(to_c):
            return math.inf

        action_power_cost = self.get_power_cost(action=action, from_c=from_c)
        return action_power_cost + self.time_to_power_cost

    def get_power_cost(self, action: Action, from_c: Coordinate) -> float:
        power_change = action.get_power_change(unit_cfg=self.unit_cfg, start_c=from_c, board=self.board)
        power_cost = max(0, -power_change)
        return power_cost

    @abstractmethod
    def heuristic(self, node: Coordinate) -> float:
        ...

    @abstractmethod
    def node_completes_goal(self, node: Coordinate) -> bool:
        ...

    def _get_distance_heuristic(self, node: Coordinate) -> float:
        min_nr_steps = node.distance_to(self.goal)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps * min_cost_per_step
        return min_distance_cost

    def _is_constraint_violated(self, to_c: TimeCoordinate) -> bool:
        to_tc2 = TimeCoordinate(x=to_c.x, y=to_c.y, t=to_c.t)
        return (
            self._is_negative_constraint_violated(to_c=to_tc2)
            or self._is_positive_constraint_violated(to_c=to_tc2)
            or self._is_power_constraint_violated(to_c)
        )

    def _is_negative_constraint_violated(self, to_c: TimeCoordinate) -> bool:
        if not self.constraints.negative:
            return False

        return to_c.t in self.constraints.negative and to_c in self.constraints.negative[to_c.t]

    def _is_positive_constraint_violated(self, to_c: TimeCoordinate) -> bool:
        if not self.constraints.positive:
            return False

        return to_c.t in self.constraints.positive and to_c != self.constraints.positive[to_c.t]

    def _is_power_constraint_violated(self, to_c: TimeCoordinate) -> bool:
        if not self.constraints.max_power_request or not isinstance(to_c, PowerTimeCoordinate):
            return False

        return to_c.power_recharged > self.constraints.max_power_request


@dataclass(kw_only=True)
class MoveToGraph(Graph):
    goal: Coordinate
    _potential_actions = [MoveAction(direction) for direction in Direction]

    def potential_actions(self, c: Coordinate) -> list[Action]:
        return self._potential_actions

    def heuristic(self, node: Coordinate) -> float:
        return self._get_distance_heuristic(node=node)

    def node_completes_goal(self, node: Coordinate) -> bool:
        return self.goal == node


@dataclass(kw_only=True)
class PickupPowerGraph(Graph):
    power_pickup_goal: int
    _potential_move_actions = [MoveAction(direction) for direction in Direction]

    def __post_init__(self):
        if self.constraints.max_power_request and self.constraints.max_power_request < self.power_pickup_goal:
            self.power_pickup_goal = self.constraints.max_power_request

        self._potential_recharge_actions = [PickupAction(amount=self.power_pickup_goal, resource=Resource.Power)]

    def potential_actions(self, c: Coordinate) -> list[Action]:
        if self.board.is_player_factory_tile(c=c):
            return self._potential_move_actions + self._potential_recharge_actions
        else:
            return self._potential_move_actions

    def heuristic(self, node: PowerTimeCoordinate) -> float:
        min_distance_cost = self._get_distance_heuristic(node=node)
        min_time_recharge_cost = self._get_time_recharge_heuristic(node=node)
        return min_distance_cost + min_time_recharge_cost

    def _get_distance_heuristic(self, node: Coordinate) -> float:
        closest_factory_tile = self.board.get_closest_factory_tile(node)
        min_distance_to_factory = closest_factory_tile.distance_to(node)
        return min_distance_to_factory

    def _get_time_recharge_heuristic(self, node: PowerTimeCoordinate) -> float:
        if self.node_completes_goal(node=node):
            return 0
        else:
            return self.time_to_power_cost

    def node_completes_goal(self, node: PowerTimeCoordinate) -> bool:
        return self.power_pickup_goal <= node.power_recharged


@dataclass(kw_only=True)
class DigAtGraph(Graph):
    goal: DigCoordinate
    _potential_move_actions = [MoveAction(direction) for direction in Direction]
    _potential_dig_actions = [DigAction()]

    def potential_actions(self, c: Coordinate) -> list[Action]:
        if self.goal.x == c.x and self.goal.y == c.y:
            return self._potential_move_actions + self._potential_dig_actions
        else:
            return self._potential_move_actions

    def heuristic(self, node: DigTimeCoordinate) -> float:
        distance_min_cost = self._get_distance_heuristic(node=node)
        digs_min_cost = self._get_digs_min_cost(node=node)

        return distance_min_cost + digs_min_cost

    def _get_digs_min_cost(self, node: DigTimeCoordinate) -> float:
        nr_digs_required = self.goal.nr_digs - node.nr_digs
        cost_per_dig = self.unit_cfg.DIG_COST + self.time_to_power_cost
        min_cost = nr_digs_required * cost_per_dig
        return min_cost

    def node_completes_goal(self, node: DigTimeCoordinate) -> bool:
        return self.goal == node


class Search:
    def __init__(self, graph: Graph) -> None:
        self.frontier = PriorityQueue()
        self.came_from: dict[TimeCoordinate, tuple[Action, TimeCoordinate]] = {}
        self.cost_so_far: dict[TimeCoordinate, float] = {}
        self.graph = graph

    def get_actions_to_complete_goal(self, start: TimeCoordinate) -> list[Action]:
        self._init_search(start)
        self._find_optimal_solution()
        return self._get_solution_actions()

    def _init_search(self, start: TimeCoordinate) -> None:
        self.cost_so_far[start] = 0
        self.frontier.put(start, 0)

    def _find_optimal_solution(self) -> None:
        while not self.frontier.empty():
            current_node: TimeCoordinate = self.frontier.get()

            if self.graph.node_completes_goal(node=current_node):
                break

            current_cost = self.cost_so_far[current_node]

            for action, next_node in self.graph.get_valid_action_nodes(current_node):
                new_cost = current_cost + self.graph.cost(action=action, from_c=current_node, to_c=next_node)
                # With a good heuristic (new_cost < cost_so_far[node]) shouldn't be relevant
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]:
                    self._add_node(node=next_node, action=action, current_node=current_node, node_cost=new_cost)

        self.final_node = current_node

    def _add_node(self, node: TimeCoordinate, action: Action, current_node: TimeCoordinate, node_cost: float) -> None:
        self.cost_so_far[node] = node_cost
        priority = node_cost + self.graph.heuristic(node)
        self.frontier.put(node, priority)
        self.came_from[node] = (action, current_node)

    def _get_solution_actions(self) -> list[Action]:
        solution = []
        cur_c = self.final_node

        while cur_c in self.came_from:
            action, cur_c = self.came_from[cur_c]
            solution.insert(0, action)

        return solution
