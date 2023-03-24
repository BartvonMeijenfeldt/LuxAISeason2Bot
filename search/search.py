from __future__ import annotations

from typing import List, Tuple, Generator, Optional, TYPE_CHECKING
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from objects.coordinate import (
    ResourcePowerTimeCoordinate,
    DigTimeCoordinate,
    TimeCoordinate,
    DigCoordinate,
    Coordinate,
)
from objects.direction import Direction
from objects.board import Board
from objects.actions.unit_action import UnitAction, MoveAction, DigAction, PickupAction, TransferAction
from objects.resource import Resource

from logic.constraints import Constraints
from utils import PriorityQueue

if TYPE_CHECKING:
    from objects.actors.unit import UnitConfig
    from logic.goal_resolution.power_availabilty_tracker import PowerAvailabilityTracker


@dataclass
class Graph(metaclass=ABCMeta):
    board: Board
    time_to_power_cost: float
    unit_cfg: UnitConfig
    constraints: Constraints
    goal: Coordinate = field(init=False)

    @abstractmethod
    def potential_actions(self, c: TimeCoordinate) -> Generator[UnitAction, None, None]:
        ...

    def get_valid_action_nodes(self, c: TimeCoordinate) -> Generator[Tuple[UnitAction, TimeCoordinate], None, None]:
        for action in self.potential_actions(c=c):
            to_c = c.add_action(action)
            if not self.constraints.tc_violates_constraint(to_c) and self.board.is_valid_c_for_player(c=to_c):
                yield ((action, to_c))

    def cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        action_power_cost = self.get_power_cost(action=action, to_c=to_c)
        return action_power_cost + self.time_to_power_cost

    def get_power_cost(self, action: UnitAction, to_c: Coordinate) -> float:
        power_change = action.get_power_change_by_end_c(unit_cfg=self.unit_cfg, end_c=to_c, board=self.board)
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
        tiebreak_cost = min_nr_steps / 100_000
        return min_distance_cost + tiebreak_cost


@dataclass
class MoveToGraph(Graph):
    goal: Coordinate
    _potential_actions = [MoveAction(direction) for direction in Direction]

    def __post_init__(self):
        if not self.constraints:
            self._potential_actions = [
                MoveAction(direction) for direction in Direction if direction != direction.CENTER
            ]

    def potential_actions(self, c: TimeCoordinate) -> List[MoveAction]:
        return self._potential_actions

    def heuristic(self, node: Coordinate) -> float:
        return self._get_distance_heuristic(node=node)

    def node_completes_goal(self, node: Coordinate) -> bool:
        return self.goal == node


class EvadeConstraintsGraph(Graph):
    _potential_actions = [MoveAction(direction) for direction in Direction]
    _move_center_action = MoveAction(Direction.CENTER)

    def potential_actions(self, c: TimeCoordinate) -> List[MoveAction]:
        return self._potential_actions

    def heuristic(self, node: TimeCoordinate) -> float:
        return -node.t

    def node_completes_goal(self, node: TimeCoordinate) -> bool:
        to_c = node.add_action(self._move_center_action)
        return not self.constraints.tc_violates_constraint(to_c)


@dataclass
class PickupPowerGraph(Graph):
    factory_power_availability_tracker: PowerAvailabilityTracker
    next_goal_c: Optional[Coordinate] = field(default=None)
    _potential_move_actions = [MoveAction(direction) for direction in Direction]

    def potential_actions(self, c: ResourcePowerTimeCoordinate) -> Generator[UnitAction, None, None]:
        if self.board.is_player_factory_tile(c=c):
            factory = self.board.get_closest_player_factory(c=c)
            power_available_in_factory = self.factory_power_availability_tracker.get_power_available(factory, c.t)
            battery_space_left = self.unit_cfg.BATTERY_CAPACITY - c.p
            power_pickup_amount = min(battery_space_left, power_available_in_factory)

            if power_pickup_amount:
                potential_recharge_action = PickupAction(amount=power_pickup_amount, resource=Resource.POWER)
                yield (potential_recharge_action)

        for action in self._potential_move_actions:
            yield (action)

    def cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        move_cost = super().cost(action, to_c)
        if self.next_goal_c is None or not isinstance(action, PickupAction):
            return move_cost

        distance_to_goal = to_c.distance_to(self.next_goal_c)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = distance_to_goal * min_cost_per_step
        return move_cost + min_distance_cost

    def heuristic(self, node: ResourcePowerTimeCoordinate) -> float:
        if self.node_completes_goal(node):
            return 0

        min_distance_cost = self._get_distance_heuristic(node=node)
        min_time_recharge_cost = self._get_time_recharge_heuristic(node=node)
        return min_distance_cost + min_time_recharge_cost

    def _get_distance_heuristic(self, node: Coordinate) -> float:
        closest_factory_tile = self.board.get_closest_player_factory_tile(node)
        distance_to_closest_factory_tiles = node.distance_to(closest_factory_tile)

        if self.next_goal_c:
            # TODO, now it calculates from closest_factory_tile the heuristic, it could be that a tile at a different
            # factory will have the min distance if you take into account the next goal
            min_distance_factory_to_next_goal = self.next_goal_c.distance_to(closest_factory_tile)
            total_distance = distance_to_closest_factory_tiles + min_distance_factory_to_next_goal
        else:
            total_distance = distance_to_closest_factory_tiles

        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = total_distance * min_cost_per_step
        tiebreak_cost = distance_to_closest_factory_tiles / 100_000

        return min_distance_cost + tiebreak_cost

    def _get_time_recharge_heuristic(self, node: ResourcePowerTimeCoordinate) -> float:
        if self.node_completes_goal(node=node):
            return 0
        else:
            return self.time_to_power_cost

    def node_completes_goal(self, node: ResourcePowerTimeCoordinate) -> bool:
        return node.q > 0


@dataclass
class TransferResourceGraph(Graph):
    _potential_move_actions = [MoveAction(direction) for direction in Direction]
    resource: Resource

    def potential_actions(self, c: ResourcePowerTimeCoordinate) -> Generator[UnitAction, None, None]:
        if self.board.get_min_distance_to_player_factory(c=c) <= 1:
            factory_tile = self.board.get_closest_player_factory_tile(c=c)
            dir = c.direction_to(factory_tile)
            transfer_action = TransferAction(direction=dir, amount=self.unit_cfg.CARGO_SPACE, resource=self.resource)
            yield (transfer_action)

        for action in self._potential_move_actions:
            yield (action)

    def heuristic(self, node: ResourcePowerTimeCoordinate) -> float:
        if self.node_completes_goal(node):
            return 0

        min_distance_cost = self._get_distance_heuristic(node=node)
        min_time_recharge_cost = self._get_transfer_heuristic(node=node)
        return min_distance_cost + min_time_recharge_cost

    def _get_distance_heuristic(self, node: Coordinate) -> float:
        min_nr_steps_to_factory = self.board.get_min_distance_to_player_factory(c=node)
        min_nr_steps_next_to_factory = max(min_nr_steps_to_factory - 1, 0)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps_next_to_factory * min_cost_per_step
        tiebreak_cost = min_nr_steps_to_factory / 100_000
        return min_distance_cost + tiebreak_cost

    def _get_transfer_heuristic(self, node: ResourcePowerTimeCoordinate) -> float:
        if self.node_completes_goal(node=node):
            return 0
        else:
            return self.time_to_power_cost

    def node_completes_goal(self, node: ResourcePowerTimeCoordinate) -> bool:
        return node.q < 0


@dataclass
class DigAtGraph(Graph):
    goal: DigCoordinate
    _potential_move_actions = [MoveAction(direction) for direction in Direction]
    _potential_dig_action = DigAction()

    def __post_init__(self):
        if not self.constraints:
            self._potential_move_actions = [
                MoveAction(direction) for direction in Direction if direction != direction.CENTER
            ]

    def potential_actions(self, c: TimeCoordinate) -> Generator[UnitAction, None, None]:
        if self.goal.x == c.x and self.goal.y == c.y:
            if self.constraints and self.constraints.max_t <= c.t:
                yield (self._potential_dig_action)
            else:
                for action in self._potential_move_actions:
                    yield (action)

                yield (self._potential_dig_action)
        else:
            for action in self._potential_move_actions:
                yield (action)

    def heuristic(self, node: DigTimeCoordinate) -> float:
        distance_min_cost = self._get_distance_heuristic(node=node)
        digs_min_cost = self._get_digs_min_cost(node=node)

        return distance_min_cost + digs_min_cost

    def _get_digs_min_cost(self, node: DigTimeCoordinate) -> float:
        nr_digs_required = self.goal.d - node.d
        cost_per_dig = self.unit_cfg.DIG_COST + self.time_to_power_cost
        min_cost = nr_digs_required * cost_per_dig
        return min_cost

    def node_completes_goal(self, node: DigTimeCoordinate) -> bool:
        return self.goal == node


class Search:
    def __init__(self, graph: Graph) -> None:
        self.frontier = PriorityQueue()

        self.came_from: dict[TimeCoordinate, tuple[UnitAction, TimeCoordinate]] = {}
        self.cost_so_far: dict[TimeCoordinate, float] = {}
        self.graph = graph

    def get_actions_to_complete_goal(self, start: TimeCoordinate) -> List[UnitAction]:
        self._init_search(start)
        self._find_optimal_solution()
        return self._get_solution_actions()

    def _init_search(self, start_tc: TimeCoordinate) -> None:
        # if self.graph.constraints:
        #     start = start_tc
        # else:
        #     start = start_tc.to_timeless_coordinate()
        start = start_tc

        self.cost_so_far[start] = 0
        self.frontier.put(start, 0)

    def _find_optimal_solution(self) -> None:
        while not self.frontier.is_empty():
            current_node: TimeCoordinate = self.frontier.pop()

            if self.graph.node_completes_goal(node=current_node):
                break

            current_cost = self.cost_so_far[current_node]

            for action, next_node in self.graph.get_valid_action_nodes(current_node):
                new_cost = current_cost + self.graph.cost(action=action, to_c=next_node)
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]:
                    self._add_node(node=next_node, action=action, current_node=current_node, node_cost=new_cost)

        self.final_node = current_node

    def _add_node(
        self, node: TimeCoordinate, action: UnitAction, current_node: TimeCoordinate, node_cost: float
    ) -> None:
        self.cost_so_far[node] = node_cost
        self.came_from[node] = (action, current_node)

        priority = node_cost + self.graph.heuristic(node)
        self.frontier.put(node, priority)

    def _get_solution_actions(self) -> List[UnitAction]:
        solution = []
        cur_c = self.final_node

        while cur_c in self.came_from:
            action, cur_c = self.came_from[cur_c]
            solution.insert(0, action)

        return solution
