from __future__ import annotations

import itertools
from typing import List, Optional

from config import CONFIG
from exceptions import NoSolutionError, SolutionNotFoundWithinBudgetError
from objects.actions.unit_action import UnitAction
from objects.coordinate import Coordinate, TimeCoordinate
from search.graph import Graph
from utils.utils import PriorityQueue


class Search:
    def __init__(self, graph: Graph) -> None:
        self.frontier = PriorityQueue()

        self.came_from: dict[Coordinate, tuple[UnitAction, Coordinate]] = {}
        self.cost_so_far: dict[Coordinate, float] = {}
        self.graph = graph

    def get_actions_to_complete_goal(self, start: Coordinate, budget: Optional[int] = None) -> List[UnitAction]:
        self._init_search(start)
        final_node = self._find_optimal_solution(budget)
        return self._get_solution_actions(final_node)

    def _init_search(self, start_tc: Coordinate) -> None:
        self._start = start_tc
        self.cost_so_far[self._start] = 0
        self.frontier.put(self._start, 0)

    def _find_optimal_solution(self, budget: Optional[int] = None) -> Coordinate:  # type: ignore
        if not budget:
            budget = CONFIG.SEARCH_BUDGET_HEAVY if self.graph.unit_type == "HEAVY" else CONFIG.SEARCH_BUDGET_LIGHT

        for i in itertools.count():
            if self.frontier.is_empty():
                raise NoSolutionError(self._start, self.graph)
            if i > budget:
                raise SolutionNotFoundWithinBudgetError(self._start, self.graph)

            current_node = self.frontier.pop()

            if self.graph.completes_goal(tc=current_node):
                return current_node

            current_cost = self.cost_so_far[current_node]

            for action, next_node in self.graph.get_valid_action_nodes(current_node):
                new_cost = current_cost + self.graph.get_cost(action=action, to_c=next_node)
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]:
                    self._add_node(node=next_node, action=action, current_node=current_node, node_cost=new_cost)

    def _add_node(
        self, node: TimeCoordinate, action: UnitAction, current_node: TimeCoordinate, node_cost: float
    ) -> None:
        self.cost_so_far[node] = node_cost
        self.came_from[node] = (action, current_node)

        priority = node_cost + self.graph.get_heuristic(node)
        self.frontier.put(node, priority)

    def _get_solution_actions(self, final_node: Coordinate) -> List[UnitAction]:
        solution = []
        cur_c = final_node

        while cur_c in self.came_from:
            action, cur_c = self.came_from[cur_c]
            solution.insert(0, action)

        return solution
