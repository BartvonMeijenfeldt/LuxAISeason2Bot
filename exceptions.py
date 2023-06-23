from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.goals.goal import Goal
    from objects.coordinate import Coordinate
    from search.graph import Graph


class InvalidGoalError(Exception):
    def __init__(self, goal: Goal) -> None:
        self.goal = goal

    def __str__(self) -> str:
        return f"Invalid Goal: {self.goal}"


class NoValidGoalFoundError(Exception):
    def __str__(self) -> str:
        return "No valid goal found"


class NoSolutionError(Exception):
    def __init__(self, start: Coordinate, graph: Graph) -> None:
        self.start = start
        self.graph = graph

    def __str__(self) -> str:
        return f"No solution to search for: start={self.start} -> graph={self.graph}"


class SolutionNotFoundWithinBudgetError(Exception):
    def __init__(self, start: Coordinate, graph: Graph) -> None:
        self.start = start
        self.graph = graph

    def __str__(self) -> str:
        return f"Solution not found within budget for: start={self.start} -> graph={self.graph}"
