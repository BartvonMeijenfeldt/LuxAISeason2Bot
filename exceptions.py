from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from logic.goal_resolution.factory_signal import Strategy
    from logic.goals.goal import Goal
    from objects.actors.actor import Actor
    from objects.actors.factory import Factory
    from objects.coordinate import Coordinate
    from search.graph import Graph


class InvalidGoalError(Exception):
    """Exception signifying that a specific goal is not valid. Either the actor is not able to find an action plan
    to fullfill the goal, or the action plan found leads to an unprofitable goal.
    """

    def __init__(self, goal: Goal, message: str = "") -> None:
        self.goal = goal
        self.message = message

    def __str__(self) -> str:
        error_str = f"Invalid Goal for actor: {self.goal.key}"
        if self.message:
            error_str += f", message={self.message}"
        return error_str


class FactorySchedulerNoValidGoalFoundError(Exception):
    """Exception signifying that the factory did not succeed to schedule a unit on the sub_strategy they wanted.

    Args:
        factory: factory.
        sub_strategy: The sub_strategy to complete a goal. E.g., strategy=increase ice mining,
            sub_strategy=clear rubble for ice mining.
        reason: The reason no valid goal was found.
    """

    def __init__(self, factory: Factory, sub_strategy: str, reason: str = "") -> None:
        self.factory = factory
        self.sub_strategy = sub_strategy
        self.reason = reason

    def __str__(self) -> str:
        error_str = f"No valid goal found for {self.factory} to complete {self.sub_strategy}"
        if self.reason:
            error_str += f"because of {self.reason}"

        return error_str


class ActorFoundNoValidGoalError(Exception):
    """Actor found no valid goal out of a group of goals.

    Args:
        actor: Actor.
        goals: Goals to select from.
    """

    def __init__(self, actor: Actor, goals: Iterable[Goal]) -> None:
        self.actor = actor
        self.goals = goals

    def __str__(self) -> str:
        actor_str = str(self.actor)
        goals_str_list = [str(goal) for goal in self.goals]
        goals_str = ", ".join(goals_str_list)
        error_str = f"No valid goal found for {actor_str} for goals: [{goals_str}]"
        return error_str


class NoValidGoalFoundForStrategyError(Exception):
    """Exception signifying that factory is not able to find a goal for one of its units to aid in completing the
    strategy."""

    def __init__(self, factory: Factory, strategy: Strategy) -> None:
        self.factory = factory
        self.strategy = strategy

    def __str__(self) -> str:
        return f"No valid goal found for {self.factory} on {self.strategy}"


class NoSolutionSearchError(Exception):
    """Exception signifying that there is no solution from start to complete the goal for the graph.

    Args:
        start: Start coordinate for search.
        graph: Graph over which to search to complete the goal.
    """

    def __init__(self, start: Coordinate, graph: Graph) -> None:
        self.start = start
        self.graph = graph

    def __str__(self) -> str:
        return f"No solution to search for: start={self.start} -> graph={self.graph}"


class SolutionSearchNotFoundWithinBudgetError(Exception):
    """Exception signifying that there is no solution found from start to complete the goal for the graph within the
    budget.

    Args:
        start: Start coordinate for search.
        graph: Graph over which to search to complete the goal.
    """

    def __init__(self, start: Coordinate, graph: Graph) -> None:
        self.start = start
        self.graph = graph

    def __str__(self) -> str:
        return f"Solution not found within budget for: start={self.start} -> graph={self.graph}"
