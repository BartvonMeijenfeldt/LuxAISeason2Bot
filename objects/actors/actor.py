from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Set, TYPE_CHECKING


from utils import PriorityQueue

if TYPE_CHECKING:
    from objects.cargo import UnitCargo
    from logic.goal_resolution.power_availabilty_tracker import PowerAvailabilityTracker
    from logic.goals.goal import Goal, GoalCollection
    from logic.constraints import Constraints
    from lux.kit import GameState


class Actor(metaclass=ABCMeta):
    def __init__(self, team_id: int, unit_id: str, power: int, cargo: UnitCargo) -> None:
        self.team_id = team_id
        self.unit_id = unit_id
        self.power = power
        self.cargo = cargo

    # TODO should remove the action_queu_goal here and let it automatically be generated in generate goals
    def get_best_goal(
        self,
        goals: list[Goal],
        game_state: GameState,
        constraints: Constraints,
        factory_power_availability_tracker: PowerAvailabilityTracker,
        reserved_goals: Set[str] = set(),
    ) -> Goal:
        # goals = self.generate_goals(game_state)
        priority_queue = self._init_priority_queue(goals, reserved_goals, game_state)

        while not priority_queue.is_empty():
            goal: Goal = priority_queue.pop()

            goal.generate_and_evaluate_action_plan(game_state, constraints, factory_power_availability_tracker)
            if not goal.is_valid:
                continue

            priority = -1 * goal.value
            priority_queue.put(goal, priority)

            if goal == priority_queue[0]:
                return goal

        raise RuntimeError("No best goal was found")

    def _init_priority_queue(
        self, goals: list[Goal], reserved_goals: Set[str], game_state: GameState
    ) -> PriorityQueue:
        goals_priority_queue = PriorityQueue()

        for goal in goals:
            if goal.key in reserved_goals:
                continue

            best_value = goal.get_best_value_per_step(game_state)
            priority = -1 * best_value
            goals_priority_queue.put(goal, priority)

        return goals_priority_queue

    @abstractmethod
    def generate_goals(self, game_state: GameState) -> GoalCollection:
        ...
