from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import List, Set, Sequence, TYPE_CHECKING
from dataclasses import dataclass

from objects.cargo import UnitCargo
from utils import PriorityQueue

if TYPE_CHECKING:
    from logic.goal_resolution.power_availabilty_tracker import PowerAvailabilityTracker
    from logic.goals.goal import Goal
    from logic.constraints import Constraints
    from lux.kit import GameState


@dataclass
class Actor(metaclass=ABCMeta):
    team_id: int
    unit_id: str
    power: int
    cargo: UnitCargo

    # TODO should remove the action_queu_goal here and let it automatically be generated in generate goals
    def get_best_goal(
        self,
        game_state: GameState,
        constraints: Constraints,
        factory_power_availability_tracker: PowerAvailabilityTracker,
        reserved_goals: Set[str] = set(),
    ) -> Goal:
        goals = self.generate_goals(game_state)
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

    def _init_priority_queue(self, goals: Sequence[Goal], reserved_goals: Set[str], game_state: GameState
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
    def generate_goals(self, game_state: GameState) -> List[Goal]:
        ...
