from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import List, Set, TYPE_CHECKING, Optional
from dataclasses import dataclass

from objects.cargo import UnitCargo
from logic.goals.unit_goal import ActionQueueGoal
from utils import PriorityQueue

if TYPE_CHECKING:
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
        reserved_goals: Set[str] = set(),
        action_queue_goal: Optional[ActionQueueGoal] = None,
    ) -> Goal:
        goals = self.generate_goals(game_state)
        if action_queue_goal:
            goals.append(action_queue_goal)
            goals = [action_queue_goal, goals[-2]]

        goals_priority_queue = PriorityQueue()

        for goal in goals:
            if goal.key in reserved_goals:
                continue

            priority = -1 * goal.best_value
            goals_priority_queue.put(goal, priority)

        while not goals_priority_queue.is_empty():
            goal = goals_priority_queue.pop()

            goal.generate_and_evaluate_action_plan(game_state, constraints)
            if not goal.is_valid:
                continue

            if goals_priority_queue.is_empty() or goal.value >= goals_priority_queue[0].best_value:
                return goal

            priority = -1 * goal.best_value
            goals_priority_queue.put(goal, priority)

        raise RuntimeError("No best goal was found")

    @abstractmethod
    def generate_goals(self, game_state: GameState) -> List[Goal]:
        ...