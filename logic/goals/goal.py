from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from objects.actions.action_plan import ActionPlan
from logic.constraints import Constraints


if TYPE_CHECKING:
    from objects.game_state import GameState


@dataclass
class Goal(metaclass=ABCMeta):
    def generate_and_evaluate_action_plan(self, game_state: GameState, constraints: Constraints) -> ActionPlan:
        self.action_plan = self.generate_action_plan(game_state=game_state, constraints=constraints)
        self._value = self.get_value_action_plan(action_plan=self.action_plan, game_state=game_state)
        return self.action_plan

    @abstractmethod
    def generate_action_plan(self, game_state: GameState, constraints: Optional[Constraints] = None) -> ActionPlan:
        ...

    @abstractmethod
    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @property
    @abstractmethod
    def key(self) -> str:
        ...

    @property
    @abstractmethod
    def is_valid(self) -> bool:
        ...

    def __lt__(self, other: Goal) -> bool:
        return self._get_best_value() < other._get_best_value()

    @property
    def best_value(self) -> float:
        if self._value is None:
            return self._get_best_value()

        return self._value

    @abstractmethod
    def _get_best_value(self) -> float:
        ...

    @property
    def value(self) -> float:
        if self._value is None:
            raise ValueError("Value is not supposed to be None here")

        return self._value

    def set_validity_plan(self, constraints: Constraints) -> None:
        for tc in self.action_plan.time_coordinates:
            if constraints.tc_violates_constraint(tc):
                self._is_valid = False
                return

        power_requested = self.action_plan.power_requested
        if constraints.max_power_request and power_requested > constraints.max_power_request:
            self._is_valid = False
            return

        self._is_valid = True


# class GoalCollection:
#     def __init__(self, goals: Sequence[Goal]) -> None:
#         self.goals_dict = {goal.key: goal for goal in goals}

#     def generate_and_evaluate_action_plans(self, game_state: GameState, constraints: Constraints) -> None:
#         for goal in self.goals_dict.values():
#             goal.generate_and_evaluate_action_plan(game_state=game_state, constraints=constraints)

#     def get_goal(self, key: str) -> Goal:
#         return self.goals_dict[key]

#     def get_keys(self) -> set[str]:
#         return {key for key, goal in self.goals_dict.items() if goal.is_valid}

#     def get_key_values(self, game_state: GameState, constraints: Constraints) -> dict[str, float]:
#         self.generate_and_evaluate_action_plans(game_state=game_state, constraints=constraints)
#         return {key: goal.value for key, goal in self.goals_dict.items() if goal.is_valid}
