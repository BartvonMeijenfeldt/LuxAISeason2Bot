from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from objects.actions.action_plan import ActionPlan
from logic.constraints import Constraints


if TYPE_CHECKING:
    from objects.game_state import GameState


@dataclass
class Goal(metaclass=ABCMeta):
    def generate_and_evaluate_action_plan(
        self, game_state: GameState, constraints: Optional[Constraints] = None
    ) -> ActionPlan:
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

    @property
    @abstractmethod
    def value(self) -> float:
        ...


class GoalCollection:
    def __init__(self, goals: Sequence[Goal]) -> None:
        self.goals_dict = {goal.key: goal for goal in goals}

    def generate_and_evaluate_action_plans(
        self, game_state: GameState, constraints: Optional[Constraints] = None
    ) -> None:
        for goal in self.goals_dict.values():
            goal.generate_and_evaluate_action_plan(game_state=game_state, constraints=constraints)

    def get_goal(self, key: str) -> Goal:
        return self.goals_dict[key]

    def get_keys(self) -> set[str]:
        return {key for key, goal in self.goals_dict.items() if goal.is_valid}

    def get_key_values(self, game_state: GameState, constraints: Optional[Constraints] = None) -> dict[str, float]:
        self.generate_and_evaluate_action_plans(game_state=game_state, constraints=constraints)
        return {key: goal.value for key, goal in self.goals_dict.items() if goal.is_valid}
