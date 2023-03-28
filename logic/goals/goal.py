from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from objects.actions.action_plan import ActionPlan
from objects.actions.unit_action_plan import UnitActionPlan
from logic.constraints import Constraints


if TYPE_CHECKING:
    from objects.game_state import GameState
    from logic.goal_resolution.power_availabilty_tracker import PowerAvailabilityTracker


@dataclass
class Goal(metaclass=ABCMeta):
    def generate_and_evaluate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        factory_power_availability_tracker: PowerAvailabilityTracker,
    ) -> ActionPlan:
        try:
            self.action_plan = self.generate_action_plan(game_state, constraints, factory_power_availability_tracker)
        except Exception:
            self._is_valid = False

        if isinstance(self.action_plan, UnitActionPlan) and not self.action_plan.is_valid_size:
            self._is_valid = False

        if self.is_valid:
            self._value = self.get_value_per_step_of_action_plan(action_plan=self.action_plan, game_state=game_state)
        return self.action_plan

    @abstractmethod
    def generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        factory_power_availability_tracker: PowerAvailabilityTracker,
    ) -> ActionPlan:
        ...

    def get_value_per_step_of_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        benefit = self.get_benefit_action_plan(action_plan=action_plan, game_state=game_state)
        cost = self.get_cost_action_plan(action_plan=action_plan, game_state=game_state)
        value = benefit - cost
        value_per_step = value / max(action_plan.nr_time_steps, 1)

        return value_per_step

    @abstractmethod
    def get_benefit_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @abstractmethod
    def get_cost_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @property
    @abstractmethod
    def key(self) -> str:
        ...

    @property
    @abstractmethod
    def is_valid(self) -> bool:
        ...

    @abstractmethod
    def get_best_value_per_step(self, game_state: GameState) -> float:
        ...

    @property
    def value(self) -> float:
        if self._value is None:
            raise ValueError("Value is not supposed to be None here")

        return self._value

    def set_validity_plan(self, constraints: Constraints) -> None:
        self._is_valid = not constraints.any_tc_violates_constraint(self.action_plan.time_coordinates)


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
