from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from objects.actions.action_plan import ActionPlan

if TYPE_CHECKING:
    from objects.game_state import GameState
    from logic.goal_resolution.schedule_info import ScheduleInfo


@dataclass
class Goal(metaclass=ABCMeta):
    @abstractmethod
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> ActionPlan:
        ...

    def get_value_per_step_of_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        benefit = self.get_power_benefit_action_plan(action_plan=action_plan, game_state=game_state)
        cost = self.get_power_cost_action_plan(action_plan=action_plan, game_state=game_state)
        value = benefit - cost
        value_per_step = value / max(action_plan.nr_time_steps, 1)

        return value_per_step

    @abstractmethod
    def get_power_benefit_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @abstractmethod
    def get_power_cost_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @property
    @abstractmethod
    def key(self) -> str:
        ...

    @abstractmethod
    def get_best_case_value_per_step(self, game_state: GameState) -> float:
        ...
