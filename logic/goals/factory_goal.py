from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from logic.goals.goal import Goal
from objects.actions.factory_action import (
    BuildHeavyAction,
    BuildLightAction,
    WaterAction,
)
from objects.actions.factory_action_plan import FactoryActionPlan
from objects.game_state import GameState

if TYPE_CHECKING:
    from logic.goal_resolution.schedule_info import ScheduleInfo
    from objects.actors.factory import Factory


@dataclass
class FactoryGoal(Goal):
    factory: Factory

    @abstractmethod
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        ...

    def get_value_per_step_of_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return super().get_value_per_step_of_action_plan(action_plan=action_plan, game_state=game_state)

    def get_power_cost_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return sum(action.get_resource_cost(self.factory) for action in action_plan)


class BuildHeavyGoal(FactoryGoal):
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        self.action_plan = FactoryActionPlan(self.factory, [BuildHeavyAction()])
        return self.action_plan

    def get_best_case_value_per_step(self, game_state: GameState) -> float:
        return 10_000

    def get_power_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 15_000

    @property
    def key(self):
        return f"Build_Heavy_{self.factory.center_tc.xy}"


class BuildLightGoal(FactoryGoal):
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        self.action_plan = FactoryActionPlan(self.factory, [BuildLightAction()])
        return self.action_plan

    def get_best_case_value_per_step(self, game_state: GameState) -> float:
        return 10_000

    def get_power_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 1_000

    @property
    def key(self):
        return f"Build_Light_{self.factory.center_tc.xy}"


@dataclass
class WaterGoal(FactoryGoal):
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        self.action_plan = FactoryActionPlan(self.factory, [WaterAction()])
        return self.action_plan

    def get_best_case_value_per_step(self, game_state: GameState) -> float:
        return 10_000

    def get_power_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 100

    @property
    def key(self):
        return f"Water_{self.factory.center_tc.xy}"


class FactoryNoGoal(FactoryGoal):
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        return FactoryActionPlan(self.factory)

    def get_best_case_value_per_step(self, game_state: GameState) -> float:
        return 0.0

    def get_power_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"No_Goal_Factory_{self.factory.center_tc.xy}"

    @property
    def key(self) -> str:
        return str(self)
