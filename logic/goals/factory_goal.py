from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import abstractmethod
from dataclasses import dataclass, field

from logic.goals.goal import Goal
from logic.constraints import Constraints
from objects.actions.factory_action import BuildHeavyAction, BuildLightAction, WaterAction
from objects.game_state import GameState
from objects.actions.factory_action_plan import FactoryActionPlan

if TYPE_CHECKING:
    from objects.actors.factory import Factory


@dataclass
class FactoryGoal(Goal):
    factory: Factory

    _value: Optional[float] = field(init=False, default=None)
    # TODO, should not be valid if constraints
    _is_valid: Optional[bool] = field(init=False, default=None)

    @abstractmethod
    def generate_action_plan(self, game_state: GameState, constraints: Constraints) -> FactoryActionPlan:
        ...

    def get_value_per_step_of_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return super().get_value_per_step_of_action_plan(action_plan=action_plan, game_state=game_state)

    def get_cost_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return sum(action.get_resource_cost(game_state, self.factory.strain_id) for action in action_plan)

    @property
    def is_valid(self) -> bool:
        if self._is_valid is None:
            raise ValueError("_is_valid is not supposed to be None here")

        return self._is_valid


class BuildHeavyGoal(FactoryGoal):
    def generate_action_plan(self, game_state: GameState, constraints: Constraints) -> FactoryActionPlan:
        self.action_plan = FactoryActionPlan(self.factory, [BuildHeavyAction()])
        self.set_validity_plan(constraints)
        return self.action_plan

    def _get_best_value(self) -> float:
        return 10_000

    def get_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 15_000

    @property
    def key(self):
        return f"Build_Heavy_{self.factory.center_tc.xy}"


class BuildLightGoal(FactoryGoal):
    def generate_action_plan(self, game_state: GameState, constraints: Constraints) -> FactoryActionPlan:
        self.action_plan = FactoryActionPlan(self.factory, [BuildLightAction()])
        self.set_validity_plan(constraints)
        return self.action_plan

    def _get_best_value(self) -> float:
        return 10_000

    def get_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 1_000

    @property
    def key(self):
        return f"Build_Light_{self.factory.center_tc.xy}"


@dataclass
class WaterGoal(FactoryGoal):
    # TODO, should not be valid if can not water, or if it is too risky, next step factory will explode
    _is_valid: Optional[bool] = field(init=False, default=True)

    def generate_action_plan(self, game_state: GameState, constraints: Constraints) -> FactoryActionPlan:
        self.action_plan = FactoryActionPlan(self.factory, [WaterAction()])
        self.set_validity_plan(constraints)
        return self.action_plan

    def _get_best_value(self) -> float:
        return 10_000

    def get_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 100

    @property
    def key(self):
        return f"Water_{self.factory.center_tc.xy}"


class FactoryNoGoal(FactoryGoal):
    _value = None
    _is_valid = True

    def generate_action_plan(self, game_state: GameState, constraints: Constraints) -> FactoryActionPlan:
        return FactoryActionPlan(self.factory)

    def _get_best_value(self) -> float:
        return 0.0

    def get_benefit_action_plan(self, action_plan: FactoryActionPlan, game_state: GameState) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"No_Goal_Factory_{self.factory.center_tc.xy}"

    @property
    def key(self) -> str:
        return str(self)
