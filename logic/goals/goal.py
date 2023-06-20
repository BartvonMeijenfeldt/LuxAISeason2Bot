from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from objects.actions.action_plan import ActionPlan

if TYPE_CHECKING:
    from logic.goal_resolution.schedule_info import ScheduleInfo
    from objects.game_state import GameState


@dataclass
class Goal(metaclass=ABCMeta):
    """Goal of an actor. What is an actor trying to achieve?"""

    @abstractmethod
    def generate_action_plan(self, schedule_info: ScheduleInfo) -> ActionPlan:
        """Create a specific action plan to complete the goal.

        Args:
            schedule_info: Schedule Info

        Returns:
            Action plan to complete the goal.
        """
        ...

    def get_value_per_step_of_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        """Value per step to complete the goal given a action plan.

        Args:
            action_plan: Action plan to carry out goal
            game_state: Current game state

        Returns:
            Value per step
        """
        benefit = self.get_power_benefit_action_plan(action_plan=action_plan, game_state=game_state)
        cost = self.get_power_cost_action_plan(action_plan=action_plan, game_state=game_state)
        value = benefit - cost
        value_per_step = value / max(action_plan.nr_time_steps, 1)

        return value_per_step

    @abstractmethod
    def get_power_benefit_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        """Computes power benefit of completing the goal given the action plan. For all goals an approximation is made
        on how much power this will generate if completed. For some goals this must be approximated.

        The idea to translate all goals to a common unit is similar to the centipawns used in chess:
        https://www.chessprogramming.org/Centipawns

        Args:
            action_plan: Action plan to complete the goal
            game_state: Current game state.

        Returns:
            Total benefit of action plan expressed in power gained
        """
        ...

    @abstractmethod
    def get_power_cost_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        """Computes power cost of completing the goal given the action plan.

        Args:
            action_plan: Action plan to complete the goal
            game_state: Current game state.

        Returns:
            Total benefit of action plan expressed in power gained
        """
        ...

    @property
    @abstractmethod
    def key(self) -> str:
        """Unique unit, goal combination."""
        ...

    @abstractmethod
    def get_best_case_value_per_step(self, game_state: GameState) -> float:
        """Computes the best case value per step. This is a quick estimation of the best possible value of completing
        this goal. This method is used to have a rough estimation of the value of goals before going through the more
        expensive process of computing action plans.

        Args:
            game_state: Game State

        Returns:
            Best case value  per step
        """
        ...
