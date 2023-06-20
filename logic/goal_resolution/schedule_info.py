from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from logic.goals.unit_goal import DigGoal
from objects.direction import Direction

if TYPE_CHECKING:
    from logic.constraints import Constraints
    from logic.goal_resolution.power_tracker import PowerTracker
    from objects.actors.unit import Unit
    from objects.coordinate import Coordinate
    from objects.game_state import GameState


@dataclass
class ScheduleInfo:
    """Class that contains all relevant info for scheduling decisions, the game_state, the constraints and the
    power_tracker and that is able to make independent copies of itself.

    Args:
        game_state: Current game state.
        constraints: Constraints, which time coordinates are promised to units
        power_tracker: Object that tracks for all factories what power is available for request.
    """

    game_state: GameState
    constraints: Constraints
    power_tracker: PowerTracker

    def copy_without_unit_scheduled_actions(self, unit: Unit) -> ScheduleInfo:
        """Creates a copy without the units scheduled actions effect on constraints and power tracker. The use of this
        is when you want to consider a new action plan for a unit without already definitively unschedeling the unit.

        Args:
            unit: Unit whose actions to remove

        Returns:
            ScheduleInfo object without the unit's scheduled actions.
        """
        if unit.is_scheduled:
            game_state = self.game_state
            constraints = copy(self.constraints)
            power_tracker = copy(self.power_tracker)
            constraints.remove_negative_constraints(unit.private_action_plan.get_time_coordinates(game_state))
            power_tracker.remove_power_requests(unit.private_action_plan.get_power_requests(game_state))
            schedule_info = replace(self, constraints=constraints, power_tracker=power_tracker)

        elif unit.can_not_move_this_step(self.game_state):
            constraints = copy(self.constraints)
            next_tc = unit.tc + Direction.CENTER
            constraints.remove_negative_constraints([next_tc])
            schedule_info = replace(self, constraints=constraints)

        else:
            schedule_info = self

        return schedule_info

    def copy_without_units_on_dig_c(self, c: Coordinate) -> ScheduleInfo:
        """Creates a copy without the scheduled actions effect on constraints and power tracker for all units planning
        to dig on the given Coordinate. The use of this is when you want to consider a new action plan for a unit to dig
        on the given coordinate, but where other units might already be planning to dig on.

        Args:
            c: Coordinate from which to remove scheduled units actions if they plan to dig there.

        Returns:
           ScheduleInfo object without any scheduled actions of units planning to dig on the given Coordinate.
        """
        game_state = self.game_state
        constraints = copy(self.constraints)
        power_tracker = copy(self.power_tracker)

        for unit in game_state.units:
            if (
                unit.is_scheduled
                and isinstance(unit.goal, DigGoal)
                and unit.goal.dig_c == c
                and unit.private_action_plan
            ):
                constraints.remove_negative_constraints(unit.private_action_plan.get_time_coordinates(game_state))
                power_tracker.remove_power_requests(unit.private_action_plan.get_power_requests(game_state))

        schedule_info = replace(self, constraints=constraints, power_tracker=power_tracker)

        return schedule_info
