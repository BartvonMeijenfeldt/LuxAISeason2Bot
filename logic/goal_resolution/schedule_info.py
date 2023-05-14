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
    game_state: GameState
    constraints: Constraints
    power_tracker: PowerTracker

    def copy_without_unit_scheduled_actions(self, unit: Unit) -> ScheduleInfo:
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
