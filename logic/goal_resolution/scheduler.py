from typing import Dict, Tuple, List
from collections import defaultdict

from objects.game_state import GameState
from objects.coordinate import TimeCoordinate
from objects.actors.factory import Factory, Strategy
from objects.actors.unit import Unit
from objects.direction import Direction
from logic.constraints import Constraints
from logic.goal_resolution.power_tracker import PowerTracker
from logic.goal_resolution.schedule_info import ScheduleInfo
from objects.actions.action_plan import ActionPlan
from logic.goals.factory_goal import BuildLightGoal
from logic.goals.unit_goal import UnitGoal, DigGoal, SupplyPowerGoal
from exceptions import NoValidGoalFoundError
from logic.goal_resolution.factory_signal import SIGNALS
from logic.goal_resolution.time_tracker import TimeTracker
from config import CONFIG


class Scheduler:
    def __init__(self, turn_start_time: float, DEBUG_MODE: bool, game_state: GameState):
        self.time_tracker = TimeTracker(turn_start_time, DEBUG_MODE=DEBUG_MODE)
        self._init_constraints_and_power_tracker(game_state)

    def schedule_goals(self) -> None:
        self._remove_completed_goals()
        self._schedule_units_too_little_power_dummy_goal()
        self._reserve_next_tc_for_units_that_can_not_move()

        self._reschedule_goals_with_no_private_action_plan()
        self._schedule_goals_still_fine()
        self._reschedule_goals_too_little_power()
        self._reschedule_goals_needs_adapting()
        self._reschedule_at_risk_units()

        self._schedule_new_goals()
        self._schedule_unassigned_units_goals()
        self._schedule_watering_or_no_goal_factories_with_colliding_build_goal()

    def _init_constraints_and_power_tracker(self, game_state: GameState) -> None:
        self.constraints = Constraints()
        self.power_tracker = PowerTracker(game_state.player_factories)
        self.schedule_info = ScheduleInfo(game_state, self.constraints, self.power_tracker)

    def _schedule_units_too_little_power_dummy_goal(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if unit.goal:
                continue

            if unit.is_on_factory(game_state) and unit.can_update_action_queue:
                continue

            if not unit.is_on_factory(game_state) and unit.can_update_action_queue_and_move:
                continue

            goal = unit.generate_no_goal_goal(self.schedule_info)
            self._schedule_unit_on_goal(goal)

    def _reserve_next_tc_for_units_that_can_not_move(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if unit.can_not_move_this_step(game_state):
                next_tc = unit.tc + Direction.CENTER
                self.constraints.add_negative_constraint(next_tc)

    def _remove_completed_goals(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if unit.goal and unit.goal.is_completed(game_state, unit.private_action_plan):
                self._unschedule_unit_goal(unit)

    def _reschedule_goals_too_little_power(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if not unit.supplied_by and not unit.private_action_plan.unit_has_enough_power(game_state) and unit.goal:
                try:
                    schedule_info = self.schedule_info.copy_without_unit_scheduled_actions(unit)
                    goal = unit.get_best_version_goal(unit.goal, schedule_info)
                except NoValidGoalFoundError:
                    self._unschedule_unit_goal(unit)
                    continue

                self._schedule_unit_on_goal(goal)

    def _reschedule_goals_needs_adapting(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if not (unit.goal and unit.goal.plan_needs_adapting(unit.private_action_plan, game_state)):
                continue

            try:
                schedule_info = self.schedule_info.copy_without_unit_scheduled_actions(unit)
                goal = unit.get_best_version_goal(unit.goal, schedule_info)
            except Exception:
                self._unschedule_unit_goal(unit)
                continue

            self._schedule_unit_on_goal(goal)

    def _schedule_goals_still_fine(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if not unit.goal:
                continue

            if unit.is_scheduled:
                continue

            if not unit.supplied_by and not unit.private_action_plan.unit_has_enough_power(game_state) and unit.goal:
                continue

            if unit.is_under_threath(game_state) and unit.next_step_is_stationary():
                continue

            if unit.next_step_walks_into_tile_where_it_might_be_captured(game_state):
                continue

            if not unit.private_action_plan:
                continue

            if unit.goal.plan_needs_adapting(unit.private_action_plan, game_state):
                continue

            time_coordinates = unit.private_action_plan.get_time_coordinates(game_state)
            if self.constraints.any_tc_not_allowed(time_coordinates):
                self._unschedule_unit_goal(unit)
            else:
                self._schedule_unit_on_goal(unit.goal)

    def _reschedule_at_risk_units(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if not unit.goal:
                continue

            if unit.is_under_threath(game_state) and unit.next_step_is_stationary():
                self._unschedule_unit_goal(unit)
                goal = unit.generate_transfer_or_dummy_goal(self.schedule_info)
                self._schedule_unit_on_goal(goal)
            elif unit.next_step_walks_into_tile_where_it_might_be_captured(game_state):
                self._unschedule_unit_goal(unit)

    def _reschedule_goals_with_no_private_action_plan(self) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.player_units:
            if unit.goal and not unit.private_action_plan:
                try:
                    goal = unit.get_best_version_goal(unit.goal, self.schedule_info)
                except Exception:
                    self._unschedule_unit_goal(unit)
                    continue

                self._schedule_unit_on_goal(goal)

    def _schedule_new_goals(self) -> None:
        game_state = self.schedule_info.game_state

        self._schedule_factory_goals()

        while True:
            if not self._exists_available_unit(game_state) or self.time_tracker.is_out_of_time_main_scheduling():
                break

            for factory, strategy in self._get_priority_sorted_strategies_factory(game_state):
                try:
                    goals = factory.schedule_units(strategy, self.schedule_info)
                except Exception:
                    continue

                for goal in goals:
                    self._schedule_unit_on_goal(goal)

                break

    def _exists_available_unit(self, game_state: GameState) -> bool:
        return any(
            factory.has_unit_available
            for factory in game_state.player_factories
            if factory.nr_schedule_failures_this_step <= CONFIG.MAX_SCHEDULING_FAILURES_ALLOWED_FACTORY
        )

    def _schedule_factory_goals(self) -> None:
        game_state = self.schedule_info.game_state

        for factory in game_state.player_factories:
            action_plan = factory.schedule_build_or_no_goal(self.schedule_info)
            self._update_constraints_and_power_tracker(game_state, action_plan)

        if game_state.real_env_steps < 6:
            self._reserve_tc_and_power_factories()

    def _reserve_tc_and_power_factories(self) -> None:
        game_state = self.schedule_info.game_state

        # Hacky method to ensure that there will be enough power available to build the first units and make sure
        # They can grab some power
        for factory in game_state.player_factories:
            for t in range(game_state.real_env_steps + 1, 7):
                goal = BuildLightGoal(factory)
                action_plan = goal.generate_action_plan(self.schedule_info)

                power_requests = action_plan.get_power_requests(game_state)
                for power_request in power_requests:
                    power_request.t = t
                    # Reserves 100 power for the new light units to use
                    power_request.p += 100
                self.power_tracker.add_power_requests(power_requests=power_requests)

                time_coordinates = goal.action_plan.get_time_coordinates(game_state)
                for time_coordinate in time_coordinates:
                    tc = TimeCoordinate(time_coordinate.x, time_coordinate.y, t)
                    self.constraints.add_negative_constraint(tc)

    def _schedule_unassigned_units_goals(self) -> None:
        game_state = self.schedule_info.game_state

        for factory in game_state.player_factories:
            for unit in factory.unscheduled_units:
                if self.time_tracker.is_out_of_time_scheduling_unassigned_units():
                    return

                goal = unit.generate_dummy_goal(self.schedule_info)
                self._schedule_unit_on_goal(goal)

    def _get_priority_sorted_strategies_factory(self, game_state: GameState) -> List[Tuple[Factory, Strategy]]:
        scores = self._score_signal_strategies_for_factories_with_available_units(game_state)
        scores = {k: v for k, v in scores.items() if v > 0}
        sorted_factory_strategies = sorted(scores, key=lambda x: -1 * scores[x])  # type: ignore
        return sorted_factory_strategies

    def _score_signal_strategies_for_factories_with_available_units(
        self, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = dict()

        for factory in game_state.player_factories:
            if (
                factory.has_unit_available
                and factory.nr_schedule_failures_this_step <= CONFIG.MAX_SCHEDULING_FAILURES_ALLOWED_FACTORY
            ):
                factory_signal = self._calculate_signal_factory(factory, game_state)
                for (factory, strategy), signal in factory_signal.items():
                    scores[(factory, strategy)] = signal

        return scores

    def _calculate_signal_factory(
        self, factory: Factory, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = defaultdict(lambda: 0.0)
        for signal in SIGNALS:
            score = signal.compute_signal(factory, game_state)
            scores[(factory, signal.strategy)] += score

        return dict(scores)

    def _schedule_watering_or_no_goal_factories_with_colliding_build_goal(self) -> None:
        next_tc_units = set()
        for unit in self.schedule_info.game_state.player_units:
            next_tc = unit.private_action_plan.next_tc
            next_tc_units.add(next_tc)

        for factory in self.schedule_info.game_state.player_factories:
            if not factory.private_action_plan:
                factory.schedule_water_or_no_goal(self.schedule_info)

            next_tc = factory.private_action_plan.next_tc
            if next_tc and next_tc in next_tc_units:
                factory.schedule_water_or_no_goal(self.schedule_info)

    def _schedule_unit_on_goal(self, goal: UnitGoal) -> None:
        game_state = self.schedule_info.game_state

        if isinstance(goal, DigGoal):
            self._remove_other_units_from_dig_goal(goal)

        if goal.unit.is_scheduled:
            self._unschedule_unit_goal(goal.unit)

        self._update_unit_info(goal)
        self._update_constraints_and_power_tracker(game_state, goal.action_plan)

    def _update_constraints_and_power_tracker(self, game_state: GameState, action_plan: ActionPlan) -> None:
        self.constraints.add_negative_constraints(action_plan.get_time_coordinates(game_state))
        self.power_tracker.add_power_requests(action_plan.get_power_requests(game_state))

    def _update_unit_info(self, goal: UnitGoal) -> None:
        if isinstance(goal, SupplyPowerGoal):
            goal.unit.supplies = goal.receiving_unit
            goal.receiving_unit.supplied_by = goal.unit

        goal.unit.schedule_goal(goal)

    def _remove_other_units_from_dig_goal(self, goal: DigGoal) -> None:
        game_state = self.schedule_info.game_state

        for unit in game_state.units:
            if unit != goal.unit and isinstance(unit.goal, DigGoal) and unit.goal.dig_c == goal.dig_c:
                self._unschedule_unit_goal(unit)

    def _unschedule_unit_goal(self, unit: Unit) -> None:
        game_state = self.schedule_info.game_state

        if unit.is_scheduled:
            self.constraints.remove_negative_constraints(unit.private_action_plan.get_time_coordinates(game_state))
            self.power_tracker.remove_power_requests(unit.private_action_plan.get_power_requests(game_state))

        supplies, supplied_by = unit.supplies, unit.supplied_by
        unit.remove_goal_and_private_action_plan()

        for connected_unit in [supplies, supplied_by]:
            if connected_unit:
                self._unschedule_unit_goal(connected_unit)
