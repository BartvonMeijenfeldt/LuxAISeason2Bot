import time
import logging

from typing import Dict, Tuple, List, Set
from collections import defaultdict
from dataclasses import dataclass, field

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
from config import CONFIG


@dataclass
class Scheduler:
    turn_start_time: float
    DEBUG_MODE: bool
    factories_unavailable: Set[Factory] = field(init=False, default_factory=set)

    def _is_out_of_time_main_scheduling(self) -> bool:
        if self.DEBUG_MODE:
            return False

        is_out_of_time = self._get_time_taken() > CONFIG.OUT_OF_TIME_MAIN_SCHEDULING

        if is_out_of_time:
            logging.critical("RAN OUT OF TIME MAIN SCHEDULING")

        return is_out_of_time

    def _is_out_of_time_scheduling_unassigned_units(self) -> bool:
        if self.DEBUG_MODE:
            return False

        is_out_of_time = self._get_time_taken() > CONFIG.OUT_OF_TIME_UNASSIGNED_SCHEDULING

        if is_out_of_time:
            logging.critical("RAN OUT OF TIME UNASSIGNED SCHEDULING")

        return is_out_of_time

    def _get_time_taken(self) -> float:
        return time.time() - self.turn_start_time

    def schedule_goals(self, game_state: GameState) -> None:
        self._init_constraints_and_power_tracker(game_state)
        self._remove_completed_goals(game_state)
        self._schedule_units_too_little_power_dummy_goal(game_state)
        self._reserve_next_tc_for_units_that_can_not_move(game_state)

        self._reschedule_goals_with_no_private_action_plan(game_state)
        self._schedule_goals_still_fine(game_state)
        self._reschedule_goals_too_little_power(game_state)
        self._reschedule_goals_needs_adapting(game_state)
        self._reschedule_at_risk_units(game_state)

        self._schedule_new_goals(game_state)
        self._schedule_unassigned_units_goals(game_state)
        self._reschedule_factories_building_on_top_of_units(game_state)

    def _init_constraints_and_power_tracker(self, game_state: GameState) -> None:
        self.game_state = game_state
        self.constraints = Constraints()
        self.power_tracker = PowerTracker(game_state.player_factories)
        self.schedule_info = ScheduleInfo(game_state, self.constraints, self.power_tracker)

    def _schedule_units_too_little_power_dummy_goal(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.goal:
                continue

            if unit.is_on_factory(game_state) and unit.can_update_action_queue:
                continue

            if not unit.is_on_factory(game_state) and unit.can_update_action_queue_and_move:
                continue

            goal = unit.generate_no_goal_goal(self.schedule_info)
            self._schedule_unit_on_goal(goal, game_state)

    def _reserve_next_tc_for_units_that_can_not_move(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.can_not_move_this_step(game_state):
                next_tc = unit.tc + Direction.CENTER
                self.constraints.add_negative_constraint(next_tc)

    def _remove_completed_goals(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.goal and unit.goal.is_completed(game_state, unit.private_action_plan):
                self._unschedule_unit_goal(unit, game_state)

    def _reschedule_goals_too_little_power(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not unit.supplied_by and not unit.private_action_plan.unit_has_enough_power(game_state) and unit.goal:
                try:
                    schedule_info = self.schedule_info.copy_without_unit_scheduled_actions(unit)
                    goal = unit.get_best_version_goal(unit.goal, schedule_info)
                except NoValidGoalFoundError:
                    self._unschedule_unit_goal(unit, game_state)
                    continue

                self._schedule_unit_on_goal(goal, game_state)

    def _reschedule_goals_needs_adapting(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not (unit.goal and unit.goal.plan_needs_adapting(unit.private_action_plan, game_state)):
                continue

            try:
                schedule_info = self.schedule_info.copy_without_unit_scheduled_actions(unit)
                goal = unit.get_best_version_goal(unit.goal, schedule_info)
            except Exception:
                self._unschedule_unit_goal(unit, game_state)
                continue

            self._schedule_unit_on_goal(goal, game_state)

    def _schedule_goals_still_fine(self, game_state: GameState) -> None:
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
                self._unschedule_unit_goal(unit, game_state)
            else:
                self._schedule_unit_on_goal(unit.goal, game_state)

    def _reschedule_at_risk_units(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not unit.goal:
                continue

            if unit.is_under_threath(game_state) and unit.next_step_is_stationary():
                self._unschedule_unit_goal(unit, game_state)
                goal = unit.generate_transfer_or_dummy_goal(self.schedule_info)
                self._schedule_unit_on_goal(goal, game_state)
            elif unit.next_step_walks_into_tile_where_it_might_be_captured(game_state):
                self._unschedule_unit_goal(unit, game_state)

    def _reschedule_goals_with_no_private_action_plan(self, game_state: GameState):
        for unit in game_state.player_units:
            if unit.goal and not unit.private_action_plan:
                try:
                    goal = unit.get_best_version_goal(unit.goal, self.schedule_info)
                except Exception:
                    self._unschedule_unit_goal(unit, game_state)
                    continue

                self._schedule_unit_on_goal(goal, game_state)

    def _schedule_new_goals(self, game_state: GameState) -> None:
        self._schedule_factory_goals(game_state)
        if game_state.real_env_steps < 6:
            self._reserve_tc_and_power_factories(game_state)

        while True:
            if not self._exists_available_unit(game_state) or self._is_out_of_time_main_scheduling():
                break

            scores = self._score_signal_strategies_for_factories_with_available_units(game_state)
            factory = self._get_highest_priority_factory(scores)
            strategies = self._get_priority_sorted_strategies_factory(factory, scores)

            try:
                goals = factory.schedule_units(strategies, self.schedule_info)
            except Exception:
                self._set_factory_unavailable(factory)
                continue

            for goal in goals:
                self._schedule_unit_on_goal(goal, game_state)

    def _set_factory_unavailable(self, factory: Factory) -> None:
        self.factories_unavailable.add(factory)

    def _exists_available_unit(self, game_state: GameState) -> bool:
        return any(
            factory.has_unit_available
            for factory in game_state.player_factories
            if factory not in self.factories_unavailable
        )

    def _schedule_factory_goals(self, game_state: GameState) -> None:
        for factory in game_state.player_factories:
            action_plan = factory.schedule_goal(self.schedule_info)
            self._update_constraints_and_power_tracker(game_state, action_plan)

    def _reserve_tc_and_power_factories(self, game_state: GameState) -> None:
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

    def _schedule_unassigned_units_goals(self, game_state: GameState) -> None:
        for factory in game_state.player_factories:
            for unit in factory.unscheduled_units:
                if self._is_out_of_time_main_scheduling():
                    return

                goal = unit.generate_dummy_goal(self.schedule_info)
                self._schedule_unit_on_goal(goal, game_state)

    def _get_highest_priority_factory(self, scores: Dict[Tuple[Factory, Strategy], float]) -> Factory:
        return max(scores, key=scores.get)[0]  # type: ignore

    def _score_signal_strategies_for_factories_with_available_units(
        self, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = dict()

        for factory in game_state.player_factories:
            if factory.has_unit_available and factory not in self.factories_unavailable:
                factory_signal = self._calculate_signal_factory(factory, game_state)
                for (factory, strategy), signal in factory_signal.items():
                    scores[(factory, strategy)] = signal

        return scores

    def _get_priority_sorted_strategies_factory(
        self, factory, signals: Dict[Tuple[Factory, Strategy], float]
    ) -> List[Strategy]:
        strategies_factory = [strat for (fact, strat), signal in signals.items() if fact == factory and signal > 0]
        strategies_factory.sort(key=lambda s: -signals[factory, s])
        return strategies_factory

    def _calculate_signal_factory(
        self, factory: Factory, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = defaultdict(lambda: 0.0)
        for signal in SIGNALS:
            score = signal.compute_signal(factory, game_state)
            scores[(factory, signal.strategy)] += score

        return dict(scores)

    def _reschedule_factories_building_on_top_of_units(self, game_state: GameState) -> None:
        next_tc_units = set()
        for unit in game_state.player_units:
            next_tc = unit.private_action_plan.next_tc
            next_tc_units.add(next_tc)

        for factory in game_state.player_factories:
            if not factory.private_action_plan:
                continue

            next_tc = factory.private_action_plan.next_tc
            if next_tc and next_tc in next_tc_units:
                action_plan = factory.schedule_goal(self.schedule_info, can_build=False)
                self._update_constraints_and_power_tracker(game_state, action_plan)

    def _schedule_unit_on_goal(self, goal: UnitGoal, game_state: GameState) -> None:
        if isinstance(goal, DigGoal):
            self._remove_other_units_from_dig_goal(goal, game_state)

        if goal.unit.is_scheduled:
            self._unschedule_unit_goal(goal.unit, game_state)

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

    def _remove_other_units_from_dig_goal(self, goal: DigGoal, game_state: GameState) -> None:
        for unit in game_state.units:
            if unit != goal.unit and isinstance(unit.goal, DigGoal) and unit.goal.dig_c == goal.dig_c:
                self._unschedule_unit_goal(unit, game_state)

    def _unschedule_unit_goal(self, unit: Unit, game_state: GameState) -> None:
        if unit.is_scheduled:
            self.constraints.remove_negative_constraints(unit.private_action_plan.get_time_coordinates(game_state))
            self.power_tracker.remove_power_requests(unit.private_action_plan.get_power_requests(game_state))

        supplies, supplied_by = unit.supplies, unit.supplied_by
        unit.remove_goal_and_private_action_plan()

        for connected_unit in [supplies, supplied_by]:
            if connected_unit:
                self._unschedule_unit_goal(connected_unit, game_state)
