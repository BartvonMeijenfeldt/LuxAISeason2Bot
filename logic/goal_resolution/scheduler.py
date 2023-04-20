import time
import logging

from typing import Dict, Tuple, List
from collections import defaultdict
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from copy import copy
from math import inf

from objects.game_state import GameState
from objects.coordinate import TimeCoordinate
from objects.actors.factory import Factory, Strategy
from objects.actors.unit import Unit
from logic.constraints import Constraints
from logic.goal_resolution.power_availabilty_tracker import PowerTracker
from objects.actions.action_plan import ActionPlan
from logic.goals.factory_goal import BuildLightGoal
from logic.goals.unit_goal import UnitGoal, DigGoal, SupplyPowerGoal
from lux.config import EnvConfig
from config import CONFIG
from exceptions import NoValidGoalFoundError


class FactoryValue(metaclass=ABCMeta):
    threshold: float
    upper_threshold: bool
    strategy: Strategy

    @abstractmethod
    def compute_value(self, factory: Factory, game_state: GameState) -> float:
        ...

    def compute_score(self, factory: Factory, game_state: GameState) -> float:
        value = self.compute_value(factory, game_state)
        normalized_value = value / self.threshold
        if self.upper_threshold and normalized_value > 1:
            return normalized_value - 1

        if not self.upper_threshold and normalized_value < 1:
            inverted_value = 1 / max(normalized_value, 0.01)
            return inverted_value - 1

        return 0


class TooMuchWaterValue(FactoryValue):
    threshold = 3
    upper_threshold = True
    strategy = Strategy.INCREASE_LICHEN_TILES

    def compute_value(self, factory: Factory, game_state: GameState) -> float:
        if factory.has_enough_space_to_increase_lichen(game_state):
            return 0.0

        return factory.water / max(30, factory.nr_lichen_tiles)


class TooLittleIceCollectionValue(FactoryValue):
    threshold = 1.3
    upper_threshold = False
    strategy = Strategy.COLLECT_ICE
    nr_steps = 50

    def compute_value(self, factory: Factory, game_state: GameState) -> float:

        water = factory.water
        safety_water_quantity = 50
        water_available = water - safety_water_quantity

        # TODO, this water collection calculates per step now, but it should be from the start of the goal
        # so we know if we can maintain our lichen in general
        water_collection_per_step = factory.get_water_collection_per_step(game_state)

        water_cost_per_step = max(6, factory.water_cost) / 2  # assume water every other turn

        return (water_available + self.nr_steps * water_collection_per_step) / (self.nr_steps * water_cost_per_step)


class PowerUnitValue(FactoryValue):
    nr_steps = 50
    threshold = 1.2

    def compute_value(self, factory: Factory, game_state: GameState) -> float:
        power = self._get_power(factory)
        power_generation_per_step = self._get_expected_power_generation_per_step(factory, game_state)
        expected_power_usage_per_step = self._get_power_usage_per_step(factory, game_state)

        value = (power + self.nr_steps * power_generation_per_step) / (self.nr_steps * expected_power_usage_per_step)
        return value

    @staticmethod
    def _get_power(factory: Factory) -> float:
        factory_power = factory.power
        units_power = sum(unit.power for unit in factory.units)
        power = factory_power + units_power
        return power

    def _get_expected_power_generation_per_step(self, factory: Factory, game_state: GameState) -> float:
        power_generation_factory = EnvConfig.FACTORY_CHARGE

        if factory.enough_water_collection_for_next_turns(game_state):
            expected_lichen = 0
        else:
            expected_lichen = factory.get_nr_connected_positions_including_being_cleared(game_state)

        total_lichen = factory.nr_connected_lichen_tiles + expected_lichen
        power_generation_lichen = EnvConfig.POWER_PER_CONNECTED_LICHEN_TILE * total_lichen
        power_generation = power_generation_factory + power_generation_lichen

        return power_generation

    def _get_power_usage_per_step(self, factory: Factory, game_state) -> float:
        metal_in_factory = factory.metal
        metal_collection = factory.get_metal_collection_per_step(game_state)

        metal_value_units = sum(unit.unit_cfg.METAL_COST for unit in factory.units)
        metal_value = metal_value_units + metal_in_factory + metal_collection

        metal_value = max(metal_value, 10)
        EXPECTED_POWER_USAGE_PER_METAL = 0.4
        expected_power_usage_per_step = metal_value * EXPECTED_POWER_USAGE_PER_METAL
        return expected_power_usage_per_step


class UpperPowerUnitValue(PowerUnitValue):
    upper_threshold = True
    strategy = Strategy.INCREASE_UNITS

    def compute_score(self, factory: Factory, game_state: GameState) -> float:
        if game_state.real_env_steps > CONFIG.LAST_STEP_SCHEDULE_ORE_MINING:
            return -inf

        return super().compute_score(factory, game_state)


class LowerPowerUnitValue(PowerUnitValue):
    upper_threshold = False
    strategy = Strategy.INCREASE_LICHEN


class AttackOpponentValue(FactoryValue):
    threshold = 0.8
    upper_threshold = True
    strategy = Strategy.ATTACK_OPPONENT

    def compute_value(self, factory: Factory, game_state: GameState) -> float:
        if not factory.nr_scheduled_units:
            return 0.85

        ratio_attacking = factory.nr_attack_scheduled_units / factory.nr_scheduled_units
        ratio_not_attacking = 1 - ratio_attacking

        # Always want attacking to be at least slightly better than 0 such that we prefer attacking over ore mining
        # After last ore step minign
        ratio = max(ratio_not_attacking, self.threshold + 0.01)

        return ratio


VALUES = [
    TooMuchWaterValue(),
    TooLittleIceCollectionValue(),
    UpperPowerUnitValue(),
    LowerPowerUnitValue(),
    AttackOpponentValue(),
]


@dataclass
class Scheduler:
    turn_start_time: float
    DEBUG_MODE: bool

    def _is_out_of_time(self) -> bool:
        if self.DEBUG_MODE:
            return False

        is_out_of_time = self._get_time_taken() > 2.5

        if is_out_of_time:
            logging.critical("RAN OUT OF TIME")

        return is_out_of_time

    def _get_time_taken(self) -> float:
        return time.time() - self.turn_start_time

    def schedule_goals(self, game_state: GameState) -> None:
        self._init_constraints_and_power_tracker(game_state)
        self._schedule_units_too_little_power_dummy_goal(game_state)
        self._remove_completed_goals(game_state)

        self._reschedule_goals_with_no_private_action_plan(game_state)
        self._schedule_goals_still_fine(game_state)
        self._reschedule_goals_too_little_power(game_state)
        self._reschedule_goals_needs_adapting(game_state)
        self._reschedule_at_risk_units(game_state)

        self._schedule_new_goals(game_state)
        self._schedule_unassigned_units_goals(game_state)
        self._reschedule_factories_building_on_top_of_units(game_state)

    def _init_constraints_and_power_tracker(self, game_state: GameState) -> None:
        self.constraints = Constraints()
        self.power_tracker = PowerTracker(game_state.player_factories)

    def _schedule_units_too_little_power_dummy_goal(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.goal:
                continue

            if unit.is_on_factory(game_state) and unit.can_update_action_queue:
                # TODO, in this case we do know the unit will be stationary although we still might want to set a plan
                # Setting this unit as stationary can help reduce collisions with own units
                continue

            if not unit.is_on_factory(game_state) and unit.can_update_action_queue_and_move:
                continue

            goal = unit.generate_no_goal_goal(game_state, self.constraints, self.power_tracker)
            self._schedule_unit_on_goal(goal, game_state)

    def _remove_completed_goals(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.goal and unit.goal.is_completed(game_state, unit.private_action_plan):
                self._unschedule_unit_goal(unit, game_state)

    def _reschedule_goals_too_little_power(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not unit.supplied_by and not unit.private_action_plan.unit_has_enough_power(game_state) and unit.goal:
                try:
                    constraints, power_tracker = self._get_constraints_power_without_unit_scheduled_actions(
                        unit, game_state
                    )
                    goal = unit.get_best_version_goal(unit.goal, game_state, constraints, power_tracker)
                except NoValidGoalFoundError:
                    self._unschedule_unit_goal(unit, game_state)
                    continue

                self._schedule_unit_on_goal(goal, game_state)

    def _reschedule_goals_needs_adapting(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not (unit.goal and unit.goal.plan_needs_adapting(unit.private_action_plan, game_state)):
                continue

            try:
                constraints, power_tracker = self._get_constraints_power_without_unit_scheduled_actions(
                    unit, game_state
                )
                goal = unit.get_best_version_goal(unit.goal, game_state, constraints, power_tracker)
            except NoValidGoalFoundError:
                self._unschedule_unit_goal(unit, game_state)
                continue

            self._schedule_unit_on_goal(goal, game_state)

    def _schedule_goals_still_fine(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not unit.goal:
                continue

            if unit.is_scheduled:
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
                goal = unit.generate_transfer_or_dummy_goal(game_state, self.constraints, self.power_tracker)
                self._schedule_unit_on_goal(goal, game_state)
            elif unit.next_step_walks_into_tile_where_it_might_be_captured(game_state):
                self._unschedule_unit_goal(unit, game_state)

    def _reschedule_goals_with_no_private_action_plan(self, game_state: GameState):
        for unit in game_state.player_units:
            if unit.goal and not unit.private_action_plan:
                try:
                    goal = unit.get_best_version_goal(unit.goal, game_state, self.constraints, self.power_tracker)
                except Exception:
                    self._unschedule_unit_goal(unit, game_state)
                    continue

                self._schedule_unit_on_goal(goal, game_state)

    def _schedule_new_goals(self, game_state: GameState) -> None:
        self._schedule_factory_goals(game_state)
        if game_state.real_env_steps < 6:
            self._reserve_tc_and_power_factories(game_state)

        while True:
            if not self._exists_available_unit(game_state) or self._is_out_of_time():
                break

            scores = self._score_strategies_for_factories_with_available_units(game_state)
            factory = self._get_highest_priority_factory(scores)
            strategies = self._get_priority_sorted_strategies_factory(factory, scores)
            goals = factory.schedule_units(strategies, game_state, self.constraints, self.power_tracker)
            for goal in goals:
                self._schedule_unit_on_goal(goal, game_state)

    def _exists_available_unit(self, game_state: GameState) -> bool:
        return any(factory.has_unit_available for factory in game_state.player_factories)

    def _schedule_factory_goals(self, game_state: GameState) -> None:
        for factory in game_state.player_factories:
            action_plan = factory.schedule_goal(game_state, self.constraints, self.power_tracker)
            self._update_constraints_and_power_tracker(game_state, action_plan)

    def _reserve_tc_and_power_factories(self, game_state: GameState) -> None:
        # Hacky method to ensure that there will be enough power available to build the first units and make sure
        # They can grab some power
        for factory in game_state.player_factories:
            for t in range(game_state.real_env_steps + 1, 7):
                goal = BuildLightGoal(factory)
                action_plan = goal.generate_action_plan(game_state, self.constraints, self.power_tracker)

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
                if self._is_out_of_time():
                    return

                goal = unit.generate_dummy_goal(game_state, self.constraints, self.power_tracker)
                self._schedule_unit_on_goal(goal, game_state)

    def _get_highest_priority_factory(self, scores: Dict[Tuple[Factory, Strategy], float]) -> Factory:
        return max(scores, key=scores.get)[0]  # type: ignore

    def _score_strategies_for_factories_with_available_units(
        self, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = dict()

        for factory in game_state.player_factories:
            if factory.has_unit_available:
                factory_score = self._calculate_score_factory(factory, game_state)
                for (factory, strategy), score in factory_score.items():
                    scores[(factory, strategy)] = score

        return scores

    def _get_priority_sorted_strategies_factory(
        self, factory, scores: Dict[Tuple[Factory, Strategy], float]
    ) -> List[Strategy]:
        strategies_factory = [strat for (fact, strat), val in scores.items() if fact == factory and val >= 0]
        strategies_factory.sort(key=lambda s: -scores[factory, s])
        return strategies_factory

    def _calculate_score_factory(
        self, factory: Factory, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = defaultdict(lambda: 0.0)
        for value in VALUES:
            score = value.compute_score(factory, game_state)
            scores[(factory, value.strategy)] += score

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
                action_plan = factory.schedule_goal(game_state, self.constraints, self.power_tracker, can_build=False)
                self._update_constraints_and_power_tracker(game_state, action_plan)

    def _schedule_unit_on_goal(self, goal: UnitGoal, game_state: GameState) -> None:
        if isinstance(goal, DigGoal):
            self._remove_other_units_from_dig_goal(goal, game_state)

        if goal != goal.unit.goal:
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

    def _get_constraints_power_without_unit_scheduled_actions(
        self, unit: Unit, game_state: GameState
    ) -> Tuple[Constraints, PowerTracker]:

        constraints = copy(self.constraints)
        power_tracker = copy(self.power_tracker)

        if unit.is_scheduled:
            constraints.remove_negative_constraints(unit.private_action_plan.get_time_coordinates(game_state))
            power_tracker.remove_power_requests(unit.private_action_plan.get_power_requests(game_state))

        return constraints, power_tracker
