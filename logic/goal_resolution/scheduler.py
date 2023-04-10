import time
import logging

from typing import Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

from objects.game_state import GameState
from objects.coordinate import TimeCoordinate
from objects.actors.factory import Factory, Strategy
from logic.constraints import Constraints
from logic.goal_resolution.power_availabilty_tracker import PowerTracker
from objects.actions.action_plan import ActionPlan
from logic.goals.factory_goal import BuildLightGoal
from lux.config import EnvConfig
from objects.direction import Direction


class FactoryRatio(metaclass=ABCMeta):
    threshold: float
    upper_threshold: bool
    strategy: Strategy

    @abstractmethod
    def compute_ratio(self, factory: Factory, game_state: GameState) -> float:
        ...

    def compute_score(self, factory: Factory, game_state: GameState) -> float:
        ratio = self.compute_ratio(factory, game_state)
        normalized_ratio = ratio / self.threshold
        if self.upper_threshold and normalized_ratio > 1:
            return normalized_ratio - 1

        if not self.upper_threshold and normalized_ratio < 1:
            inverted_ratio = 1 / max(normalized_ratio, 0.01)
            return inverted_ratio - 1

        return 0


class WaterLichenTilesRatio(FactoryRatio):
    threshold = 3
    upper_threshold = True
    strategy = Strategy.INCREASE_LICHEN

    def compute_ratio(self, factory: Factory, game_state: GameState) -> float:
        return factory.water / max(30, factory.nr_lichen_tiles)


class WaterNextStepsLichenTilesRatio(FactoryRatio):
    threshold = 1.3
    upper_threshold = False
    strategy = Strategy.COLLECT_ICE
    nr_steps = 50

    def compute_ratio(self, factory: Factory, game_state: GameState) -> float:

        water = factory.water
        safety_water_quantity = 50
        water_available = water - safety_water_quantity

        water_collection_per_step = factory.get_water_collection_per_step(game_state)

        water_cost_per_step = max(6, factory.water_cost) / 2  # assume water every other turn

        return (water_available + self.nr_steps * water_collection_per_step) / (self.nr_steps * water_cost_per_step)


class PowerUnitRatio(FactoryRatio):
    nr_steps = 50
    threshold = 1.2

    def compute_ratio(self, factory: Factory, game_state: GameState) -> float:
        power = self._get_power(factory)
        power_generation_per_step = self._get_power_generation_per_step(factory, game_state)
        expected_power_usage_per_step = self._get_power_usage_per_step(factory, game_state)

        return (power + self.nr_steps * power_generation_per_step) / (self.nr_steps * expected_power_usage_per_step)

    @staticmethod
    def _get_power(factory: Factory) -> float:
        factory_power = factory.power
        units_power = sum(unit.power for unit in factory.units)
        power = factory_power + units_power
        return power

    def _get_power_generation_per_step(self, factory: Factory, game_state: GameState) -> float:
        power_generation_factory = EnvConfig.FACTORY_CHARGE

        if factory.enough_water_collection_for_next_turns(game_state):
            expected_lichen = 0
        else:
            nr_can_spread_to_positions_being_cleared = factory.nr_can_spread_to_positions_being_cleared
            expected_lichen = factory.nr_can_spread_positions + nr_can_spread_to_positions_being_cleared

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


class UpperPowerUnitRatio(PowerUnitRatio):
    upper_threshold = True
    strategy = Strategy.INCREASE_UNITS


class LowerPowerUnitRatio(PowerUnitRatio):
    upper_threshold = False
    strategy = Strategy.INCREASE_LICHEN


RATIOS = [
    WaterLichenTilesRatio(),
    WaterNextStepsLichenTilesRatio(),
    UpperPowerUnitRatio(),
    LowerPowerUnitRatio(),
]


@dataclass
class Scheduler:
    turn_start_time: float
    DEBUG_MODE: bool

    def _is_out_of_time(self) -> bool:
        if self.DEBUG_MODE:
            return False

        is_out_of_time = self._get_time_taken() > 2.9

        if is_out_of_time:
            logging.critical("RAN OUT OF TIME")

        return is_out_of_time

    def _get_time_taken(self) -> float:
        return time.time() - self.turn_start_time

    def schedule_goals(self, game_state: GameState) -> None:
        self._init_constraints_and_power_tracker(game_state)
        self._remove_completed_goals(game_state)
        self._update_goals_at_risk_units(game_state)
        self._schedule_new_goals(game_state)
        self._schedule_unassigned_units_goals(game_state)
        self._reschedule_factories_building_on_top_of_units(game_state)

    def _init_constraints_and_power_tracker(self, game_state: GameState) -> None:
        self.constraints = Constraints()
        self.power_tracker = PowerTracker(game_state.player_factories)

    def _remove_completed_goals(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.goal and unit.goal.is_completed(game_state):
                unit.remove_goal_and_private_action_plan()

    def _update_goals_at_risk_units(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if not unit.goal or not unit.private_action_plan:
                continue

            if unit.is_under_threath(game_state) and unit.next_step_is_stationary():
                goal = unit.generate_flee_transfer_goal_or_dummy_goal(game_state, self.constraints, self.power_tracker)
                unit.set_goal(goal)
                unit.set_private_action_plan(goal.action_plan)
                self._update_constraints_and_power_tracker(game_state, goal.action_plan)
            elif unit.next_step_walks_into_tile_where_it_might_be_captured(game_state):
                unit.remove_goal_and_private_action_plan()
            else:
                time_coordinates = unit.private_action_plan.time_coordinates
                if self.constraints.any_tc_in_negative_constraints(time_coordinates):
                    unit.remove_goal_and_private_action_plan()
                else:
                    self._update_constraints_and_power_tracker(game_state, unit.goal.action_plan)

    def _update_constraints_and_power_tracker(self, game_state: GameState, action_plan: ActionPlan) -> None:
        self.constraints.add_negative_constraints(action_plan.time_coordinates)
        self.power_tracker.update_power_available(action_plan.get_power_requests(game_state))

    def _schedule_new_goals(self, game_state: GameState) -> None:
        self._schedule_factory_goals(game_state)
        if game_state.real_env_steps < 6:
            self._reserve_tc_and_power_factories(game_state)

        self._reserve_tc_and_power_units_with_private_action_plan(game_state)

        # while True:
        # TODO, this should not need these i < 100 construction but eventually break because
        # all units are scheduled or unavailable.
        i = 0
        while i < 100:
            scores = self._score_strategies(game_state)
            if not scores or self._is_out_of_time():
                break

            factory, strategy = self._get_highest_priority_factory_and_strategy(scores)
            try:
                action_plan = factory.schedule_unit(strategy, game_state, self.constraints, self.power_tracker)
            except Exception:
                i += 1
                continue

            self._update_constraints_and_power_tracker(game_state, action_plan)

    def _schedule_factory_goals(self, game_state: GameState) -> None:
        for factory in game_state.player_factories:
            action_plan = factory.set_goal(game_state, self.constraints, self.power_tracker)
            self._update_constraints_and_power_tracker(game_state, action_plan)

    def _reserve_tc_and_power_factories(self, game_state: GameState) -> None:
        for factory in game_state.player_factories:
            for t in range(game_state.real_env_steps + 1, 7):
                goal = BuildLightGoal(factory)
                goal.generate_and_evaluate_action_plan(game_state, self.constraints, self.power_tracker)

                power_requests = goal.action_plan.get_power_requests(game_state)
                for power_request in power_requests:
                    power_request.t = t
                self.power_tracker.update_power_available(power_requests=power_requests)

                time_coordinates = goal.action_plan.time_coordinates
                for time_coordinate in time_coordinates:
                    tc = TimeCoordinate(time_coordinate.x, time_coordinate.y, t)
                    self.constraints.add_negative_constraint(tc)

    def _schedule_unassigned_units_goals(self, game_state: GameState) -> None:
        for factory in game_state.player_factories:
            for unit in factory.unassigned_units:
                goal = unit.generate_dummy_goal(game_state, self.constraints, self.power_tracker)
                unit.set_goal(goal)
                unit.set_private_action_plan(goal.action_plan)
                self._update_constraints_and_power_tracker(game_state, goal.action_plan)

    def _reserve_tc_and_power_units_with_private_action_plan(self, game_state: GameState) -> None:
        for unit in game_state.player_units:
            if unit.private_action_plan:
                self._update_constraints_and_power_tracker(game_state, unit.private_action_plan)

    def _get_highest_priority_factory_and_strategy(
        self, scores: Dict[Tuple[Factory, Strategy], float]
    ) -> Tuple[Factory, Strategy]:
        return max(scores, key=scores.get)  # type: ignore

    def _score_strategies(self, game_state: GameState) -> Dict[Tuple[Factory, Strategy], float]:
        scores = dict()

        for factory in game_state.player_factories:
            if factory.has_unassigned_units:
                factory_score = self._calculate_score_factory(factory, game_state)
                for (factory, strategy), score in factory_score.items():
                    scores[(factory, strategy)] = score

        return scores

    def _calculate_score_factory(
        self, factory: Factory, game_state: GameState
    ) -> Dict[Tuple[Factory, Strategy], float]:
        scores = defaultdict(lambda: 0.0)
        for ratio in RATIOS:
            score = ratio.compute_score(factory, game_state)
            scores[(factory, ratio.strategy)] += score

        return dict(scores)

    def _reschedule_factories_building_on_top_of_units(self, game_state: GameState) -> None:
        next_tc_units = set()
        for unit in game_state.player_units:
            if not unit.private_action_plan:
                next_tc = unit.tc + Direction.CENTER
            else:
                next_tc = unit.private_action_plan.next_tc

            next_tc_units.add(next_tc)

        for factory in game_state.player_factories:
            if not factory.private_action_plan:
                continue

            next_tc = factory.private_action_plan.next_tc
            if next_tc and next_tc in next_tc_units:
                action_plan = factory.set_goal(game_state, self.constraints, self.power_tracker, can_build=False)
                # Should also remove constraints and power tracker if I want to continue planning after this
                self._update_constraints_and_power_tracker(game_state, action_plan)
