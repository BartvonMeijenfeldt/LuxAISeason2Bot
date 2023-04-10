from typing import List, Dict, Tuple
from collections import defaultdict
from abc import ABCMeta, abstractmethod

from objects.game_state import GameState
from objects.actors.factory import Factory, Strategy
from logic.constraints import Constraints
from logic.goal_resolution.power_availabilty_tracker import PowerTracker
from lux.config import EnvConfig


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
    threshold = 1
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


class Scheduler:
    ratios: List[FactoryRatio] = [
        WaterLichenTilesRatio(),
        WaterNextStepsLichenTilesRatio(),
        UpperPowerUnitRatio(),
        LowerPowerUnitRatio(),
    ]

    def schedule_goals(self, game_state: GameState):
        constraints = Constraints()
        power_tracker = PowerTracker(game_state.player_factories)
        for factory in game_state.player_factories:
            action_plan = factory.set_goal(game_state, constraints, power_tracker)
            constraints.add_negative_constraints(action_plan.time_coordinates)
            power_tracker.update_power_available(action_plan.get_power_requests(game_state))

        # while True:
        i = 0
        while i < 10:
            scores = self._score_strategies(game_state)
            if not scores:
                break

            factory, strategy = self._get_highest_priority_factory_and_strategy(scores)
            try:
                action_plan = factory.schedule_unit(strategy, game_state, constraints, power_tracker)
            except Exception:
                i += 1
                continue

            constraints.add_negative_constraints(action_plan.time_coordinates)
            power_tracker.update_power_available(action_plan.get_power_requests(game_state))

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
        for ratio in self.ratios:
            score = ratio.compute_score(factory, game_state)
            scores[(factory, ratio.strategy)] += score

        return dict(scores)
