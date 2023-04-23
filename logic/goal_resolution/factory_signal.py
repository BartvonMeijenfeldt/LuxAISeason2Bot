from abc import ABCMeta, abstractmethod
from math import inf, sqrt
from typing import List

from objects.actors.factory import Factory, Strategy
from objects.game_state import GameState
from config import CONFIG
from lux.config import EnvConfig


class FactorySignal(metaclass=ABCMeta):
    strategy: Strategy

    @abstractmethod
    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        """Returns a signal around 1.0, which will be used to schedule goals based on highest values and if I have
        enough time to multiply goals value per step with this amount to make a smart decision based on what tasks
        are available around me and what the factory needs."""
        ...


class AlmostOutOfWaterSignal(FactorySignal):
    strategy = Strategy.IMMEDIATELY_RETURN_ICE

    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        if game_state.real_env_steps > 900:
            return 0.0

        water_supply_factory = factory.water + factory.ice * EnvConfig.ICE_WATER_RATIO
        if water_supply_factory < CONFIG.TOO_LITTLE_WATER_DISTRESS_LEVEL:
            water_supply_and_incoming = factory.get_incoming_ice_before_no_water(game_state) * EnvConfig.ICE_WATER_RATIO
            if water_supply_and_incoming < CONFIG.TOO_LITTLE_WATER_DISTRESS_LEVEL:
                return CONFIG.DISTRESS_SIGNAL

        return 0.0


class TooLittleLichenTilesSignal(FactorySignal):
    strategy = Strategy.INCREASE_LICHEN_TILES

    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        nr_tiles_needed_to_grow = factory.nr_tiles_needed_to_grow_to_lichen_target(game_state)
        signal = min(sqrt(nr_tiles_needed_to_grow), CONFIG.MAX_SIGNAL_TOO_LITTE_LICHEN)
        return signal


class IceCollectionSignal(FactorySignal):
    strategy = Strategy.COLLECT_ICE

    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        water_collection_per_step = factory.get_water_collection_per_step()
        water_cost_per_step = EnvConfig.FACTORY_WATER_CONSUMPTION + factory.water_cost / 2  # water every other turn
        water_collection_usage_ratio = water_collection_per_step / water_cost_per_step
        ratio_collection_vs_target = water_collection_usage_ratio / CONFIG.WATER_COLLECTION_VERSUS_USAGE_MIN_TARGET

        # Signal always at least 1, because we are expected to get our power back due to the lichen
        signal = max(1.0, 2 - ratio_collection_vs_target)

        return signal


class PowerUnitSignal(FactorySignal):
    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        if game_state.real_env_steps <= 2:
            return 0.0

        power = factory.power_including_units
        power_generation_per_step = factory.get_expected_power_generation(game_state)
        expected_power_usage_per_step = max(1, factory.get_expected_power_consumption())

        nr_steps = CONFIG.POWER_UNIT_RATIO_NR_STEPS
        signal = (power + nr_steps * power_generation_per_step) / (nr_steps * expected_power_usage_per_step)

        return signal


class CollectOreSignal(PowerUnitSignal):
    strategy = Strategy.COLLECT_ORE

    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        if game_state.real_env_steps > CONFIG.LAST_STEP_SCHEDULE_ORE_MINING:
            return -inf

        power_unit_signal = max(
            CONFIG.UNIT_IMPORTANCE_MIN_LEVEL_POWER_UNIT, super().compute_signal(factory, game_state)
        )
        unit_importance_signal = self._get_unit_importance_signal(game_state)
        return power_unit_signal * unit_importance_signal

    def _get_unit_importance_signal(self, game_state: GameState) -> float:
        slope = CONFIG.START_UNIT_IMPORTANCE_SIGNAL / CONFIG.LAST_TURN_UNIT_IMPORTANCE
        t = game_state.real_env_steps
        return CONFIG.START_UNIT_IMPORTANCE_SIGNAL - t * slope


class AttackOpponentSignal(FactorySignal):
    strategy = Strategy.ATTACK_OPPONENT

    def compute_signal(self, factory: Factory, game_state: GameState) -> float:
        if game_state.real_env_steps >= CONFIG.ATTACK_EN_MASSE_START_STEP:
            return CONFIG.ATTACK_EN_MASSE_SIGNAL

        return 0.8


SIGNALS: List[FactorySignal] = [
    AlmostOutOfWaterSignal(),
    TooLittleLichenTilesSignal(),
    IceCollectionSignal(),
    CollectOreSignal(),
    AttackOpponentSignal(),
]
