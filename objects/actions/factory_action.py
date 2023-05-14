from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from lux.config import HEAVY_CONFIG, LIGHT_CONFIG, EnvConfig
from objects.actions.action import Action

if TYPE_CHECKING:
    from objects.actors.factory import Factory


COST_WATER = 1
COST_METAL = 1


class FactoryAction(Action):
    quantity_metal_cost: int

    @abstractmethod
    def to_lux_output(self) -> int:
        ...

    @staticmethod
    @abstractmethod
    def get_water_cost(factory: Factory) -> int:
        ...

    @abstractmethod
    def get_resource_cost(self, factory: Factory) -> float:
        ...


class BuildAction(FactoryAction):
    @staticmethod
    def get_water_cost(factory: Factory) -> int:
        return 0

    def get_resource_cost(self, factory: Factory) -> float:
        return self.quantity_metal_cost * COST_METAL


class BuildLightAction(BuildAction):
    quantity_metal_cost = LIGHT_CONFIG.METAL_COST

    @property
    def requested_power(self) -> int:
        return LIGHT_CONFIG.POWER_COST

    @staticmethod
    def to_lux_output() -> int:
        return 0


class BuildHeavyAction(BuildAction):
    quantity_metal_cost = HEAVY_CONFIG.METAL_COST

    @property
    def requested_power(self) -> int:
        return HEAVY_CONFIG.POWER_COST

    @staticmethod
    def to_lux_output() -> int:
        return 1


class WaterAction(FactoryAction):
    quantity_metal_cost = 0

    @property
    def requested_power(self) -> int:
        return 0

    def get_resource_cost(self, factory: Factory) -> float:
        return self.get_water_cost(factory=factory) * COST_WATER

    @staticmethod
    def get_water_cost(factory: Factory) -> int:
        # Might be less if you water at the same time as another factory
        max_nr_tiles_to_water = factory.max_nr_tiles_to_water
        water_cost = np.ceil(max_nr_tiles_to_water / EnvConfig.LICHEN_WATERING_COST_FACTOR)
        return max(1, water_cost)

    @staticmethod
    def to_lux_output() -> int:
        return 2
