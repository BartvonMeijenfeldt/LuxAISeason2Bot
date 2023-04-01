from abc import abstractmethod
import numpy as np

from objects.game_state import GameState
from objects.actions.action import Action
from lux.config import EnvConfig

ENV_CFG = EnvConfig()
LIGHT_CFG = ENV_CFG.LIGHT_ROBOT
HEAVY_CFG = ENV_CFG.HEAVY_ROBOT

COST_WATER = 1
COST_METAL = 1


class FactoryAction(Action):
    quantity_metal_cost: int

    @abstractmethod
    def to_lux_output(self) -> int:
        ...

    @staticmethod
    @abstractmethod
    def get_water_cost_from_strain_id(game_state: GameState, strain_id: int) -> int:
        ...

    @abstractmethod
    def get_resource_cost(self, game_state: GameState, strain_id: int) -> float:
        ...


class BuildAction(FactoryAction):
    @staticmethod
    def get_water_cost_from_strain_id(game_state: GameState, strain_id: int) -> int:
        return 0

    def get_resource_cost(self, game_state: GameState, strain_id: int) -> float:
        return self.quantity_metal_cost * COST_METAL


class BuildLightAction(BuildAction):
    quantity_metal_cost = LIGHT_CFG.METAL_COST

    @property
    def requested_power(self) -> int:
        return LIGHT_CFG.POWER_COST

    @staticmethod
    def to_lux_output() -> int:
        return 0


class BuildHeavyAction(BuildAction):
    quantity_metal_cost = HEAVY_CFG.METAL_COST

    @property
    def requested_power(self) -> int:
        return HEAVY_CFG.POWER_COST

    @staticmethod
    def to_lux_output() -> int:
        return 1


class WaterAction(FactoryAction):
    quantity_metal_cost = 0

    @property
    def requested_power(self) -> int:
        return 0

    def get_resource_cost(self, game_state: GameState, strain_id: int) -> float:
        return self.get_water_cost_from_strain_id(game_state=game_state, strain_id=strain_id) * COST_WATER

    @staticmethod
    def get_water_cost_from_strain_id(game_state: GameState, strain_id: int) -> int:
        # Might be less if you water at the same time as another factory
        max_nr_tiles_to_water = game_state.board.get_max_nr_tiles_to_water(strain_id)
        return np.ceil(max_nr_tiles_to_water / ENV_CFG.LICHEN_WATERING_COST_FACTOR)

    @staticmethod
    def to_lux_output() -> int:
        return 2
