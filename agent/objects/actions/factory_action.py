from abc import abstractmethod
import numpy as np

from objects.game_state import GameState
from objects.actions.action import Action
from lux.config import EnvConfig

ENV_CFG = EnvConfig()
LIGHT_CFG = ENV_CFG.ROBOTS["LIGHT"]
HEAVY_CFG = ENV_CFG.ROBOTS["HEAVY"]


class FactoryAction(Action):
    metal_cost: int

    @abstractmethod
    def to_lux_output(self) -> int:
        ...

    @staticmethod
    @abstractmethod
    def get_water_cost(game_state: GameState, strain_id: int) -> int:
        ...


class BuildAction(FactoryAction):
    @staticmethod
    def get_water_cost(game_state: GameState, strain_id: int) -> int:
        return 0


class BuildLightAction(BuildAction):
    metal_cost = LIGHT_CFG.METAL_COST

    @property
    def requested_power(self) -> int:
        return LIGHT_CFG.POWER_COST

    @staticmethod
    def to_lux_output() -> int:
        return 0


class BuildHeavyAction(BuildAction):
    metal_cost = HEAVY_CFG.METAL_COST

    @property
    def requested_power(self) -> int:
        return HEAVY_CFG.POWER_COST

    @staticmethod
    def to_lux_output() -> int:
        return 1


class WaterAction(FactoryAction):
    metal_cost = 0

    @property
    def requested_power(self) -> int:
        return 0

    @staticmethod
    def get_water_cost(game_state: GameState, strain_id: int) -> int:
        # TODO, this underestimates the cost, does not take into account the new water tiles
        owned_lichen_tiles = (game_state.board.lichen_strains == strain_id).sum()
        return np.ceil(owned_lichen_tiles / ENV_CFG.LICHEN_WATERING_COST_FACTOR)

    @staticmethod
    def to_lux_output() -> int:
        return 2
