import numpy as np

from dataclasses import dataclass

from lux.config import EnvConfig
from objects.cargo import UnitCargo
from objects.coordinate import Coordinate, CoordinateList
from objects.game_state import GameState


@dataclass
class Factory:
    team_id: int
    unit_id: str
    strain_id: int
    power: int
    cargo: UnitCargo
    center: Coordinate
    env_cfg: EnvConfig
    radius = 1

    def act(self, game_state: GameState) -> np.array:
        if game_state.real_env_steps == 0:
            return self.build_heavy()
        elif game_state.real_env_steps in [5, 10, 15, 20]:
            return self.build_light()
        elif game_state.env_steps > 775 and self.can_water(game_state):
            return self.water()
        else:
            return None

    def build_heavy_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.METAL_COST

    def build_heavy_power_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["HEAVY"]
        return unit_cfg.POWER_COST

    def can_build_heavy(self, game_state):
        return self.power >= self.build_heavy_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_heavy_metal_cost(game_state)

    def build_heavy(self):
        return 1

    def build_light_metal_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.METAL_COST

    def build_light_power_cost(self, game_state):
        unit_cfg = self.env_cfg.ROBOTS["LIGHT"]
        return unit_cfg.POWER_COST

    def can_build_light(self, game_state):
        return self.power >= self.build_light_power_cost(
            game_state
        ) and self.cargo.metal >= self.build_light_metal_cost(game_state)

    def build_light(self):
        return 0

    def water_cost(self, game_state):
        """
        Water required to perform water action
        """
        owned_lichen_tiles = (game_state.board.lichen_strains == self.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.env_cfg.LICHEN_WATERING_COST_FACTOR)

    def can_water(self, game_state):
        return self.cargo.water >= self.water_cost(game_state)

    def water(self):
        return 2

    @property
    def pos_slice(self):
        return self.x_slice, self.y_slice

    @property
    def x_slice(self):
        return slice(self.center.x - self.radius, self.center.x + self.radius + 1)

    @property
    def y_slice(self):
        return slice(self.center.y - self.radius, self.center.y + self.radius + 1)

    @property
    def pos_x_range(self):
        return range(self.center.x - self.radius, self.center.x + self.radius + 1)

    @property
    def pos_y_range(self):
        return range(self.center.y - self.radius, self.center.y + self.radius + 1)

    @property
    def coordinates(self) -> CoordinateList:
        return CoordinateList([Coordinate(x, y) for x in self.pos_x_range for y in self.pos_y_range])

    def is_on_factory(self, c: Coordinate) -> bool:
        return c in self.coordinates

    def min_dis_to(self, c: Coordinate) -> int:
        return self.coordinates.min_dis_to(c)

    def dis_to_tiles(self, c: Coordinate) -> list[int]:
        return self.coordinates.dis_to_tiles(c)

    def get_all_closest_factory_tiles(self, c: Coordinate) -> CoordinateList:
        return self.coordinates.get_all_closest_tiles(c)

    def get_closest_factory_tile(self, c: Coordinate) -> Coordinate:
        return self.coordinates.get_closest_tile(c)
