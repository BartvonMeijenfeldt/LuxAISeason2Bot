from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from typing import Tuple
from lux.config import EnvConfig
from objects.actions.factory_action import WaterAction, LIGHT_CFG, HEAVY_CFG
from objects.actors.actor import Actor
from objects.coordinate import TimeCoordinate, Coordinate, CoordinateList
from objects.game_state import GameState
from logic.goals.goal import GoalCollection
from logic.goals.factory_goal import BuildHeavyGoal, BuildLightGoal, WaterGoal, FactoryNoGoal, FactoryGoal


@dataclass
class Factory(Actor):
    strain_id: int
    center_tc: TimeCoordinate
    env_cfg: EnvConfig
    radius = 1

    def __hash__(self) -> int:
        return hash(self.unit_id)

    def __eq__(self, __o: Factory) -> bool:
        return self.unit_id == __o.unit_id

    def generate_goals(self, game_state: GameState) -> GoalCollection:
        water_cost = self.water_cost(game_state)
        goals = []

        if self.can_build_heavy:
            goals.append(BuildHeavyGoal(self))

        elif self.can_build_light:
            goals.append(BuildLightGoal(self))

        elif self.cargo.water - water_cost > 50 and water_cost < 5:
            goals.append(WaterGoal(self))

        elif (
            game_state.env_steps > 750
            and self.can_water(game_state)
            and self.cargo.water - water_cost > game_state.steps_left
        ):
            goals.append(WaterGoal(self))

        goals += self._get_dummy_goals()
        return GoalCollection(goals)

    def _get_dummy_goals(self) -> list[FactoryGoal]:
        return [FactoryNoGoal(self)]

    def get_expected_power_available(self, n=50) -> np.ndarray:
        # TODO add the expected effect of Lichen
        expected_increase_power = np.arange(n) * 50
        return expected_increase_power + self.power

    @property
    def daily_charge(self) -> int:
        return self.env_cfg.FACTORY_CHARGE

    @property
    def can_build_heavy(self) -> bool:
        return self.power >= HEAVY_CFG.POWER_COST and self.cargo.metal >= HEAVY_CFG.METAL_COST

    @property
    def can_build_light(self) -> bool:
        return self.power >= LIGHT_CFG.POWER_COST and self.cargo.metal >= LIGHT_CFG.METAL_COST

    def water_cost(self, game_state: GameState) -> int:
        return WaterAction.get_water_cost_from_strain_id(game_state=game_state, strain_id=self.strain_id)

    def can_water(self, game_state):
        return self.cargo.water >= self.water_cost(game_state)

    @property
    def pos_slice(self) -> Tuple[slice, slice]:
        return self.x_slice, self.y_slice

    @property
    def x_slice(self) -> slice:
        return slice(self.center_tc.x - self.radius, self.center_tc.x + self.radius + 1)

    @property
    def y_slice(self) -> slice:
        return slice(self.center_tc.y - self.radius, self.center_tc.y + self.radius + 1)

    @property
    def pos_x_range(self):
        return range(self.center_tc.x - self.radius, self.center_tc.x + self.radius + 1)

    @property
    def pos_y_range(self):
        return range(self.center_tc.y - self.radius, self.center_tc.y + self.radius + 1)

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

    def __repr__(self) -> str:
        return f"Factory[id={self.unit_id}, center={self.center_tc.xy}]"
