from __future__ import annotations

from dataclasses import dataclass

from lux.config import EnvConfig
from objects.actions.factory_action import WaterAction, LIGHT_CFG, HEAVY_CFG
from objects.actors.actor import Actor
from objects.coordinate import TimeCoordinate, Coordinate, CoordinateList
from objects.game_state import GameState
from logic.goals.factory_goal import BuildHeavyGoal, BuildLightGoal, WaterGoal, FactoryNoGoal
from logic.goals.goal import GoalCollection


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
        if game_state.real_env_steps == 0:
            goals = [BuildHeavyGoal(self)]
        elif game_state.real_env_steps in [2, 4, 6, 8, 10]:
            goals = [BuildLightGoal(self)]
        elif self.can_build_light and game_state.real_env_steps > 11:
            goals = [BuildLightGoal(self)]
        elif (
            game_state.env_steps > 700
            and self.can_water(game_state)
            and self.cargo.water - self.water_cost(game_state) > game_state.steps_left
        ):
            goals = [WaterGoal(self)]
        else:
            goals = []

        goals.append(FactoryNoGoal(self))
        return GoalCollection(goals)

    @property
    def can_build_heavy(self):
        return self.power >= HEAVY_CFG.POWER_COST and self.cargo.metal >= HEAVY_CFG.METAL_COST

    @property
    def can_build_light(self):
        return self.power >= LIGHT_CFG.POWER_COST and self.cargo.metal >= LIGHT_CFG.METAL_COST

    def water_cost(self, game_state: GameState):
        return WaterAction.get_water_cost(game_state=game_state, strain_id=self.strain_id)

    def can_water(self, game_state):
        return self.cargo.water >= self.water_cost(game_state)

    @property
    def pos_slice(self):
        return self.x_slice, self.y_slice

    @property
    def x_slice(self):
        return slice(self.center_tc.x - self.radius, self.center_tc.x + self.radius + 1)

    @property
    def y_slice(self):
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
