from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from agent.objects.board import Board
    from agent.lux.factory import Factory
    from agent.lux.unit import Unit
    from agent.lux.team import Team
    from agent.objects.coordinate import Coordinate, CoordinateList
    from agent.lux.config import EnvConfig


@dataclass
class GameState:
    env_steps: int
    env_cfg: EnvConfig
    board: Board
    player_units: list[Unit]
    opp_units: list[Unit]
    player_factories: list[Factory]
    opp_factories: list[Factory]
    player_team: Team
    opp_team: Team

    @property
    def real_env_steps(self):
        """
        the actual env step in the environment, which subtracts the time spent bidding and placing factories
        """
        if self.env_cfg.BIDDING_SYSTEM:
            # + 1 for extra factory placement and + 1 for bidding step
            return self.env_steps - (self.board.factories_per_team * 2 + 1)
        else:
            return self.env_steps

    def is_day(self):
        return self.real_env_steps % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH

    @property
    def ice_coordinates(self) -> CoordinateList:
        return self.board.ice_coordinates

    @property
    def player_factory_tiles(self) -> CoordinateList:
        return CoordinateList([c for factory in self.player_factories for c in factory.coordinates])

    def get_all_closest_factory_tiles(self, c: Coordinate) -> CoordinateList:
        return self.player_factory_tiles.get_all_closest_tiles(c)

    def get_closest_factory_tile(self, c: Coordinate) -> Coordinate:
        return self.player_factory_tiles.get_closest_tile(c)
