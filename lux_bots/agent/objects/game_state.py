from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from agent.objects.board import Board
    from agent.objects.factory import Factory
    from agent.objects.unit import Unit
    from agent.lux.team import Team
    from agent.objects.coordinate import Coordinate, CoordinateList
    from agent.lux.config import EnvConfig


@dataclass
class GameState:
    env_steps: int
    env_cfg: EnvConfig
    board: Board
    player_team: Team
    opp_team: Team

    @property
    def player_units(self) -> list[Unit]:
        return self.board.player_units

    @property
    def opp_units(self) -> list[Unit]:
        return self.board.opp_units

    @property
    def player_factories(self) -> list[Factory]:
        return self.board.player_factories

    @property
    def opp_factories(self) -> list[Factory]:
        return self.board.opp_factories

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
        return self.board.player_factory_tiles

    def get_all_closest_factory_tiles(self, c: Coordinate) -> CoordinateList:
        return self.player_factory_tiles.get_all_closest_tiles(c)

    def get_closest_factory_tile(self, c: Coordinate) -> Coordinate:
        return self.board.get_closest_factory_tile(c)

    def get_closest_ice_tile(self, c: Coordinate) -> Coordinate:
        return self.board.get_closest_ice_tile(c=c)

    def get_closest_rubble_tile(self, c: Coordinate) -> Coordinate:
        return self.board.get_closest_rubble_tile(c=c)
