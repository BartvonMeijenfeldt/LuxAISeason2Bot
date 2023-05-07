from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence, Set
from logic.goals.unit_goal import DefendTileGoal, DefendLichenTileGoal

from dataclasses import dataclass
from lux.config import EnvConfig

if TYPE_CHECKING:
    from objects.board import Board
    from objects.actors.actor import Actor
    from objects.actors.factory import Factory
    from objects.actors.unit import Unit
    from lux.team import Team
    from objects.coordinate import Coordinate, CoordinateList


@dataclass
class GameState:
    env_steps: int
    env_cfg: EnvConfig
    board: Board
    player_team: Team
    opp_team: Team

    def __hash__(self) -> int:
        return self.real_env_steps

    def __eq__(self, o: GameState) -> bool:
        return self.real_env_steps == o.real_env_steps

    def __repr__(self) -> str:
        return f"Gamestate [t={self.real_env_steps}]"

    @property
    def actors(self) -> Sequence[Actor]:
        return self.units + self.factories

    @property
    def units(self) -> list[Unit]:
        return self.player_units + self.opp_units

    @property
    def player_units(self) -> list[Unit]:
        return self.board.player_units

    @property
    def opp_units(self) -> list[Unit]:
        return self.board.opp_units

    @property
    def factories(self) -> list[Factory]:
        return self.player_factories + self.opp_factories

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

    @property
    def steps_left(self) -> int:
        return EnvConfig.max_episode_length - self.real_env_steps - 1

    @property
    def opp_lichen_tiles(self) -> CoordinateList:
        return self.board.opp_lichen_tiles

    def is_player_factory_tile(self, c: Coordinate) -> bool:
        return self.board.is_player_factory_tile(c)

    def is_opponent_factory_tile(self, c: Coordinate) -> bool:
        return self.board.is_opponent_factory_tile(c)

    def get_player_unit_on_c(self, c: Coordinate) -> Optional[Unit]:
        return self.board.get_player_unit_on_c(c)

    def get_closest_player_factory(self, c: Coordinate) -> Factory:
        return self.board.get_closest_player_factory(c=c)

    def get_closest_player_factory_c(self, c: Coordinate) -> Coordinate:
        return self.board.get_closest_player_factory_tile(c)

    def get_dis_to_closest_opp_heavy(self, c: Coordinate) -> float:
        return self.board.get_min_dis_to_opp_heavy(c=c)

    def c_is_undefended(self, c: Coordinate) -> bool:
        return not self.c_is_defended(c)

    def c_is_defended(self, c: Coordinate) -> bool:
        return c.xy in self.defended_c

    @property
    def defended_c(self) -> Set[tuple]:
        return {
            unit.goal.tile_c.xy
            for unit in self.player_units
            if unit.is_scheduled
            and (isinstance(unit.goal, DefendTileGoal) or isinstance(unit.goal, DefendLichenTileGoal))
        }

    def is_opponent_heavy_on_tile(self, c: Coordinate) -> bool:
        return self.board.is_opponent_heavy_on_tile(c=c)

    def get_neighboring_opponents(self, c: Coordinate) -> list[Unit]:
        return self.board.get_neighboring_opponents(c=c)

    def is_rubble_tile(self, c: Coordinate) -> bool:
        return self.board.is_rubble_tile(c)

    def is_opponent_lichen_tile(self, c: Coordinate) -> bool:
        return self.board.is_opponent_lichen_tile(c)

    @property
    def positions_in_dig_goals(self) -> set[tuple]:
        return self.board.positions_in_dig_goals

    @property
    def positions_in_heavy_dig_goals(self) -> set[tuple]:
        return self.board.positions_in_heavy_dig_goals

    def get_min_distance_to_any_opp_factory(self, c: Coordinate) -> int:
        return self.board.get_min_distance_to_any_opp_factory(c)

    def get_min_distance_to_any_player_factory(self, c: Coordinate) -> int:
        return self.board.get_min_distance_to_any_player_factory(c)
