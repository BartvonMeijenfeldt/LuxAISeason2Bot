from dataclasses import dataclass
from typing import Literal

from objects.cargo import UnitCargo
from lux.config import UnitConfig
from objects.coordinate import TimeCoordinate, CoordinateList
from objects.game_state import GameState
from objects.action import Action

from logic.goal import GoalCollection, CollectIceGoal, ClearRubbleGoal, NoGoalGoal


@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: Literal["LIGHT", "HEAVY"]
    tc: TimeCoordinate
    power: int
    cargo: UnitCargo
    unit_cfg: UnitConfig
    action_queue: list[Action]
    time_to_power_cost = 50

    @property
    def agent_id(self):
        if self.team_id == 0:
            return "player_0"
        return "player_1"

    @property
    def has_actions_in_queue(self) -> bool:
        return len(self.action_queue) > 0

    @property
    def action_queue_cost(self):
        cost = self.unit_cfg.ACTION_QUEUE_POWER_COST
        return cost

    def generate_goals(self, game_state: GameState) -> GoalCollection:
        if game_state.env_steps <= 725 and self.unit_type == "HEAVY":
            target_ice_c = game_state.get_closest_ice_tile(c=self.tc)
            target_factory_c = game_state.get_closest_factory_c(c=target_ice_c)
            goals = [CollectIceGoal(unit=self, ice_c=target_ice_c, factory_c=target_factory_c)]

        else:
            closest_rubble_tiles = game_state.get_n_closest_rubble_tiles(c=self.tc, n=10)
            # TODO add something here to do a feasibility check if
            # they can ever clear the first rubble, even with a full capacity
            goals = [
                ClearRubbleGoal(unit=self, rubble_positions=CoordinateList([rubble_tile]))
                for rubble_tile in closest_rubble_tiles
            ]

        goals += [NoGoalGoal(unit=self)]
        goals = GoalCollection(goals)

        return goals

    @property
    def power_space_left(self) -> int:
        return self.unit_cfg.BATTERY_CAPACITY - self.power

    @property
    def cargo_space_left(self) -> int:
        return self.unit_cfg.CARGO_SPACE - self.cargo.total

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.tc}"
        return out

    def __repr__(self) -> str:
        return self.unit_id
