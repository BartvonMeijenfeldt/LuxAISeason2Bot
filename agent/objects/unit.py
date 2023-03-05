from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from objects.cargo import UnitCargo
from lux.config import UnitConfig
from objects.coordinate import TimeCoordinate, CoordinateList
from objects.game_state import GameState
from objects.action import Action

from logic.goal import (
    GoalCollection,
    CollectIceGoal,
    ClearRubbleGoal,
    CollectOreGoal,
    NoGoalGoal,
    ActionQueueGoal,
    FleeGoal,
)


@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    tc: TimeCoordinate
    power: int
    cargo: UnitCargo
    unit_cfg: UnitConfig
    action_queue: List[Action]
    time_to_power_cost = 50
    _is_under_threath: Optional[bool] = field(init=False, default=None)

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

    def generate_goals(self, game_state: GameState, action_queue_goal: Optional[ActionQueueGoal]) -> GoalCollection:
        if action_queue_goal:
            goals = [action_queue_goal]
            if self.is_under_threath(game_state) and self.next_step_is_stationary():
                # TODO, this should be getting all threatening opponents and the flee goal should be adapted to 
                # take multiple opponents into account
                neighboring_opponents = self._get_neighboring_opponents(game_state)
                randomly_picked_neighboring_opponent = neighboring_opponents[0]
                flee_goal = FleeGoal(unit=self, opp_c=randomly_picked_neighboring_opponent.tc)
                goals.append(flee_goal)

        elif game_state.env_steps <= 900 and self.unit_type == "HEAVY":
            target_ice_c = game_state.get_closest_ice_tile(c=self.tc)
            target_factory_c = game_state.get_closest_factory_c(c=target_ice_c)
            goals = [CollectIceGoal(unit=self, resource_c=target_ice_c, factory_c=target_factory_c)]

        else:
            closest_rubble_tiles = game_state.get_n_closest_rubble_tiles(c=self.tc, n=5)
            goals = [
                ClearRubbleGoal(unit=self, rubble_positions=CoordinateList([rubble_tile]))
                for rubble_tile in closest_rubble_tiles
            ]

            target_ore_c = game_state.get_closest_ore_tile(c=self.tc)
            target_factory_c = game_state.get_closest_factory_c(c=target_ore_c)
            collect_ore_goal = CollectOreGoal(unit=self, resource_c=target_ore_c, factory_c=target_factory_c)
            goals.append(collect_ore_goal)

        goals += [NoGoalGoal(unit=self)]
        goals = GoalCollection(goals)

        return goals

    def is_under_threath(self, game_state: GameState) -> bool:
        if self._is_under_threath is None:
            self._is_under_threath = self._get_is_under_threath(game_state)

        return self._is_under_threath

    def _get_is_under_threath(self, game_state: GameState) -> bool:
        neighboring_opponents = self._get_neighboring_opponents(game_state)

        for opponent in neighboring_opponents:
            if not self.is_stronger_than(opponent) and self.is_not_on_factory(game_state):
                return True

        return False

    def next_step_is_stationary(self) -> bool:
        return self.has_actions_in_queue and self.action_queue[0].is_stationary

    def is_not_on_factory(self, game_state: GameState) -> bool:
        return not game_state.is_player_factory_tile(self.tc)

    def _get_neighboring_opponents(self, game_state: GameState) -> list[Unit]:
        neighboring_opponents = []

        for tc in self.tc.neighbors:
            opponent_on_c = game_state.get_opponent_on_c(tc)
            if opponent_on_c:
                neighboring_opponents.append(opponent_on_c)

        return neighboring_opponents

    def is_stronger_than(self, other: Unit) -> bool:
        return self.unit_type == "HEAVY" and other.unit_type == "LIGHT"

    @property
    def recharge_power(self) -> int:
        return self.unit_cfg.CHARGE

    @property
    def dig_power_cost(self) -> int:
        return self.unit_cfg.DIG_COST

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
        return f"Unit[id={self.unit_id}]"
