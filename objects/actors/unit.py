from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from objects.actors.actor import Actor
from lux.config import UnitConfig
from objects.coordinate import TimeCoordinate
from objects.game_state import GameState
from objects.resource import Resource
from objects.actions.unit_action import UnitAction
from objects.actions.unit_action_plan import UnitActionPlan
from logic.goals.unit_goal import (
    UnitGoal,
    CollectIceGoal,
    ClearRubbleGoal,
    CollectOreGoal,
    UnitNoGoal,
    ActionQueueGoal,
    FleeGoal,
    EvadeConstraintsGoal,
)


@dataclass
class Unit(Actor):
    unit_type: str  # "LIGHT" or "HEAVY"
    tc: TimeCoordinate
    unit_cfg: UnitConfig
    action_queue: List[UnitAction]
    prev_step_goal: Optional[UnitGoal]
    _is_under_threath: Optional[bool] = field(init=False, default=None)

    def __post_init__(self):
        self.time_to_power_cost = 5 if self.unit_type == "LIGHT" else 50
        self.recharge_power = self.unit_cfg.CHARGE
        self.move_power_cost = self.unit_cfg.MOVE_COST
        self.move_time_and_power_cost = self.move_power_cost + self.time_to_power_cost
        self.dig_power_cost = self.unit_cfg.DIG_COST
        self.dig_time_and_power_cost = self.dig_power_cost + self.time_to_power_cost
        self.action_queue_cost = self.unit_cfg.ACTION_QUEUE_POWER_COST
        self.battery_capacity = self.unit_cfg.BATTERY_CAPACITY
        self.power_space_left = self.battery_capacity - self.power
        self.cargo_space_left = self.unit_cfg.CARGO_SPACE - self.cargo.total
        self.rubble_removed_per_dig = self.unit_cfg.DIG_RUBBLE_REMOVED
        self.resources_gained_per_dig = self.unit_cfg.DIG_RESOURCE_GAIN
        self.has_actions_in_queue = len(self.action_queue) > 0
        self.agent_id = "player_0" if self.team_id == 0 else "player_1"

    def generate_goals(self, game_state: GameState) -> List[UnitGoal]:
        if self.action_queue:
            if not self.prev_step_goal:
                raise RuntimeError()

            action_plan = UnitActionPlan(original_actions=self.action_queue, actor=self, is_set=True)
            action_queue_goal = ActionQueueGoal(unit=self, action_plan=action_plan, goal=self.prev_step_goal)
            goals = [action_queue_goal]

            if self.is_under_threath(game_state) and self.next_step_is_stationary():
                # TODO, this should be getting all threatening opponents and the flee goal should be adapted to
                # take multiple opponents into account
                neighboring_opponents = self._get_neighboring_opponents(game_state)
                randomly_picked_neighboring_opponent = neighboring_opponents[0]
                flee_goal = FleeGoal(unit=self, opp_c=randomly_picked_neighboring_opponent.tc)
                goals.append(flee_goal)

        elif game_state.env_steps <= 920 and self.unit_type == "HEAVY":
            targets_ice_c = game_state.get_n_closest_ice_tiles(c=self.tc, n=2)
            goals = []
            for target_ice_c in targets_ice_c:
                for pickup_power in [False, True]:
                    target_factory_c = game_state.get_closest_player_factory_c(c=target_ice_c)
                    goals.append(
                        CollectIceGoal(
                            unit=self, pickup_power=pickup_power, dig_c=target_ice_c, factory_c=target_factory_c
                        )
                    )

        elif game_state.env_steps > 920 and self.unit_type == "HEAVY":
            closest_rubble_tiles = game_state.get_n_closest_rubble_tiles(c=self.tc, n=5)
            goals = []
            for pickup_power in [False, True]:
                goals.extend(
                    ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=rubble_tile)
                    for rubble_tile in closest_rubble_tiles
                )

        else:
            closest_rubble_tiles = game_state.get_n_closest_rubble_tiles(c=self.tc, n=5)
            goals = []
            for pickup_power in [False, True]:
                goals.extend(
                    ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=rubble_tile)
                    for rubble_tile in closest_rubble_tiles
                )

            closest_ice_tiles = game_state.get_n_closest_ice_tiles(c=self.tc, n=1)

            for pickup_power in [False, True]:

                ice_goals = [
                    CollectIceGoal(
                        unit=self,
                        pickup_power=pickup_power,
                        dig_c=ice_tile,
                        factory_c=game_state.get_closest_player_factory_c(c=ice_tile),
                    )
                    for ice_tile in closest_ice_tiles
                ]

            goals.extend(ice_goals)

            target_ore_c = game_state.get_closest_ore_tile(c=self.tc)
            target_factory_c = game_state.get_closest_player_factory_c(c=target_ore_c)

            for pickup_power in [False, True]:
                collect_ore_goal = CollectOreGoal(
                    unit=self, pickup_power=pickup_power, dig_c=target_ore_c, factory_c=target_factory_c
                )
                goals.append(collect_ore_goal)

        goals += [UnitNoGoal(unit=self)]
        goals += [EvadeConstraintsGoal(unit=self)]

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

    def get_quantity_resource_in_cargo(self, resource: Resource) -> int:
        if resource.name == "ICE":
            return self.cargo.ice
        elif resource.name == "ORE":
            return self.cargo.ore
        elif resource.name == "WATER":
            return self.cargo.water
        elif resource.name == "METAL":
            return self.cargo.metal
        else:
            raise ValueError("Unexpexcted resoruce")

    def _get_neighboring_opponents(self, game_state: GameState) -> list[Unit]:
        neighboring_opponents = []

        for tc in self.tc.neighbors:
            opponent_on_c = game_state.get_opponent_on_c(tc)
            if opponent_on_c:
                neighboring_opponents.append(opponent_on_c)

        return neighboring_opponents

    def is_stronger_than(self, other: Unit) -> bool:
        return self.unit_type == "HEAVY" and other.unit_type == "LIGHT"

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.tc}"
        return out

    def __repr__(self) -> str:
        return f"Unit[id={self.unit_id}]"
