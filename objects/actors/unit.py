from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Optional
from functools import lru_cache
from math import ceil

import logging
from objects.actors.actor import Actor
from lux.config import UnitConfig
from objects.coordinate import TimeCoordinate, Coordinate
from objects.game_state import GameState
from objects.resource import Resource
from objects.actions.unit_action import UnitAction
from objects.actions.unit_action_plan import UnitActionPlan
from objects.cargo import UnitCargo
from logic.goals.goal import GoalCollection
from logic.goals.unit_goal import (
    UnitGoal,
    DigGoal,
    CollectIceGoal,
    ClearRubbleGoal,
    CollectOreGoal,
    DestroyLichenGoal,
    UnitNoGoal,
    ActionQueueGoal,
    FleeGoal,
    TransferIceGoal,
    TransferOreGoal,
    EvadeConstraintsGoal,
)
from config import CONFIG


@dataclass
class Unit(Actor):
    unit_type: str  # "LIGHT" or "HEAVY"
    tc: TimeCoordinate
    unit_cfg: UnitConfig
    action_queue: List[UnitAction]
    goal: Optional[UnitGoal] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._set_unit_final_variables()
        self._set_unit_state_variables()

    def update_state(self, tc: TimeCoordinate, power: int, cargo: UnitCargo, action_queue: List[UnitAction]) -> None:
        self.tc = tc
        self.power = power
        self.cargo = cargo
        self.action_queue = action_queue
        self._set_unit_state_variables()

    def _set_unit_final_variables(self) -> None:
        self.id = int(self.unit_id[5:])
        self.is_light = self.unit_type == "LIGHT"
        self.is_heavy = self.unit_type == "HEAVY"
        self.time_to_power_cost = CONFIG.LIGHT_TIME_TO_POWER_COST if self.is_light else CONFIG.HEAVY_TIME_TO_POWER_COST
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
        self.lichen_removed_per_dig = self.unit_cfg.DIG_LICHEN_REMOVED

    def _set_unit_state_variables(self) -> None:
        self.x = self.tc.x
        self.y = self.tc.y
        self.has_actions_in_queue = len(self.action_queue) > 0
        self.agent_id = f"player_{self.team_id}"

    def generate_goals(self, game_state: GameState) -> GoalCollection:
        goals = self._generate_goals(game_state)
        goals = self._filter_goals(goals, game_state)
        return GoalCollection(goals)

    def _generate_goals(self, game_state: GameState) -> List[UnitGoal]:
        self._init_goals()

        if self.action_queue and self.goal and not self.goal.is_completed(game_state):
            if self.is_under_threath(game_state) and self.next_step_is_stationary():
                self._add_flee_goal(game_state)
                self._add_relevant_transfer_goals(game_state)

            elif self.next_step_walks_into_tile_where_it_might_be_captured(game_state):
                self._add_base_goals(game_state)
            else:
                self._add_action_queue_goal()

        else:
            self._add_base_goals(game_state)

        self._add_dummy_goals()

        return self.goals

    def _init_goals(self) -> None:
        self.goals = []

    def _add_action_queue_goal(self) -> None:
        if not self.goal:
            logging.critical("Action queue found but no prev step goal")
            return

        prev_step_goal = self.goal
        prev_goal = prev_step_goal.goal if isinstance(prev_step_goal, ActionQueueGoal) else prev_step_goal
        action_plan = UnitActionPlan(original_actions=self.action_queue, actor=self, is_set=True)
        action_queue_goal = ActionQueueGoal(unit=self, action_plan=action_plan, goal=prev_goal)
        self.goals.append(action_queue_goal)

    def next_step_walks_into_tile_where_it_might_be_captured(self, game_state: GameState) -> bool:
        return (self.is_light and self.next_step_walks_into_opponent_heavy(game_state)) or (
            self.next_step_walks_next_to_opponent_unit_that_can_capture_self(game_state)
        )

    def next_step_walks_next_to_opponent_unit_that_can_capture_self(self, game_state: GameState) -> bool:
        next_action = self.action_queue[0]
        #  TODO put this in proper function or something
        next_c = self.tc + next_action.unit_direction
        if game_state.is_player_factory_tile(next_c):
            return False

        return self.is_next_c_next_to_opponent_that_can_capture_self(c=next_c, game_state=game_state)

    def is_next_c_next_to_opponent_that_can_capture_self(self, c: Coordinate, game_state: GameState) -> bool:
        if game_state.is_player_factory_tile(c):
            return False

        strongest_neighboring_opponent = self.get_strongest_neighboring_opponent(c, game_state)
        if not strongest_neighboring_opponent:
            return False

        if self.tc.xy == c.xy:
            return strongest_neighboring_opponent.can_capture_opponent_stationary(self)
        else:
            return strongest_neighboring_opponent.can_capture_opponent_moving(self)

    def get_strongest_neighboring_opponent(self, c: Coordinate, game_state: GameState) -> Optional[Unit]:
        neighboring_opponents = game_state.get_neighboring_opponents(c)
        if not neighboring_opponents:
            return None

        strongest_neighboring_opponent = max(neighboring_opponents, key=lambda x: x.is_heavy * 10_000 + x.power)
        return strongest_neighboring_opponent

    def can_capture_opponent_stationary(self, other: Unit) -> bool:
        return not other.is_stronger_than(self)

    def can_capture_opponent_moving(self, other: Unit) -> bool:
        return self.is_stronger_than(other) or (not other.is_stronger_than(self) and self.power >= other.power)

    def next_step_walks_into_opponent_heavy(self, game_state: GameState) -> bool:
        if not self.action_queue:
            return False

        next_action = self.action_queue[0]
        #  TODO put this in proper function or something
        next_c = self.tc + next_action.unit_direction
        return game_state.is_opponent_heavy_on_tile(next_c)

    def _add_flee_goal(self, game_state: GameState) -> None:
        # TODO, this should be getting all threatening opponents and the flee goal should be adapted to
        # take multiple opponents into account
        neighboring_opponents = self._get_neighboring_opponents(game_state)
        randomly_picked_neighboring_opponent = neighboring_opponents[0]
        flee_goal = FleeGoal(unit=self, opp_c=randomly_picked_neighboring_opponent.tc)
        self.goals.append(flee_goal)

    def _add_rubble_goals(self, game_state: GameState, max_distance: int = 10) -> None:
        rubble_positions = game_state.board.get_rubble_to_remove_positions(c=self.tc, max_distance=max_distance)
        rubble_goals = [
            ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=Coordinate(*rubble_pos))
            for rubble_pos in rubble_positions
            if game_state.board.is_rubble_tile(Coordinate(*rubble_pos))
            for pickup_power in [False, True]
        ]

        self.goals.extend(rubble_goals)

    def _add_base_goals(self, game_state: GameState) -> None:
        if self.is_light:
            self._add_rubble_goals(game_state)
            self._add_ice_goals(game_state, n=2, return_to_current_closest_factory=True)
            self._add_ore_goals(game_state, n=2, return_to_current_closest_factory=True)
            self._add_relevant_transfer_goals(game_state)
            if game_state.real_env_steps >= CONFIG.START_STEP_DESTROYING_LICHEN:
                self._add_destroy_lichen_goals(game_state, n=10)
        else:
            self._add_ice_goals(game_state, n=2, return_to_current_closest_factory=True)

    def _add_ice_goals(self, game_state: GameState, n: int, return_to_current_closest_factory: bool = True) -> None:
        closest_ice_tiles = game_state.get_n_closest_ice_tiles(c=self.tc, n=n)
        factory = game_state.get_closest_player_factory(c=self.tc) if return_to_current_closest_factory else None

        ice_goals = [
            CollectIceGoal(
                unit=self,
                pickup_power=pickup_power,
                dig_c=ice_tile,
                factory=factory,
            )
            for ice_tile in closest_ice_tiles
            for pickup_power in [False, True]
        ]

        self.goals.extend(ice_goals)

    def _add_ore_goals(self, game_state: GameState, n: int, return_to_current_closest_factory: bool = True) -> None:
        closest_ore_tiles = game_state.get_n_closest_ore_tiles(c=self.tc, n=n)
        factory = game_state.get_closest_player_factory(c=self.tc) if return_to_current_closest_factory else None

        ore_goals = [
            CollectOreGoal(unit=self, pickup_power=pickup_power, dig_c=ore_tile, factory=factory)
            for ore_tile in closest_ore_tiles
            for pickup_power in [False, True]
        ]

        self.goals.extend(ore_goals)

    def _add_destroy_lichen_goals(self, game_state: GameState, n: int) -> None:
        closest_lichen_tiles = game_state.get_n_closest_opp_lichen_tiles(c=self.tc, n=n)
        destroy_lichen_goals = [
            DestroyLichenGoal(unit=self, pickup_power=pickup_power, dig_c=lichen_tile)
            for lichen_tile in closest_lichen_tiles
            for pickup_power in [False, True]
        ]
        self.goals.extend(destroy_lichen_goals)

    def _add_dummy_goals(self) -> None:
        none_goals = [UnitNoGoal(self), EvadeConstraintsGoal(self)]
        self.goals.extend(none_goals)

    def _add_relevant_transfer_goals(self, game_state: GameState) -> None:
        if self.cargo.ice:
            self._add_transfer_ice_goal(game_state)
        if self.cargo.ore:
            self._add_transfer_ore_goal(game_state)

    def _add_transfer_ice_goal(self, game_state: GameState, return_to_current_closest_factory: bool = True) -> None:
        factory = game_state.get_closest_player_factory(c=self.tc) if return_to_current_closest_factory else None
        goal = TransferIceGoal(self, factory)
        self.goals.append(goal)

    def _add_transfer_ore_goal(self, game_state: GameState, return_to_current_closest_factory: bool = True) -> None:
        factory = game_state.get_closest_player_factory(c=self.tc) if return_to_current_closest_factory else None
        goal = TransferOreGoal(self, factory)
        self.goals.append(goal)

    def _filter_goals(self, goals: Sequence[UnitGoal], game_state: GameState) -> List[UnitGoal]:
        return [
            goal
            for goal in goals
            if not (
                self.is_light and isinstance(goal, DigGoal) and game_state.get_dis_to_closest_opp_heavy(goal.dig_c) <= 1
            )
        ]

    @lru_cache(8)
    def is_under_threath(self, game_state: GameState) -> bool:
        if self.is_on_factory(game_state):
            return False

        neighboring_opponents = self._get_neighboring_opponents(game_state)
        return any(opponent.can_capture_opponent_stationary(self) for opponent in neighboring_opponents)

    def _get_neighboring_opponents(self, game_state: GameState) -> list[Unit]:
        return game_state.get_neighboring_opponents(self.tc)

    def next_step_is_stationary(self) -> bool:
        return self.has_actions_in_queue and self.action_queue[0].is_stationary

    def is_on_factory(self, game_state: GameState) -> bool:
        return game_state.is_player_factory_tile(self.tc)

    def get_quantity_resource_in_cargo(self, resource: Resource) -> int:
        return self.cargo.get_resource(resource)

    def get_nr_digs_to_fill_cargo(self) -> int:
        return ceil(self.cargo_space_left / self.resources_gained_per_dig)

    def get_danger_tcs(self, game_state) -> dict[TimeCoordinate, float]:
        return {
            neighbor_c: 10_000
            for neighbor_c in self.tc.neighbors
            if self.is_next_c_next_to_opponent_that_can_capture_self(neighbor_c, game_state)
        }

    def is_stronger_than(self, other: Unit) -> bool:
        return self.is_heavy and other.is_light

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.tc}"
        return out

    def __repr__(self) -> str:
        return f"Unit[id={self.unit_id}]"
