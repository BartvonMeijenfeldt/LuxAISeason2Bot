from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Optional
from math import ceil

from objects.actors.actor import Actor
from lux.config import UnitConfig
from objects.coordinate import TimeCoordinate, Coordinate
from objects.game_state import GameState
from objects.resource import Resource
from objects.actions.unit_action import UnitAction
from objects.actions.unit_action_plan import UnitActionPlan
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

    def __post_init__(self) -> None:
        self.x = self.tc.x
        self.y = self.tc.y
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
        self.lichen_removed_per_dig = self.unit_cfg.DIG_LICHEN_REMOVED
        self.has_actions_in_queue = len(self.action_queue) > 0
        self.agent_id = f"player_{self.team_id}"

    def generate_goals(self, game_state: GameState) -> GoalCollection:
        goals = self._generate_goals(game_state)
        goals = self._filter_goals(goals, game_state)
        return GoalCollection(goals)

    def _generate_goals(self, game_state: GameState) -> List[UnitGoal]:
        self._init_goals()

        if self.action_queue:
            self._add_action_queue_goal()
            if self.is_under_threath(game_state) and self.next_step_is_stationary():
                self._add_flee_goal(game_state)

        elif self.unit_type == "LIGHT":
            self._add_factory_rubble_to_clear_goals(game_state)
            self._add_rubble_goals(game_state, n=10)

            if game_state.real_env_steps > 50:
                self._add_ice_goals(game_state, n=2)
                self._add_ore_goals(game_state, n=2)

            if game_state.real_env_steps > 500:
                self._add_destroy_lichen_goals(game_state, n=10)

        else:
            self._add_ice_goals(game_state, n=2, closest_factory_to_ice=False)

            # if game_state.real_env_steps > 20:
            #     self._add_ore_goals(game_state, n=2)

        self._add_dummy_goals()

        return self.goals

    def _init_goals(self) -> None:
        self.goals = []

    def _add_action_queue_goal(self) -> None:
        if not self.prev_step_goal:
            raise RuntimeError()

        action_plan = UnitActionPlan(original_actions=self.action_queue, actor=self, is_set=True)
        action_queue_goal = ActionQueueGoal(unit=self, action_plan=action_plan, goal=self.prev_step_goal)
        self.goals.append(action_queue_goal)

    def _add_flee_goal(self, game_state: GameState) -> None:
        # TODO, this should be getting all threatening opponents and the flee goal should be adapted to
        # take multiple opponents into account
        neighboring_opponents = self._get_neighboring_opponents(game_state)
        randomly_picked_neighboring_opponent = neighboring_opponents[0]
        flee_goal = FleeGoal(unit=self, opp_c=randomly_picked_neighboring_opponent.tc)
        self.goals.append(flee_goal)

    def _add_factory_rubble_to_clear_goals(self, game_state: GameState, max_distance: int = 10) -> None:
        rubble_positions = game_state.board.get_rubble_to_remove_positions(c=self.tc, max_distance=max_distance)
        rubble_goals = [
            ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=Coordinate(*rubble_pos))
            for rubble_pos in rubble_positions
            for pickup_power in [False, True]
        ]

        self.goals.extend(rubble_goals)

    def _add_rubble_goals(self, game_state: GameState, n: int) -> None:
        closest_rubble_tiles = game_state.get_n_closest_rubble_tiles(c=self.tc, n=n)
        rubble_goals = [
            ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=rubble_tile)
            for rubble_tile in closest_rubble_tiles
            for pickup_power in [False, True]
        ]

        self.goals.extend(rubble_goals)

    def _add_ice_goals(self, game_state: GameState, n: int, closest_factory_to_ice: bool = True) -> None:
        closest_ice_tiles = game_state.get_n_closest_ice_tiles(c=self.tc, n=n)

        if closest_factory_to_ice:
            ice_goals = [
                CollectIceGoal(unit=self, pickup_power=pickup_power, dig_c=ice_tile)
                for ice_tile in closest_ice_tiles
                for pickup_power in [False, True]
            ]
        else:
            ice_goals = [
                CollectIceGoal(
                    unit=self,
                    pickup_power=pickup_power,
                    dig_c=ice_tile,
                    factory=game_state.get_closest_player_factory(c=self.tc),
                )
                for ice_tile in closest_ice_tiles
                for pickup_power in [False, True]
            ]

        self.goals.extend(ice_goals)

    def _add_ore_goals(self, game_state: GameState, n: int) -> None:
        closest_ore_tiles = game_state.get_n_closest_ore_tiles(c=self.tc, n=n)
        ice_goals = [
            CollectOreGoal(unit=self, pickup_power=pickup_power, dig_c=ore_tile)
            for ore_tile in closest_ore_tiles
            for pickup_power in [False, True]
        ]

        self.goals.extend(ice_goals)

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

    def _filter_goals(self, goals: Sequence[UnitGoal], game_state: GameState) -> List[UnitGoal]:
        return [
            goal
            for goal in goals
            if not (
                self.unit_type == "LIGHT"
                and isinstance(goal, DigGoal)
                and game_state.get_dis_to_closest_opp_heavy(goal.dig_c) <= 1
            )
        ]

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

    def get_nr_digs_to_fill_cargo(self) -> int:
        return ceil(self.cargo_space_left / self.resources_gained_per_dig)

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
