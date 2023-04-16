from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Optional, TYPE_CHECKING, Iterable
from functools import lru_cache
from math import ceil

from copy import copy
from utils import PriorityQueue
from objects.actors.actor import Actor
from lux.config import UnitConfig
from objects.coordinate import TimeCoordinate, Coordinate
from objects.game_state import GameState
from objects.resource import Resource
from objects.actions.unit_action import UnitAction
from objects.actions.unit_action_plan import UnitActionPlan
from objects.cargo import Cargo
from logic.goal_resolution.power_availabilty_tracker import PowerTracker
from logic.constraints import Constraints
from logic.goals.unit_goal import (
    UnitGoal,
    DigGoal,
    CollectIceGoal,
    ClearRubbleGoal,
    CollectOreGoal,
    DestroyLichenGoal,
    UnitNoGoal,
    FleeGoal,
    TransferIceGoal,
    TransferOreGoal,
    EvadeConstraintsGoal,
    SupplyPowerGoal,
)
from config import CONFIG
from exceptions import NoValidGoalFoundError

if TYPE_CHECKING:
    from objects.actors.factory import Factory


@dataclass(eq=False)
class Unit(Actor):
    unit_type: str  # "LIGHT" or "HEAVY"
    tc: TimeCoordinate
    unit_cfg: UnitConfig
    action_queue: List[UnitAction] = field(init=False, default_factory=list)
    goal: Optional[UnitGoal] = field(init=False, default=None)
    can_be_assigned: bool = field(init=False, default=True)
    is_supplied_by: Optional[Unit] = field(init=False, default=None)
    private_action_plan: UnitActionPlan = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.private_action_plan = UnitActionPlan(self, [], is_set=True)
        self._set_unit_final_variables()
        self._set_unit_state_variables()

    def update_state(self, tc: TimeCoordinate, power: int, cargo: Cargo, action_queue: List[UnitAction]) -> None:
        self.tc = tc
        self.power = power
        self.cargo = cargo
        # For opponent units private_action_plan will be None
        if self._last_action_was_carried_out(action_queue) and self.private_action_plan:
            self.private_action_plan.step()

        self.action_queue = action_queue
        if self.private_action_plan and self.action_queue == self.private_action_plan.actions:
            self.private_action_plan.is_set = True

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
        self.rubble_removed_per_dig = self.unit_cfg.DIG_RUBBLE_REMOVED
        self.resources_gained_per_dig = self.unit_cfg.DIG_RESOURCE_GAIN
        self.lichen_removed_per_dig = self.unit_cfg.DIG_LICHEN_REMOVED
        self.update_action_queue_power_cost = self.unit_cfg.ACTION_QUEUE_POWER_COST

    def _set_unit_state_variables(self) -> None:
        self.is_scheduled = False
        self.x = self.tc.x
        self.y = self.tc.y
        self.power_space_left = self.battery_capacity - self.power
        self.cargo_space_left = self.unit_cfg.CARGO_SPACE - self.cargo.total
        self.can_update_action_queue = self.power >= self.update_action_queue_power_cost
        self.can_update_action_queue_and_move = self.power >= self.update_action_queue_power_cost + self.move_power_cost
        self.has_actions_in_queue = len(self.action_queue) > 0
        self.can_be_assigned = not self.goal and self.can_update_action_queue
        self.agent_id = f"player_{self.team_id}"

    def _last_action_was_carried_out(self, action_queue: list[UnitAction]) -> bool:
        if not self.action_queue:
            return True

        return self.action_queue != action_queue

    @property
    def first_action_of_queue_and_private_action_plan_same(self) -> bool:
        if not self.action_queue or not self.private_action_plan:
            return False

        first_action_of_queue = self.action_queue[0]
        first_action_of_plan = self.private_action_plan.primitive_actions[0]
        return first_action_of_queue.next_step_equal(first_action_of_plan)

    def generate_goals(self, game_state: GameState, factory: Factory) -> list[UnitGoal]:
        goals = self._generate_goals(game_state, factory)
        goals = self._filter_goals(goals, game_state)
        return goals

    def _generate_goals(self, game_state: GameState, factory: Factory) -> List[UnitGoal]:
        self._init_goals()
        self._add_base_goals(game_state, factory)
        self._add_dummy_goals()

        return self.goals

    def _init_goals(self) -> None:
        self.goals = []

    def next_step_walks_into_tile_where_it_might_be_captured(self, game_state: GameState) -> bool:
        return (self.is_light and self.next_step_walks_into_opponent_heavy(game_state)) or (
            self.next_step_walks_next_to_opponent_unit_that_can_capture_self(game_state)
        )

    def next_step_walks_next_to_opponent_unit_that_can_capture_self(self, game_state: GameState) -> bool:
        if not self.private_action_plan:
            return False

        next_action = self.private_action_plan.primitive_actions[0]
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

    # Dummy Goal added in case we can not reach factory, should add partial fleeing to solve this problem
    def generate_transfer_or_dummy_goal(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        transfer_goals = self._get_relevant_transfer_goals(game_state)
        dummy_goals = self._get_dummy_goals(game_state)
        goals = transfer_goals + dummy_goals
        goal = self.get_best_goal(goals, game_state, constraints, power_tracker)
        return goal

    def _get_flee_goal(self, game_state: GameState) -> FleeGoal:
        # TODO, this should be getting all threatening opponents and the flee goal should be adapted to
        # take multiple opponents into account
        neighboring_opponents = self._get_neighboring_opponents(game_state)
        randomly_picked_neighboring_opponent = neighboring_opponents[0]
        flee_goal = FleeGoal(unit=self, opp_c=randomly_picked_neighboring_opponent.tc)
        return flee_goal

    def _get_relevant_transfer_goals(self, game_state: GameState) -> List[UnitGoal]:
        goals = []
        if self.cargo.ice:
            ice_goal = self._get_transfer_ice_goal(game_state)
            goals.append(ice_goal)
        if self.cargo.ore:
            ore_goal = self._get_transfer_ore_goal(game_state)
            goals.append(ore_goal)

        return goals

    def _get_transfer_ice_goal(self, game_state: GameState, return_to_current_closest_factory: bool = True) -> UnitGoal:
        factory = game_state.get_closest_player_factory(c=self.tc) if return_to_current_closest_factory else None
        goal = TransferIceGoal(self, factory)
        return goal

    def _get_transfer_ore_goal(self, game_state: GameState, return_to_current_closest_factory: bool = True) -> UnitGoal:
        factory = game_state.get_closest_player_factory(c=self.tc) if return_to_current_closest_factory else None
        goal = TransferOreGoal(self, factory)
        return goal

    def generate_clear_rubble_goal(
        self, game_state: GameState, c: Coordinate, constraints: Constraints, power_tracker: PowerTracker
    ) -> ClearRubbleGoal:
        rubble_goals = self._get_clear_rubble_goals(c)
        goal = self.get_best_goal(rubble_goals, game_state, constraints, power_tracker)
        return goal  # type: ignore

    def get_best_goal(
        self,
        goals: Iterable[UnitGoal],
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitGoal:
        goals = list(goals)
        # goals = self.generate_goals(game_state)
        priority_queue = self._init_priority_queue(goals, game_state)

        constraints_with_danger = copy(constraints)
        unit_danger_coordinates = self.get_danger_tcs(game_state)
        constraints_with_danger.add_danger_coordinates(unit_danger_coordinates)

        while not priority_queue.is_empty():
            goal: UnitGoal = priority_queue.pop()

            try:
                goal.generate_and_evaluate_action_plan(game_state, constraints_with_danger, power_tracker)
            except Exception:
                continue

            priority = -1 * goal.value
            priority_queue.put(goal, priority)

            if goal == priority_queue[0]:
                return goal

        raise NoValidGoalFoundError

    def _init_priority_queue(self, goals: list[UnitGoal], game_state: GameState) -> PriorityQueue:
        goals_priority_queue = PriorityQueue()

        for goal in goals:
            best_value = goal.get_best_value_per_step(game_state)
            priority = -1 * best_value
            goals_priority_queue.put(goal, priority)

        return goals_priority_queue

    def _get_clear_rubble_goals(self, c: Coordinate) -> list[ClearRubbleGoal]:
        rubble_goals = [
            ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=c) for pickup_power in [False, True]
        ]

        return rubble_goals

    def _add_rubble_goals(self, factory: Factory, game_state: GameState) -> None:
        rubble_positions = factory.get_rubble_positions_to_clear(game_state)
        rubble_goals = [
            ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=Coordinate(*rubble_pos))
            for rubble_pos in rubble_positions
            for pickup_power in [False, True]
        ]

        self.goals.extend(rubble_goals)

    def _add_base_goals(self, game_state: GameState, factory: Factory) -> None:
        # if self.is_light:
        self._add_rubble_goals(factory, game_state)
        self._add_ice_goals(game_state, factory)
        self._add_ore_goals(game_state, factory)
        self._add_relevant_transfer_goals(game_state)
        # if game_state.real_env_steps >= CONFIG.START_STEP_DESTROYING_LICHEN:
        #     self._add_destroy_lichen_goals(game_state, n=10)
        # else:
        #     self._add_ice_goals(game_state, n=2, return_to_current_closest_factory=True)

    def generate_dummy_goal(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        dummy_goals = self._get_dummy_goals(game_state)
        goal = self.get_best_goal(dummy_goals, game_state, constraints, power_tracker)
        return goal

    def _get_dummy_goals(self, game_state: GameState) -> list[UnitGoal]:
        dummy_goals = [UnitNoGoal(self), EvadeConstraintsGoal(self)]
        if self.is_under_threath(game_state):
            flee_goal = self._get_flee_goal(game_state)
            dummy_goals.append(flee_goal)

        return dummy_goals

    def generate_no_goal_goal(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitNoGoal:
        no_goal_goal = UnitNoGoal(self)
        goal = self.get_best_goal([no_goal_goal], game_state, constraints, power_tracker)
        return goal  # type: ignore

    def generate_collect_ore_goal(
        self,
        game_state: GameState,
        c: Coordinate,
        is_supplied: bool,
        constraints: Constraints,
        power_tracker: PowerTracker,
        factory: Factory,
    ) -> CollectOreGoal:
        ore_goals = self._get_collect_ore_goals(c, game_state, factory, is_supplied)
        goal = self.get_best_goal(ore_goals, game_state, constraints, power_tracker)
        return goal  # type: ignore

    def generate_supply_power_goal(
        self,
        game_state: GameState,
        receiving_unit: Unit,
        receiving_action_plan: UnitActionPlan,
        receiving_c: Coordinate,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> SupplyPowerGoal:
        supply_goal = self._get_supply_power_goal(receiving_unit, receiving_action_plan, receiving_c)
        goal = self.get_best_goal([supply_goal], game_state, constraints, power_tracker)
        return goal  # type: ignore

    def _get_supply_power_goal(
        self, receiving_unit: Unit, receiving_action_plan: UnitActionPlan, receiving_c: Coordinate
    ) -> SupplyPowerGoal:
        return SupplyPowerGoal(self, receiving_unit, receiving_action_plan, receiving_c)

    def _get_collect_ore_goals(
        self, c: Coordinate, game_state: GameState, factory: Factory, is_supplied: bool
    ) -> list[CollectOreGoal]:
        ore_goals = [
            CollectOreGoal(unit=self, pickup_power=pickup_power, dig_c=c, factory=factory, is_supplied=is_supplied)
            for pickup_power in [False, True]
            if self._is_feasible_dig_c(c, game_state)
        ]

        return ore_goals

    def generate_collect_ice_goal(
        self,
        game_state: GameState,
        c: Coordinate,
        is_supplied: bool,
        constraints: Constraints,
        power_tracker: PowerTracker,
        factory: Factory,
    ) -> CollectIceGoal:
        ice_goals = self._get_collect_ice_goals(c, game_state, factory, is_supplied)
        goal = self.get_best_goal(ice_goals, game_state, constraints, power_tracker)
        return goal  # type: ignore

    def generate_destroy_lichen_goal(
        self, game_state: GameState, c: Coordinate, constraints: Constraints, power_tracker: PowerTracker
    ) -> DestroyLichenGoal:

        ice_goals = self._get_destroy_lichen_goals(c, game_state)
        goal = self.get_best_goal(ice_goals, game_state, constraints, power_tracker)
        return goal  # type: ignore

    def _get_collect_ice_goals(
        self, c: Coordinate, game_state: GameState, factory: Factory, is_supplied: bool
    ) -> list[CollectIceGoal]:
        ice_goals = [
            CollectIceGoal(unit=self, pickup_power=pickup_power, dig_c=c, factory=factory, is_supplied=is_supplied)
            for pickup_power in [False, True]
            if self._is_feasible_dig_c(c, game_state)
        ]

        return ice_goals

    def _add_ice_goals(self, game_state: GameState, factory: Factory) -> None:
        ice_positions = game_state.board.ice_positions_set - game_state.positions_in_dig_goals

        ice_goals = [
            CollectIceGoal(
                unit=self, pickup_power=pickup_power, dig_c=Coordinate(*ice_pos), factory=factory, is_supplied=False
            )
            for ice_pos in ice_positions
            for pickup_power in [False, True]
        ]

        self.goals.extend(ice_goals)

    def _add_ore_goals(self, game_state: GameState, factory: Factory) -> None:
        ore_positions = game_state.board.ore_positions_set - game_state.positions_in_dig_goals

        ore_goals = [
            CollectOreGoal(
                unit=self, pickup_power=pickup_power, dig_c=Coordinate(*ore_pos), factory=factory, is_supplied=False
            )
            for ore_pos in ore_positions
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

    def _get_destroy_lichen_goals(self, c: Coordinate, game_state: GameState) -> List[DestroyLichenGoal]:
        return [
            DestroyLichenGoal(self, pickup_power, c)
            for pickup_power in [False, True]
            if self._is_feasible_dig_c(c, game_state)
        ]

    def _is_feasible_dig_c(self, c: Coordinate, game_state: GameState) -> bool:
        return not (self.is_light and game_state.get_dis_to_closest_opp_heavy(c) <= 1)

    def _add_dummy_goals(self) -> None:
        dummy_goals = [UnitNoGoal(self), EvadeConstraintsGoal(self)]
        self.goals.extend(dummy_goals)

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
        if not self.private_action_plan:
            return False

        return self.private_action_plan.is_first_action_stationary

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

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.tc}"
        return out

    def __repr__(self) -> str:
        return f"Unit[id={self.unit_id}]"

    @property
    def ice(self) -> int:
        return self.cargo.ice

    @property
    def water(self) -> int:
        return self.cargo.water

    @property
    def ore(self) -> int:
        return self.cargo.ore

    @property
    def metal(self) -> int:
        return self.cargo.metal

    def set_action_queue(self, action_plan: UnitActionPlan) -> None:
        self.action_queue = action_plan.actions

    def set_private_action_plan(self, action_plan: UnitActionPlan) -> None:
        self.private_action_plan = action_plan

    def remove_goal_and_private_action_plan(self) -> None:
        if self.is_supplied_by:
            self.is_supplied_by.remove_goal_and_private_action_plan()

        if isinstance(self.goal, SupplyPowerGoal):
            self.goal.receiving_unit.is_supplied_by = None

        self.goal = None
        self.is_scheduled = False
        self.private_action_plan = UnitActionPlan(self, [])
        self.can_be_assigned = True

        if not self.action_queue:
            self.private_action_plan.is_set = True
