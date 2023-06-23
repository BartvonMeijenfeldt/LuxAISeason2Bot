from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass, field, replace
from math import ceil
from typing import TYPE_CHECKING, Iterable, List, Optional

from config import CONFIG
from exceptions import ActorFoundNoValidGoalError, InvalidGoalError
from logic.constraints import Constraints
from logic.goal_resolution.schedule_info import ScheduleInfo
from logic.goals.unit_goal import (
    ClearRubbleGoal,
    CollectGoal,
    CollectIceGoal,
    CollectOreGoal,
    DefendLichenTileGoal,
    DefendTileGoal,
    DestroyLichenGoal,
    EvadeConstraintsGoal,
    FleeGoal,
    SupplyPowerGoal,
    TransferGoal,
    TransferIceGoal,
    TransferOreGoal,
    UnitGoal,
    UnitNoGoal,
)
from lux.config import UnitConfig
from objects.actions.unit_action import UnitAction
from objects.actions.unit_action_plan import (
    UnitActionPlan,
    get_primitive_actions_from_list,
)
from objects.actors.actor import Actor
from objects.cargo import Cargo
from objects.coordinate import Coordinate, TimeCoordinate
from objects.direction import Direction
from objects.game_state import GameState
from objects.resource import Resource
from utils.utils import PriorityQueue

if TYPE_CHECKING:
    from objects.actors.factory import Factory


logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Unit(Actor):
    """Unit.

    Args:
        unit_type: Whether the unit is a light or heavy type.
        tc: Time coordinate of unit.
        unit_cfg: Config of the unit.
        action_queue: Actions currently in the queue of the unit.
        goal: The strategic goal of the unit.
        can_be_assigned: Whether the unit can be assigned to a(nother) goal.
        supplies: Whether the unit is currently planning to supply power to another unit.
        supplied_by: Whether there is another unit planning to supply power to this unit.
        private_action_plan: The private action plan of the unit for the next steps.

    Returns:
        _description_
    """

    unit_type: str  # "LIGHT" or "HEAVY"
    tc: TimeCoordinate
    unit_cfg: UnitConfig
    action_queue: List[UnitAction] = field(init=False, default_factory=list)
    goal: Optional[UnitGoal] = field(init=False, default=None)
    can_be_assigned: bool = field(init=False, default=True)
    supplies: Optional[Unit] = field(init=False, default=None)
    supplied_by: Optional[Unit] = field(init=False, default=None)
    private_action_plan: UnitActionPlan = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.private_action_plan = UnitActionPlan(self, [], is_set=True)
        self.send_action_queue = None
        self.infeasible_assignments = set()
        self._set_unit_final_variables()
        self._set_unit_state_variables()

    def update_state(self, tc: TimeCoordinate, power: int, cargo: Cargo, action_queue: List[UnitAction]) -> None:
        """Update the current state of the unit.

        Args:
            tc: Time coordinate of unit.
            power: Power of unit.
            cargo: Cargo of unit.
            action_queue: Action queue of unit.
        """
        self.tc = tc
        self.power = power
        self.cargo = cargo

        if self._last_player_private_action_was_carried_out(action_queue):
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
        self.dig_power_cost = self.unit_cfg.DIG_COST
        self.action_queue_cost = self.unit_cfg.ACTION_QUEUE_POWER_COST
        self.battery_capacity = self.unit_cfg.BATTERY_CAPACITY
        self.rubble_removed_per_dig = self.unit_cfg.DIG_RUBBLE_REMOVED
        self.resources_gained_per_dig = self.unit_cfg.DIG_RESOURCE_GAIN
        self.lichen_removed_per_dig = self.unit_cfg.DIG_LICHEN_REMOVED
        self.update_action_queue_power_cost = self.unit_cfg.ACTION_QUEUE_POWER_COST
        self.cargo_space = self.unit_cfg.CARGO_SPACE
        self.nr_digs_empty_to_full_cargo = ceil(self.cargo_space / self.resources_gained_per_dig)
        self.main_cargo_threshold = self._get_main_cargo_threshold()
        self.min_value_per_step = self._get_min_value_per_step()

    def _set_unit_state_variables(self) -> None:
        self.is_scheduled = False
        self.infeasible_assignments.clear()
        self.x = self.tc.x
        self.y = self.tc.y
        self.cargo_space_left = self.unit_cfg.CARGO_SPACE - self.cargo.total
        self.can_update_action_queue = self.power >= self.update_action_queue_power_cost
        self.can_update_action_queue_and_move = self.power >= self.update_action_queue_power_cost + self.move_power_cost
        self.can_be_assigned = not self.goal and self.can_update_action_queue
        self.primitive_actions_in_queue = get_primitive_actions_from_list(self.action_queue)
        self.main_cargo = self._get_main_cargo()

    def _get_main_cargo_threshold(self) -> int:
        return CONFIG.MAIN_CARGO_THRESHOLD_LIGHT if self.is_light else CONFIG.MAIN_CARGO_THRESHOLD_HEAVY

    def _get_min_value_per_step(self) -> float:
        return CONFIG.MIN_VALUE_PER_STEP_LIGHT if self.is_light else CONFIG.MIN_VALUE_PER_STEP_HEAVY

    def _get_main_cargo(self) -> Optional[Resource]:
        if not self.cargo.main_resource:
            return None

        quantity = self.get_quantity_resource(self.cargo.main_resource)

        if quantity < self.main_cargo_threshold:
            return None

        return self.cargo.main_resource

    def has_too_little_power_for_first_action_in_queue(self, game_state: GameState) -> bool:
        if not self.primitive_actions_in_queue:
            return False

        first_action = self.primitive_actions_in_queue[0]
        power_after_action = self.power + first_action.get_power_change(self.unit_cfg, self.tc, game_state.board)
        return power_after_action < 0

    def _last_player_private_action_was_carried_out(self, action_queue: list[UnitAction]) -> bool:
        if self.private_action_plan.is_first_action_move_center():
            return True

        last_action_queue = self.send_action_queue if self.send_action_queue else self.action_queue

        if not last_action_queue:
            return True

        return last_action_queue != action_queue

    @property
    def are_first_action_of_queue_and_private_action_plan_same(self) -> bool:
        """Whether the first action in the action queue and the private action plan are the same. If so that means that
        for the current turn no power needs to be spent to update the action queue.
        """
        if not self.action_queue or not self.private_action_plan:
            return False

        first_action_of_queue = self.action_queue[0]
        first_action_of_plan = self.private_action_plan.primitive_actions[0]
        return first_action_of_queue.next_step_equal(first_action_of_plan)

    def next_step_queue_dangerous(self, game_state: GameState) -> bool:
        """Whether the units next step, if following the current action queue, is a tile where it might perish.

        Args:
            game_state: Current game state.

        Returns:
            Whether next step if following the current action queue might lead to the unit perishing.
        """
        return (self.is_light and self.next_step_queue_walks_into_opponent_heavy(game_state)) or (
            self._next_step_next_to_dangerous_opponent_unit(game_state)
        )

    def _next_step_next_to_dangerous_opponent_unit(self, game_state: GameState) -> bool:
        if not self.private_action_plan:
            return False

        next_action = self.private_action_plan.primitive_actions[0]
        next_c = self.tc + next_action.unit_direction
        if game_state.is_player_factory_tile(next_c):
            return False

        return self._is_next_c_next_to_dangerous_opponent(c=next_c, game_state=game_state)

    def _is_next_c_next_to_dangerous_opponent(self, c: Coordinate, game_state: GameState) -> bool:
        if game_state.is_player_factory_tile(c):
            return False

        strongest_neighboring_opponent = self._get_strongest_neighboring_opponent(c, game_state)
        if not strongest_neighboring_opponent:
            return False

        if self.tc.xy == c.xy:
            return strongest_neighboring_opponent.can_capture_opponent_stationary(self)
        else:
            return strongest_neighboring_opponent.can_capture_opponent_moving(self)

    def _get_strongest_neighboring_opponent(self, c: Coordinate, game_state: GameState) -> Optional[Unit]:
        neighboring_opponents = game_state.get_neighboring_opponents(c)
        if not neighboring_opponents:
            return None

        strongest_neighboring_opponent = max(neighboring_opponents, key=lambda x: x.is_heavy * 10_000 + x.power)
        return strongest_neighboring_opponent

    def can_capture_opponent_stationary(self, other: Unit) -> bool:
        return not other.is_stronger_type_than(self)

    def can_capture_opponent_moving(self, other: Unit) -> bool:
        return self.is_stronger_type_than(other) or (
            not other.is_stronger_type_than(self) and self.power >= other.power
        )

    def next_step_queue_walks_into_opponent_heavy(self, game_state: GameState) -> bool:
        if not self.action_queue:
            return False

        next_action = self.action_queue[0]
        next_c = self.tc + next_action.unit_direction
        return game_state.is_opponent_heavy_on_tile(next_c)

    def _get_flee_goal(self) -> FleeGoal:
        flee_goal = FleeGoal(unit=self)
        return flee_goal

    def _get_relevant_transfer_goals(self, game_state: GameState) -> List[UnitGoal]:
        goals = []
        if self.cargo.ice:
            ice_goal = self._get_transfer_ice_goal()
            goals.append(ice_goal)
        if self.cargo.ore:
            ore_goal = self._get_transfer_ore_goal()
            goals.append(ore_goal)

        return self._filter_out_invalid_goals(goals, game_state)

    def generate_transfer_ice_goal(self, schedule_info: ScheduleInfo, factory: Factory) -> UnitGoal:
        goal = TransferIceGoal(self, factory)
        if not self._is_valid_goal(goal, schedule_info.game_state):
            raise ActorFoundNoValidGoalError(self, [goal])

        return self.get_best_goal([goal], schedule_info)

    def generate_transfer_ore_goal(self, schedule_info: ScheduleInfo, factory: Factory) -> UnitGoal:
        goal = TransferOreGoal(self, factory)
        if not self._is_valid_goal(goal, schedule_info.game_state):
            raise ActorFoundNoValidGoalError(self, [goal])

        return self.get_best_goal([goal], schedule_info)

    def _get_transfer_ice_goal(self, factory: Optional[Factory] = None) -> UnitGoal:
        goal = TransferIceGoal(self, factory)
        return goal

    def _get_transfer_ore_goal(self, factory: Optional[Factory] = None) -> UnitGoal:
        goal = TransferOreGoal(self, factory)
        return goal

    def get_best_version_goal(self, goal: UnitGoal, schedule_info: ScheduleInfo) -> UnitGoal:
        if hasattr(goal, "pickup_power"):
            goals = [copy(goal), copy(goal)]
            for goal, pickup_power in zip(goals, [True, False]):
                goal.pickup_power = pickup_power  # type: ignore
        else:
            goals = [goal]

        return self.get_best_goal(goals, schedule_info)

    def get_best_goal(self, goals: Iterable[UnitGoal], schedule_info: ScheduleInfo) -> UnitGoal:
        """Get the goal that has the highest value per step.

        Args:
            goals: Goals to select from.
            schedule_info: Schedule info.

        Raises:
            NoValidGoalFoundError: None of the goals leads to a valid action plan.

        Returns:
            Best goal
        """
        game_state = schedule_info.game_state
        constraints = schedule_info.constraints

        goals = list(goals)
        priority_queue = self._init_priority_queue(goals, game_state)
        constraints_with_danger = self._get_constraints_with_danger_tcs(constraints, game_state)
        schedule_info = replace(schedule_info, constraints=constraints_with_danger)

        while not priority_queue.is_empty():
            goal: UnitGoal = priority_queue.pop()

            try:
                action_plan = goal.generate_action_plan(schedule_info)
            except Exception as e:
                logger.debug(str(e))
                self._if_non_dummy_goal_add_to_infeasible_assignments(goal)
                continue

            value = goal.get_value_per_step_of_action_plan(action_plan, game_state)
            if value <= self.min_value_per_step and not goal.is_dummy_goal:
                e = InvalidGoalError(goal, message="Negative value non-dummy goal")
                logger.debug(str(e))
                self._if_non_dummy_goal_add_to_infeasible_assignments(goal)
                continue

            priority = -1 * value
            priority_queue.put(goal, priority)

            if goal == priority_queue[0]:
                return goal

        raise ActorFoundNoValidGoalError(self, goals)

    def _if_non_dummy_goal_add_to_infeasible_assignments(self, goal: UnitGoal) -> None:
        if not goal.is_dummy_goal:
            self.infeasible_assignments.add(goal.key)

    def _get_constraints_with_danger_tcs(self, constraints: Constraints, game_state: GameState) -> Constraints:
        constraints_with_danger = copy(constraints)
        stationary_danger_tcs = self.get_stationary_danger_tcs(game_state)
        constraints_with_danger.add_stationary_danger_coordinates(stationary_danger_tcs)
        moving_danger_tcs = self.get_moving_danger_tcs(game_state)
        constraints_with_danger.add_moving_danger_coordinates(moving_danger_tcs)
        return constraints_with_danger

    def _init_priority_queue(self, goals: list[UnitGoal], game_state: GameState) -> PriorityQueue:
        goals_priority_queue = PriorityQueue()

        for goal in goals:
            if not self.is_feasible_assignment(goal):
                continue

            best_value = goal.get_best_case_value_per_step(game_state)
            if best_value < 0 and not goal.is_dummy_goal:
                continue

            priority = -1 * best_value
            goals_priority_queue.put(goal, priority)

        return goals_priority_queue

    def get_clear_rubble_goals(self, game_state: GameState, c: Coordinate) -> list[ClearRubbleGoal]:
        goals = [ClearRubbleGoal(unit=self, pickup_power=pickup_power, dig_c=c) for pickup_power in [False, True]]

        return self._filter_out_invalid_goals(goals, game_state)  # type: ignore

    def generate_transfer_or_dummy_goal(self, schedule_info: ScheduleInfo) -> UnitGoal:
        transfer_goals = self._get_relevant_transfer_goals(schedule_info.game_state)
        dummy_goals = self._get_dummy_goals(schedule_info.game_state)
        goals = transfer_goals + dummy_goals
        goal = self.get_best_goal(goals, schedule_info)
        return goal

    def _get_dummy_goals(self, game_state: GameState) -> list[UnitGoal]:
        dummy_goals = [UnitNoGoal(self), EvadeConstraintsGoal(self)]
        if self.is_under_threath(game_state):
            flee_goal = self._get_flee_goal()
            dummy_goals.append(flee_goal)

        return self._filter_out_invalid_goals(dummy_goals, game_state)

    def generate_no_goal_goal(self, schedule_info: ScheduleInfo) -> UnitNoGoal:
        no_goal_goal = UnitNoGoal(self)
        if not self._is_valid_goal(no_goal_goal, schedule_info.game_state):
            raise ActorFoundNoValidGoalError(self, [no_goal_goal])

        goal = self.get_best_goal([no_goal_goal], schedule_info)
        return goal  # type: ignore

    def generate_collect_ore_goal(
        self,
        schedule_info: ScheduleInfo,
        c: Coordinate,
        is_supplied: bool,
        factory: Factory,
        quantity: Optional[int] = None,
    ) -> CollectOreGoal:
        ore_goals = self.get_collect_ore_goals(c, schedule_info.game_state, factory, is_supplied, quantity)
        goal = self.get_best_goal(ore_goals, schedule_info)
        return goal  # type: ignore

    def generate_supply_power_goal(
        self,
        schedule_info: ScheduleInfo,
        receiving_unit: Unit,
        receiving_action_plan: UnitActionPlan,
        receiving_c: Coordinate,
    ) -> SupplyPowerGoal:
        supply_c = schedule_info.game_state.get_closest_player_factory_c(c=receiving_c)
        supply_goals = self.get_supply_power_goals(
            schedule_info.game_state, receiving_unit, receiving_action_plan, receiving_c=receiving_c, supply_c=supply_c
        )
        goal = self.get_best_goal(supply_goals, schedule_info)
        return goal  # type: ignore

    def get_supply_power_goals(
        self,
        game_state: GameState,
        receiving_unit: Unit,
        receiving_action_plan: UnitActionPlan,
        receiving_c: Coordinate,
        supply_c: Coordinate,
    ) -> List[SupplyPowerGoal]:
        goals = [
            SupplyPowerGoal(
                self,
                receiving_unit=receiving_unit,
                receiving_action_plan=receiving_action_plan,
                receiving_c=receiving_c,
                supply_c=supply_c,
                pickup_power=pickup_power,
            )
            for pickup_power in [True, False]
        ]

        return self._filter_out_invalid_goals(goals, game_state)  # type: ignore

    def get_collect_ore_goals(
        self, c: Coordinate, game_state: GameState, factory: Factory, is_supplied: bool, quantity: Optional[int] = None
    ) -> list[CollectOreGoal]:
        ore_goals = [
            CollectOreGoal(
                unit=self,
                pickup_power=pickup_power,
                dig_c=c,
                factory=factory,
                is_supplied=is_supplied,
                quantity=quantity,
            )
            for pickup_power in [False, True]
        ]

        return self._filter_out_invalid_goals(ore_goals, game_state)  # type: ignore

    def generate_collect_ice_goal(
        self,
        schedule_info: ScheduleInfo,
        c: Coordinate,
        is_supplied: bool,
        factory: Factory,
        quantity: Optional[int] = None,
    ) -> CollectIceGoal:
        ice_goals = self.get_collect_ice_goals(c, schedule_info.game_state, factory, is_supplied, quantity)
        goal = self.get_best_goal(ice_goals, schedule_info)
        return goal  # type: ignore

    def get_collect_ice_goals(
        self, c: Coordinate, game_state: GameState, factory: Factory, is_supplied: bool, quantity: Optional[int] = None
    ) -> list[CollectIceGoal]:
        ice_goals = [
            CollectIceGoal(
                unit=self,
                pickup_power=pickup_power,
                dig_c=c,
                factory=factory,
                is_supplied=is_supplied,
                quantity=quantity,
            )
            for pickup_power in [False, True]
        ]

        return self._filter_out_invalid_goals(ice_goals, game_state)  # type: ignore

    def get_ice_goals(
        self, c: Coordinate, game_state: GameState, factory: Factory, is_supplied: bool, quantity: Optional[int] = None
    ) -> list[UnitGoal]:
        if (
            self.is_heavy
            and game_state.get_min_distance_to_any_player_factory(c) == 1
            and game_state.get_dis_to_closest_opp_heavy(c) <= 1
            and game_state.c_is_undefended(c)
        ):
            heavy_opp = next(u for u in game_state.get_neighboring_opponents(c) if u.is_heavy)
            goals = [DefendTileGoal(self, c, heavy_opp, factory, pickup_power) for pickup_power in [False, True]]

        else:
            goals = [
                CollectIceGoal(
                    unit=self,
                    pickup_power=pickup_power,
                    dig_c=c,
                    factory=factory,
                    is_supplied=is_supplied,
                    quantity=quantity,
                )
                for pickup_power in [False, True]
            ]

        return self._filter_out_invalid_goals(goals, game_state)  # type: ignore

    def get_destroy_lichen_goals(self, c: Coordinate, game_state: GameState) -> List[DestroyLichenGoal]:
        goals = [DestroyLichenGoal(self, pickup_power, c) for pickup_power in [False, True]]

        return self._filter_out_invalid_goals(goals, game_state)  # type: ignore

    def get_defend_lichen_goals(self, game_state: GameState, opp: Unit) -> List[DefendLichenTileGoal]:
        goals = [DefendLichenTileGoal(self, opp.tc, opp, pickup_power) for pickup_power in [False, True]]
        return self._filter_out_invalid_goals(goals, game_state)  # type: ignore

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

    def get_quantity_resource(self, resource: Resource) -> int:
        return self.cargo.get_resource(resource)

    def get_nr_digs_to_fill_cargo(self) -> int:
        return ceil(self.cargo_space_left / self.resources_gained_per_dig)

    def get_moving_danger_tcs(self, game_state: GameState) -> dict[TimeCoordinate, float]:
        return {
            neighbor_c: 10_000
            for neighbor_c in self.tc.non_stationary_neighbors
            if self._is_next_c_next_to_dangerous_opponent(neighbor_c, game_state)
        }

    def get_stationary_danger_tcs(self, game_state: GameState) -> dict[TimeCoordinate, float]:
        danger_tcs = {}
        stationary_tc = self.tc + Direction.CENTER
        if self._is_next_c_next_to_dangerous_opponent(stationary_tc, game_state):
            danger_tcs[stationary_tc] = 10_000
        return danger_tcs

    def is_stronger_type_than(self, other: Unit) -> bool:
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

    def set_send_action_queue(self, action_plan: UnitActionPlan) -> None:
        self.send_action_queue = action_plan.actions

    def set_send_no_action_queue(self) -> None:
        self.send_action_queue = None

    def schedule_goal(self, goal: UnitGoal) -> None:
        """Schedule a unit on a goal and set the corresponding private action plan.

        Args:
            goal: Goal to set unit on.
        """
        self.goal = goal
        self.private_action_plan = goal.action_plan
        self.is_scheduled = True
        self.can_be_assigned = False

    def remove_goal_and_private_action_plan(self) -> None:
        """Remove the goal of the unit and its private action plan. Also handles unsetting potential supplied or
        supplying units state to not be supplying/supplied by this unit.
        """
        if self.supplied_by:
            self.supplied_by.set_not_supplying()

        if self.supplies:
            self.supplies.set_unsupplied()

        if isinstance(self.goal, SupplyPowerGoal):
            self.goal.receiving_unit.supplied_by = None

        self.goal = None
        self.is_scheduled = False
        self.private_action_plan = UnitActionPlan(self, [])
        self.supplies = None
        self.supplied_by = None
        self.can_be_assigned = True

        if not self.action_queue:
            self.private_action_plan.is_set = True

    def set_not_supplying(self) -> None:
        self.supplies = None

    def set_unsupplied(self) -> None:
        self.supplied_by = None

    def get_nr_digs_to_quantity_resource(self, resource: Resource, q: int) -> int:
        """Get the number of digs required to get to at least q amount of the given resource.

        Args:
            resource: Resource to dig.
            q: Quantity of resource to dig.

        Returns:
            Number of digs required.
        """
        resource_in_cargo = self.get_quantity_resource(resource)
        if resource_in_cargo >= q:
            return 0

        quantity_to_mine = q - resource_in_cargo
        nr_digs_required = ceil(quantity_to_mine / self.resources_gained_per_dig)
        return nr_digs_required

    def is_feasible_assignment(self, goal: UnitGoal) -> bool:
        """Whether the assignment can potentially have a feasible action plan.

        Args:
            goal: Goal to potentially assign unit on.

        Returns:
            Whether goal is a feasible assignment.
        """
        return goal.key not in self.infeasible_assignments

    def can_not_move_this_step(self, game_state: GameState) -> bool:
        """Whether the unit can potentially move this step.

        Args:
            game_state: Current game state.

        Returns:
            Whether the unit can potentially move this step.
        """
        if self.can_update_action_queue_and_move:
            return False

        if not self.primitive_actions_in_queue:
            return False

        if self.primitive_actions_in_queue[0].is_stationary:
            return False

        return UnitActionPlan(self, self.primitive_actions_in_queue[:1]).unit_has_enough_power(game_state)

    def _filter_out_invalid_goals(self, goals: Iterable[UnitGoal], game_state: GameState) -> List[UnitGoal]:
        return [goal for goal in goals if self._is_valid_goal(goal, game_state)]

    def _is_valid_goal(self, goal: UnitGoal, game_state: GameState) -> bool:
        if isinstance(self.goal, DefendTileGoal) and not isinstance(goal, DefendTileGoal):
            return False

        if (
            isinstance(self.goal, DefendTileGoal)
            and self.goal.tile_c.distance_to(self.tc) > CONFIG.MAX_DISTANCE_COLLECTING
        ):
            return False

        if not self._is_valid_goal_given_cargo(goal):
            return False

        if isinstance(goal, CollectGoal):
            return self._is_valid_collect_goal(goal, game_state)

        if isinstance(goal, DestroyLichenGoal):
            return self._is_valid_destroy_lichen_goal(goal, game_state)

        return True

    def _is_valid_collect_goal(self, goal: CollectGoal, game_state: GameState) -> bool:
        if self.is_light and game_state.get_dis_to_closest_opp_heavy(goal.dig_c) <= 1:
            return False

        dig_c = goal.dig_c
        distance_to_dig_c = goal.factory.min_distance_to_c(dig_c) if goal.factory else self.tc.distance_to(dig_c)
        return distance_to_dig_c <= CONFIG.MAX_DISTANCE_COLLECTING

    def _is_valid_destroy_lichen_goal(self, goal: DestroyLichenGoal, game_state: GameState) -> bool:
        return (
            self._is_feasible_dig_c(goal.dig_c, game_state)
            and self.tc.distance_to(goal.dig_c) < CONFIG.MAX_DISTANCE_DESTROY_LICHEN
            and game_state.get_dis_to_closest_opp_heavy(goal.dig_c) > 1
        )

    def _is_feasible_dig_c(self, c: Coordinate, game_state: GameState) -> bool:
        return not (self.is_light and game_state.get_dis_to_closest_opp_heavy(c) <= 1)

    def _is_valid_goal_given_cargo(self, goal: UnitGoal) -> bool:
        if not self.main_cargo or goal.is_dummy_goal:
            return True

        return (isinstance(goal, CollectGoal) or isinstance(goal, TransferGoal)) and goal.resource == self.main_cargo
