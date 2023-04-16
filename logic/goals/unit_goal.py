from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from math import ceil, inf
from copy import copy
from functools import lru_cache
import numpy as np

from search.search import (
    Search,
    EvadeConstraintsGraph,
    MoveToGraph,
    DigAtGraph,
    PickupPowerGraph,
    Graph,
    TransferToFactoryResourceGraph,
    TransferPowerToUnitResourceGraph,
)
from objects.actions.unit_action import DigAction, MoveAction, TransferAction
from objects.actions.unit_action_plan import UnitActionPlan
from objects.direction import Direction
from objects.resource import Resource
from objects.coordinate import (
    ResourcePowerTimeCoordinate,
    DigCoordinate,
    DigTimeCoordinate,
    TimeCoordinate,
    Coordinate,
    ResourceTimeCoordinate,
)
from logic.constraints import Constraints
from logic.goals.goal import Goal
from config import CONFIG
from exceptions import InvalidGoalError


if TYPE_CHECKING:
    from objects.actors.unit import Unit
    from objects.actors.factory import Factory
    from objects.game_state import GameState
    from objects.board import Board
    from lux.config import UnitConfig
    from objects.actions.unit_action import UnitAction
    from logic.goal_resolution.power_availabilty_tracker import PowerTracker


@dataclass
class UnitGoal(Goal):
    unit: Unit
    _value: Optional[float] = field(init=False, default=None)

    @abstractmethod
    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        ...

    def generate_action_plan(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitActionPlan:
        self._generate_action_plan(game_state, constraints, power_tracker)

        if not self.action_plan.is_valid_size:
            raise InvalidGoalError

        return self.action_plan

    @abstractmethod
    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:
        ...

    def get_best_value_per_step(self, game_state: GameState) -> float:
        benefit = self._get_max_benefit(game_state)
        cost, min_nr_steps = self._get_min_cost_and_steps(game_state)

        if min_nr_steps == 0:
            return -inf

        value = benefit - cost
        value_per_step = value / min_nr_steps

        return value_per_step

    @abstractmethod
    def _get_max_benefit(self, game_state: GameState) -> float:
        ...

    @abstractmethod
    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        ...

    def _get_move_to_graph(self, board: Board, goal: Coordinate, constraints: Constraints) -> MoveToGraph:
        graph = MoveToGraph(
            unit_type=self.unit.unit_type,
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            goal=goal,
            constraints=constraints,
        )

        return graph

    def _get_pickup_power_graph(
        self,
        board: Board,
        power_tracker: PowerTracker,
        constraints: Constraints,
        next_goal_c: Optional[Coordinate] = None,
    ) -> PickupPowerGraph:
        graph = PickupPowerGraph(
            unit_type=self.unit.unit_type,
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            constraints=constraints,
            next_goal_c=next_goal_c,
            power_tracker=power_tracker,
        )

        return graph

    def _get_dig_graph(self, board: Board, goal: DigCoordinate, constraints: Constraints) -> DigAtGraph:
        graph = DigAtGraph(
            unit_type=self.unit.unit_type,
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            constraints=constraints,
            goal=goal,
        )

        return graph

    @staticmethod
    def _search_graph(graph: Graph, start: TimeCoordinate) -> list[UnitAction]:
        search = Search(graph=graph)
        optimal_actions = search.get_actions_to_complete_goal(start=start)
        return optimal_actions

    def _add_power_pickup_actions(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
        next_goal_c: Optional[Coordinate] = None,
    ) -> None:
        actions = self._get_power_pickup_actions(game_state, constraints, power_tracker, next_goal_c)
        self.action_plan.extend(actions)

    def _get_power_pickup_actions(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
        next_goal_c: Optional[Coordinate],
    ) -> list[UnitAction]:
        p = self.action_plan.get_final_ptc(game_state).p
        if p == self.unit.battery_capacity:
            return []

        graph = self._get_pickup_power_graph(
            board=game_state.board,
            power_tracker=power_tracker,
            constraints=constraints,
            next_goal_c=next_goal_c,
        )

        recharge_tc = ResourcePowerTimeCoordinate(
            *self.action_plan.final_tc.xyt,
            p=p,
            unit_cfg=self.unit.unit_cfg,
            game_state=game_state,
            q=0,
            resource=Resource.POWER,
        )
        actions = self._search_graph(graph=graph, start=recharge_tc)
        return actions

    def _get_move_to_plan(
        self,
        start_tc: TimeCoordinate,
        goal: Coordinate,
        constraints: Constraints,
        board: Board,
    ) -> list[UnitAction]:

        graph = self._get_move_to_graph(board=board, goal=goal, constraints=constraints)
        actions = self._search_graph(graph=graph, start=start_tc)
        return actions

    def _init_action_plan(self) -> None:
        self.action_plan = UnitActionPlan(actor=self.unit)
        # TODO remove power due to updating action plan?

    def get_cost_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_used(board=game_state.board)
        total_cost = number_of_steps * self.unit.time_to_power_cost + power_cost
        return total_cost

    def _get_valid_actions(self, actions: list[UnitAction], game_state: GameState) -> list[UnitAction]:
        potential_action_plan = self.action_plan + actions
        nr_valid_primitive_actions = potential_action_plan.get_nr_valid_primitive_actions(game_state)
        nr_original_primitive_actions = len(self.action_plan.primitive_actions)
        return potential_action_plan.primitive_actions[nr_original_primitive_actions:nr_valid_primitive_actions]

    @abstractmethod
    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        ...

    @abstractmethod
    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        ...


@dataclass
class DigGoal(UnitGoal):
    pickup_power: bool
    dig_c: Coordinate

    @abstractmethod
    def _get_benefit_n_digs(self, n_digs: int, game_state: GameState) -> float:
        ...

    def _get_nr_digs_to_clear_rubble(self, board: Board) -> int:
        rubble_at_pos = board.rubble[self.dig_c.xy]
        nr_digs = ceil(rubble_at_pos / self.unit.rubble_removed_per_dig)
        return nr_digs

    def _get_dig_plan(
        self, start_tc: TimeCoordinate, dig_c: Coordinate, nr_digs: int, constraints: Constraints, board: Board
    ) -> list[UnitAction]:
        actions = []

        for _ in range(nr_digs):
            start_dtc = DigTimeCoordinate(*start_tc.xyt, d=0)
            dig_coordinate = DigCoordinate(x=dig_c.x, y=dig_c.y, d=1)
            graph = self._get_dig_graph(board=board, goal=dig_coordinate, constraints=constraints)

            try:
                new_actions = self._search_graph(graph=graph, start=start_dtc)
            except Exception:
                if actions:
                    return actions

                raise

            actions.extend(new_actions)

            for action in new_actions:
                start_tc = start_tc.add_action(action)

        return actions

    def find_max_dig_actions_can_still_reach_factory(
        self, actions: Sequence[UnitAction], game_state: GameState, constraints: Constraints
    ) -> list[UnitAction]:
        # TODO, see if when we first start with the upper limit, if that were to improve the speed

        low = 0
        high = self._get_nr_digs_in_actions(actions)

        closest_factory_c = game_state.get_closest_player_factory_c(c=self.dig_c)
        actions_back = get_actions_a_to_b(
            self.dig_c,
            closest_factory_c,
            game_state,
            self.unit.unit_type,
            self.unit.time_to_power_cost,
            self.unit.unit_cfg,
        )

        while low < high:
            mid = (high + low) // 2
            if mid == low:
                mid += 1

            potential_actions = self._get_actions_up_to_n_digs(actions, mid)
            potential_action_plan = self.action_plan + potential_actions + actions_back

            if potential_action_plan.unit_has_enough_power(game_state):
                low = mid
            else:
                high = mid - 1

        actions = self._get_actions_up_to_n_digs(actions, low)
        return actions

    def _get_nr_digs_in_actions(self, actions: Sequence[UnitAction]) -> int:
        return sum(dig_action.n for dig_action in actions if isinstance(dig_action, DigAction))

    def _get_actions_up_to_n_digs(self, actions: Sequence[UnitAction], n: int) -> list[UnitAction]:
        if n == 0:
            return []

        return_actions = []
        nr_added_actions = 0
        for action in actions:
            return_actions.append(action)

            if isinstance(action, DigAction):
                nr_added_actions += 1
                if nr_added_actions == n:
                    return return_actions

        raise ValueError(f"Only found {nr_added_actions}, of the required {n} actions")

    def _get_max_nr_digs_current_ptc(self, game_state: GameState) -> int:
        power_available = self.action_plan.get_final_ptc(game_state).p
        return self._get_max_nr_digs(power_available=power_available, game_state=game_state)

    def _get_max_nr_digs_possible(self, power_available: int) -> int:
        dig_power_cost = self.unit.dig_power_cost
        recharge_power = self.unit.recharge_power
        min_power_change_per_dig = dig_power_cost - recharge_power

        quotient, remainder = divmod(power_available, min_power_change_per_dig)
        if remainder >= recharge_power:
            max_nr_digs = quotient
        else:
            max_nr_digs = max(0, quotient - 1)

        return max_nr_digs

    def _get_max_benefit(self, game_state: GameState) -> float:
        max_nr_digs = self._get_best_max_nr_digs(game_state)
        max_benefit = self._get_benefit_n_digs(max_nr_digs, game_state)
        return max_benefit

    def _get_best_max_nr_digs(self, game_state: GameState) -> int:
        power_available = self.unit.battery_capacity if self.pickup_power else self.unit.power
        return self._get_max_nr_digs(power_available, game_state)

    def _get_max_nr_digs(self, power_available: int, game_state: GameState) -> int:
        max_nr_digs = self._get_max_nr_digs_possible(power_available)
        max_useful_digs = self._get_max_useful_digs(game_state)
        return min(max_nr_digs, max_useful_digs)

    @abstractmethod
    def _get_max_useful_digs(self, game_state: GameState) -> int:
        ...

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        return self._get_benefit_n_digs(action_plan.nr_digs, game_state)

    def _get_min_cost_and_steps_go_to_dig_c(self, game_state: GameState) -> tuple[float, int]:
        nr_steps_moving, nr_steps_power_pickup = self._get_min_nr_steps_moving_and_power_pickup(game_state)
        min_nr_steps = nr_steps_moving + nr_steps_power_pickup

        min_cost_moving = nr_steps_moving * self.unit.move_time_and_power_cost
        min_cost_power_pickup = nr_steps_power_pickup * self.unit.time_to_power_cost
        min_cost = min_cost_moving + min_cost_power_pickup

        return min_cost, min_nr_steps

    def _get_min_nr_steps_to_dig_c(self, game_state: GameState) -> int:
        nr_steps_moving, nr_steps_power_pickup = self._get_min_nr_steps_moving_and_power_pickup(game_state)
        return nr_steps_moving + nr_steps_power_pickup

    def _get_min_nr_steps_moving_and_power_pickup(self, game_state: GameState) -> tuple[int, int]:
        if self.pickup_power:
            return self._get_min_nr_steps_moving_and_power_pickup_with_pickup_power(game_state)
        else:
            return self._get_min_nr_steps_moving_and_power_pickup_without_pickup_power(game_state)

    def _get_min_nr_steps_moving_and_power_pickup_without_pickup_power(self, game_state: GameState) -> tuple[int, int]:
        nr_steps_moving = self.unit.tc.distance_to(self.dig_c)
        nr_steps_power_pickup = 0
        return nr_steps_moving, nr_steps_power_pickup

    def _get_min_nr_steps_moving_and_power_pickup_with_pickup_power(self, game_state: GameState) -> tuple[int, int]:
        nr_steps_moving = self._get_min_steps_moving_to_dig_c_with_pickup_power(game_state)
        nr_steps_power_pickup = 1
        return nr_steps_moving, nr_steps_power_pickup

    def _get_min_steps_moving_to_dig_c_with_pickup_power(self, game_state: GameState) -> int:
        closest_factory_tile = game_state.board.get_closest_player_factory_tile(self.unit.tc)
        nr_steps_to_factory_tile = self.unit.tc.distance_to(closest_factory_tile)
        nr_steps_to_resource_c = closest_factory_tile.distance_to(self.dig_c)
        nr_steps_moving = nr_steps_to_factory_tile + nr_steps_to_resource_c
        return nr_steps_moving

    def _get_min_cost_and_steps_max_nr_digs(self, game_state: GameState) -> tuple[float, int]:
        max_nr_digs = self._get_best_max_nr_digs(game_state)
        min_cost_digging = max_nr_digs * self.unit.dig_time_and_power_cost

        return min_cost_digging, max_nr_digs


@dataclass
class CollectGoal(DigGoal):
    is_supplied: bool
    factory: Optional[Factory] = field(default=None)
    # quantity: Optional[int] = None
    resource: Resource = field(init=False)

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        if action_plan:
            return False

        return True

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:

        self._init_action_plan()
        if self.pickup_power:
            self._add_power_pickup_actions(
                game_state=game_state,
                constraints=constraints,
                next_goal_c=self.dig_c,
                power_tracker=power_tracker,
            )

            if not self.action_plan.unit_has_enough_power(game_state):
                raise InvalidGoalError

        self._add_dig_actions(game_state=game_state, constraints=constraints)
        self._add_transfer_resources_to_factory_actions(game_state=game_state, constraints=constraints)
        return self.action_plan

    def _add_dig_actions(self, game_state: GameState, constraints: Constraints) -> None:
        max_nr_digs = self._get_max_nr_digs_current_ptc(game_state)

        actions_max_nr_digs = self._get_dig_plan(
            start_tc=self.action_plan.final_tc,
            dig_c=self.dig_c,
            nr_digs=max_nr_digs,
            constraints=constraints,
            board=game_state.board,
        )

        if self.is_supplied or self._is_heavy_startup(game_state):
            self.action_plan.extend(actions_max_nr_digs)
            return

        max_valid_digs_actions = self._get_valid_actions(actions_max_nr_digs, game_state)
        max_valid_digs_actions = self.find_max_dig_actions_can_still_reach_factory(
            max_valid_digs_actions, game_state, constraints
        )

        if len(max_valid_digs_actions) == 0:
            raise InvalidGoalError

        self.action_plan.extend(max_valid_digs_actions)

    def _add_transfer_resources_to_factory_actions(self, game_state: GameState, constraints: Constraints) -> None:
        actions = self._get_transfer_resources_to_factory_actions(board=game_state.board, constraints=constraints)
        self.action_plan.extend(actions=actions)
        if not (self.is_supplied or self._is_heavy_startup(game_state)) and not self.action_plan.unit_has_enough_power(
            game_state
        ):
            raise InvalidGoalError

    def _is_heavy_startup(self, game_state: GameState) -> bool:
        return self.unit.is_heavy and game_state.real_env_steps in [1, 2]

    def _get_transfer_resources_to_factory_actions(self, board: Board, constraints: Constraints) -> list[UnitAction]:
        return self._get_transfer_plan(start_tc=self.action_plan.final_tc, constraints=constraints, board=board)

    def _get_transfer_plan(
        self,
        start_tc: TimeCoordinate,
        constraints: Constraints,
        board: Board,
    ) -> list[UnitAction]:
        start = ResourceTimeCoordinate(start_tc.x, start_tc.y, start_tc.t, q=0, resource=self.resource)
        graph = self._get_transfer_graph(board=board, constraints=constraints)
        actions = self._search_graph(graph=graph, start=start)
        return actions

    def _get_transfer_graph(self, board: Board, constraints: Constraints) -> TransferToFactoryResourceGraph:
        graph = TransferToFactoryResourceGraph(
            unit_type=self.unit.unit_type,
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            constraints=constraints,
            resource=self.resource,
            q=self.unit.unit_cfg.CARGO_SPACE,
            factory=self.factory,
        )

        return graph

    def _get_max_nr_digs(self, power_available: int, game_state: GameState) -> int:
        if self._is_heavy_startup(game_state):
            return CONFIG.TURN_1_NR_DIGS_HEAVY

        if self.is_supplied:
            return self._get_max_useful_digs(game_state)

        return super()._get_max_nr_digs(power_available, game_state)

    def _get_max_useful_digs(self, game_state: GameState) -> int:
        return self._get_total_nr_digs_to_fill_cargo(game_state)

    def _get_total_nr_digs_to_fill_cargo(self, game_state: GameState) -> int:
        nr_digs_to_clear_rubble = self._get_nr_digs_to_clear_rubble(game_state.board)
        nr_digs_to_fill_cargo = self.unit.get_nr_digs_to_fill_cargo()
        total_nr_digs_to_fill_cargo = nr_digs_to_clear_rubble + nr_digs_to_fill_cargo
        return total_nr_digs_to_fill_cargo

    def _get_resources_collected_by_n_digs(self, n_digs, game_state: GameState) -> int:
        nr_digs_required_to_clear_rubble = self._get_nr_digs_to_clear_rubble(game_state.board)
        nr_digs_for_collecting_resources = max(n_digs - nr_digs_required_to_clear_rubble, 0)
        return nr_digs_for_collecting_resources * self.unit.resources_gained_per_dig

    def _get_benefit_n_digs(self, n_digs: int, game_state: GameState) -> float:
        nr_resources_digged = n_digs * self.unit.resources_gained_per_dig
        nr_resources_unit = self.unit.get_quantity_resource_in_cargo(self.resource)
        nr_resources_to_return = nr_resources_digged + nr_resources_unit

        benefit_resource = self.get_benefit_resource(game_state)

        return benefit_resource * nr_resources_to_return

    @abstractmethod
    def get_benefit_resource(self, game_state: GameState) -> float:
        ...

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        min_cost_digging, max_nr_digs = self._get_min_cost_and_steps_max_nr_digs(game_state)
        min_cost_go_to_c, min_steps_go_to_c = self._get_min_cost_and_steps_go_to_dig_c(game_state)
        min_cost_transfer, min_steps_transfer = self._get_min_cost_and_steps_transfer_resource(game_state)

        min_cost = min_cost_digging + min_cost_go_to_c + min_cost_transfer
        min_steps = max_nr_digs + min_steps_go_to_c + min_steps_transfer

        return min_cost, min_steps

    def _get_min_cost_and_steps_transfer_resource(self, game_state: GameState) -> tuple[float, int]:
        nr_steps_to_closest_factory = game_state.board.get_min_distance_to_any_player_factory(self.dig_c)
        nr_steps_next_to_closest_factory = max(nr_steps_to_closest_factory - 1, 0)
        nr_steps_transfer = 1
        nr_steps = nr_steps_next_to_closest_factory + nr_steps_transfer

        move_cost = nr_steps_next_to_closest_factory * self.unit.move_time_and_power_cost
        transfer_cost = self.unit.time_to_power_cost
        cost = move_cost + transfer_cost

        return cost, nr_steps


@lru_cache(256)
def get_actions_a_to_b(
    a: Coordinate, b: Coordinate, game_state: GameState, unit_type: str, time_to_power_cost: int, unit_cfg: UnitConfig
) -> list[UnitAction]:
    fake_tc = TimeCoordinate(a.x, a.y, 0)
    graph = MoveToGraph(
        unit_type=unit_type,
        board=game_state.board,
        time_to_power_cost=time_to_power_cost,
        unit_cfg=unit_cfg,
        goal=b,
        constraints=Constraints(),
    )

    return UnitGoal._search_graph(graph=graph, start=fake_tc)


@dataclass
class SupplyPowerGoal(UnitGoal):
    receiving_unit: Unit
    receiving_action_plan: UnitActionPlan
    receiving_c: Coordinate

    def __repr__(self) -> str:
        return f"supply_power_to_{self.receiving_unit}"

    @property
    def key(self) -> str:
        return str(self)

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        receiving_unit = self.receiving_unit
        return receiving_unit.goal.is_completed(game_state, receiving_unit.private_action_plan)  # type: ignore

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:

        self._init_action_plan()
        self._add_move_to_supply_c_actions(game_state=game_state, constraints=constraints)
        self._add_supply_actions(game_state, constraints, power_tracker)

        return self.action_plan

    def _add_move_to_supply_c_actions(self, game_state: GameState, constraints: Constraints) -> None:
        supply_c = game_state.get_closest_player_factory_c(c=self.receiving_c)

        if self.unit.tc.xy == supply_c.xy:
            return

        move_actions = self._get_move_to_plan(
            start_tc=self.unit.tc, goal=supply_c, constraints=constraints, board=game_state.board
        )
        self.action_plan.extend(move_actions)
        if not self.action_plan.unit_has_enough_power(game_state):
            raise InvalidGoalError

    def _add_supply_actions(self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker) -> None:
        receiving_unit_ptcs = self.receiving_action_plan.get_power_time_coordinates(game_state)
        receiving_unit_powers = np.array([ptc.p for ptc in receiving_unit_ptcs])

        power_pickup_turn = True
        while True:
            if power_pickup_turn:
                power_tracker_up_to_date = copy(power_tracker)
                power_requests_up_to_now = self.action_plan.get_power_requests(game_state)
                power_tracker_up_to_date.add_power_requests(power_requests_up_to_now)
                actions = self._get_power_pickup_actions(
                    game_state, constraints, power_tracker_up_to_date, self.receiving_c
                )
            else:
                actions = self._get_transfer_resources_to_unit_actions(game_state, constraints)
                index_transfer = self.action_plan.nr_primitive_actions + len(actions) - 1

                if index_transfer >= len(receiving_unit_ptcs):
                    break

                power_transfer_action: TransferAction = actions[-1]  # type: ignore
                receiving_unit_power_after_transfer = receiving_unit_powers[index_transfer]

                power_transfer_amount = self._get_adjusted_power_transfer_amount(
                    receiving_unit_power=receiving_unit_power_after_transfer,
                    power_transfer=power_transfer_action.amount,
                    game_state=game_state,
                )
                power_transfer_action.amount = power_transfer_amount

                if receiving_unit_ptcs[index_transfer].xy != self.receiving_c.xy:
                    power_transfer_action.amount = 0

                receiving_unit_powers[index_transfer:] += power_transfer_action.amount

            potential_action_plan = self.action_plan + actions
            if not potential_action_plan.is_valid_size:
                break

            self.action_plan.extend(actions)
            power_pickup_turn = not power_pickup_turn

        max_transfer_index = self.action_plan.nr_primitive_actions - 1
        if receiving_unit_powers[:max_transfer_index].min() < 0:
            raise InvalidGoalError

    def _get_transfer_resources_to_unit_actions(
        self, game_state: GameState, constraints: Constraints
    ) -> list[UnitAction]:
        start_tc = self.action_plan.final_tc
        start = ResourceTimeCoordinate(start_tc.x, start_tc.y, start_tc.t, q=0, resource=Resource.POWER)
        graph = self._get_transfer_to_unit_graph(game_state=game_state, constraints=constraints)
        actions = self._search_graph(graph=graph, start=start)
        return actions

    def _get_transfer_to_unit_graph(
        self, game_state: GameState, constraints: Constraints
    ) -> TransferPowerToUnitResourceGraph:
        power_end_action_plan = self.action_plan.get_final_ptc(game_state).p
        graph = TransferPowerToUnitResourceGraph(
            game_state.board,
            self.unit.time_to_power_cost,
            self.unit.unit_cfg,
            self.unit.unit_type,
            constraints,
            Resource.POWER,
            power_end_action_plan,
            self.receiving_c,
        )

        return graph

    def _get_adjusted_power_transfer_amount(
        self, receiving_unit_power: int, power_transfer: int, game_state: GameState
    ) -> int:

        power_transfer = self._remove_surplus_power(receiving_unit_power, power_transfer)
        power_transfer = self._remove_safety_reduction(power_transfer, game_state)
        power_transfer = self._remove_min_power_income_for_other_units_factory(power_transfer, game_state)
        return power_transfer

    def _remove_surplus_power(self, receiving_unit_power, power_transfer: int) -> int:
        surplus_power = max(0, receiving_unit_power - self.receiving_unit.battery_capacity)
        power_transfer_minus_surplus = power_transfer - surplus_power
        return power_transfer_minus_surplus

    def _remove_safety_reduction(self, power_transfer: int, game_state: GameState) -> int:
        supplying_unit_power_left = self.action_plan.get_final_ptc(game_state).p - power_transfer
        safety_level = self.unit.action_queue_cost + self.unit.move_power_cost
        safety_reduction = max(0, safety_level - supplying_unit_power_left)
        power_transfer_minus_safety = power_transfer - safety_reduction
        return power_transfer_minus_safety

    def _remove_min_power_income_for_other_units_factory(self, power_transfer: int, game_state: GameState) -> int:
        final_tc = self.action_plan.final_tc
        closest_factory = game_state.get_closest_player_factory(final_tc)
        factory_power_income = closest_factory.expected_power_gain
        MIN_POWER_STREAM_FOR_OTHER_UNITS = 0
        max_power_transfer = 2 * (factory_power_income - MIN_POWER_STREAM_FOR_OTHER_UNITS)
        power_transfer_minus_income = min(power_transfer, max_power_transfer)
        return power_transfer_minus_income

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        return 100

    def _get_max_benefit(self, game_state: GameState) -> float:
        return 100

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        nr_steps_move_to_to_receiving_unit = max(0, self.unit.tc.distance_to(self.receiving_c) - 1)
        min_cost_moving = nr_steps_move_to_to_receiving_unit * self.unit.move_time_and_power_cost
        min_steps_including_transfer_power = nr_steps_move_to_to_receiving_unit + 1
        return min_cost_moving, min_steps_including_transfer_power

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


@dataclass
class TransferResourceGoal(UnitGoal):
    factory: Optional[Factory] = field(default=None)
    resource: Resource = field(init=False)

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        return self.unit.get_quantity_resource_in_cargo(self.resource) == 0

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:

        self._init_action_plan()
        self._add_transfer_resources_to_factory_actions(board=game_state.board, constraints=constraints)
        return self.action_plan

    def _add_transfer_resources_to_factory_actions(self, board: Board, constraints: Constraints) -> None:
        actions = self._get_transfer_resources_to_factory_actions(board=board, constraints=constraints)
        self.action_plan.extend(actions=actions)

    def _get_transfer_resources_to_factory_actions(self, board: Board, constraints: Constraints) -> list[UnitAction]:
        start_tc = self.action_plan.final_tc
        start = ResourceTimeCoordinate(start_tc.x, start_tc.y, start_tc.t, q=0, resource=self.resource)
        graph = self._get_transfer_to_factory_graph(board=board, constraints=constraints)
        actions = self._search_graph(graph=graph, start=start)
        return actions

    def _get_transfer_to_factory_graph(self, board: Board, constraints: Constraints) -> TransferToFactoryResourceGraph:
        graph = TransferToFactoryResourceGraph(
            unit_type=self.unit.unit_type,
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            constraints=constraints,
            resource=self.resource,
            q=self.unit.unit_cfg.CARGO_SPACE,
            factory=self.factory,
        )

        return graph

    def _get_max_benefit(self, game_state: GameState) -> float:
        nr_resources_unit = self.unit.get_quantity_resource_in_cargo(self.resource)
        benefit_resource = self.get_benefit_resource(game_state)
        return benefit_resource * nr_resources_unit

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        return self._get_max_benefit(game_state)

    @abstractmethod
    def get_benefit_resource(self, game_state: GameState) -> float:
        ...

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        min_cost_transfer, min_steps_transfer = self._get_min_cost_and_steps_transfer_resource(game_state)
        return min_cost_transfer, min_steps_transfer

    def _get_min_cost_and_steps_transfer_resource(self, game_state: GameState) -> tuple[float, int]:
        nr_steps_to_closest_factory = game_state.board.get_min_distance_to_any_player_factory(self.unit.tc)
        nr_steps_next_to_closest_factory = max(nr_steps_to_closest_factory - 1, 0)
        nr_steps_transfer = 1
        nr_steps = nr_steps_next_to_closest_factory + nr_steps_transfer

        move_cost = nr_steps_next_to_closest_factory * self.unit.move_time_and_power_cost
        transfer_cost = self.unit.time_to_power_cost
        cost = move_cost + transfer_cost

        return cost, nr_steps


@dataclass
class CollectIceGoal(CollectGoal):
    resource = Resource.ICE

    def __repr__(self) -> str:
        return f"collect_ice_[{self.dig_c}]"

    @property
    def key(self) -> str:
        return str(self)

    def get_benefit_resource(self, game_state: GameState) -> float:
        return get_benefit_ice(game_state)

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return self._get_resources_collected_by_n_digs(self.action_plan.nr_digs, game_state)

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


@dataclass
class TransferIceGoal(TransferResourceGoal):
    resource = Resource.ICE

    def __repr__(self) -> str:
        return f"transfer_ice_[{self.unit}]"

    @property
    def key(self) -> str:
        return str(self)

    def get_benefit_resource(self, game_state: GameState) -> float:
        return get_benefit_ice(game_state)

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return self.unit.ice

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


def get_benefit_ice(game_state: GameState) -> float:
    return CONFIG.BENEFIT_ICE


@dataclass
class CollectOreGoal(CollectGoal):
    resource = Resource.ORE

    def __repr__(self) -> str:
        return f"collect_ore_[{self.dig_c}]"

    @property
    def key(self) -> str:
        return str(self)

    def get_benefit_resource(self, game_state: GameState) -> float:
        return get_benefit_ore(game_state)

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return self._get_resources_collected_by_n_digs(self.action_plan.nr_digs, game_state)


@dataclass
class TransferOreGoal(TransferResourceGoal):
    resource = Resource.ORE

    def __repr__(self) -> str:
        return f"transfer_ore_[{self.unit}]"

    @property
    def key(self) -> str:
        return str(self)

    def get_benefit_resource(self, game_state: GameState) -> float:
        return get_benefit_ore(game_state)

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return self.unit.ore


def get_benefit_ore(game_state: GameState) -> float:
    return CONFIG.BASE_BENEFIT_ORE - game_state.real_env_steps * CONFIG.BENEFIT_ORE_REDUCTION_PER_T


class ClearRubbleGoal(DigGoal):
    def __repr__(self) -> str:
        return f"clear_rubble_[{self.dig_c}]"

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        if not action_plan:
            return True

        return not game_state.is_rubble_tile(self.dig_c)

    @property
    def key(self) -> str:
        return str(self)

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:
        self._init_action_plan()

        if self.pickup_power:
            self._add_power_pickup_actions(
                game_state=game_state,
                constraints=constraints,
                next_goal_c=self.dig_c,
                power_tracker=power_tracker,
            )

            if not self.action_plan.unit_has_enough_power(game_state):
                raise InvalidGoalError

        self._add_clear_rubble_actions(game_state=game_state, constraints=constraints)
        return self.action_plan

    def _get_max_useful_digs(self, game_state: GameState) -> int:
        return self._get_nr_digs_to_clear_rubble(game_state.board)

    def _add_clear_rubble_actions(self, game_state: GameState, constraints: Constraints) -> None:
        max_nr_digs = self._get_max_nr_digs_current_ptc(game_state)

        potential_dig_actions = self._get_dig_plan(
            start_tc=self.action_plan.final_tc,
            dig_c=self.dig_c,
            nr_digs=max_nr_digs,
            constraints=constraints,
            board=game_state.board,
        )

        max_valid_digs_actions = self._get_valid_actions(potential_dig_actions, game_state)

        max_valid_digs_actions = self.find_max_dig_actions_can_still_reach_factory(
            max_valid_digs_actions, game_state, constraints
        )

        if len(max_valid_digs_actions) == 0:
            raise InvalidGoalError

        self.action_plan.extend(max_valid_digs_actions)

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        return self._get_benefit_n_digs(action_plan.nr_digs, game_state)

    def _get_benefit_n_digs(self, n_digs: int, game_state: GameState) -> float:
        rubble_removed = self._get_rubble_removed(n_digs, game_state)
        bonus_clear_rubble = (
            CONFIG.RUBBLE_CLEAR_FOR_LICHEN_BONUS_CLEARING if self._clears_rubble(rubble_removed, game_state) else 0
        )
        score = (rubble_removed + bonus_clear_rubble) * 100

        # benefit_rubble_removed = self._get_benefit_removing_rubble(rubble_removed, game_state)
        return score

    def _get_rubble_removed(self, n_digs: int, game_state: GameState) -> int:
        max_rubble_removed = self.unit.rubble_removed_per_dig * n_digs
        rubble_at_pos = game_state.board.rubble[self.dig_c.xy]
        rubble_removed = min(max_rubble_removed, rubble_at_pos)
        return rubble_removed

    # def _get_benefit_removing_rubble(self, rubble_removed: int, game_state: GameState) -> float:
    #     benefit_pathing = self._get_benefit_removing_rubble_pathing(rubble_removed, game_state)
    #     benefit_lichen = self._get_benefit_removing_rubble_for_lichen_growth(rubble_removed, game_state)
    #     return benefit_pathing + benefit_lichen

    # def _get_benefit_removing_rubble_pathing(self, rubble_removed: int, game_state: GameState) -> float:
    #     importance_pathing = game_state.get_importance_removing_rubble_for_pathing(self.dig_c)
    #     return rubble_removed * importance_pathing

    # def _get_benefit_removing_rubble_for_lichen_growth(self, rubble_removed: int, game_state: GameState) -> float:
    #     importance_lichen = game_state.get_importance_removing_rubble_for_lichen_growth(self.dig_c)
    #     score_lichen_removed = self._get_score_rubble_removed(rubble_removed, game_state)
    #     return importance_lichen * score_lichen_removed

    def _get_score_rubble_removed(self, rubble_removed: int, game_state: GameState) -> float:
        if not self._clears_rubble(rubble_removed, game_state):
            return rubble_removed

        return rubble_removed + CONFIG.RUBBLE_CLEAR_FOR_LICHEN_BONUS_CLEARING

    def _clears_rubble(self, rubble_removed: int, game_state: GameState) -> bool:
        rubble_at_pos = game_state.board.rubble[self.dig_c.xy]
        return rubble_removed >= rubble_at_pos

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        min_cost_digging, max_nr_digs = self._get_min_cost_and_steps_max_nr_digs(game_state)
        min_cost_go_to_c, min_steps_go_to_c = self._get_min_cost_and_steps_go_to_dig_c(game_state)

        min_cost = min_cost_digging + min_cost_go_to_c
        min_steps = max_nr_digs + min_steps_go_to_c

        return min_cost, min_steps

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


class DestroyLichenGoal(DigGoal):
    def __repr__(self) -> str:
        return f"destroy_lichen[{self.dig_c}]"

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        return not game_state.is_opponent_lichen_tile(self.dig_c)

    @property
    def key(self) -> str:
        return str(self)

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:
        self._init_action_plan()

        if self.pickup_power:
            self._add_power_pickup_actions(
                game_state=game_state,
                constraints=constraints,
                next_goal_c=self.dig_c,
                power_tracker=power_tracker,
            )

            if not self.action_plan.unit_has_enough_power(game_state):
                raise InvalidGoalError

        self._add_destroy_lichen_actions(game_state=game_state, constraints=constraints)
        return self.action_plan

    def _add_destroy_lichen_actions(self, game_state: GameState, constraints: Constraints) -> None:
        max_nr_digs = self._get_max_nr_digs_current_ptc(game_state)

        potential_dig_actions = self._get_dig_plan(
            start_tc=self.action_plan.final_tc,
            dig_c=self.dig_c,
            nr_digs=max_nr_digs,
            constraints=constraints,
            board=game_state.board,
        )

        max_valid_digs_actions = self._get_valid_actions(potential_dig_actions, game_state)

        max_valid_digs_actions = self.find_max_dig_actions_can_still_reach_factory(
            max_valid_digs_actions, game_state, constraints
        )

        if len(max_valid_digs_actions) == 0:
            raise InvalidGoalError

        self.action_plan.extend(max_valid_digs_actions)

    def _get_max_useful_digs(self, game_state: GameState) -> int:
        return self._get_nr_max_digs_to_destroy_lichen(game_state)

    def _get_nr_max_digs_to_destroy_lichen(self, game_state: GameState) -> int:
        # Can underestimate the amount of digs when constraints make the unit appear a move later there
        nr_steps_to_lichen = self._get_min_nr_steps_to_dig_c(game_state)
        max_lichen_upon_arrival = self._get_max_lichen_in_n_steps(game_state.board, nr_steps_to_lichen)
        return self._get_nr_max_dig_to_destroy_lichen_unit_at_lichen(max_lichen_upon_arrival)

    def _get_nr_max_dig_to_destroy_lichen_unit_at_lichen(self, lichen_at_tile: int) -> int:
        lichen_removed_per_dig = self.unit.lichen_removed_per_dig
        potential_regain_lichen_per_turn = 1
        min_lichen_change_per_dig = lichen_removed_per_dig - potential_regain_lichen_per_turn

        quotient, remainder = divmod(lichen_at_tile, min_lichen_change_per_dig)

        if remainder <= potential_regain_lichen_per_turn:
            max_nr_digs = quotient
        else:
            max_nr_digs = quotient + 1

        return max_nr_digs

    def _get_max_lichen_in_n_steps(self, board: Board, n_steps: int) -> int:
        max_lichen = 100
        current_lichen = board.lichen[self.dig_c.xy]
        return min(max_lichen, current_lichen + n_steps)

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        return self._get_benefit_n_digs(action_plan.nr_digs, game_state)

    def _get_benefit_n_digs(self, n_digs: int, game_state: GameState) -> float:
        lichen_removed = self._get_lichen_removed(n_digs, game_state)
        benefit_lichen_removed = self._get_benefit_removing_lichen(lichen_removed, game_state)
        return benefit_lichen_removed

    def _get_lichen_removed(self, n_digs: int, game_state: GameState) -> int:
        max_lichen_removed = self.unit.rubble_removed_per_dig * n_digs
        lichen_at_pos = game_state.board.lichen[self.dig_c.xy]
        lichen_removed = min(max_lichen_removed, lichen_at_pos)
        return lichen_removed

    def _get_benefit_removing_lichen(self, lichen_removed: int, game_state: GameState) -> float:
        benefit_lichen_removed = 20
        bonus_cleared_lichen = 20 * benefit_lichen_removed

        lichen_removed_benefit = benefit_lichen_removed * lichen_removed

        lichen_at_pos = game_state.board.lichen[self.dig_c.xy]
        if lichen_removed == lichen_at_pos:
            return lichen_removed_benefit + bonus_cleared_lichen
        else:
            return lichen_removed_benefit

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        min_cost_digging, max_nr_digs = self._get_min_cost_and_steps_max_nr_digs(game_state)
        min_cost_go_to_c, min_steps_go_to_c = self._get_min_cost_and_steps_go_to_dig_c(game_state)

        min_cost = min_cost_digging + min_cost_go_to_c
        min_steps = max_nr_digs + min_steps_go_to_c

        return min_cost, min_steps

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


@dataclass
class FleeGoal(UnitGoal):
    opp_c: TimeCoordinate

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        return not self.unit.is_under_threath(game_state)

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:
        self._init_action_plan()
        constraints = self._add_flee_constraints(constraints)
        self._go_to_factory_actions(game_state, constraints)

        return self.action_plan

    def _add_flee_constraints(self, constraints: Constraints) -> Constraints:
        current_tc = self.unit.tc
        opp_c = self.opp_c
        current_c_next_t = self.unit.tc + Direction.CENTER

        constraints = copy(constraints)
        constraints.add_negative_constraints([current_tc, opp_c, current_c_next_t])
        return constraints

    def _go_to_factory_actions(self, game_state: GameState, constraints: Constraints) -> None:
        closest_factory_c = game_state.get_closest_player_factory_c(c=self.action_plan.final_tc)
        potential_move_actions = self._get_move_to_plan(
            start_tc=self.unit.tc, goal=closest_factory_c, constraints=constraints, board=game_state.board
        )
        potential_move_actions = self._get_valid_actions(potential_move_actions, game_state)

        while potential_move_actions:
            potential_action_plan = self.action_plan + potential_move_actions

            if potential_action_plan.is_valid_size:
                self.action_plan.extend(potential_move_actions)
                self.cur_tc = self.action_plan.final_tc
                break

            potential_move_actions = potential_move_actions[:-1]
        else:
            raise InvalidGoalError

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        return CONFIG.BENEFIT_FLEEING

    def __repr__(self) -> str:
        return f"No_Goal_{self.unit.unit_id}"

    @property
    def key(self) -> str:
        return str(self)

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        min_steps = game_state.board.get_min_distance_to_any_player_factory(self.unit.tc)
        min_cost = min_steps * self.unit.move_time_and_power_cost

        return min_cost, min_steps

    def _get_max_benefit(self, game_state: GameState) -> float:
        return CONFIG.BENEFIT_FLEEING

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


class UnitNoGoal(UnitGoal):
    _value = None
    # TODO, what should be the value of losing a unit?
    PENALTY_VIOLATING_CONSTRAINT = -10_000

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        return True

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:
        self.action_plan = UnitActionPlan(actor=self.unit, original_actions=[MoveAction(Direction.CENTER)])
        self._invalidates_constraint = constraints.any_tc_violates_constraint(self.action_plan.time_coordinates)
        return self.action_plan

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        if action_plan.is_first_action_stationary and self.unit.is_under_threath(game_state):
            return -CONFIG.COST_POTENTIALLY_LOSING_UNIT

        return 0.0 if not self._invalidates_constraint else self.PENALTY_VIOLATING_CONSTRAINT

    def __repr__(self) -> str:
        return f"No_Goal_{self.unit.unit_id}"

    def _get_max_benefit(self, game_state: GameState) -> float:
        return 0

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        return self.unit.time_to_power_cost, 1

    @property
    def key(self) -> str:
        return str(self)

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0


class EvadeConstraintsGoal(UnitGoal):
    _value = None

    def is_completed(self, game_state: GameState, action_plan: UnitActionPlan) -> bool:
        return True

    def _generate_action_plan(
        self,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitActionPlan:
        self._init_action_plan()
        if constraints.any_tc_violates_constraint(self.action_plan.time_coordinates):
            self._add_evade_actions(game_state, constraints)
        else:
            self.action_plan = UnitActionPlan(actor=self.unit, original_actions=[MoveAction(Direction.CENTER)])
        return self.action_plan

    def _add_evade_actions(self, game_state: GameState, constraints: Constraints):
        potential_move_actions = self._get_evade_plan(
            start_tc=self.unit.tc, constraints=constraints, board=game_state.board
        )
        self.action_plan.extend(potential_move_actions)

        if not self.action_plan.actor_can_carry_out_plan(game_state):
            raise InvalidGoalError

    def _get_evade_plan(
        self,
        start_tc: TimeCoordinate,
        constraints: Constraints,
        board: Board,
    ) -> list[UnitAction]:
        graph = self._get_evade_constraints_graph(board=board, constraints=constraints)
        actions = self._search_graph(graph=graph, start=start_tc)
        return actions

    def _get_evade_constraints_graph(self, board: Board, constraints: Constraints) -> EvadeConstraintsGraph:
        graph = EvadeConstraintsGraph(
            unit_type=self.unit.unit_type,
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            constraints=constraints,
        )

        return graph

    def get_benefit_action_plan(self, action_plan: UnitActionPlan, game_state: GameState) -> float:
        if action_plan.is_first_action_stationary and self.unit.is_under_threath(game_state):
            return -CONFIG.COST_POTENTIALLY_LOSING_UNIT

        return 0.0

    def __repr__(self) -> str:
        return f"No_Goal_{self.unit.unit_id}"

    def _get_max_benefit(self, game_state: GameState) -> float:
        return 0

    def _get_min_cost_and_steps(self, game_state: GameState) -> tuple[float, int]:
        return self.unit.time_to_power_cost, 1

    @property
    def key(self) -> str:
        return str(self)

    def quantity_ice_to_transfer(self, game_state: GameState) -> int:
        return 0

    def quantity_ore_to_transfer(self, game_state: GameState) -> int:
        return 0
