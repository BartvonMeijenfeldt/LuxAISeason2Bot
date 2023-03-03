from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from math import ceil
from itertools import count

from search import Search, MoveToGraph, DigAtGraph, PickupPowerGraph, Graph
from objects.action import DigAction, TransferAction
from objects.action_plan import ActionPlan
from objects.direction import Direction
from objects.resource import Resource
from objects.coordinate import (
    PowerTimeCoordinate,
    DigCoordinate,
    DigTimeCoordinate,
    TimeCoordinate,
    Coordinate,
    CoordinateList,
)
from logic.constraints import Constraints


if TYPE_CHECKING:
    from objects.unit import Unit
    from objects.game_state import GameState
    from objects.board import Board
    from objects.action import Action


@dataclass
class Goal(metaclass=ABCMeta):
    unit: Unit

    _value: Optional[float] = field(init=False, default=None)
    _is_valid: Optional[bool] = field(init=False, default=None)
    solution_hash: dict[str, ActionPlan] = field(init=False, default_factory=dict)

    def generate_and_evaluate_action_plan(
        self, game_state: GameState, constraints: Optional[Constraints] = None
    ) -> ActionPlan:
        self.generate_action_plan(game_state=game_state, constraints=constraints)
        self._value = self.get_value_action_plan(action_plan=self.action_plan, game_state=game_state)
        return self.action_plan

    def generate_action_plan(self, game_state: GameState, constraints: Optional[Constraints] = None) -> ActionPlan:
        if not constraints:
            constraints = Constraints()

        if constraints.key in self.solution_hash:
            return self.solution_hash[constraints.key]

        if constraints.parent in self.solution_hash:
            parent_solution = self.solution_hash[constraints.parent]
            if self._parent_solution_is_valid(parent_solution, constraints):
                self.solution_hash[constraints.key] = parent_solution
                return parent_solution

        action_plan = self._generate_action_plan(game_state, constraints=constraints)
        self.solution_hash[constraints.key] = action_plan
        return action_plan

    def _parent_solution_is_valid(self, parent_solution: ActionPlan, constraints: Constraints) -> bool:
        for tc in parent_solution.get_time_coordinates():
            if constraints.tc_violates_constraint(tc):
                return False

        if constraints.max_power_request is not None:
            if parent_solution.power_requested > constraints.max_power_request:
                return False

        return True

    @abstractmethod
    def _generate_action_plan(self, game_state: GameState, constraints: Constraints = None) -> ActionPlan:
        ...

    @abstractmethod
    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @property
    @abstractmethod
    def key(self) -> str:
        ...

    @property
    def value(self) -> float:
        if self._value is None:
            raise ValueError("Value is not supposed to be None here")

        return self._value

    @property
    def is_valid(self) -> bool:
        if self._is_valid is None:
            raise ValueError("_is_valid is not supposed to be None here")

        return self._is_valid

    def _get_move_graph(self, board: Board, goal: Coordinate, constraints: Constraints) -> MoveToGraph:
        graph = MoveToGraph(
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            goal=goal,
            constraints=constraints,
        )

        return graph

    def _get_recharge_graph(self, board: Board, recharge_amount: int, constraints: Constraints) -> PickupPowerGraph:
        graph = PickupPowerGraph(
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            power_pickup_goal=recharge_amount,
            constraints=constraints,
        )

        return graph

    def _get_dig_graph(self, board: Board, goal: DigCoordinate, constraints: Constraints) -> DigAtGraph:
        graph = DigAtGraph(
            board=board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            constraints=constraints,
            goal=goal,
        )

        return graph

    def _search_graph(self, graph: Graph, start: TimeCoordinate) -> list[Action]:
        search = Search(graph=graph)
        optimal_actions = search.get_actions_to_complete_goal(start=start)
        return optimal_actions

    def _optional_add_power_pickup_action(self, game_state: GameState, constraints: Constraints) -> None:
        if constraints.max_power_request is not None and constraints.max_power_request < 10:
            return

        power_space_left = self.unit.power_space_left
        if not power_space_left:
            return

        closest_factory = game_state.get_closest_factory(c=self.unit.tc)
        if not closest_factory.is_on_factory(c=self.unit.tc):
            return

        power_in_factory = closest_factory.power
        power_to_pickup = min(power_space_left, power_in_factory)

        if power_to_pickup / self.unit.power < 0.1:
            return

        graph = self._get_recharge_graph(
            board=game_state.board, recharge_amount=power_to_pickup, constraints=constraints
        )

        recharge_tc = PowerTimeCoordinate(*self.action_plan.final_tc, power_recharged=0)
        new_actions = self._search_graph(graph=graph, start=recharge_tc)
        self.action_plan.extend(new_actions)

    def _init_action_plan(self) -> None:
        self.action_plan = ActionPlan(unit=self.unit)

    def __lt__(self, other: Goal):
        return self.value < other.value


@dataclass
class CollectIceGoal(Goal):
    ice_c: Coordinate
    factory_c: Coordinate
    quantity: Optional[int] = None

    def __repr__(self) -> str:
        return f"collect_ice_[{self.ice_c}]"

    @property
    def key(self) -> str:
        return str(self)

    def _generate_action_plan(self, game_state: GameState, constraints: Optional[Constraints] = None) -> ActionPlan:
        if constraints is None:
            constraints = Constraints()

        self._is_valid = True
        self._init_action_plan()
        self._optional_add_power_pickup_action(game_state=game_state, constraints=constraints)
        self._add_dig_actions(game_state=game_state, constraints=constraints)
        self._add_ice_to_factory_actions(board=game_state.board, constraints=constraints)
        self._add_transfer_action()
        return self.action_plan

    def _get_transfer_action(self) -> TransferAction:
        max_cargo = self.unit.unit_cfg.CARGO_SPACE
        return TransferAction(direction=Direction.CENTER, amount=max_cargo, resource=Resource.Ice)

    def _add_dig_actions(self, game_state: GameState, constraints: Constraints) -> None:
        # TODO make this a binary search or something otherwise more efficient
        best_plan = None

        for nr_digs in count(start=1):
            dig_coordinate = DigCoordinate(x=self.ice_c.x, y=self.ice_c.y, nr_digs=nr_digs)
            graph = self._get_dig_graph(board=game_state.board, goal=dig_coordinate, constraints=constraints)
            dig_time_coordinate = DigTimeCoordinate(*self.action_plan.final_tc, nr_digs=0)
            dig_plan_actions = self._search_graph(graph=graph, start=dig_time_coordinate)

            potential_action_plan = self.action_plan + dig_plan_actions
            final_pos_digging = potential_action_plan.final_tc
            actions_after_digging = self._get_actions_after_digging(
                board=game_state.board, constraints=constraints, start=final_pos_digging
            )
            potential_action_plan = potential_action_plan + actions_after_digging

            if not potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
                break

            best_plan = dig_plan_actions

        if best_plan:
            self.action_plan.extend(best_plan)
        else:
            self._is_valid = False

    def _get_actions_after_digging(self, board: Board, constraints: Constraints, start: TimeCoordinate) -> list[Action]:
        ice_to_factory_actions = self._get_move_to_factory_actions(board=board, constraints=constraints, start=start)
        transfer_action = self._get_transfer_action()
        return ice_to_factory_actions + [transfer_action]

    def _add_ice_to_factory_actions(self, board: Board, constraints: Constraints) -> None:
        actions = self._get_move_to_factory_actions(board=board, constraints=constraints)
        self.action_plan.extend(actions=actions)

    def _get_move_to_factory_actions(
        self, board: Board, constraints: Constraints, start: Optional[TimeCoordinate] = None
    ) -> list[Action]:

        if not start:
            start = self.action_plan.final_tc

        graph = self._get_move_graph(board=board, goal=self.factory_c, constraints=constraints)
        optimal_actions = self._search_graph(graph=graph, start=start)
        return optimal_actions

    def _add_transfer_action(self) -> None:
        max_cargo = self.unit.unit_cfg.CARGO_SPACE
        transfer_action = TransferAction(direction=Direction.CENTER, amount=max_cargo, resource=Resource.Ice)
        self.action_plan.append(transfer_action)

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_used(board=game_state.board)
        return number_of_steps + 0.1 * power_cost


@dataclass
class ClearRubbleGoal(Goal):
    rubble_positions: CoordinateList

    def __repr__(self) -> str:
        first_rubble_c = self.rubble_positions[0]
        return f"clear_rubble_[{first_rubble_c}]"

    @property
    def key(self) -> str:
        return str(self)

    def _generate_action_plan(self, game_state: GameState, constraints: Optional[Constraints] = None) -> ActionPlan:
        if constraints is None:
            constraints = Constraints()

        self._init_action_plan()
        self._optional_add_power_pickup_action(game_state=game_state, constraints=constraints)
        self._add_clear_initial_rubble_actions(game_state=game_state, constraints=constraints)
        # self._add_additional_rubble_actions(game_state=game_state, constraints=constraints)
        self._optional_add_go_to_factory_actions(game_state=game_state, constraints=constraints)
        return self.action_plan

    def _add_clear_initial_rubble_actions(self, game_state: GameState, constraints: Constraints) -> None:
        for rubble_c in self.rubble_positions:
            start = DigTimeCoordinate(*self.action_plan.final_tc, nr_digs=0)
            nr_required_digs = self._get_nr_required_digs(rubble_c=rubble_c, board=game_state.board)

            potential_dig_actions = self._get_rubble_actions(
                rubble_c=rubble_c,
                nr_digs=nr_required_digs,
                constraints=constraints,
                board=game_state.board,
                start=start,
            )

            potential_dig_actions = self._get_valid_actions(potential_dig_actions, game_state)

            while potential_dig_actions:
                potential_action_plan = self.action_plan + potential_dig_actions

                if potential_action_plan.is_valid_size and self._unit_can_still_reach_factory(
                    action_plan=potential_action_plan, game_state=game_state, constraints=constraints
                ):
                    self.action_plan.extend(potential_dig_actions)
                    self.cur_tc = self.action_plan.final_tc
                    break

                potential_dig_actions = potential_dig_actions[:-1]
            else:
                self._is_valid = False
                return

        self._is_valid = True

    def _unit_can_still_reach_factory(
        self, action_plan: ActionPlan, game_state: GameState, constraints: Constraints
    ) -> bool:
        return action_plan.unit_can_add_reach_factory_to_plan(
            game_state=game_state, constraints=constraints
        ) or action_plan.unit_can_reach_factory_after_action_plan(game_state=game_state, constraints=constraints)

    def _get_nr_required_digs(self, rubble_c: Coordinate, board: Board) -> int:

        rubble_at_pos = board.rubble[rubble_c.xy]
        nr_required_digs = ceil(rubble_at_pos / self.unit.unit_cfg.DIG_RUBBLE_REMOVED)
        return nr_required_digs

    def _get_rubble_actions(
        self, rubble_c: Coordinate, nr_digs: int, constraints: Constraints, board: Board, start: DigTimeCoordinate,
    ) -> list[Action]:
        dig_coordinate = DigCoordinate(x=rubble_c.x, y=rubble_c.y, nr_digs=nr_digs)

        graph = self._get_dig_graph(board=board, goal=dig_coordinate, constraints=constraints)
        actions = self._search_graph(graph=graph, start=start)
        return actions

    def _get_valid_actions(self, actions: list[Action], game_state: GameState) -> list[Action]:
        potential_action_plan = self.action_plan + actions
        nr_valid_primitive_actions = potential_action_plan.get_nr_valid_primitive_actions(game_state)
        nr_original_primitive_actions = len(self.action_plan.primitive_actions)
        return potential_action_plan.primitive_actions[nr_original_primitive_actions:nr_valid_primitive_actions]

    def _add_additional_rubble_actions(self, game_state: GameState, constraints: Constraints):
        if len(self.action_plan.actions) == 0 or not isinstance(self.action_plan.actions[-1], DigAction):
            return

        while True:
            closest_rubble = game_state.board.get_closest_rubble_tile(self.cur_tc, exclude_c=self.rubble_positions)
            dig_time_coordinate = DigTimeCoordinate(*self.cur_tc, nr_digs=0)
            potential_dig_rubble_actions = self._get_rubble_actions(
                start=dig_time_coordinate, rubble_c=closest_rubble, board=game_state.board, constraints=constraints
            )

            potential_action_plan = self.action_plan + potential_dig_rubble_actions

            if potential_action_plan.unit_can_carry_out_plan(
                game_state=game_state
            ) and self._unit_can_still_reach_factory(
                action_plan=potential_action_plan, game_state=game_state, constraints=constraints
            ):
                self.action_plan.extend(potential_dig_rubble_actions)
                self.rubble_positions.append(c=closest_rubble)
                self.cur_tc = self.action_plan.final_tc
            else:
                return

    def _optional_add_go_to_factory_actions(self, game_state: GameState, constraints: Constraints) -> None:
        closest_factory_c = game_state.get_closest_factory_c(c=self.action_plan.final_tc)
        graph = self._get_move_graph(board=game_state.board, goal=closest_factory_c, constraints=constraints)
        potential_move_actions = self._search_graph(graph=graph, start=self.action_plan.final_tc)

        potential_action_plan = self.action_plan + potential_move_actions

        if potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
            self.action_plan.extend(potential_move_actions)

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        if number_of_steps == 0:
            return -10000000000

        first_rubble_bonus = self.unit.tc.distance_to(self.rubble_positions[0]) * -1000

        power_cost = action_plan.get_power_used(board=game_state.board)
        number_of_rubble_cleared = len(self.rubble_positions)
        rubble_cleared_per_step = number_of_rubble_cleared / number_of_steps
        rubble_cleared_per_power = number_of_rubble_cleared / power_cost
        return rubble_cleared_per_step + rubble_cleared_per_power + first_rubble_bonus + 100_000


@dataclass
class ActionQueueGoal(Goal):
    """Goal currently in action queue"""

    goal: Goal
    action_plan: ActionPlan
    _is_valid = True

    def _generate_action_plan(self, game_state: GameState, constraints: Optional[Constraints] = None) -> ActionPlan:
        # TODO add something to generation infeasible if it violates constraints
        return self.action_plan

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        value_including_update_cost = self.goal.get_value_action_plan(self.action_plan, game_state)
        return value_including_update_cost + 100_000
        # return value_including_update_cost + self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

    @property
    def key(self) -> str:
        # This will cause trouble when we allow goal switching, those goals will have the same ID
        # Can probably be solved by just picking the highest one / returning highest one by the goal collection
        return self.goal.key


class NoGoalGoal(Goal):
    _value = None
    _is_valid = True

    def _generate_action_plan(self, game_state: GameState, constraints: Optional[Constraints] = None) -> ActionPlan:
        self._init_action_plan()
        return self.action_plan

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"No_Goal_{self.unit.unit_id}"

    @property
    def key(self) -> str:
        return str(self)


class GoalCollection:
    def __init__(self, goals: Sequence[Goal]) -> None:
        self.goals_dict = {goal.key: goal for goal in goals}

    def generate_and_evaluate_action_plans(
        self, game_state: GameState, constraints: Optional[Constraints] = None
    ) -> None:
        for goal in self.goals_dict.values():
            goal.generate_and_evaluate_action_plan(game_state=game_state, constraints=constraints)

    def get_goal(self, key: str) -> Goal:
        return self.goals_dict[key]

    def get_keys(self) -> set[str]:
        return {key for key, goal in self.goals_dict.items() if goal.is_valid}

    def get_key_values(self, game_state: GameState, constraints: Optional[Constraints] = None) -> dict[str, float]:
        self.generate_and_evaluate_action_plans(game_state=game_state, constraints=constraints)
        return {key: goal.value for key, goal in self.goals_dict.items() if goal.is_valid}
