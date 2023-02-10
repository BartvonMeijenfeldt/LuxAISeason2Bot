from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from math import ceil

from search import get_actions_a_to_b, PowerCostGraph
from objects.action import DigAction, MoveAction, TransferAction, PickupAction
from objects.action_plan import ActionPlan
from objects.coordinate import Direction
from objects.resource import Resource

if TYPE_CHECKING:
    from objects.unit import Unit
    from objects.game_state import GameState
    from objects.board import Board
    from objects.coordinate import Coordinate, CoordinateList


@dataclass(kw_only=True)
class Goal(metaclass=ABCMeta):
    unit: Unit

    action_plan: ActionPlan = field(init=False)
    _value: Optional[float] = field(init=False, default=None)
    _is_valid: Optional[bool] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._init_action_plan()

    def generate_and_evaluate_action_plan(self, game_state: GameState) -> None:
        self.generate_action_plan(game_state=game_state)
        self._value = self.get_value_action_plan(action_plan=self.action_plan, game_state=game_state)

    @abstractmethod
    def generate_action_plan(self, game_state: GameState) -> None:
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

    def _optional_add_power_pickup_action(self, game_state: GameState) -> None:
        power_space_left = self.unit.power_space_left
        closest_factory = game_state.get_closest_factory(c=self.unit.c)

        if closest_factory.is_on_factory(c=self.unit.c) and power_space_left:
            power_in_factory = closest_factory.power
            cargo_to_pickup = min(power_space_left, power_in_factory)
            self.action_plan.append(PickupAction(cargo_to_pickup, Resource.Power))

    def _init_action_plan(self) -> None:
        self.action_plan = ActionPlan(unit=self.unit)

    def __lt__(self, other: Goal):
        return self.value < other.value


@dataclass
class CollectIceGoal(Goal):
    ice_c: Coordinate
    factory_pos: Coordinate
    quantity: Optional[int] = None

    def __str__(self) -> str:
        return f"collect_ice_[{self.ice_c}]"

    @property
    def key(self) -> str:
        return str(self)

    def generate_action_plan(self, game_state: GameState) -> None:
        self.graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        self._is_valid = True  # TODO
        self._init_action_plan()
        self._optional_add_power_pickup_action(game_state=game_state)
        self._add_pos_to_ice_actions()
        self._add_max_dig_action(game_state=game_state)
        self._add_ice_to_factory_actions()
        self._add_transfer_action()

    def _add_pos_to_ice_actions(self) -> None:
        actions = get_actions_a_to_b(graph=self.graph, start=self.unit.c, end=self.ice_c)
        self.action_plan.extend(actions=actions)

    def _add_ice_to_factory_actions(self) -> None:
        actions = self._get_ice_to_factory_actions()
        self.action_plan.extend(actions=actions)

    def _get_ice_to_factory_actions(self) -> list[MoveAction]:
        return get_actions_a_to_b(graph=self.graph, start=self.ice_c, end=self.factory_pos)

    def _add_transfer_action(self) -> None:
        max_cargo = self.unit.unit_cfg.CARGO_SPACE
        transfer_action = TransferAction(direction=Direction.CENTER, amount=max_cargo, resource=Resource.Ice)
        self.action_plan.append(transfer_action)

    def _get_transfer_action(self) -> TransferAction:
        max_cargo = self.unit.unit_cfg.CARGO_SPACE
        return TransferAction(direction=Direction.CENTER, amount=max_cargo, resource=Resource.Ice)

    def _get_actions_after_digging(self) -> list[MoveAction | TransferAction]:
        ice_to_factory_actions = self._get_ice_to_factory_actions()
        transfer_action = self._get_transfer_action()
        return ice_to_factory_actions + [transfer_action]

    def _add_max_dig_action(self, game_state: GameState) -> None:
        # TODO make this a binary search or something otherwise more efficient
        best_n = None
        n_digging = 0

        actions_after_digging = self._get_actions_after_digging()

        while True:
            potential_dig_action = DigAction(n=n_digging)
            new_actions = [potential_dig_action] + actions_after_digging
            potential_action_plan = self.action_plan + new_actions
            if not potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
                break

            best_n = n_digging
            n_digging += 1

        if best_n:
            dig_action = DigAction(n=best_n)
            self.action_plan.append(dig_action)

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_used(board=game_state.board)
        return number_of_steps + 0.1 * power_cost


@dataclass
class ClearRubbleGoal(Goal):
    rubble_positions: CoordinateList

    def __str__(self) -> str:
        first_rubble_c = self.rubble_positions[0]
        return f"clear_rubble_[{first_rubble_c}]"

    @property
    def key(self) -> str:
        return str(self)

    def generate_action_plan(self, game_state: GameState):
        self.graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        self._init_action_plan()
        self._optional_add_power_pickup_action(game_state=game_state)
        self._add_clear_initial_rubble_actions(game_state=game_state)
        self._add_additional_rubble_actions(game_state=game_state)
        self._optional_add_go_to_factory_actions(game_state=game_state)

    def _add_clear_initial_rubble_actions(self, game_state: GameState) -> None:
        self.cur_c = self.unit.c

        for rubble_c in self.rubble_positions:
            potential_dig_actions = self._get_rubble_actions(
                start_c=self.cur_c, rubble_c=rubble_c, board=game_state.board
            )
            potential_action_plan = self.action_plan + potential_dig_actions

            if not potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
                self._is_valid = False
                return

            if self._unit_can_still_reach_factory(action_plan=potential_action_plan, game_state=game_state):
                self.action_plan.extend(potential_dig_actions)
                self.cur_c = rubble_c
            else:
                self._is_valid = False
                return

        self._is_valid = True

    def _unit_can_still_reach_factory(self, action_plan: ActionPlan, game_state: GameState) -> bool:
        return action_plan.unit_can_add_reach_factory_to_plan(
            game_state=game_state, graph=self.graph
        ) or action_plan.unit_can_reach_factory_after_action_plan(game_state=game_state, graph=self.graph)

    def _get_rubble_actions(
        self, start_c: Coordinate, rubble_c: Coordinate, board: Board
    ) -> list[MoveAction | DigAction]:
        pos_to_rubble_actions = get_actions_a_to_b(graph=self.graph, start=start_c, end=rubble_c)

        rubble_at_pos = board.rubble[tuple(rubble_c)]
        nr_required_digs = ceil(rubble_at_pos / self.unit.unit_cfg.DIG_RUBBLE_REMOVED)
        dig_action = [DigAction(n=nr_required_digs)]

        actions = pos_to_rubble_actions + dig_action
        return actions

    def _add_additional_rubble_actions(self, game_state: GameState):
        if len(self.action_plan.actions) == 0 or not isinstance(self.action_plan.actions[-1], DigAction):
            return

        while True:
            closest_rubble = game_state.board.get_closest_rubble_tile(self.cur_c, exclude_c=self.rubble_positions)
            potential_dig_rubble_actions = self._get_rubble_actions(
                start_c=self.cur_c, rubble_c=closest_rubble, board=game_state.board
            )

            potential_action_plan = self.action_plan + potential_dig_rubble_actions

            if not potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
                return

            if self._unit_can_still_reach_factory(action_plan=potential_action_plan, game_state=game_state):
                self.action_plan.extend(potential_dig_rubble_actions)
                self.rubble_positions.append(c=closest_rubble)
                self.cur_c = closest_rubble
            else:
                return

    def _optional_add_go_to_factory_actions(self, game_state: GameState) -> None:
        closest_factory_c = game_state.get_closest_factory_c(c=self.cur_c)
        potential_rubble_to_factory_actions = get_actions_a_to_b(self.graph, start=self.cur_c, end=closest_factory_c)
        potential_action_plan = self.action_plan + potential_rubble_to_factory_actions

        if potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
            self.action_plan.extend(potential_rubble_to_factory_actions)

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        if number_of_steps == 0:
            return -10000000000

        first_rubble_bonus = self.unit.c.distance_to(self.rubble_positions[0]) * -1000

        power_cost = action_plan.get_power_used(board=game_state.board)
        number_of_rubble_cleared = len(self.rubble_positions)
        rubble_cleared_per_step = number_of_rubble_cleared / number_of_steps
        rubble_cleared_per_power = number_of_rubble_cleared / power_cost
        return rubble_cleared_per_step + rubble_cleared_per_power + first_rubble_bonus + 100_000


class NoGoalGoal(Goal):
    _value = None
    _is_valid = True

    def generate_action_plan(self, game_state: GameState) -> None:
        return None

    def get_value_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        return 0.0

    def __str__(self) -> str:
        return f"No_Goal_{self.unit.unit_id}"

    @property
    def key(self) -> str:
        return str(self)


@dataclass
class GoalCollection:
    goals: Sequence[Goal]

    def generate_and_evaluate_action_plans(self, game_state: GameState) -> None:
        for goal in self.goals:
            goal.generate_and_evaluate_action_plan(game_state=game_state)

        self.goals = [goal for goal in self.goals if goal.is_valid]

    @property
    def best_action_plan(self) -> ActionPlan:
        best_goal = max(self.goals)
        return best_goal.action_plan

    def __getitem__(self, key: int):
        return self.goals[key]

    def get_goal(self, key: str) -> Goal:
        for goal in self.goals:
            if goal.key == key:
                return goal

        raise ValueError(f"{key} not key of goals")

    def get_keys(self) -> set[str]:
        return {goal.key for goal in self.goals}

    def get_key_values(self) -> dict[str, float]:
        return {goal.key: goal.value for goal in self.goals}
