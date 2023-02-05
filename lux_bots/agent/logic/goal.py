from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from math import ceil

from search import get_actions_a_to_b, PowerCostGraph
from objects.action import Action, DigAction, MoveAction, TransferAction, PickupAction
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

    action_plans: list[ActionPlan] = field(init=False, default_factory=list)
    cur_actions: list[Action] = field(init=False, default_factory=list)

    @abstractmethod
    def generate_action_plans(self, game_state: GameState) -> None:
        ...

    @abstractmethod
    def evaluate_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        ...

    @property
    def best_action_plan(self) -> ActionPlan:
        return max(self.action_plans)

    @property
    def __eq__(self, other: "Goal") -> bool:
        self.best_action_plan < other.best_action_plan

    def _add_power_pickup_action(self, game_state: GameState) -> list[PickupAction]:
        power_space_left = self.unit.power_space_left
        closest_factory = game_state.get_closest_factory(c=self.unit.c)

        if closest_factory.is_on_factory(c=self.unit.c):
            power_in_factory = closest_factory.power
            cargo_to_pickup = min(power_space_left, power_in_factory)
            self._append_action(PickupAction(cargo_to_pickup, Resource.Power))

    def _extend_actions(self, actions: list[Action]) -> None:
        self.cur_actions.extend(actions)

    def _append_action(self, action: Action) -> None:
        self.cur_actions.append(action)

    def _create_action_plan(self, actions: list[Action], game_state: GameState) -> ActionPlan:
        return ActionPlan(original_actions=actions, unit=self.unit, goal=self, game_state=game_state)

    @property
    def value(self):
        return max(self.action_plans).value

    def __lt__(self, other: Goal):
        return self.value < other.value


@dataclass
class CollectIceGoal(Goal):
    ice_c: Coordinate
    factory_pos: Coordinate
    quantity: Optional[int] = None

    def generate_action_plans(self, game_state: GameState):
        self.action_plans = [self._generate_plan(game_state=game_state)]

    def _generate_plan(self, game_state: GameState) -> ActionPlan:
        self.graph = PowerCostGraph(game_state.board, time_to_power_cost=20)

        self._add_power_pickup_action(game_state=game_state)
        self._add_pos_to_ice_actions()
        self._add_max_dig_action(game_state=game_state)
        self._add_ice_to_factory_actions()
        self._add_transfer_action()

        return self._create_action_plan(actions=self.cur_actions, game_state=game_state)

    def _add_pos_to_ice_actions(self) -> None:
        actions = get_actions_a_to_b(graph=self.graph, start=self.unit.c, end=self.ice_c)
        self._extend_actions(actions=actions)

    def _add_ice_to_factory_actions(self) -> None:
        actions = self._get_ice_to_factory_actions()
        self._extend_actions(actions=actions)

    def _get_ice_to_factory_actions(self) -> list[MoveAction]:
        return get_actions_a_to_b(graph=self.graph, start=self.ice_c, end=self.factory_pos)

    def _add_transfer_action(self) -> None:
        max_cargo = self.unit.unit_cfg.CARGO_SPACE
        transfer_action = TransferAction(direction=Direction.CENTER, amount=max_cargo, resource=Resource.Ice)
        self._append_action(transfer_action)

    def _get_transfer_action(self) -> TransferAction:
        max_cargo = self.unit.unit_cfg.CARGO_SPACE
        return TransferAction(direction=Direction.CENTER, amount=max_cargo, resource=Resource.Ice)

    def _get_actions_after_digging(self) -> list[Action]:
        ice_to_factory_actions = self._get_ice_to_factory_actions()
        transfer_action = self._get_transfer_action()
        return ice_to_factory_actions + [transfer_action]

    def _add_max_dig_action(self, game_state: GameState) -> None:
        best_n = None
        n_digging = 0

        actions_after_digging = self._get_actions_after_digging()

        while True:
            potential_dig_action = DigAction(n=n_digging)
            potential_action = self.cur_actions + [potential_dig_action] + actions_after_digging
            action_plan = self._create_action_plan(actions=potential_action, game_state=game_state)
            if not action_plan.unit_can_carry_out_plan(game_state=game_state):
                break

            best_n = n_digging
            n_digging += 1

        if best_n:
            potential_dig_action = DigAction(n=best_n)
            self._append_action(DigAction(n=best_n))

    def evaluate_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_used(board=game_state.board)
        return number_of_steps + 0.1 * power_cost


@dataclass
class ClearRubbleGoal(Goal):
    rubble_positions: CoordinateList

    def generate_action_plans(self, game_state: GameState):
        self.action_plans = [self._generate_plan(game_state=game_state)]

    def _generate_plan(self, game_state: GameState) -> ActionPlan:
        self.graph = PowerCostGraph(game_state.board, time_to_power_cost=20)

        self._add_power_pickup_action(game_state=game_state)
        self._add_clear_initial_rubble_actions(game_state=game_state)
        self._add_additional_rubble_actions(game_state=game_state)
        self._optional_add_go_to_factory_actions(game_state=game_state)

        return self._create_action_plan(actions=self.cur_actions, game_state=game_state)

    def _add_clear_initial_rubble_actions(self, game_state: GameState) -> None:
        self.cur_c = self.unit.c

        for rubble_c in self.rubble_positions:
            potential_dig_rubble_actions = self._get_rubble_actions(
                start_c=self.cur_c, rubble_c=rubble_c, board=game_state.board
            )
            potential_actions = self.cur_actions + potential_dig_rubble_actions
            potential_action_plan = self._create_action_plan(actions=potential_actions, game_state=game_state)

            if not potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
                return

            if self._can_add_factory(
                new_actions=potential_dig_rubble_actions, new_c=rubble_c, game_state=game_state
            ) or potential_action_plan.unit_can_reach_factory_after_action_plan(
                game_state=game_state, graph=self.graph
            ):
                self._extend_actions(potential_dig_rubble_actions)
                self.cur_c = rubble_c
            else:
                return

    def _get_rubble_actions(self, start_c: Coordinate, rubble_c: Coordinate, board: Board) -> list[Action]:
        pos_to_rubble_actions = get_actions_a_to_b(graph=self.graph, start=start_c, end=rubble_c)

        rubble_at_pos = board.rubble[tuple(rubble_c)]
        nr_required_digs = ceil(rubble_at_pos / self.unit.unit_cfg.DIG_RUBBLE_REMOVED)
        dig_action = [DigAction(n=nr_required_digs)]

        actions = pos_to_rubble_actions + dig_action
        return actions

    def _add_additional_rubble_actions(self, game_state: GameState) -> list[Action]:
        while True:
            closest_rubble = game_state.board.get_closest_rubble_tile(self.cur_c, exclude_c=self.rubble_positions)
            potential_dig_rubble_actions = self._get_rubble_actions(
                start_c=self.cur_c, rubble_c=closest_rubble, board=game_state.board
            )

            potential_actions = self.cur_actions + potential_dig_rubble_actions
            potential_plan = self._create_action_plan(actions=potential_actions, game_state=game_state)

            if not potential_plan.unit_can_carry_out_plan(game_state=game_state):
                return

            if self._can_add_factory(
                new_actions=potential_dig_rubble_actions, new_c=closest_rubble, game_state=game_state
            ) or potential_plan.unit_can_reach_factory_after_action_plan(game_state=game_state, graph=self.graph):

                self.cur_actions.extend(potential_dig_rubble_actions)
                self.rubble_positions.append(c=closest_rubble)
                self.cur_c = closest_rubble
            else:
                return

    def _can_add_factory(self, new_actions: list[Action], new_c: Coordinate, game_state: GameState) -> bool:
        closest_factory_c = game_state.get_closest_factory_tile(c=new_c)
        rubble_to_factory_actions = get_actions_a_to_b(self.graph, start=new_c, end=closest_factory_c)
        actions = self.cur_actions + new_actions + rubble_to_factory_actions
        action_plan = self._create_action_plan(actions=actions, game_state=game_state)

        return action_plan.unit_can_carry_out_plan(game_state=game_state)

    def _optional_add_go_to_factory_actions(self, game_state: GameState) -> None:
        closest_factory_c = game_state.get_closest_factory_tile(c=self.cur_c)
        potential_rubble_to_factory_actions = get_actions_a_to_b(self.graph, start=self.cur_c, end=closest_factory_c)
        potential_actions = self.cur_actions + potential_rubble_to_factory_actions
        potential_action_plan = self._create_action_plan(actions=potential_actions, game_state=game_state)

        if potential_action_plan.unit_can_carry_out_plan(game_state=game_state):
            self._extend_actions(potential_rubble_to_factory_actions)

    def evaluate_action_plan(self, action_plan: ActionPlan, game_state: GameState) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_used(board=game_state.board)
        number_of_rubble_cleared = len(self.rubble_positions)
        rubble_cleared_per_step = number_of_rubble_cleared / number_of_steps
        rubble_cleared_per_power = number_of_rubble_cleared / power_cost
        return rubble_cleared_per_step + rubble_cleared_per_power


@dataclass
class GoalCollection:
    goals: list[Goal]

    def generate_action_plans(self, game_state: GameState) -> None:
        for goal in self.goals:
            goal.generate_action_plans(game_state=game_state)

    @property
    def best_action_plan(self) -> ActionPlan:
        best_goal = max(self.goals)
        return best_goal.best_action_plan
