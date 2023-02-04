from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
from math import floor, ceil

from search import get_actions_a_to_b, Graph, PowerCostGraph
from objects.action import Action, DigAction, MoveAction, TransferAction, PickupAction, ActionPlan
from objects.coordinate import Direction
from objects.resource import Resource

if TYPE_CHECKING:
    from objects.unit import Unit
    from objects.game_state import GameState
    from objects.board import Board
    from objects.coordinate import Coordinate, CoordinateList


class Goal(metaclass=ABCMeta):
    action_plans: list[ActionPlan]

    @abstractmethod
    def generate_action_plans(self, unit: Unit, game_state: GameState) -> None:
        ...

    def evaluate_action_plans(self, unit: Unit, game_state: GameState) -> None:
        for action_plan in self.action_plans:
            value_action_plan = self._evaluate_action_plan(unit=unit, game_state=game_state, action_plan=action_plan)
            action_plan.value = value_action_plan

    @property
    def best_action_plan(self) -> ActionPlan:
        return max(self.action_plans)

    @property
    def __eq__(self, other: "Goal") -> bool:
        self.best_action_plan < other.best_action_plan

    @abstractmethod
    def _evaluate_action_plan(self, unit: Unit, game_state: GameState) -> float:
        ...


@dataclass
class CollectIceGoal(Goal):
    ice_c: Coordinate
    factory_pos: Coordinate
    quantity: Optional[int] = None

    def generate_action_plans(self, unit: Unit, game_state: GameState):
        self.action_plans = [self._generate_plan(unit=unit, game_state=game_state)]

    def _generate_plan(self, unit: Unit, game_state: GameState) -> ActionPlan:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)

        power_pickup_action = self._get_power_pickup_action(unit=unit, game_state=game_state)
        pos_to_ice_actions = self._get_pos_to_ice_actions(unit=unit, graph=graph)
        ice_to_factory_actions = self._get_ice_to_factory_actions(graph=graph)

        dig_action = self._get_dig_action(
            unit=unit,
            game_state=game_state,
            power_pickup=power_pickup_action,
            move_actions=pos_to_ice_actions + ice_to_factory_actions,
        )

        transfer_action = TransferAction(direction=Direction.CENTER, amount=3000, resource=Resource.Ice, repeat=0, n=1)

        actions = [power_pickup_action] + pos_to_ice_actions + [dig_action] + ice_to_factory_actions + [transfer_action]
        return ActionPlan(actions)

    def _get_power_pickup_action(self, unit: Unit, game_state: GameState) -> PickupAction:
        power_space_left = unit.power_space_left
        power_in_factory = game_state.get_closest_factory(c=unit.c).power
        cargo_to_pickup = min(power_space_left, power_in_factory)

        return PickupAction(cargo_to_pickup, Resource.Power)

    def _get_pos_to_ice_actions(self, unit: Unit, graph: Graph) -> list[MoveAction]:
        return get_actions_a_to_b(graph=graph, start=unit.c, end=self.ice_c)

    def _get_ice_to_factory_actions(self, graph: Graph) -> list[MoveAction]:
        return get_actions_a_to_b(graph=graph, start=self.ice_c, end=self.factory_pos)

    def _get_dig_action(
        self, unit: Unit, game_state: GameState, power_pickup: PickupAction, move_actions: list[MoveAction]
    ) -> DigAction:
        power_after_pickup = unit.power + power_pickup.amount
        power_required_moving = sum(
            [move.get_power_required(unit=unit, start_c=unit.c, board=game_state.board) for move in move_actions]
        )

        # TODO adjust for charging on the way
        power_left_for_digging = power_after_pickup - power_required_moving
        n_digging = floor(power_left_for_digging / unit.unit_cfg.DIG_COST)
        n_digging = min(n_digging, ceil(unit.cargo_space_left / unit.unit_cfg.DIG_RESOURCE_GAIN))
        return DigAction(repeat=0, n=n_digging)

    def _evaluate_action_plan(self, unit: Unit, game_state: GameState, action_plan: ActionPlan) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_required(unit=unit, board=game_state.board)
        return number_of_steps + 0.1 * power_cost


@dataclass
class ClearRubbleGoal(Goal):
    rubble_positions: CoordinateList
    factory_pos: Coordinate

    def generate_action_plans(self, unit: Unit, game_state: GameState):
        self.action_plans = [self._generate_plan(unit=unit, game_state=game_state)]

    def _generate_plan(self, unit: Unit, game_state: GameState) -> ActionPlan:
        self.graph = PowerCostGraph(game_state.board, time_to_power_cost=20)

        pickup_action = self._get_power_pickup_action(unit=unit, game_state=game_state)
        rubble_actions = self._clear_initial_rubble_actions(unit=unit, game_state=game_state)

        self.cur_actions = [pickup_action] + rubble_actions

        self._add_additional_rubble_actions(unit=unit, pickup_power=pickup_action.amount, game_state=game_state)
        rubble_to_factory_actions = get_actions_a_to_b(self.graph, start=self.cur_c, end=self.factory_pos)

        # rubble_to_factory_actions = get_actions_a_to_b(graph, start=rubble_c, end=self.factory_pos)
        actions = self.cur_actions + rubble_to_factory_actions
        return ActionPlan(actions)

    def _get_power_pickup_action(self, unit: Unit, game_state: GameState) -> PickupAction:
        power_space_left = unit.power_space_left
        power_in_factory = game_state.get_closest_factory(c=unit.c).power
        cargo_to_pickup = min(power_space_left, power_in_factory)

        return PickupAction(cargo_to_pickup, Resource.Power)

    def _clear_initial_rubble_actions(self, unit: Unit, game_state: GameState) -> list[Action]:
        rubble_actions = []
        starts = [unit.c] + self.rubble_positions[:-1]

        for start_c, rubble_c in zip(starts, self.rubble_positions):
            new_actions = self._get_rubble_actions(
                unit=unit, start_c=start_c, rubble_c=rubble_c, board=game_state.board
            )
            rubble_actions += new_actions

        self.cur_c = rubble_c
        return rubble_actions

    def _get_rubble_actions(self, unit: Unit, start_c: Coordinate, rubble_c: Coordinate, board: Board) -> list[Action]:
        pos_to_rubble_actions = get_actions_a_to_b(graph=self.graph, start=start_c, end=rubble_c)

        rubble_at_pos = board.rubble[tuple(rubble_c)]
        nr_required_digs = ceil(rubble_at_pos / unit.unit_cfg.DIG_RUBBLE_REMOVED)
        dig_action = [DigAction(n=nr_required_digs)]

        actions = pos_to_rubble_actions + dig_action
        return actions

    def _add_additional_rubble_actions(self, unit: Unit, pickup_power: int, game_state: GameState) -> list[Action]:
        power_after_pickup = unit.power + pickup_power

        while True:
            closest_rubble = game_state.board.get_closest_rubble_tile(self.cur_c, exclude_c=self.rubble_positions)
            dig_rubble_actions = self._get_rubble_actions(
                unit=unit, start_c=self.cur_c, rubble_c=closest_rubble, board=game_state.board
            )

            if self._can_still_go_back(
                unit=unit,
                new_actions=dig_rubble_actions,
                new_c=closest_rubble,
                game_state=game_state,
                power_available=power_after_pickup,
            ):
                self.cur_actions += dig_rubble_actions
                self.rubble_positions.append(c=closest_rubble)
                self.cur_c = closest_rubble
            else:
                return

    def _can_still_go_back(
        self, unit: Unit, new_actions: list[Action], new_c: Coordinate, game_state: GameState, power_available: int
    ) -> bool:
        rubble_to_factory_actions = get_actions_a_to_b(self.graph, start=new_c, end=self.factory_pos)
        action_plan = ActionPlan(self.cur_actions + new_actions + rubble_to_factory_actions)
        if not action_plan.is_valid:
            return False

        total_power_required = action_plan.get_power_required(unit=unit, board=game_state.board)
        return total_power_required <= power_available

    def _evaluate_action_plan(self, unit: Unit, game_state: GameState, action_plan: ActionPlan) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_required(unit=unit, board=game_state.board)
        return number_of_steps + 0.1 * power_cost


@dataclass
class GoalCollection:
    goals: list[Goal]

    def generate_action_plans(self, unit: Unit, game_state: GameState) -> None:
        for goal in self.goals:
            goal.generate_action_plans(unit=unit, game_state=game_state)
            goal.evaluate_action_plans(unit=unit, game_state=game_state)

    @property
    def best_action_plan(self) -> ActionPlan:
        best_goal = max(self.goals)
        return best_goal.best_action_plan
