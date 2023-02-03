from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
from math import ceil

from search import get_actions_a_to_b, Graph, PowerCostGraph
from objects.action import Action, DigAction, TransferAction, PickupAction, ActionPlan

if TYPE_CHECKING:
    from objects.unit import Unit
    from objects.game_state import GameState
    from objects.board import Board
    from objects.coordinate import Coordinate, CoordinateList


class Goal(metaclass=ABCMeta):
    action_plans: ActionPlan

    @abstractmethod
    def generate_action_plans(self, unit: Unit, game_state: GameState) -> None:
        ...

    def evaluate_action_plans(self, unit: Unit, game_state: GameState) -> None:
        self.action_plan_evaluations = [
            self._evaluate_action_plan(unit=unit, game_state=game_state, action_plan=action_plan)
            for action_plan in self.action_plans
        ]

    @property
    def best_action_plan(self) -> ActionPlan:
        index_best_plan = self.action_plan_evaluations.index(max(self.action_plan_evaluations))
        return self.action_plans[index_best_plan]

    @abstractmethod
    def _evaluate_action_plan(self, unit: Unit, game_state: GameState) -> float:
        ...


@dataclass
class CollectIceGoal(Goal):
    ice_pos: Coordinate
    factory_pos: Coordinate
    quantity: Optional[int] = None

    def generate_action_plans(self, unit: Unit, game_state: GameState):
        self.action_plans = [self._generate_plan(unit=unit, game_state=game_state)]

    def _generate_plan(self, unit: Unit, game_state: GameState) -> ActionPlan:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        pickup_action = [PickupAction(4, 850, 0, 1)]
        pos_to_ice_actions = get_actions_a_to_b(graph=graph, start=unit.c, end=self.ice_pos)
        dig_action = [DigAction(repeat=0, n=11)]
        ice_to_factory_actions = get_actions_a_to_b(graph, start=self.ice_pos, end=self.factory_pos)
        transfer_action = [
            TransferAction(direction=ice_to_factory_actions[0].direction, resource=0, amount=100, repeat=0, n=1)
        ]
        actions = pickup_action + pos_to_ice_actions + dig_action + ice_to_factory_actions + transfer_action
        return ActionPlan(actions)

    def _evaluate_action_plan(self, unit: Unit, game_state: GameState, action_plan: ActionPlan) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_required(unit_cfg=unit.unit_cfg, unit_c=unit.c, board=game_state.board)
        return number_of_steps + 0.1 * power_cost


@dataclass
class ClearRubbleGoal(Goal):
    rubble_positions: CoordinateList
    factory_pos: Coordinate

    def generate_action_plans(self, unit: Unit, game_state: GameState):
        self.action_plans = [self._generate_plan(unit=unit, game_state=game_state)]

    def _generate_plan(self, unit: Unit, game_state: GameState) -> ActionPlan:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        pickup_action = [PickupAction(4, 1000, 0, 1)]

        rubble_actions = []

        for i in range(len(self.rubble_positions)):
            if i == 0:
                start = unit.c
            else:
                start = self.rubble_positions[i - 1]
            rubble_c = self.rubble_positions[i]

            new_actions = self._get_rubble_actions(start=start, rubble_c=rubble_c, graph=graph, board=game_state.board)
            rubble_actions += new_actions

        rubble_to_factory_actions = get_actions_a_to_b(graph, start=rubble_c, end=self.factory_pos)
        actions = pickup_action + rubble_actions + rubble_to_factory_actions
        return ActionPlan(actions)

    def _get_rubble_actions(self, start: Coordinate, rubble_c: Coordinate, graph: Graph, board: Board) -> list[Action]:
        pos_to_rubble_actions = get_actions_a_to_b(graph=graph, start=start, end=rubble_c)

        rubble_at_pos = board.rubble[tuple(rubble_c)]
        required_digs = ceil(rubble_at_pos / 20)
        dig_action = [DigAction(repeat=0, n=required_digs)]

        actions = pos_to_rubble_actions + dig_action
        return actions

    def _evaluate_action_plan(self, unit: Unit, game_state: GameState, action_plan: ActionPlan) -> float:
        number_of_steps = len(action_plan)
        power_cost = action_plan.get_power_required(unit_cfg=unit.unit_cfg, unit_c=unit.c, board=game_state.board)
        return number_of_steps + 0.1 * power_cost
