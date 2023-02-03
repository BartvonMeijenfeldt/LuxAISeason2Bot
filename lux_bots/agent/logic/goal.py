from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
from math import ceil

from objects.game_state import GameState
from objects.board import Board
from objects.action import Action, DigAction, TransferAction, PickupAction
from objects.coordinate import Coordinate, CoordinateList
from search import get_plan_a_to_b, Graph, PowerCostGraph


class Goal(metaclass=ABCMeta):
    @abstractmethod
    def generate_action_plans(self, game_state) -> list[list[Action]]:
        ...

    @abstractmethod
    def generate_plan(self, game_state) -> list[Action]:
        ...


@dataclass
class CollectIceGoal(Goal):
    unit_pos: Coordinate
    ice_pos: Coordinate
    factory_pos: Coordinate
    quantity: Optional[int] = None

    def generate_action_plans(self, game_state) -> list[list[Action]]:
        return [self.generate_plan(game_state=game_state)]

    def generate_plan(self, game_state: GameState) -> list[Action]:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        pickup_action = [PickupAction(4, 250, 0, 1)]
        pos_to_ice_actions = get_plan_a_to_b(graph=graph, start=self.unit_pos, end=self.ice_pos)
        dig_action = [DigAction(repeat=0, n=4)]
        ice_to_factory_actions = get_plan_a_to_b(graph, start=self.ice_pos, end=self.factory_pos)
        transfer_action = [
            TransferAction(direction=ice_to_factory_actions[0].direction, resource=0, amount=100, repeat=0, n=1)
        ]
        actions = pickup_action + pos_to_ice_actions + dig_action + ice_to_factory_actions + transfer_action
        return [a.to_array() for a in actions]


@dataclass
class ClearRubbleGoal(Goal):
    unit_pos: Coordinate
    rubble_positions: CoordinateList
    factory_pos: Coordinate

    def generate_action_plans(self, game_state) -> list[list[Action]]:
        return [self.generate_plan(game_state=game_state)]

    def generate_plan(self, game_state: GameState) -> list[Action]:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        pickup_action = [PickupAction(4, 1000, 0, 1)]

        rubble_actions = []

        for i in range(len(self.rubble_positions)):
            if i == 0:
                start = self.unit_pos
            else:
                start = self.rubble_positions[i - 1]
            rubble_pos = self.rubble_positions[i]

            new_actions = self._get_rubble_actions(
                start=start, rubble_pos=rubble_pos, graph=graph, board=game_state.board
            )
            rubble_actions += new_actions

        rubble_to_factory_actions = get_plan_a_to_b(graph, start=rubble_pos, end=self.factory_pos)
        actions = pickup_action + rubble_actions + rubble_to_factory_actions
        return [a.to_array() for a in actions]

    def _get_rubble_actions(
        self, start: Coordinate, rubble_pos: Coordinate, graph: Graph, board: Board
    ) -> list[Action]:
        pos_to_rubble_actions = get_plan_a_to_b(graph=graph, start=start, end=rubble_pos)

        rubble_at_pos = board.rubble[tuple(rubble_pos)]
        required_digs = ceil(rubble_at_pos / 20)
        dig_action = [DigAction(repeat=0, n=required_digs)]
        return pos_to_rubble_actions + dig_action
