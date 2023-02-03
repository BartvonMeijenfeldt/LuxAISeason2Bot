from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional
from math import ceil

from objects.game_state import GameState
from objects.action import Action, DigAction, TransferAction, PickupAction
from objects.coordinate import Coordinate
from search import get_plan_a_to_b, PowerCostGraph


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
    rubble_pos: Coordinate
    factory_pos: Coordinate

    def generate_action_plans(self, game_state) -> list[list[Action]]:
        return [self.generate_plan(game_state=game_state)]

    def generate_plan(self, game_state: GameState) -> list[Action]:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        pickup_action = [PickupAction(4, 1000, 0, 1)]
        pos_to_ice_actions = get_plan_a_to_b(graph=graph, start=self.unit_pos, end=self.rubble_pos)

        rubble_at_pos = game_state.board.rubble[tuple(self.rubble_pos)]
        required_digs = ceil(rubble_at_pos / 20)
        dig_action = [DigAction(repeat=0, n=required_digs)]
        ice_to_factory_actions = get_plan_a_to_b(graph, start=self.rubble_pos, end=self.factory_pos)
        actions = pickup_action + pos_to_ice_actions + dig_action + ice_to_factory_actions
        return [a.to_array() for a in actions]
