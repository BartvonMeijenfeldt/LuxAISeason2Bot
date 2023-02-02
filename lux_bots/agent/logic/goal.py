from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from agent.objects.game_state import GameState
from agent.objects.action import Action, MoveAction, DigAction
from agent.objects.coordinate import Coordinate, Direction
from agent.search import get_plan_a_to_b, PowerCostGraph


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
    quantity: Optional[int] = None

    def generate_action_plans(self, game_state) -> list[list[Action]]:
        return [self.generate_plan(game_state=game_state)]

    def generate_plan(self, game_state: GameState) -> list[Action]:
        graph = PowerCostGraph(game_state.board, time_to_power_cost=20)
        move_actions = get_plan_a_to_b(graph=graph, start=self.unit_pos, end=self.ice_pos)
        dig_action = [DigAction(repeat=0, n=1)]
        actions = move_actions + dig_action 
        return [a.to_array() for a in actions]

    def get_x_actions(self, dx: int) -> list[Action]:
        if dx == 0:
            return []

        direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        nr_actions = abs(dx)
        return [MoveAction(direction=direction, repeat=0, n=1) for _ in range(nr_actions)]

    def get_y_actions(self, dy: int) -> list[Action]:
        if dy == 0:
            return []

        direction = Direction.DOWN if dy > 0 else Direction.UP
        nr_actions = abs(dy)
        return [MoveAction(direction=direction, repeat=0, n=1) for _ in range(abs(nr_actions))]
