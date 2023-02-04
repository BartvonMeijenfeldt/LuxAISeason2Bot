from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import floor
from collections.abc import Iterator
from objects.resource import Resource

from objects.coordinate import Coordinate

if TYPE_CHECKING:
    from objects.unit import Unit
    from objects.coordinate import Direction
    from objects.board import Board


@dataclass(kw_only=True)
class Action(metaclass=ABCMeta):
    repeat: int = 0
    n: int = 1

    @abstractmethod
    def to_array(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        ...

    def get_final_pos(self, start_c: Coordinate) -> Coordinate:
        return start_c


@dataclass
class MoveAction(Action):
    direction: Direction

    def to_array(self) -> np.ndarray:
        action_identifier = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, self.direction.number, resource, amount, self.repeat, self.n])

    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        power_required = 0
        cur_pos = start_c

        for _ in range(self.n):
            cur_pos = cur_pos + self.direction
            rubble_at_target = board.rubble[tuple(cur_pos)]
            power_required_single_action = floor(
                unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
            )
            power_required += power_required_single_action

        return power_required

    def get_final_pos(self, start_c: Coordinate) -> Coordinate:
        cur_pos = start_c

        for _ in range(self.n):
            cur_pos = cur_pos + self.direction

        return cur_pos


@dataclass
class TransferAction(Action):
    direction: Direction
    amount: int
    resource: Resource

    def to_array(self) -> np.ndarray:
        action_identifier = 1
        return np.array(
            [action_identifier, self.direction.number, self.resource.value, self.amount, self.repeat, self.n]
        )

    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        return 0


@dataclass
class PickupAction(Action):
    amount: int
    resource: Resource

    def to_array(self) -> np.ndarray:
        action_identifier = 2
        direction = 0
        return np.array([action_identifier, direction, self.resource.value, self.amount, self.repeat, self.n])

    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        return 0


@dataclass
class DigAction(Action):
    def to_array(self) -> np.ndarray:
        action_identifier = 3
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        return unit.unit_cfg.DIG_COST * self.n


@dataclass
class SelfDestructAction(Action):
    def to_array(self) -> np.ndarray:
        action_identifier = 4
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        return unit.unit_cfg.SELF_DESTRUCT_COST * self.n


@dataclass
class RechargeAction(Action):
    amount: int

    def to_array(self) -> np.ndarray:
        action_identifier = 5
        direction = 0
        resource = 0
        return np.array([action_identifier, direction, resource, self.amount, self.repeat, self.n])

    def get_power_required(self, unit: Unit, start_c: Coordinate, board: Board) -> float:
        return 0


class ActionPlan:
    def __init__(self, actions: list[Action]) -> None:
        self.base_actions = actions
        self.actions = self.get_condensed_action_plan(self.base_actions)
        self.value: float = None

    def get_condensed_action_plan(self, actions: list[Action]) -> list[Action]:
        condensed_actions = []

        for i, action in enumerate(actions):
            if i == 0:
                self._init_current_action(action=action)
                continue

            if action == self.cur_action:
                self.repeat_count += action.n
            else:
                condensed_action = self._get_condensed_action()
                condensed_actions.append(condensed_action)

                self._init_current_action(action=action)

        condensed_action = self._get_condensed_action()
        condensed_actions.append(condensed_action)

        return condensed_actions

    def get_power_required(self, unit: Unit, board: Board) -> float:
        cur_c = unit.c
        total_power = 0

        for action in self:
            power_action = action.get_power_required(unit=unit, start_c=cur_c, board=board)
            total_power += power_action
            cur_c = action.get_final_pos(start_c=cur_c)

        return total_power

    def _init_current_action(self, action: Action) -> None:
        self.cur_action: Action = action
        self.repeat_count: int = action.n

    def _get_condensed_action(self) -> Action:
        condensed_action = replace(self.cur_action)
        condensed_action.n = self.repeat_count
        return condensed_action

    def to_action_arrays(self) -> list[np.array]:
        return [action.to_array() for action in self.actions]

    def __lt__(self, other: "ActionPlan") -> bool:
        self.value < other.value

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def is_valid(self) -> bool:
        return len(self) <= 20
