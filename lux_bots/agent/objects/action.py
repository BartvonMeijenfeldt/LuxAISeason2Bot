from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import floor

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
    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        ...

    def get_final_pos(self, start_c: Coordinate) -> Coordinate:
        return start_c

    @classmethod
    def from_array(cls, x: np.ndarray) -> Action:
        action_identifier, direction, resource, amount, repeat, n = x
        resource = Resource(resource)

        match action_identifier:
            case 0:
                return MoveAction(direction=direction, repeat=repeat, n=n)
            case 1:
                return TransferAction(direction=direction, amount=amount, resource=resource, repeat=repeat, n=n)
            case 2:
                return PickupAction(amount=amount, resource=resource, repeat=repeat, n=n)
            case 3:
                return DigAction(repeat=repeat, n=n)
            case 4:
                return SelfDestructAction(repeat=repeat, n=n)
            case 5:
                return RechargeAction(amount=amount, repeat=repeat, n=n)
            case _:
                raise ValueError(f"Action identifier {action_identifier} is not an int between 0 and 5 (inc.)")


@dataclass
class MoveAction(Action):
    direction: Direction

    def to_array(self) -> np.ndarray:
        action_identifier = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, self.direction.number, resource, amount, self.repeat, self.n])

    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        power_change = 0
        cur_pos = start_c

        for _ in range(self.n):
            cur_pos = cur_pos + self.direction
            rubble_at_target = board.rubble[tuple(cur_pos)]
            power_required_single_action = floor(
                unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
            )
            power_change -= power_required_single_action

        return power_change

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

    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        if self.resource == Resource.Power:
            return -self.amount
        else:
            return 0


@dataclass
class PickupAction(Action):
    amount: int
    resource: Resource

    def to_array(self) -> np.ndarray:
        action_identifier = 2
        direction = 0
        return np.array([action_identifier, direction, self.resource.value, self.amount, self.repeat, self.n])

    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        if self.resource == Resource.Power:
            return self.amount
        else:
            return 0


@dataclass
class DigAction(Action):
    def to_array(self) -> np.ndarray:
        action_identifier = 3
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        return -unit.unit_cfg.DIG_COST * self.n


@dataclass
class SelfDestructAction(Action):
    def to_array(self) -> np.ndarray:
        action_identifier = 4
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        return -unit.unit_cfg.SELF_DESTRUCT_COST * self.n


@dataclass
class RechargeAction(Action):
    amount: int

    def to_array(self) -> np.ndarray:
        action_identifier = 5
        direction = 0
        resource = 0
        return np.array([action_identifier, direction, resource, self.amount, self.repeat, self.n])

    def get_power_change(self, unit: Unit, start_c: Coordinate, board: Board) -> int:
        return 0
