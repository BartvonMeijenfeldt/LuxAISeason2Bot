from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import floor

from objects.resource import Resource
from objects.direction import Direction, NUMBER_DIRECTION

if TYPE_CHECKING:
    from objects.coordinate import Coordinate
    from objects.unit import UnitConfig
    from objects.board import Board
    TCoordinate = TypeVar('TCoordinate', bound=Coordinate)


@dataclass
class Action(metaclass=ABCMeta):

    @property
    @abstractmethod
    def unit_direction(self) -> Direction:
        ...

    @property
    @abstractmethod
    def power_requested(self) -> int:
        ...

    @abstractmethod
    def to_array(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        ...

    @abstractmethod
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        ...

    def get_final_c(self, start_c: TCoordinate) -> TCoordinate:
        return start_c + Direction.CENTER

    @classmethod
    def from_array(cls, x: np.ndarray) -> Action:
        action_identifier, direction, resource, amount, repeat, n = x
        direction = NUMBER_DIRECTION[direction]
        resource = Resource(resource)

        if action_identifier == 0:
            return MoveAction(direction=direction, repeat=repeat, n=n)
        elif action_identifier == 1:
            return TransferAction(direction=direction, amount=amount, resource=resource, repeat=repeat, n=n)
        elif action_identifier == 2:
            return PickupAction(amount=amount, resource=resource, repeat=repeat, n=n)
        elif action_identifier == 3:
            return DigAction(repeat=repeat, n=n)
        elif action_identifier == 4:
            return SelfDestructAction(repeat=repeat, n=n)
        elif action_identifier == 5:
            return RechargeAction(amount=amount, repeat=repeat, n=n)
        else:
            raise ValueError(f"Action identifier {action_identifier} is not an int between 0 and 5 (inc.)")


@dataclass
class MoveAction(Action):
    direction: Direction
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return self.direction

    @property
    def power_requested(self) -> int:
        return 0

    def to_array(self) -> np.ndarray:
        action_identifier = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, self.direction.number, resource, amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        if self.direction == Direction.CENTER:
            return 0

        power_change = 0
        cur_pos = start_c

        for _ in range(self.n):
            cur_pos = cur_pos + self.direction
            rubble_at_target = board.rubble[cur_pos.xy]
            power_required_single_action = self.get_power_cost(rubble_at_target, unit_cfg)
            power_change -= power_required_single_action

        return power_change
    
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1
    
        if self.direction == Direction.CENTER:
            return 0
        
        rubble_at_target = board.rubble[end_c.xy]
        return self.get_power_cost(rubble_at_target, unit_cfg)


    @staticmethod
    def get_power_cost(rubble_to: int, unit_cfg: UnitConfig) -> int:
        return floor(unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_to)

    def get_final_c(self, start_c: Coordinate) -> Coordinate:
        cur_pos = start_c

        for _ in range(self.n):
            cur_pos = cur_pos + self.direction

        return cur_pos


@dataclass
class TransferAction(Action):
    direction: Direction
    amount: int
    resource: Resource
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def power_requested(self) -> int:
        return 0

    def to_array(self) -> np.ndarray:
        action_identifier = 1
        return np.array(
            [action_identifier, self.direction.number, self.resource.value, self.amount, self.repeat, self.n]
        )

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        if self.resource == Resource.Power:
            return -self.amount * self.n
        else:
            return 0
        
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1

        if self.resource == Resource.Power:
            return -self.amount
        else:
            return 0


@dataclass
class PickupAction(Action):
    amount: int
    resource: Resource
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def power_requested(self) -> int:
        if self.resource == Resource.Power:
            return self.amount

        return 0

    def to_array(self) -> np.ndarray:
        action_identifier = 2
        direction = 0
        return np.array([action_identifier, direction, self.resource.value, self.amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        if self.resource == Resource.Power:
            return self.amount  * self.n
        else:
            return 0
        
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1

        if self.resource == Resource.Power:
            return self.amount
        else:
            return 0       


@dataclass
class DigAction(Action):
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def power_requested(self) -> int:
        return 0

    def to_array(self) -> np.ndarray:
        action_identifier = 3
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        return -unit_cfg.DIG_COST * self.n
    
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1

        return -unit_cfg.DIG_COST


@dataclass
class SelfDestructAction(Action):
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def power_requested(self) -> int:
        return 0

    def to_array(self) -> np.ndarray:
        action_identifier = 4
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        return -unit_cfg.SELF_DESTRUCT_COST * self.n

    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1
        return -unit_cfg.SELF_DESTRUCT_COST


@dataclass
class RechargeAction(Action):
    amount: int
    repeat: int = 0
    n: int = 1

    @property
    def power_requested(self) -> int:
        return 0

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    def to_array(self) -> np.ndarray:
        action_identifier = 5
        direction = 0
        resource = 0
        return np.array([action_identifier, direction, resource, self.amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        return 0
    
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1
        return 0
