from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from math import floor
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from objects.actions.action import Action
from objects.direction import NUMBER_DIRECTION, Direction
from objects.resource import Resource

if TYPE_CHECKING:
    from lux.config import UnitConfig
    from objects.board import Board
    from objects.coordinate import Coordinate

    TCoordinate = TypeVar("TCoordinate", bound=Coordinate)


class UnitAction(Action):
    n: int  # Number of times to repeat the action

    def next_step_equal(self, other: UnitAction) -> bool:
        self_one_step = replace(self, n=1)
        other_one_step = replace(other, n=1)
        return self_one_step == other_one_step

    @abstractmethod
    def to_lux_output(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def unit_direction(self) -> Direction:
        """The direction the unit will move towards to carry out the action."""
        ...

    @abstractmethod
    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        """Get the power change of the unit to carry out the action. Defined as:

        power_change = power_after_action - power_before_action.

        So units that pick up power will have a positive power change, other actions will leed to non-positive power
        changes.

        Args:
            unit_cfg: Unit config.
            start_c: _Start coordinate of unit.
            board: Current board.

        Returns:
            The power change of the unit to carry out the action.
        """
        ...

    @abstractmethod
    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        """Calculates power change, like the get_power_change method. But calculates it by using the coordinate the
        unit is moving to. This method exists as a speed upgrade, when the the end coordinate is already calculated.

        Args:
            unit_cfg: Unit config.
            end_c: End coordinate of unit.
            board: Current board.

        Returns:
            The power change of the unit to carry out the action.
        """
        ...

    def get_final_c(self, start_c: TCoordinate) -> TCoordinate:
        """Get the final coordinate of the unit.

        Args:
            start_c: Start coordinate.

        Returns:
            Final coordinate of the unit.
        """
        return start_c.__add__(Direction.CENTER)

    @property
    def is_stationary(self) -> bool:
        return self.unit_direction == Direction.CENTER

    @classmethod
    def from_array(cls, x: np.ndarray) -> UnitAction:
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
class MoveAction(UnitAction):
    direction: Direction
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return self.direction

    @property
    def requested_power(self) -> int:
        return 0

    def to_lux_output(self) -> np.ndarray:
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
        return -self.get_power_cost_by_end_c(unit_cfg, end_c, board)

    def get_power_cost_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1

        if self.direction == Direction.CENTER:
            return 0

        return self.get_move_onto_cost(unit_cfg, end_c, board)

    @classmethod
    def get_move_onto_cost(cls, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        rubble_at_target = board.rubble[end_c.xy]
        power_cost = cls.get_power_cost(rubble_at_target, unit_cfg)
        return power_cost

    @staticmethod
    def get_power_cost(rubble_to: int, unit_cfg: UnitConfig) -> int:
        return floor(unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_to)

    def get_final_c(self, start_c: Coordinate) -> Coordinate:
        cur_pos = start_c

        for _ in range(self.n):
            cur_pos = cur_pos + self.direction

        return cur_pos


@dataclass
class TransferAction(UnitAction):
    direction: Direction
    amount: int
    resource: Resource
    repeat: int = 0
    n: int = 1

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"amount is {self.amount}, amount can not be negative")

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def requested_power(self) -> int:
        return 0

    def to_lux_output(self) -> np.ndarray:
        action_identifier = 1
        return np.array(
            [action_identifier, self.direction.number, self.resource.value, self.amount, self.repeat, self.n]
        )

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        if self.resource == Resource.POWER:
            return -self.amount * self.n
        else:
            return 0

    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1

        if self.resource == Resource.POWER:
            return -self.amount
        else:
            return 0


@dataclass
class PickupAction(UnitAction):
    amount: int
    resource: Resource
    repeat: int = 0
    n: int = 1

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"amount is {self.amount}, amount can not be negative")

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def requested_power(self) -> int:
        if self.resource == Resource.POWER:
            return self.amount * self.n

        return 0

    def to_lux_output(self) -> np.ndarray:
        action_identifier = 2
        direction = 0
        return np.array([action_identifier, direction, self.resource.value, self.amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        if self.resource == Resource.POWER:
            return self.amount * self.n
        else:
            return 0

    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1

        if self.resource == Resource.POWER:
            return self.amount
        else:
            return 0


@dataclass
class DigAction(UnitAction):
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def requested_power(self) -> int:
        return 0

    def to_lux_output(self) -> np.ndarray:
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
class SelfDestructAction(UnitAction):
    repeat: int = 0
    n: int = 1

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    @property
    def requested_power(self) -> int:
        return 0

    def to_lux_output(self) -> np.ndarray:
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
class RechargeAction(UnitAction):
    amount: int
    repeat: int = 0
    n: int = 1

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"amount is {self.amount}, amount can not be negative")

    @property
    def requested_power(self) -> int:
        return 0

    @property
    def unit_direction(self) -> Direction:
        return Direction.CENTER

    def to_lux_output(self) -> np.ndarray:
        action_identifier = 5
        direction = 0
        resource = 0
        return np.array([action_identifier, direction, resource, self.amount, self.repeat, self.n])

    def get_power_change(self, unit_cfg: UnitConfig, start_c: Coordinate, board: Board) -> int:
        return 0

    def get_power_change_by_end_c(self, unit_cfg: UnitConfig, end_c: Coordinate, board: Board) -> int:
        assert self.n == 1
        return 0
