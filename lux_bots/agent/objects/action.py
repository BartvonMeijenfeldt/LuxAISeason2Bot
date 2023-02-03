import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, replace
from math import floor
from collections.abc import Iterator

from lux.config import UnitConfig
from objects.coordinate import Direction, Coordinate
from objects.board import Board


@dataclass
class Action(metaclass=ABCMeta):
    @abstractmethod
    def to_array(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        ...


@dataclass
class MoveAction(Action):
    direction: Direction
    repeat: int
    n: int

    def to_array(self) -> np.ndarray:
        action_identifier = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, self.direction.number, resource, amount, self.repeat, self.n])

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        target_c = unit_c + self.direction
        rubble_at_target = board.rubble[tuple(target_c)]
        return floor(unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)


@dataclass
class TransferAction(Action):
    direction: Direction
    resource: int
    amount: int
    repeat: int
    n: int

    def to_array(self) -> np.ndarray:
        action_identifier = 1
        return np.array([action_identifier, self.direction.number, self.resource, self.amount, self.repeat, self.n])

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        return 0


@dataclass
class PickupAction(Action):
    resource: int
    amount: int
    repeat: int
    n: int

    def to_array(self) -> np.ndarray:
        action_identifier = 2
        direction = 0
        return np.array([action_identifier, direction, self.resource, self.amount, self.repeat, self.n])

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        return 0


@dataclass
class DigAction(Action):
    repeat: int
    n: int

    def to_array(self) -> np.ndarray:
        action_identifier = 3
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        return unit_cfg.DIG_COST


@dataclass
class SelfDestructAction(Action):
    repeat: int
    n: int

    action_identifier: int = field(default=4, init=False)
    direction: int = field(default=None, init=False)
    resource: int = field(default=0, init=False)
    amount: int = field(default=0, init=False)

    def to_array(self) -> np.ndarray:
        action_identifier = 4
        direction = 0
        resource = 0
        amount = 0
        return np.array([action_identifier, direction, resource, amount, self.repeat, self.n])

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        return unit_cfg.SELF_DESTRUCT_COST


@dataclass
class RechargeAction(Action):
    amount: int
    repeat: int
    n: int

    def to_array(self) -> np.ndarray:
        action_identifier = 5
        direction = 0
        resource = 0
        return np.array([action_identifier, direction, resource, self.amount, self.repeat, self.n])

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
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

    def get_power_required(self, unit_cfg: UnitConfig, unit_c: Coordinate, board: Board) -> float:
        return sum([action.get_power_required(unit_cfg=unit_cfg, unit_c=unit_c, board=board) for action in self])

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
