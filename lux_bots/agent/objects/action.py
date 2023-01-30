import numpy as np

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from agent.objects.coordinate import Direction


class Action(metaclass=ABCMeta):
    @abstractmethod
    def to_array(self) -> np.ndarray:
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


@dataclass
class DestructAction(Action):
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
