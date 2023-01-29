import numpy as np

from dataclasses import dataclass, field


@dataclass
class Action:
    action_identifier: int
    direction: int
    resource: int
    amount: int
    repeat: int
    n: int

    def to_array(self) -> np.ndarray:
        return np.array([self.action_identifier, self.direction, self.resource, self.amount, self.repeat, self.n])


@dataclass
class MoveAction(Action):
    action_identifier: int = field(default=0, init=False)
    resource: int = field(default=0, init=False)
    amount: int = field(default=0, init=False)


@dataclass
class TransferAction(Action):
    action_identifier: int = field(default=1, init=False)


@dataclass
class PickupAction(Action):
    action_identifier: int = field(default=2, init=False)
    direction: int = field(default=0, init=False)


@dataclass
class DigAction(Action):
    action_identifier: int = field(default=3, init=False)
    direction: int = field(default=0, init=False)
    resource: int = field(default=0, init=False)
    amount: int = field(default=0, init=False)


@dataclass
class DestructAction(Action):
    action_identifier: int = field(default=4, init=False)
    direction: int = field(default=0, init=False)
    resource: int = field(default=0, init=False)
    amount: int = field(default=0, init=False)


@dataclass
class RechargeAction(Action):
    action_identifier: int = field(default=5, init=False)
    direction: int = field(default=0, init=False)
    resource: int = field(default=0, init=False)
