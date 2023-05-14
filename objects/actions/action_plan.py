from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from objects.coordinate import TimeCoordinate

if TYPE_CHECKING:
    from objects.actions.action import Action
    from objects.actors.actor import Actor
    from objects.actors.factory import Factory
    from objects.game_state import GameState


@dataclass
class PowerRequest:
    factory: Factory
    t: int
    p: int


@dataclass
class ActionPlan(metaclass=ABCMeta):
    actor: Actor
    actions: list[Action] = field(init=False)

    @property
    @abstractmethod
    def next_tc(self) -> Optional[TimeCoordinate]:
        ...

    @abstractmethod
    def get_time_coordinates(self, game_state: GameState) -> List[TimeCoordinate]:
        ...

    @abstractmethod
    def to_lux_output(self):
        ...

    @abstractmethod
    def get_power_requests(self, game_state: GameState) -> List[PowerRequest]:
        ...

    @property
    def nr_time_steps(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)
