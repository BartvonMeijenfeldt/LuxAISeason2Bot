from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from collections.abc import Iterator
from objects.coordinate import TimeCoordinate


if TYPE_CHECKING:
    from objects.actors.actor import Actor
    from objects.actions.action import Action
    from objects.game_state import GameState


@dataclass
class ActionPlan(metaclass=ABCMeta):
    actor: Actor

    @property
    @abstractmethod
    def actions(self) -> list[Action]:
        ...

    @abstractmethod
    def actor_can_carry_out_plan(self, game_state: GameState) -> bool:
        ...

    @property
    @abstractmethod
    def time_coordinates(self) -> set[TimeCoordinate]:
        ...

    def to_lux_output(self):
        return [action.to_lux_output() for action in self.actions]

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

