from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
from collections.abc import Iterator
from objects.coordinate import TimeCoordinate


if TYPE_CHECKING:
    from objects.actors.actor import Actor
    from objects.actions.action import Action
    from objects.game_state import GameState


@dataclass
class ActionPlan(metaclass=ABCMeta):
    actor: Actor
    actions: list[Action] = field(init=False)

    @abstractmethod
    def actor_can_carry_out_plan(self, game_state: GameState) -> bool:
        ...

    @property
    @abstractmethod
    def time_coordinates(self) -> set[TimeCoordinate]:
        ...

    @abstractmethod
    def to_lux_output(self):
        ...

    @property
    def nr_time_steps(self) -> int:
        return len(self.actions)

    @property
    def power_requested(self) -> int:
        return sum(action.requested_power for action in self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)
