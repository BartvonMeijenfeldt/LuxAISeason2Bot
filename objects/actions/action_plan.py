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
    """Actions that the unit is planning to carry out. These actions are a private plan and not necessarily communicated
    with the environment yet. A reason to not fully communicate the plan can either be to safe power (every update of
    the action queue cost power) or to hide the intentino from the opponent.

    Args:
        actor: Actor.
        actions: List of actions the actor is planning to carry out.
    """

    actor: Actor
    actions: list[Action] = field(init=False)

    @property
    @abstractmethod
    def next_tc(self) -> Optional[TimeCoordinate]:
        """Time coordinate the actor will be at next step, given the action plan."""
        ...

    @abstractmethod
    def get_time_coordinates(self, game_state: GameState) -> List[TimeCoordinate]:
        """Get a list of all time coordinates the actor will be at given its action plan.

        Args:
            game_state: Current game state

        Returns:
            List of all time coordinates the actor will be at given its action plan.
        """
        ...

    @abstractmethod
    def to_lux_output(self):
        """Convert the action plan to the output the Lux environment expects."""
        ...

    @abstractmethod
    def get_power_requests(self, game_state: GameState) -> List[PowerRequest]:
        """Get the power requests the unit will make to carry out its action plan.

        Args:
            game_state: Current game state.

        Returns:
            The power requests the unit will make to carry out its action plan.
        """
        ...

    @property
    def nr_time_steps(self) -> int:
        """The number of time steps needed to complete the action plan"""
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)
