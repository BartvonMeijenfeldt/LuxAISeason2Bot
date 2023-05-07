from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, List, Optional
from dataclasses import dataclass, field

# from objects.actors.factory import Factory
from objects.coordinate import TimeCoordinate
from objects.actions.action_plan import ActionPlan, PowerRequest
from objects.actions.factory_action import FactoryAction, BuildAction

if TYPE_CHECKING:
    from objects.actors.factory import Factory
    from objects.game_state import GameState


@dataclass
class FactoryActionPlan(ActionPlan):
    actor: Factory
    actions: list[FactoryAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        # TODO allow for multiple actions
        assert len(self.actions) <= 1

    def __iter__(self) -> Iterator[FactoryAction]:
        return iter(self.actions)

    def get_resource_cost(self) -> float:
        return sum(action.get_resource_cost(self.actor) for action in self.actions)

    @property
    def next_tc(self) -> Optional[TimeCoordinate]:
        if not self.actions:
            return None

        if isinstance(self.actions[0], BuildAction):
            return TimeCoordinate(*self.actor.center_tc.xy, self.actor.center_tc.t + 1)

        return None

    def get_time_coordinates(self, game_state: GameState) -> List[TimeCoordinate]:
        return [
            TimeCoordinate(*self.actor.center_tc.xy, t)
            for t, action in enumerate(self.actions, start=self.actor.center_tc.t + 1)
            if isinstance(action, BuildAction)
        ]

    def get_power_requests(self, game_state: GameState) -> List[PowerRequest]:
        return [
            PowerRequest(self.actor, t=t, p=action.requested_power)
            for t, action in enumerate(self.actions, start=self.actor.center_tc.t)
        ]

    def to_lux_output(self):
        if not self.actions:
            return None

        return self.actions[0].to_lux_output()
