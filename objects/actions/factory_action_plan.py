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

    def actor_can_carry_out_plan(self, game_state: GameState) -> bool:
        return self.actor_has_enough_resources()

    def actor_has_enough_resources(self) -> bool:
        return self.actor_has_enough_power and self.actor_has_enough_metal and self.actor_has_enough_water()

    @property
    def actor_has_enough_power(self) -> bool:
        power_requested = sum(action.requested_power for action in self.actions)
        return power_requested <= self.actor.power

    @property
    def actor_has_enough_metal(self) -> bool:
        metal_requested = sum(action.quantity_metal_cost for action in self.actions)
        return metal_requested <= self.actor.cargo.metal

    def actor_has_enough_water(self) -> bool:
        power_requested = sum(action.get_water_cost(factory=self.actor) for action in self.actions)
        return power_requested <= self.actor.cargo.water

    @property
    def next_tc(self) -> Optional[TimeCoordinate]:
        if not self.actions:
            return None

        if isinstance(self.actions[0], BuildAction):
            return TimeCoordinate(*self.actor.center_tc.xy, self.actor.center_tc.t + 1)

        return None

    @property
    def time_coordinates(self) -> List[TimeCoordinate]:
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
