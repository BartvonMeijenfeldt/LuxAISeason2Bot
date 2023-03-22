from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from collections import defaultdict

from objects.actors.actor import Actor
from objects.actors.unit import Unit
from objects.coordinate import TimeCoordinate
from objects.actions.action_plan import ActionPlan
from objects.actions.unit_action_plan import UnitActionPlan
from objects.game_state import GameState
from logic.constraints import Constraints
from logic.goals.goal import Goal


@dataclass
class Solution:
    constraints: dict[Actor, Constraints]
    goals: dict[Actor, Goal]
    action_plans: dict[Actor, ActionPlan]
    value: float

    @property
    def joint_action_plan(self) -> dict[Actor, ActionPlan]:
        return {unit: action_plan for unit, action_plan in self.action_plans.items()}

    def __lt__(self, other: Solution) -> bool:
        return self.value < other.value


@dataclass
class PowerCollision:
    actors: Sequence[Actor]
    power_deficit: int


@dataclass
class ActorCollision:
    actors: list[Actor]
    tc: TimeCoordinate

    @property
    def constraint_actor(self) -> Actor:
        return self.actors[0]

    @property
    def non_constraint_actors(self) -> list[Actor]:
        return self.actors[1:]


def get_power_collision(actor_action_plans: dict[Actor, ActionPlan], game_state: GameState):
    factories_power_available = {factory: factory.power for factory in game_state.board.player_factories}
    factories_requested_by_units = defaultdict(list)
    unit_action_plans: list[tuple[Unit, UnitActionPlan]] = [
        (unit, action_plan)
        for unit, action_plan in actor_action_plans.items()
        if isinstance(unit, Unit) and isinstance(action_plan, UnitActionPlan)
    ]

    for unit, action_plan in unit_action_plans:
        if not action_plan.primitive_actions:
            continue

        first_action = action_plan.primitive_actions[0]
        unit_power_requested = first_action.requested_power

        if not unit_power_requested:
            continue

        factory = game_state.get_closest_player_factory(unit.tc)
        factories_requested_by_units[factory].append(unit)

        if unit_power_requested > factories_power_available[factory]:
            power_deficit = unit_power_requested - factories_power_available[factory]
            return PowerCollision(actors=factories_requested_by_units[factory], power_deficit=power_deficit)

        factories_power_available[factory] -= unit_power_requested

    return None


def get_unit_collision(unit_action_plans: dict[Actor, ActionPlan]) -> Optional[ActorCollision]:
    all_time_coordinates = set()

    for action_plan in unit_action_plans.values():
        time_coordinates = set(action_plan.time_coordinates)

        collisions = all_time_coordinates & time_coordinates
        if collisions:
            collision_tc = next(iter(collisions))
            return _get_unit_collision(unit_action_plans=unit_action_plans, collision_tc=collision_tc)

        all_time_coordinates.update(time_coordinates)

    return None


def _get_unit_collision(unit_action_plans: dict[Actor, ActionPlan], collision_tc: TimeCoordinate) -> ActorCollision:
    collision_units = []
    for unit, action_plan in unit_action_plans.items():
        time_coordinates = action_plan.time_coordinates
        if collision_tc in time_coordinates:
            collision_units.append(unit)

    assert len(collision_units) >= 2

    return ActorCollision(actors=collision_units, tc=collision_tc)
