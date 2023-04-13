from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field


if TYPE_CHECKING:
    from logic.goals.goal import Goal
    from objects.cargo import Cargo
    from objects.actions.action_plan import ActionPlan


@dataclass
class Actor(metaclass=ABCMeta):
    team_id: int
    unit_id: str
    power: int
    cargo: Cargo
    goal: Optional[Goal] = field(init=False, default=None)
    private_action_plan: Optional[ActionPlan] = field(init=False, default=None)

    def set_goal(self, goal: Goal) -> None:
        self.goal = goal
