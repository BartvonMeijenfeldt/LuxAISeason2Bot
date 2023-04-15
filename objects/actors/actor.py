from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass, field


if TYPE_CHECKING:
    from logic.goals.goal import Goal
    from objects.cargo import Cargo
    from objects.actions.action_plan import ActionPlan


@dataclass
class Actor:
    team_id: int
    unit_id: str
    power: int
    cargo: Cargo
    goal: Optional[Goal] = field(init=False, default=None)
    private_action_plan: ActionPlan = field(init=False)

    def __post_init__(self) -> None:
        self.id = int(self.unit_id.split("_")[1])

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, __o: Actor) -> bool:
        return self.id == __o.id

    def schedule_goal(self, goal: Goal) -> None:
        self.goal = goal
        self.private_action_plan = goal.action_plan
        self.is_scheduled = True
