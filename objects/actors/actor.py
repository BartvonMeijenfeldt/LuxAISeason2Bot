from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from logic.goals.goal import Goal
    from objects.actions.action_plan import ActionPlan
    from objects.cargo import Cargo


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
