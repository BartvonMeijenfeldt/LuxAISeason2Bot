from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Set, TYPE_CHECKING, Iterable, Optional
from dataclasses import dataclass, field


from utils import PriorityQueue

if TYPE_CHECKING:
    from objects.cargo import Cargo
    from objects.actions.action_plan import ActionPlan
    from logic.goal_resolution.power_availabilty_tracker import PowerTracker
    from logic.goals.goal import Goal, GoalCollection
    from logic.constraints import Constraints
    from lux.kit import GameState


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
