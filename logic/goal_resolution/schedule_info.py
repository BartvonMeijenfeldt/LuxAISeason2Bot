from dataclasses import dataclass


from objects.game_state import GameState
from logic.constraints import Constraints
from logic.goal_resolution.power_tracker import PowerTracker


@dataclass
class ScheduleInfo:
    game_state: GameState
    constraints: Constraints
    power_tracker: PowerTracker
