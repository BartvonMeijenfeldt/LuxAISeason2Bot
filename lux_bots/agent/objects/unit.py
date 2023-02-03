from dataclasses import dataclass

from objects.cargo import UnitCargo
from lux.config import UnitConfig
from objects.coordinate import Coordinate
from objects.game_state import GameState

from logic.goal import Goal, CollectIceGoal, ClearRubbleGoal


@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    c: Coordinate
    power: int
    cargo: UnitCargo
    unit_cfg: UnitConfig
    action_queue: list

    @property
    def agent_id(self):
        if self.team_id == 0:
            return "player_0"
        return "player_1"

    @property
    def has_actions_in_queue(self) -> bool:
        return len(self.action_queue) > 0

    @property
    def action_queue_cost(self):
        cost = self.unit_cfg.ACTION_QUEUE_POWER_COST
        return cost

    def generate_goals(self, game_state: GameState) -> list[Goal]:
        if game_state.env_steps <= 800:
            target_ice_c = game_state.get_closest_ice_tile(c=self.c)
            target_factory_c = game_state.get_closest_factory_tile(c=target_ice_c)
            goals = [CollectIceGoal(ice_pos=target_ice_c, factory_pos=target_factory_c)]
            for goal in goals:
                goal.generate_action_plans(unit=self, game_state=game_state)
                goal.evaluate_action_plans(unit=self, game_state=game_state)

        else:
            closest_rubble_tiles = game_state.get_n_closest_rubble_tiles(c=self.c, n=4)
            target_factory_c = game_state.get_closest_factory_tile(c=closest_rubble_tiles[-1])
            goals = [
                ClearRubbleGoal(rubble_positions=closest_rubble_tiles, factory_pos=target_factory_c)
            ]

            for goal in goals:
                goal.generate_action_plans(unit=self, game_state=game_state)
                goal.evaluate_action_plans(unit=self, game_state=game_state)

        return goals

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.c}"
        return out
