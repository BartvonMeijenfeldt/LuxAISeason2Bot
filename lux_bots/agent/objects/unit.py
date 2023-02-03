import math
from dataclasses import dataclass

from objects.action import MoveAction, TransferAction, PickupAction, DigAction, DestructAction, RechargeAction
from objects.cargo import UnitCargo
from lux.config import UnitConfig
from objects.coordinate import Coordinate, Direction
from objects.game_state import GameState

from logic.goal import Goal, CollectIceGoal, ClearRubbleGoal


@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    pos: Coordinate
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
            target_ice_c = game_state.get_closest_ice_tile(c=self.pos)
            target_factory_c = game_state.get_closest_factory_tile(c=target_ice_c)
            goals = [CollectIceGoal(unit_pos=self.pos, ice_pos=target_ice_c, factory_pos=target_factory_c)]
        else:
            closest_rubble_c = game_state.get_closest_rubble_tile(c=self.pos)
            target_factory_c = game_state.get_closest_factory_tile(c=closest_rubble_c)
            goals = [ClearRubbleGoal(unit_pos=self.pos, rubble_pos=closest_rubble_c, factory_pos=target_factory_c)]

        return goals

    def move_cost(self, game_state: GameState, direction: Direction):
        board = game_state.board
        target_pos: Coordinate = self.pos + direction.value
        if board.is_off_the_board(c=target_pos):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos.x, target_pos.y]
        if factory_there not in game_state.player_team.factory_strains and factory_there != -1:
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return None
        rubble_at_target = board.rubble[target_pos.x][target_pos.y]

        return math.floor(self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)

    def move(self, direction, repeat=0, n=1):
        assert isinstance(direction, Direction)
        return MoveAction(direction=direction, repeat=repeat, n=n).to_array()

    def transfer(self, transfer_direction: Direction, transfer_resource, transfer_amount, repeat=0, n=1):
        assert transfer_resource < 5 and transfer_resource >= 0
        return TransferAction(
            direction=transfer_direction, resource=transfer_resource, amount=transfer_amount, repeat=repeat, n=n
        ).to_array()

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        assert pickup_resource < 5 and pickup_resource >= 0
        return PickupAction(
            action_identifier=2, resource=pickup_resource, amount=pickup_amount, repeat=repeat, n=n
        ).to_array()

    @property
    def dig_cost(self):
        return self.unit_cfg.DIG_COST

    def dig(self, repeat=0, n=1):
        return DigAction(repeat=repeat, n=n).to_array()

    @property
    def self_destruct_cost(self):
        return self.unit_cfg.SELF_DESTRUCT_COST

    def self_destruct(self, repeat=0, n=1):
        return DestructAction(repeat=repeat, n=n).to_array()

    def recharge(self, charge_amount, repeat=0, n=1):
        return RechargeAction(amount=charge_amount, repeat=repeat, n=n).to_array()

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out
