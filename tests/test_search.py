import unittest
from typing import Optional, Sequence

from logic.constraints import Constraints
from logic.goal_resolution.power_tracker import PowerTracker
from lux.config import EnvConfig
from lux.kit import GameState
from objects.actions.unit_action import DigAction as DA
from objects.actions.unit_action import MoveAction as MA
from objects.actions.unit_action import PickupAction as PA
from objects.actions.unit_action import TransferAction as TA
from objects.actions.unit_action import UnitAction
from objects.coordinate import Coordinate as C
from objects.coordinate import DigCoordinate as DC
from objects.coordinate import DigTimeCoordinate as DTC
from objects.coordinate import PowerTimeCoordinate as PTC
from objects.coordinate import ResourcePowerTimeCoordinate as RPTC
from objects.coordinate import ResourceTimeCoordinate as RTC
from objects.coordinate import TimeCoordinate as TC
from objects.direction import Direction as D
from objects.resource import Resource
from search.search import (
    DigAtGraph,
    MoveToGraph,
    PickupPowerGraph,
    Search,
    TransferToFactoryResourceGraph,
)
from tests.generate_game_state import FactoryPos, FactoryPositions
from tests.generate_game_state import RubbleTile as RT
from tests.generate_game_state import Tiles, get_state
from tests.init_constraints import init_constraints

ENV_CFG = EnvConfig()
LIGHT_CFG = ENV_CFG.LIGHT_ROBOT


class TestMoveToSearch(unittest.TestCase):
    def _test_move_to_search(
        self,
        state: GameState,
        start: TC,
        goal: C,
        expected_actions: Sequence[UnitAction],
        time_to_power_cost: float = 50,
        unit_type: str = "LIGHT",
        constraints: Optional[Constraints] = None,
    ):
        if constraints is None:
            constraints = Constraints()

        expected_actions = list(expected_actions)

        unit_cfg = ENV_CFG.get_unit_config(unit_type)

        move_to_graph = MoveToGraph(
            unit_type=unit_type,
            board=state.board,
            time_to_power_cost=time_to_power_cost,
            unit_cfg=unit_cfg,
            constraints=constraints,
            goal=goal,
        )
        search = Search(move_to_graph)
        actions = search.get_actions_to_complete_goal(start=start)
        self.assertEqual(actions, expected_actions)

    def test_already_there_path(self):
        start = TC(3, 3, 0)
        goal = C(3, 3)
        state = get_state()
        expected_actions = []

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_one_down_path(self):

        start = TC(3, 2, 0)
        goal = C(3, 3)
        state = get_state()
        expected_actions = [MA(D.DOWN)]

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_through_own_factory(self):

        start = TC(1, 3, 0)
        goal = C(1, 7)
        factory_positions = FactoryPositions(player=[FactoryPos(2, 5)])
        expected_actions = [MA(D.DOWN)] * 4

        state = get_state(factory_positions=factory_positions)
        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_around_opponent_factory(self):
        start = TC(1, 3, 0)
        goal = C(1, 7)
        factory_positions = FactoryPositions(opp=[FactoryPos(2, 5)])
        expected_actions = [MA(D.LEFT)] + [MA(D.DOWN)] * 4 + [MA(D.RIGHT)]

        state = get_state(factory_positions=factory_positions)

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_through_the_rubble(self):
        start = TC(2, 2, 0)
        goal = C(5, 5)
        rubble_tiles = [RT(3, 2, 20), RT(2, 4, 20), RT(4, 3, 20), RT(3, 5, 20), RT(5, 4, 20)]

        tiles = Tiles(rubble=rubble_tiles)
        expected_actions = [MA(D.DOWN), MA(D.RIGHT)] * 3

        state = get_state(tiles=tiles)

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_neg_constraint_wait_now(self):
        start = TC(2, 2, 0)
        goal = C(5, 2)
        constraints = init_constraints(negative_constraints=[TC(3, 2, 1)])
        expected_actions = [MA(D.CENTER), MA(D.RIGHT), MA(D.RIGHT), MA(D.RIGHT)]

        state = get_state()

        self._test_move_to_search(
            state=state, start=start, goal=goal, expected_actions=expected_actions, constraints=constraints
        )

    def test_neg_constraint_wait_in_3_steps(self):
        start = TC(2, 2, 0)
        goal = C(5, 2)
        constraints = init_constraints(negative_constraints=[TC(5, 2, 3)])

        expected_actions = [MA(D.RIGHT), MA(D.RIGHT), MA(D.CENTER), MA(D.RIGHT)]

        state = get_state()

        self._test_move_to_search(
            state=state, start=start, goal=goal, expected_actions=expected_actions, constraints=constraints
        )

    def test_low_time_to_power_cost_move_around_rubble(self):
        start = TC(2, 2, 0)
        goal = C(5, 2)
        time_to_power_cost = 3
        unit_type = "LIGHT"
        rubble_tiles = [RT(3, 2, 100), RT(4, 2, 100), RT(3, 3, 100)]
        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.UP), MA(D.RIGHT), MA(D.RIGHT), MA(D.RIGHT), MA(D.DOWN)]

        state = get_state(tiles=tiles)

        self._test_move_to_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )

    def test_high_time_to_power_cost_move_through_rubble(self):
        start = TC(2, 2, 0)
        goal = C(5, 2)
        time_to_power_cost = 5
        unit_type = "LIGHT"
        rubble_tiles = [RT(3, 2, 100), RT(4, 2, 100), RT(3, 3, 100)]
        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.RIGHT), MA(D.RIGHT), MA(D.RIGHT)]

        state = get_state(tiles=tiles)

        self._test_move_to_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )

    def test_low_time_to_power_cost_move_around_many_rubble(self):
        start = TC(2, 2, 0)
        goal = C(20, 2)
        time_to_power_cost = 41.499
        unit_type = "LIGHT"
        rubble_tiles = [RT(x, 2, 100) for x in range(2, 20)] + [RT(3, 3, 20)]

        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.UP)] + [MA(D.RIGHT)] * 18 + [MA(D.DOWN)]

        state = get_state(tiles=tiles)

        self._test_move_to_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )

    def test_high_time_to_power_cost_move_through_many_rubble(self):
        start = TC(2, 2, 0)
        goal = C(20, 2)
        time_to_power_cost = 41.501
        unit_type = "LIGHT"
        rubble_tiles = [RT(x, 2, 100) for x in range(2, 20)]
        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.RIGHT)] * 18

        state = get_state(tiles=tiles)

        self._test_move_to_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )


class DigAtSearch(unittest.TestCase):
    def _test_dig_at_search(
        self,
        state: GameState,
        start: DTC,
        goal: DC,
        expected_actions: Sequence[UnitAction],
        time_to_power_cost: float = 50,
        unit_type: str = "LIGHT",
        constraints: Optional[Constraints] = None,
    ):
        if constraints is None:
            constraints = Constraints()

        expected_actions = list(expected_actions)

        unit_cfg = ENV_CFG.get_unit_config(unit_type)

        dig_at_graph = DigAtGraph(
            unit_type=unit_type,
            board=state.board,
            time_to_power_cost=time_to_power_cost,
            unit_cfg=unit_cfg,
            constraints=constraints,
            goal=goal,
        )
        search = Search(dig_at_graph)
        actions = search.get_actions_to_complete_goal(start=start)
        self.assertEqual(actions, expected_actions)

    def test_already_there_path(self):
        start = DTC(3, 3, 0, 0)
        goal = DC(3, 3, 3)
        state = get_state()
        expected_actions = [DA()] * 3

        self._test_dig_at_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_one_down_path(self):
        start = DTC(3, 2, 0, 0)
        goal = DC(3, 3, 3)
        state = get_state()
        expected_actions = [MA(D.DOWN)] + [DA()] * 3

        self._test_dig_at_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_through_own_factory(self):
        start = DTC(1, 3, 0, 0)
        goal = DC(1, 7, 2)
        factory_positions = FactoryPositions(player=[FactoryPos(2, 5)])
        expected_actions = [MA(D.DOWN)] * 4 + [DA()] * 2

        state = get_state(factory_positions=factory_positions)
        self._test_dig_at_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_around_opponent_factory(self):
        start = DTC(1, 3, 0, 0)
        goal = DC(1, 7, 2)
        factory_positions = FactoryPositions(opp=[FactoryPos(2, 5)])

        expected_actions = [MA(D.LEFT)] + [MA(D.DOWN)] * 4 + [MA(D.RIGHT)] + [DA()] * 2

        state = get_state(factory_positions=factory_positions)

        self._test_dig_at_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_through_the_rubble(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 5, 3)
        rubble_tiles = [
            RT(3, 2, 20),
            RT(2, 4, 20),
            RT(4, 3, 20),
            RT(3, 5, 20),
            RT(5, 4, 20),
        ]

        tiles = Tiles(rubble=rubble_tiles)
        expected_actions = [MA(D.DOWN), MA(D.RIGHT)] * 3 + [DA()] * 3

        state = get_state(tiles=tiles)

        self._test_dig_at_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_neg_constraint_wait_now(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 2, 3)
        constraints = init_constraints(negative_constraints=[TC(3, 2, 1)])

        expected_actions = [MA(D.CENTER)] + [MA(D.RIGHT)] * 3 + [DA()] * 3

        state = get_state()

        self._test_dig_at_search(
            state=state, start=start, goal=goal, expected_actions=expected_actions, constraints=constraints
        )

    def test_neg_constraint_wait_in_3_steps(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 2, 3)
        constraints = init_constraints(negative_constraints=[TC(5, 2, 3)])
        expected_actions = [MA(D.RIGHT), MA(D.RIGHT), MA(D.CENTER), MA(D.RIGHT)] + [DA()] * 3

        state = get_state()

        self._test_dig_at_search(
            state=state, start=start, goal=goal, expected_actions=expected_actions, constraints=constraints
        )

    def test_neg_constraint_at_first_dig_possibility(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 2, 3)
        constraints = init_constraints(negative_constraints=[TC(5, 2, 4)])
        expected_actions = [MA(D.RIGHT), MA(D.RIGHT), MA(D.CENTER), MA(D.CENTER), MA(D.RIGHT)] + [DA()] * 3

        state = get_state()

        self._test_dig_at_search(
            state=state, start=start, goal=goal, expected_actions=expected_actions, constraints=constraints
        )

    def test_neg_constraint_at_second_dig_possibility(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 2, 3)
        constraints = init_constraints(negative_constraints=[TC(5, 2, 5)])
        rubble_tiles = [RT(5, 1, 20), RT(5, 3, 20), RT(6, 2, 20)]
        tiles = Tiles(rubble=rubble_tiles)
        expected_actions = [MA(D.RIGHT)] * 3 + [DA(), MA(D.LEFT), MA(D.RIGHT)] + [DA()] * 2

        state = get_state(tiles=tiles)

        self._test_dig_at_search(
            state=state, start=start, goal=goal, expected_actions=expected_actions, constraints=constraints
        )

    def test_low_time_to_power_cost_move_around_rubble(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 2, 3)
        time_to_power_cost = 3
        unit_type = "LIGHT"
        rubble_tiles = [RT(3, 2, 100), RT(4, 2, 100), RT(3, 3, 100)]
        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.UP)] + [MA(D.RIGHT)] * 3 + [MA(D.DOWN)] + [DA()] * 3

        state = get_state(tiles=tiles)

        self._test_dig_at_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )

    def test_high_time_to_power_cost_move_through_rubble(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(5, 2, 3)
        time_to_power_cost = 5
        unit_type = "LIGHT"
        rubble_tiles = [RT(3, 2, 100), RT(4, 2, 100), RT(3, 3, 100)]
        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.RIGHT)] * 3 + [DA()] * 3

        state = get_state(tiles=tiles)

        self._test_dig_at_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )

    def test_low_time_to_power_cost_move_around_many_rubble(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(20, 2, 3)
        time_to_power_cost = 41.499
        unit_type = "LIGHT"
        rubble_tiles = [RT(x, 2, 100) for x in range(2, 20)] + [RT(3, 3, 20)]

        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.UP)] + [MA(D.RIGHT)] * 18 + [MA(D.DOWN)] + [DA()] * 3

        state = get_state(tiles=tiles)

        self._test_dig_at_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )

    def test_high_time_to_power_cost_move_through_many_rubble(self):
        start = DTC(2, 2, 0, 0)
        goal = DC(20, 2, 3)
        time_to_power_cost = 41.501
        unit_type = "LIGHT"
        rubble_tiles = [RT(x, 2, 100) for x in range(2, 20)]
        tiles = Tiles(rubble=rubble_tiles)

        expected_actions = [MA(D.RIGHT)] * 18 + [DA()] * 3

        state = get_state(tiles=tiles)

        self._test_dig_at_search(
            state=state,
            start=start,
            goal=goal,
            unit_type=unit_type,
            expected_actions=expected_actions,
            time_to_power_cost=time_to_power_cost,
        )


class TestPowerPickupSearch(unittest.TestCase):
    def _test_power_pickup_search(
        self,
        state: GameState,
        start: PTC,
        expected_actions: Sequence[UnitAction],
        next_goal_c: Optional[C] = None,
        time_to_power_cost: float = 50,
        unit_type: str = "LIGHT",
        constraints: Optional[Constraints] = None,
    ):
        if constraints is None:
            constraints = Constraints()

        expected_actions = list(expected_actions)

        start_ptc = RPTC(
            start.x, start.y, start.t, start.p, start.unit_cfg, start.game_state, q=0, resource=Resource.POWER
        )
        unit_cfg = ENV_CFG.get_unit_config(unit_type)

        power_availability_tracker = PowerTracker(state.board.player_factories)

        pick_up_power_graph = PickupPowerGraph(
            board=state.board,
            time_to_power_cost=time_to_power_cost,
            unit_cfg=unit_cfg,
            unit_type=unit_type,
            constraints=constraints,
            next_goal_c=next_goal_c,
            power_tracker=power_availability_tracker,
            later_pickup=True,
        )
        search = Search(pick_up_power_graph)
        actions = search.get_actions_to_complete_goal(start=start_ptc)
        self.assertEqual(actions, expected_actions)

    def test_already_there_path(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=3, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        expected_actions = [PA(amount=49, resource=Resource.POWER)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_already_there_no_power_available_day(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, p=0)])
        state = get_state(factory_positions=factory_positions, real_env_steps=1)

        start = PTC(x=3, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        expected_actions = [MA(D.CENTER), PA(amount=48, resource=Resource.POWER)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_already_there_no_power_available_night(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, p=0)])
        state = get_state(factory_positions=factory_positions, real_env_steps=31)

        start = PTC(x=3, y=3, t=31, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        expected_actions = [MA(D.CENTER), PA(amount=49, resource=Resource.POWER)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_move_through_rubble_pickup_more_power(self):
        rubble_tiles = [RT(1, 0, 100), RT(2, 0, 100), RT(0, 1, 100), RT(1, 1, 100), RT(0, 2, 100)]
        tiles = Tiles(rubble=rubble_tiles)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 1, p=0)])
        state = get_state(factory_positions=factory_positions, tiles=tiles)

        start = PTC(x=0, y=0, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        expected_actions = [MA(D.RIGHT), MA(D.RIGHT), PA(amount=59, resource=Resource.POWER)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_move_to_factory_day(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=1, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        expected_actions = [MA(D.RIGHT), PA(amount=49, resource=Resource.POWER)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_move_to_factory_night(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=1, y=3, t=31, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        expected_actions = [MA(D.RIGHT), PA(amount=50, resource=Resource.POWER)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_move_take_next_goal_into_account_right(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        constraints = init_constraints(negative_constraints=[TC(3, 3, 2)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=3, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        next_goal_c = C(5, 3)
        expected_actions = [MA(D.RIGHT), PA(amount=49, resource=Resource.POWER)]

        state = get_state(factory_positions=factory_positions)

        self._test_power_pickup_search(
            state=state,
            start=start,
            expected_actions=expected_actions,
            constraints=constraints,
            next_goal_c=next_goal_c,
        )

    def test_move_take_next_goal_into_account_up(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        constraints = init_constraints(negative_constraints=[TC(3, 3, 2)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=3, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        next_goal_c = C(3, 1)
        expected_actions = [MA(D.UP), PA(amount=49, resource=Resource.POWER)]

        state = get_state(factory_positions=factory_positions)

        self._test_power_pickup_search(
            state=state,
            start=start,
            expected_actions=expected_actions,
            constraints=constraints,
            next_goal_c=next_goal_c,
        )

    def test_move_take_next_goal_into_account_down(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        constraints = init_constraints(negative_constraints=[TC(3, 3, 2)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=3, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        next_goal_c = C(3, 5)
        expected_actions = [MA(D.DOWN), PA(amount=49, resource=Resource.POWER)]

        state = get_state(factory_positions=factory_positions)

        self._test_power_pickup_search(
            state=state,
            start=start,
            expected_actions=expected_actions,
            constraints=constraints,
            next_goal_c=next_goal_c,
        )

    def test_move_take_next_goal_into_account_left(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        constraints = init_constraints(negative_constraints=[TC(3, 3, 2)])
        state = get_state(factory_positions=factory_positions)

        start = PTC(x=3, y=3, t=1, p=100, unit_cfg=LIGHT_CFG, game_state=state)
        next_goal_c = C(1, 3)
        expected_actions = [MA(D.LEFT), PA(amount=49, resource=Resource.POWER)]

        state = get_state(factory_positions=factory_positions)

        self._test_power_pickup_search(
            state=state,
            start=start,
            expected_actions=expected_actions,
            constraints=constraints,
            next_goal_c=next_goal_c,
        )


class TestTransferResearchesSearch(unittest.TestCase):
    def _test_power_pickup_search(
        self,
        state: GameState,
        start: TC,
        expected_actions: Sequence[UnitAction],
        time_to_power_cost: float = 50,
        unit_type: str = "LIGHT",
        resource: Resource = Resource.ICE,
        constraints: Optional[Constraints] = None,
    ):
        if constraints is None:
            constraints = Constraints()

        expected_actions = list(expected_actions)

        start_ptc = RTC(start.x, start.y, start.t, q=0, resource=resource)
        unit_cfg = ENV_CFG.get_unit_config(unit_type)

        move_to_graph = TransferToFactoryResourceGraph(
            unit_type=unit_type,
            board=state.board,
            time_to_power_cost=time_to_power_cost,
            unit_cfg=unit_cfg,
            constraints=constraints,
            resource=resource,
            q=unit_cfg.CARGO_SPACE,
        )
        search = Search(move_to_graph)
        actions = search.get_actions_to_complete_goal(start=start_ptc)
        self.assertEqual(actions, expected_actions)

    def test_already_there_path(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = TC(x=3, y=3, t=1)
        expected_actions = [TA(direction=D.CENTER, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_next_to_factory_left(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = TC(x=1, y=3, t=1)
        expected_actions = [TA(direction=D.RIGHT, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_next_to_factory_right(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = TC(x=5, y=3, t=1)
        expected_actions = [TA(direction=D.LEFT, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_next_to_factory_up(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = TC(x=3, y=1, t=1)
        expected_actions = [TA(direction=D.DOWN, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_next_to_factory_down(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = TC(x=3, y=5, t=1)
        expected_actions = [TA(direction=D.UP, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_move_to_factory(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)

        start = TC(x=0, y=3, t=1)
        expected_actions = [MA(D.RIGHT), TA(direction=D.RIGHT, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(state=state, start=start, expected_actions=expected_actions)

    def test_move_on_factory_and_transfer(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)
        constraints = init_constraints(negative_constraints=[TC(2, 2, 2), TC(2, 3, 2), TC(2, 4, 2), TC(1, 3, 2)])

        start = TC(x=2, y=3, t=1)
        expected_actions = [MA(D.RIGHT), TA(direction=D.CENTER, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(
            state=state, start=start, expected_actions=expected_actions, constraints=constraints
        )

    def test_move_away_from_factory_and_transfer(self):
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        state = get_state(factory_positions=factory_positions)
        constraints = init_constraints(negative_constraints=[TC(2, 2, 2), TC(2, 3, 2), TC(2, 4, 2), TC(3, 3, 2)])

        start = TC(x=2, y=3, t=1)
        expected_actions = [MA(D.LEFT), TA(direction=D.RIGHT, amount=LIGHT_CFG.CARGO_SPACE, resource=Resource.ICE)]

        self._test_power_pickup_search(
            state=state, start=start, expected_actions=expected_actions, constraints=constraints
        )


if __name__ == "__main__":
    unittest.main()
