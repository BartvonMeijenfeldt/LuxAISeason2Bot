import unittest

from typing import Optional, Sequence

from logic.constraints import Constraints
from objects.coordinate import Coordinate, TimeCoordinate
from objects.actions.unit_action import MoveAction, UnitAction
from objects.direction import Direction as D
from search.search import MoveToGraph, Search
from lux.kit import GameState
from lux.config import EnvConfig
from tests.generate_game_state import get_state, FactoryPositions, UnitPos, Tiles, RubbleTile


ENV_CFG = EnvConfig()


class TestMoveToSearch(unittest.TestCase):
    def _test_move_to_search(
        self,
        state: GameState,
        start: TimeCoordinate,
        goal: Coordinate,
        expected_actions: Sequence[UnitAction],
        time_to_power_cost: int = 50,
        unit_type: str = "LIGHT",
        constraints: Optional[Constraints] = None,
    ):
        if constraints is None:
            constraints = Constraints()

        expected_actions = list(expected_actions)

        unit_cfg = ENV_CFG.ROBOTS[unit_type]

        move_to_graph = MoveToGraph(
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
        start = TimeCoordinate(3, 3, 0)
        goal = Coordinate(3, 3)
        state = get_state(board_width=9)
        expected_actions = []

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_one_down_path(self):

        start = TimeCoordinate(3, 2, 0)
        goal = Coordinate(3, 3)
        state = get_state(board_width=9)
        expected_actions = [MoveAction(D.DOWN)]

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_through_own_factory(self):

        start = TimeCoordinate(1, 3, 0)
        goal = Coordinate(1, 7)
        factory_positions = FactoryPositions(player=[UnitPos(2, 5)])

        directions = [D.DOWN] * 4
        expected_actions = [MoveAction(d) for d in directions]

        state = get_state(board_width=9, factory_positions=factory_positions)
        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_around_opponent_factory(self):
        start = TimeCoordinate(1, 3, 0)
        goal = Coordinate(1, 7)
        factory_positions = FactoryPositions(opp=[UnitPos(2, 5)])

        directions = [D.LEFT] + [D.DOWN] * 4 + [D.RIGHT]
        expected_actions = [MoveAction(d) for d in directions]

        state = get_state(board_width=9, factory_positions=factory_positions)

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)

    def test_through_the_rubble(self):
        start = TimeCoordinate(2, 2, 0)
        goal = Coordinate(5, 5)
        rubble_tiles = [
            RubbleTile(3, 2, 20),
            RubbleTile(2, 4, 20),
            RubbleTile(4, 3, 20),
            RubbleTile(3, 5, 20),
            RubbleTile(5, 4, 20),
        ]

        tiles = Tiles(rubble=rubble_tiles)

        directions = [D.DOWN, D.RIGHT] * 3
        expected_actions = [MoveAction(d) for d in directions]

        state = get_state(board_width=9, tiles=tiles)

        self._test_move_to_search(state=state, start=start, goal=goal, expected_actions=expected_actions)


if __name__ == "__main__":
    unittest.main()
