import unittest

from objects.coordinate import Coordinate as C
from lux.config import EnvConfig
from tests.generate_game_state import get_state, Tiles, LichenTile as LT, FactoryPositions, UnitPos


ENV_CFG = EnvConfig()


class TestMinDistances(unittest.TestCase):
    def test_only_factory(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1)])
        expected_distance = 2

        state = get_state(board_width=9, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_multiple_factories(self):
        c = C(5, 9)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1), UnitPos(3, 9, id=2)])
        expected_distance = 1

        state = get_state(board_width=12, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_and_lichen(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1)])
        lichen_tiles = [LT(x=3, y=5, lichen=1, strain=1)]
        expected_distance = 1

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(board_width=9, tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_and_multiple_lichen(self):
        c = C(3, 8)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1)])
        lichen_tiles = [LT(x=3, y=5, lichen=1, strain=1),
                        LT(x=3, y=6, lichen=1, strain=1),
                        LT(x=3, y=7, lichen=1, strain=1)]
        expected_distance = 1

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(board_width=9, tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_both_player_factory(self):
        c = C(3, 8)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1)], opp=[UnitPos(3, 10, id=2)])
        expected_distance = 4

        state = get_state(board_width=12, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_both_factory_and_both_lichen_1_distance(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1)], opp=[UnitPos(3, 10, id=2)])
        lichen_tiles = [LT(x=3, y=5, lichen=1, strain=1),
                        LT(x=3, y=6, lichen=1, strain=2)]
        expected_distance = 1

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(board_width=12, tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_both_factory_and_both_lichen_0_distance(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[UnitPos(3, 3, id=1)], opp=[UnitPos(3, 10, id=2)])
        lichen_tiles = [LT(x=3, y=6, lichen=1, strain=1),
                        LT(x=3, y=5, lichen=1, strain=2)]
        expected_distance = 0

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(board_width=12, tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)


if __name__ == "__main__":
    unittest.main()
