import unittest

from lux.config import EnvConfig
from objects.coordinate import Coordinate as C
from tests.generate_game_state import FactoryPos, FactoryPositions
from tests.generate_game_state import LichenTile as LT
from tests.generate_game_state import Tiles, get_state

ENV_CFG = EnvConfig()


class TestMinDistancesPlayerFactoryOrLichen(unittest.TestCase):
    def test_only_factory(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        expected_distance = 2

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_multiple_factories(self):
        c = C(5, 9)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1), FactoryPos(3, 9, id=2)])
        expected_distance = 1

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_and_lichen(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1)])
        lichen_tiles = [LT(x=3, y=5, lichen=1, strain=1)]
        expected_distance = 1

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_and_multiple_lichen(self):
        c = C(3, 8)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1)])
        lichen_tiles = [
            LT(x=3, y=5, lichen=1, strain=1),
            LT(x=3, y=6, lichen=1, strain=1),
            LT(x=3, y=7, lichen=1, strain=1),
        ]
        expected_distance = 1

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_both_player_factory(self):
        c = C(3, 8)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1)], opp=[FactoryPos(3, 10, id=2)])
        expected_distance = 4

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_both_factory_and_both_lichen_1_distance(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1)], opp=[FactoryPos(3, 10, id=2)])
        lichen_tiles = [LT(x=3, y=5, lichen=1, strain=1), LT(x=3, y=6, lichen=1, strain=2)]
        expected_distance = 1

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)

    def test_both_factory_and_both_lichen_0_distance(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1)], opp=[FactoryPos(3, 10, id=2)])
        lichen_tiles = [LT(x=3, y=6, lichen=1, strain=1), LT(x=3, y=5, lichen=1, strain=2)]
        expected_distance = 0

        tiles = Tiles(lichen=lichen_tiles)
        state = get_state(tiles=tiles, factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_player_factory_or_lichen(c)

        self.assertEqual(expected_distance, distance)


class TestMinDistancesPlayerFactory(unittest.TestCase):
    def test_one_factory(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        expected_distance = 2

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_any_player_factory(c)

        self.assertEqual(expected_distance, distance)

    def test_multiple_factories(self):
        c = C(5, 9)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1), FactoryPos(3, 9)])
        expected_distance = 1

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_any_player_factory(c)

        self.assertEqual(expected_distance, distance)

    def test_both_player_factory(self):
        c = C(3, 8)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)], opp=[FactoryPos(3, 10)])
        expected_distance = 4

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_any_player_factory(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_on_edge_tile(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 5)])
        expected_distance = 0

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_any_player_factory(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_on_center_tile(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 6)])
        expected_distance = 0

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_any_player_factory(c)

        self.assertEqual(expected_distance, distance)

    def test_factory_on_corner_tile(self):
        c = C(4, 5)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 6)])
        expected_distance = 0

        state = get_state(factory_positions=factory_positions)
        board = state.board

        distance = board.get_min_distance_to_any_player_factory(c)

        self.assertEqual(expected_distance, distance)


class TestClosestPlayerFactoryTile(unittest.TestCase):
    def test_one_factory(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)])
        expected_tile = C(3, 4)

        state = get_state(factory_positions=factory_positions)
        board = state.board

        closest_tile = board.get_closest_player_factory_tile(c)
        self.assertEqual(expected_tile, closest_tile)

    def test_multiple_factories(self):
        c = C(5, 9)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3, id=1), FactoryPos(3, 9)])
        expected_tile = C(4, 9)

        state = get_state(factory_positions=factory_positions)
        board = state.board

        closest_tile = board.get_closest_player_factory_tile(c)
        self.assertEqual(expected_tile, closest_tile)

    def test_both_player_factory(self):
        c = C(3, 8)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 3)], opp=[FactoryPos(3, 10)])
        expected_tile = C(3, 4)

        state = get_state(factory_positions=factory_positions)
        board = state.board

        closest_tile = board.get_closest_player_factory_tile(c)
        self.assertEqual(expected_tile, closest_tile)

    def test_factory_on_edge_tile(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 5)])
        expected_tile = c

        state = get_state(factory_positions=factory_positions)
        board = state.board

        closest_tile = board.get_closest_player_factory_tile(c)
        self.assertEqual(expected_tile, closest_tile)

    def test_factory_on_center_tile(self):
        c = C(3, 6)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 6)])
        expected_tile = c

        state = get_state(factory_positions=factory_positions)
        board = state.board

        closest_tile = board.get_closest_player_factory_tile(c)
        self.assertEqual(expected_tile, closest_tile)

    def test_factory_on_corner_tile(self):
        c = C(4, 5)
        factory_positions = FactoryPositions(player=[FactoryPos(3, 6)])
        expected_tile = c

        state = get_state(factory_positions=factory_positions)
        board = state.board

        closest_tile = board.get_closest_player_factory_tile(c)
        self.assertEqual(expected_tile, closest_tile)


if __name__ == "__main__":
    unittest.main()
