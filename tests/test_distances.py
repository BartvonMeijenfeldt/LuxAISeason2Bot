import unittest
import numpy as np

from typing import Tuple

from distances import (
    get_distances_between_positions,
    get_min_distance_between_positions,
    get_closest_pos_and_pos_between_positions,
    get_closest_pos_between_pos_and_positions,
    get_n_closests_positions_between_positions,
    get_distance_between_pos_and_pos,
    get_distances_between_pos_and_positions,
    get_positions_on_optimal_path_between_pos_and_pos,
)

from tests.generate_game_state import get_state, Tiles, RubbleTile as RT


class TestGetDistancesBetweenPositions(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, expected: np.ndarray) -> None:
        distances = get_distances_between_positions(a, b)
        np.testing.assert_equal(expected, distances)

    def test_two_close_two_far_away(self):
        a = np.array([[0, 0], [24, 24]])
        b = np.array([[25, 25], [47, 47]])
        expected = np.array([[50, 94], [2, 46]])
        self._test_expected(a, b, expected)

    def test_3_by_3(self):
        a = np.array([[0, 0], [15, 15], [30, 30]])
        b = np.array([[0, 0], [15, 15], [30, 30]])

        expected = np.array([[0, 30, 60], [30, 0, 30], [60, 30, 0]])
        self._test_expected(a, b, expected)


class TestGetMinDistanceBetweenPositions(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, expected: int) -> None:
        min_distance = get_min_distance_between_positions(a, b)
        self.assertEqual(expected, min_distance)

    def test_two_close_two_far_away(self):
        a = np.array([[0, 0], [24, 24]])
        b = np.array([[25, 25], [47, 47]])
        expected = 2
        self._test_expected(a, b, expected)

    def test_3_by_3(self):
        a = np.array([[0, 0], [15, 15], [30, 30]])
        b = np.array([[0, 0], [15, 15], [30, 30]])

        expected = 0
        self._test_expected(a, b, expected)

    def test_all_close(self):
        a = np.array([[23, 24], [24, 24]])
        b = np.array([[24, 25], [25, 25]])

        expected = 1
        self._test_expected(a, b, expected)

    def test_all_far(self):
        a = np.array([[1, 1], [1, 2]])
        b = np.array([[21, 2], [21, 3]])

        expected = 20
        self._test_expected(a, b, expected)

    def test_one_versus_three(self):
        a = np.array([[24, 24]])
        b = np.array([[24, 25], [25, 25], [25, 26]])

        expected = 1
        self._test_expected(a, b, expected)

    def test_same_positions(self):
        a = np.array([[23, 24], [24, 24]])
        b = np.array([[24, 24], [24, 25]])

        expected = 0
        self._test_expected(a, b, expected)


class TestClosestPosAndPosBetweenPositions(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, expected: Tuple[np.ndarray, np.ndarray]) -> None:
        closest_positions = get_closest_pos_and_pos_between_positions(a, b)
        np.testing.assert_equal(expected, closest_positions)

    def test_two_close_two_far_away(self):
        a = np.array([[0, 0], [24, 24]])
        b = np.array([[25, 25], [47, 47]])

        expected = np.array([24, 24]), np.array([25, 25])
        self._test_expected(a, b, expected)

    def test_all_close(self):
        a = np.array([[23, 24], [24, 24]])
        b = np.array([[24, 25], [25, 25]])

        expected = np.array([24, 24]), np.array([24, 25])
        self._test_expected(a, b, expected)

    def test_all_far(self):
        a = np.array([[1, 1], [1, 2]])
        b = np.array([[21, 2], [21, 3]])

        expected = np.array([1, 2]), np.array([21, 2])
        self._test_expected(a, b, expected)

    def test_one_versus_three(self):
        a = np.array([[24, 24]])
        b = np.array([[24, 25], [25, 25], [25, 26]])

        expected = np.array([24, 24]), np.array([24, 25])
        self._test_expected(a, b, expected)

    def test_same_positions(self):
        a = np.array([[23, 24], [24, 24]])
        b = np.array([[24, 24], [24, 25]])

        expected = np.array([24, 24]), np.array([24, 24])
        self._test_expected(a, b, expected)


class TestGetClosestPosBetweenPosAndPositions(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, expected: np.ndarray) -> None:
        pos = get_closest_pos_between_pos_and_positions(a, b)
        np.testing.assert_equal(expected, pos)

    def test_two_far_away(self):
        a = np.array([0, 0])
        b = np.array([[25, 25], [47, 47]])

        expected = np.array([25, 25])
        self._test_expected(a, b, expected)

    def test_both_close(self):
        a = np.array([24, 24])
        b = np.array([[24, 25], [25, 25]])

        expected = np.array([24, 25])
        self._test_expected(a, b, expected)

    def test_three(self):
        a = np.array([24, 24])
        b = np.array([[25, 25], [24, 25], [25, 26]])

        expected = np.array([24, 25])
        self._test_expected(a, b, expected)

    def test_same_positions(self):
        a = np.array([24, 24])
        b = np.array([[24, 24], [24, 25]])

        expected = np.array([24, 24])
        self._test_expected(a, b, expected)


class TestGetNClosestPositionsBetweenPositions(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, n: int, expected: np.ndarray) -> None:
        positions = get_n_closests_positions_between_positions(a, b, n)
        np.testing.assert_equal(expected, positions)

    def test_two_close_two_far_away(self):
        a = np.array([[0, 0], [24, 24]])
        b = np.array([[25, 25], [47, 47]])
        n = 1

        expected = np.array([[24, 24]])
        self._test_expected(a, b, n, expected)

    def test_3_by_3_n_is_1(self):
        a = np.array([[0, 0], [15, 15], [30, 30]])
        b = np.array([[0, 0], [15, 16], [30, 32]])
        n = 1

        expected = np.array([[0, 0]])
        self._test_expected(a, b, n, expected)

    def test_3_by_3_n_is_2(self):
        a = np.array([[0, 0], [15, 15], [30, 30]])
        b = np.array([[0, 0], [15, 16], [30, 32]])
        n = 2

        expected = np.array([[0, 0], [15, 15]])
        self._test_expected(a, b, n, expected)

    def test_all_close(self):
        a = np.array([[23, 24], [24, 24]])
        b = np.array([[24, 25], [25, 25]])
        n = 1

        expected = np.array([[24, 24]])
        self._test_expected(a, b, n, expected)

    def test_all_far(self):
        a = np.array([[1, 1], [1, 2]])
        b = np.array([[21, 2], [21, 3]])
        n = 1

        expected = np.array([[1, 2]])
        self._test_expected(a, b, n, expected)


class TestGetDistanceBetweenPosAndPos(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, expected: int) -> None:
        positions = get_distance_between_pos_and_pos(a, b)
        np.testing.assert_equal(expected, positions)

    def test_same_position(self):
        a = np.array([24, 24])
        b = np.array([24, 24])
        expected = 0
        return self._test_expected(a, b, expected)

    def test_pos_a_bigger_x_bigger_y(self):
        a = np.array([26, 29])
        b = np.array([24, 24])
        expected = 7
        return self._test_expected(a, b, expected)

    def test_pos_a_bigger_x_smaller_y(self):
        a = np.array([26, 21])
        b = np.array([24, 24])
        expected = 5
        return self._test_expected(a, b, expected)

    def test_pos_a_smaller_x_smaller_y(self):
        a = np.array([18, 21])
        b = np.array([24, 24])
        expected = 9
        return self._test_expected(a, b, expected)

    def test_pos_a_smaller_x_bigger_y(self):
        a = np.array([18, 33])
        b = np.array([24, 24])
        expected = 15
        return self._test_expected(a, b, expected)


class TestGetDistancesBetweenPosAndPositions(unittest.TestCase):
    def _test_expected(self, a: np.ndarray, b: np.ndarray, expected: np.ndarray) -> None:
        pos = get_distances_between_pos_and_positions(a, b)
        np.testing.assert_equal(expected, pos)

    def test_two_far_away(self):
        a = np.array([0, 0])
        b = np.array([[25, 25], [47, 47]])

        expected = np.array([50, 94])
        self._test_expected(a, b, expected)

    def test_both_close(self):
        a = np.array([24, 24])
        b = np.array([[24, 25], [25, 25]])

        expected = np.array([1, 2])
        self._test_expected(a, b, expected)

    def test_three(self):
        a = np.array([24, 24])
        b = np.array([[25, 25], [24, 25], [25, 26]])

        expected = np.array([2, 1, 3])
        self._test_expected(a, b, expected)

    def test_same_positions(self):
        a = np.array([24, 24])
        b = np.array([[24, 24], [24, 25]])

        expected = np.array([0, 1])
        self._test_expected(a, b, expected)


class TestGetPositionsOnOptimalPathBetweenPosAndPos(unittest.TestCase):
    def _test_get_positions(self, a: np.ndarray, b: np.ndarray, tiles: Tiles, expected: np.ndarray):
        game_state = get_state(tiles=tiles)
        board = game_state.board
        positions = get_positions_on_optimal_path_between_pos_and_pos(a, b, board)
        np.testing.assert_equal(expected, positions)

    def test_one_rubble_away(self):
        rubble_tiles = [RT(24, y, 20) for y in range(48)]
        tiles = Tiles(rubble=rubble_tiles)
        start = np.array([23, 24])
        goal = np.array([25, 24])
        expected = np.array([[24, 24]])

        self._test_get_positions(start, goal, tiles, expected)

    # def test_go_through_lowest(self):
    #     rubble_tiles = [RT(24, 46, 1)] + [RT(24, y, 100) for y in range(48) if y != 46]
    #     tiles = Tiles(rubble=rubble_tiles)
    #     start = np.array([23, 45])
    #     goal = np.array([25, 45])
    #     expected = np.array([[23, 46], [24, 46], [25, 46]])

    #     self._test_get_positions(start, goal, tiles, expected)


if __name__ == "__main__":
    unittest.main()
