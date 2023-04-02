import numpy as np

from scipy.signal import convolve2d
from scipy.ndimage.filters import minimum_filter

from objects.board import Board
from objects.coordinate import Coordinate, CoordinateList


def get_factory_spawn_loc(board: Board, valid_spawns: np.ndarray) -> tuple:
    rubble_score = get_rubble_score(board)
    ice_score = get_ice_score(board)
    ore_score = get_ore_score(board)
    scores = get_scores(rubble_score, ice_score, ore_score, valid_spawns=valid_spawns)
    spawn_loc = get_coordinate_biggest(scores)
    return spawn_loc


def get_rubble_score(board: Board) -> np.ndarray:
    neighbouring_inverted_rubble = sum_neighbouring_inverted_rubble(board.rubble)
    rubble_score = neighbouring_inverted_rubble / 100000
    return rubble_score


def sum_neighbouring_inverted_rubble(rubble: np.ndarray) -> np.ndarray:
    inverted_rubble = 100 - rubble
    neighbouring_inverted_rubble = sum_closest_numbers(inverted_rubble, r=5)
    directly_neighbouring_inverted_rubble = sum_closest_numbers(inverted_rubble, r=1)

    # TODO bonus connected / easy to connect?
    return neighbouring_inverted_rubble + 100 * directly_neighbouring_inverted_rubble


def get_ice_score(board: Board) -> np.ndarray:
    tiles_array = board._get_tiles_xy_array()
    min_distances = _get_min_distances_placing_factory_to_positions(tiles_array, board.ice_positions)
    base_scores = get_base_scores(min_distances, base_score=20, max_distance=10, best_n_valid=4)
    direct_neighbor_bonus = get_direct_neighbor_bonus(min_distances, bonus=300)
    return base_scores + direct_neighbor_bonus


def get_base_scores(min_distances: np.ndarray, base_score: int, max_distance: int, best_n_valid: int) -> np.ndarray:
    min_distances = min_distances.astype("float")
    base_array = np.array([base_score])
    valid_mask = (min_distances != 0) & (min_distances <= max_distance)
    scores = np.divide(base_array, min_distances, out=np.zeros_like(min_distances), where=valid_mask)
    best_n_scores = np.partition(scores, -best_n_valid, axis=2)[:, :, -best_n_valid:]
    base_scores = best_n_scores.sum(axis=2)
    return base_scores


def get_direct_neighbor_bonus(min_distances: np.ndarray, bonus: int) -> np.ndarray:
    has_direct_neighbor = min_distances.min(axis=2) == 1
    return bonus * has_direct_neighbor


def _get_min_distances_placing_factory_to_positions(tiles_array: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = positions.transpose()
    diff = tiles_array[..., None] - positions[None, None, ...]
    abs_dis = np.abs(diff)
    distances = np.sum(abs_dis, axis=2)

    f = np.ones((3, 3, 1))
    min_distances = minimum_filter(distances, footprint=f)
    return min_distances


def get_ore_score(board: Board) -> np.ndarray:
    tiles_array = board._get_tiles_xy_array()
    min_distances = _get_min_distances_placing_factory_to_positions(tiles_array, board.ore_positions)
    base_scores = get_base_scores(min_distances, base_score=40, max_distance=10, best_n_valid=5)
    return base_scores


def sum_closest_numbers(x: np.ndarray, r: int) -> np.ndarray:
    conv_array = _get_conv_filter_surrounding_factory(r=r)
    sum_closest_numbers = convolve2d(x, conv_array, mode="same")
    return sum_closest_numbers


def _get_conv_filter_surrounding_factory(r: int) -> np.ndarray:
    """Get the convolutional filter of coordinates surrounding a 3x3 tile up to distance r"""
    array_size = 2 * r + 3
    coordinates_factory = _get_coordinates_factory(array_size)
    distance_array = _get_min_distance_to_object_array(array_size, object=coordinates_factory)
    conv_filter = _convert_min_distance_to_conv_filter(distance_array, r)

    return conv_filter


def _get_coordinates_factory(array_size: int) -> CoordinateList:
    """Get the 3x3 coordinates of the factory in the middle of an array of odd size"""
    assert array_size % 2 == 1
    center = array_size // 2
    coordinates = CoordinateList([Coordinate(center + i, center + j) for i in [-1, 0, 1] for j in [-1, 0, 1]])

    return coordinates


def _get_min_distance_to_object_array(array_size: int, object: CoordinateList) -> np.ndarray:
    min_distance_array = np.empty((array_size, array_size))

    for i in range(array_size):
        for j in range(array_size):
            c_ij = Coordinate(i, j)
            min_distance_array[i, j] = object.min_dis_to(c_ij)

    return min_distance_array


def _convert_min_distance_to_conv_filter(distance_array: np.ndarray, r: int) -> np.ndarray:
    between_0_and_r = (distance_array > 0) & (distance_array <= r)
    return np.where(between_0_and_r, 1, 0)


def get_scores(
    rubble_score: np.ndarray, ice_score: np.ndarray, ore_score: np.ndarray, valid_spawns: np.ndarray
) -> np.ndarray:
    score = rubble_score + ice_score + ore_score
    score = zero_invalid_spawns(score, valid_spawns)
    return score


def zero_invalid_spawns(x: np.ndarray, valid_spawns: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[~valid_spawns] = 0
    return x


def get_coordinate_biggest(x_: np.ndarray):
    biggest_loc_int = np.argmax(x_)
    x, y = np.unravel_index(biggest_loc_int, x_.shape)
    return (x, y)
