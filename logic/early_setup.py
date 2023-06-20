from typing import List, Optional

import numpy as np
from scipy.ndimage.filters import minimum_filter

from config import CONFIG
from lux.config import EnvConfig
from objects.board import Board
from utils.distances import get_distances_between_positions
from utils.image_processing import get_islands


def get_factory_spawn_loc(board: Board, valid_spawns: np.ndarray) -> tuple:
    """Get the best factory spawn (starting position) location by taking into account the rubble, ice, ore, closeness
    to the border into account.

    Args:
        board: Board
        valid_spawns: Valid spawn locations

    Returns:
        Spawn location
    """

    rubble_score = _get_rubble_score(board)
    ice_score = _get_ice_score(board)
    ore_score = _get_ore_score(board)
    border_score = _get_closeness_to_border_score(board)
    scores = _get_scores(rubble_score, ice_score, ore_score, border_score, valid_spawns=valid_spawns)
    spawn_loc = _get_coordinate_highest_score(scores)
    return spawn_loc


def _get_rubble_score(board: Board) -> np.ndarray:
    rubble_adjusted = _set_unspreadable_positions_to_inf_rubble(board)
    factory_pos_connects_to_empty_pos = get_factory_pos_connects_to_empty_pos(rubble_adjusted)
    rubble_value = _get_empty_rubble_value(rubble_adjusted, factory_pos_connects_to_empty_pos)
    score = rubble_value * CONFIG.VALUE_CONNECTED_TILE

    return score


def _get_empty_rubble_value(rubble: np.ndarray, factory_pos_connects_to_empty_pos: np.ndarray) -> np.ndarray:
    # Gets a score for empty rubble. Connected empty tiles count for 1, then based on distance and closeness to empty
    ratio_empty = 1 - rubble / EnvConfig.MAX_RUBBLE
    distances_between_all_positions = _get_distances_between_all_positions()
    scores = np.divide(ratio_empty, distances_between_all_positions, where=distances_between_all_positions != 0)
    scores[factory_pos_connects_to_empty_pos] = 1
    scores = scores.reshape(48 * 48, 48 * 48)
    best_indexes = np.argpartition(-scores, CONFIG.BEST_N_RUBBLE_TILES)[..., : CONFIG.BEST_N_RUBBLE_TILES]
    best_scores = scores[np.arange(scores.shape[0])[:, None], best_indexes]
    sum_scores = best_scores.sum(axis=1)
    return sum_scores.reshape(48, 48)


def _get_distances_between_all_positions() -> np.ndarray:
    positions_board = np.argwhere(np.ones((EnvConfig.map_size, EnvConfig.map_size)))
    distances_between_all_positions = get_distances_between_positions(positions_board, positions_board)
    distances_between_all_positions = distances_between_all_positions.reshape((48, 48, 48, 48))
    return distances_between_all_positions


def _set_unspreadable_positions_to_inf_rubble(board: Board) -> np.ndarray:
    rubble = board.rubble.copy()
    rubble = rubble.astype("float")
    unspreadable_positions = board.unspreadable_positions
    rubble[unspreadable_positions[:, 0], unspreadable_positions[:, 1]] = np.inf
    return rubble


def get_factory_pos_connects_to_empty_pos(rubble: np.ndarray) -> np.ndarray:
    factory_positions = _get_potential_factory_positions()
    islands = get_islands(rubble == 0)

    factory_pos_connects_to_empty_pos = _get_factory_pos_connects_to_island_pos(islands, factory_positions)
    return factory_pos_connects_to_empty_pos


def _get_factory_pos_connects_to_island_pos(
    islands_positions_list: List[np.ndarray], factory_positions: np.ndarray
) -> np.ndarray:
    # Returns Shape 48 x 48 x 48 x 48: Does (x1, y1) connects to (x2, y2)?
    factory_pos_connected_to_rubble_pos = np.zeros([EnvConfig.map_size] * 4)
    # nr_cleared_tiles = np.zeros((EnvConfig.map_size, EnvConfig.map_size))

    for island_positions in islands_positions_list:
        connected_tiles_mask = _get_is_connected_mask(factory_positions, island_positions)
        island_tiles_mask = _get_island_tiles_mask(island_positions)
        booleans_to_add = _get_booleans_to_add(connected_tiles_mask, island_tiles_mask)
        factory_pos_connected_to_rubble_pos += booleans_to_add

    return factory_pos_connected_to_rubble_pos.astype(bool)


def _get_booleans_to_add(connected_tiles_mask: np.ndarray, island_tiles_mask: np.ndarray) -> np.ndarray:
    booleans_to_add = np.zeros((EnvConfig.map_size,) * 4, dtype=bool)
    for i in range(island_tiles_mask.shape[0]):
        connected_tiles_mask_i = connected_tiles_mask[..., i]
        tile_i_boolean = island_tiles_mask[i]
        booleans_to_add[connected_tiles_mask_i, tile_i_boolean] += True

    return booleans_to_add


def _get_is_connected_mask(factory_positions: np.ndarray, island_positions: np.ndarray) -> np.ndarray:
    # Return shape is 48 x 48 x nr_island_positions
    differences_xy = factory_positions[..., None] - island_positions.transpose()[None, None, None, ...]
    distances_factory_positions_to_island_positions = np.abs(differences_xy).sum(axis=3)
    min_distances_factory_to_tile = distances_factory_positions_to_island_positions.min(axis=2)
    min_distance_factory_to_island = min_distances_factory_to_tile.min(axis=2)
    connected_to_island_mask = min_distance_factory_to_island <= 1
    not_on_factory_mask = min_distances_factory_to_tile > 0
    connected_tiles_mask = connected_to_island_mask[:, :, None] & not_on_factory_mask
    return connected_tiles_mask


def _get_island_tiles_mask(island_positions: np.ndarray) -> np.ndarray:
    # Return shape is nr_island_positions x 48 x 48
    tiles_mask = np.zeros([island_positions.shape[0], EnvConfig.map_size, EnvConfig.map_size], dtype=bool)
    index_ax_0 = np.arange(len(island_positions))
    tiles_mask[index_ax_0, island_positions[:, 0], island_positions[:, 1]] = True
    return tiles_mask


def _get_potential_factory_positions() -> np.ndarray:
    # Shape 48 x 48 x 9 x 2
    positions = []
    for i in range(EnvConfig.map_size):
        positions_i = []
        for j in range(EnvConfig.map_size):
            positions_ij = [(x, y) for x in range(i - 1, i + 2) for y in range(j - 1, j + 2)]
            positions_i.append(positions_ij)

        positions.append(positions_i)

    positions = np.array(positions)
    return positions


def _get_ice_score(board: Board) -> np.ndarray:
    tiles_array = board._get_tiles_xy_array()
    min_distances = _get_min_distances_placing_factory_to_positions(tiles_array, board.ice_positions)
    base_scores = get_base_scores(min_distances, base_score=CONFIG.BASE_SCORE_ICE)
    closest_distance_penalty = get_closest_distances_penalty_score(
        min_distances, penalty=CONFIG.PENALTY_DISTANCE_CLOSEST_ICE
    )

    scores = np.subtract(base_scores, closest_distance_penalty)
    return scores


def get_base_scores(min_distances: np.ndarray, base_score: int, best_n_valid: Optional[int] = None) -> np.ndarray:
    min_distances = min_distances.astype("float")
    base_array = np.array([base_score])
    valid_mask = min_distances != 0
    scores = np.divide(base_array, min_distances, out=np.zeros_like(min_distances), where=valid_mask)
    scores = -np.sort(-scores, axis=2)
    if best_n_valid:
        scores = scores[:, :, :best_n_valid]
    scores_weighted_on_closest_pos = scores / np.arange(1, scores.shape[2] + 1)
    base_scores = scores_weighted_on_closest_pos.sum(axis=2)
    return base_scores


def get_closest_distances_penalty_score(min_distances: np.ndarray, penalty: int) -> np.ndarray:
    closest_distances = min_distances.min(axis=2)
    penalty_score = penalty * np.clip(closest_distances - 1, 0, np.inf)
    return penalty_score


def _get_min_distances_placing_factory_to_positions(tiles_array: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = positions.transpose()
    diff = tiles_array[..., None] - positions[None, None, ...]
    abs_dis = np.abs(diff)
    distances = np.sum(abs_dis, axis=2)

    f = np.ones((3, 3, 1))
    min_distances = minimum_filter(distances, footprint=f)
    return min_distances


def _get_ore_score(board: Board) -> np.ndarray:
    tiles_array = board._get_tiles_xy_array()
    min_distances = _get_min_distances_placing_factory_to_positions(tiles_array, board.ore_positions)
    base_scores = get_base_scores(min_distances, base_score=CONFIG.BASE_SCORE_ORE)
    additional_bonus_closest_ore = CONFIG.BONUS_CLOSEST_NEIGHBOR_ORE - CONFIG.BASE_SCORE_ORE
    closest_neighbor_score = get_base_scores(min_distances, additional_bonus_closest_ore, best_n_valid=1)
    return base_scores + closest_neighbor_score


def _get_closeness_to_border_score(board: Board) -> np.ndarray:
    scores_x = np.ones_like(board.rubble) * 6
    scores_x[1:-1] = 4
    scores_x[2:-2] = 3
    scores_x[3:-3] = 1
    scores_x[4:-4] = 0

    scores_y = scores_x.transpose()
    scores = scores_x + scores_y
    scores = scores * -1 * CONFIG.BORDER_PENALTY

    return scores


def _get_scores(
    rubble_score: np.ndarray,
    ice_score: np.ndarray,
    ore_score: np.ndarray,
    border_scores: np.ndarray,
    valid_spawns: np.ndarray,
) -> np.ndarray:
    score = rubble_score + ice_score + ore_score + border_scores
    score = set_invalid_spawns_minus_inf(score, valid_spawns)
    return score


def set_invalid_spawns_minus_inf(x: np.ndarray, valid_spawns: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[~valid_spawns] = -np.inf
    return x


def _get_coordinate_highest_score(x_: np.ndarray):
    biggest_loc_int = np.argmax(x_)
    x, y = np.unravel_index(biggest_loc_int, x_.shape)
    return (x, y)
