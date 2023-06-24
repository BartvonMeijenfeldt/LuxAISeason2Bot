from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np

from objects.coordinate import Coordinate
from search.graph import TilesToClearGraph
from search.search import Search
from utils.positions import init_empty_positions

if TYPE_CHECKING:
    from objects.board import Board


logger = logging.getLogger(__name__)


def get_distances_between_positions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b.transpose()
    diff = a[..., None] - b[None, ...]
    abs_diff = np.abs(diff)
    distances = abs_diff.sum(axis=1)
    return distances


def get_min_distance_between_positions(a: np.ndarray, b: np.ndarray) -> int:
    distances = get_distances_between_positions(a, b)
    return distances.min()


def get_closest_pos_and_pos_between_positions(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distances = get_distances_between_positions(a, b)
    index_a, index_b = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    pos_a = a[index_a]
    pos_b = b[index_b]

    return pos_a, pos_b


def get_min_distance_between_pos_and_positions(pos: np.ndarray, positions: np.ndarray) -> int:
    distances = get_distances_between_pos_and_positions(pos=pos, positions=positions)
    return distances.min()


def get_closest_pos_between_pos_and_positions(pos: np.ndarray, positions: np.ndarray) -> np.ndarray:
    distances = get_distances_between_pos_and_positions(pos=pos, positions=positions)
    index_closest = np.argmin(distances)
    return positions[index_closest]


def get_min_distances_between_positions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    "For each position in a get the min distances to b"
    distances = get_distances_between_positions(a, b)
    min_distances = distances.min(axis=1)
    return min_distances


def get_n_closests_positions_between_positions(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Return the n positions in a that are closest to b"""
    min_distances = get_min_distances_between_positions(a, b)
    closest_indexes = np.argpartition(min_distances, n)[:n]
    closest_positions = a[closest_indexes]
    return closest_positions


def get_distance_between_pos_and_pos(a: np.ndarray, b: np.ndarray) -> int:
    diff = np.subtract(a, b)
    abs_diff = np.abs(diff)
    distance = abs_diff.sum()
    return distance


def get_distances_between_pos_and_positions(pos: np.ndarray, positions: np.ndarray) -> np.ndarray:
    diff = np.subtract(pos, positions)
    abs_diff = np.abs(diff)
    distances = np.sum(abs_diff, axis=1)
    return distances


def get_positions_on_optimal_path_between_pos_and_pos(a: np.ndarray, b: np.ndarray, board: Board) -> np.ndarray:
    start = Coordinate(*a)
    goal = Coordinate(*b)

    graph = TilesToClearGraph(board=board, goal=goal)
    search = Search(graph=graph)
    try:
        optimal_actions = search.get_actions_to_complete_goal(start=start, budget=500)
    except Exception as e:
        logger.warning(f"Positions on optimal path failed due to {str(e)} from {a} to {b}")
        return init_empty_positions()

    positions = []
    c = start

    for action in optimal_actions[:-1]:
        c = c.add_action(action)
        positions.append(c.xy)

    if not positions:
        return init_empty_positions()

    positions = np.array(positions)

    return positions
