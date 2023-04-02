import numpy as np

from typing import Tuple

from search.search import TilesToClearGraph, Search, SolutionNotFoundWithinBudgetError
from objects.coordinate import Coordinate
from positions import init_empty_positions


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


def get_positions_on_optimal_path_between_pos_and_pos(a: np.ndarray, b: np.ndarray, board) -> np.ndarray:
    start = Coordinate(a[0], a[1])
    goal = Coordinate(b[0], b[1])

    graph = TilesToClearGraph(board=board, goal=goal)
    search = Search(graph=graph)
    try:
        optimal_actions = search.get_actions_to_complete_goal(start=start)
    except SolutionNotFoundWithinBudgetError:
        return init_empty_positions()

    positions = []
    c = start

    for action in optimal_actions[:-1]:
        c = c.add_action(action)
        positions.append(c.xy)

    if not positions:
        return np.empty((0, 2), dtype=int)

    positions = np.array(positions)

    return positions
