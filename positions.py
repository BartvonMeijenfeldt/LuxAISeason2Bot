import numpy as np

from typing import Set

from lux.config import EnvConfig


NEIGHBORING_DIRECTIONS_POSITIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def init_empty_positions() -> np.ndarray:
    return np.empty((0, 2), dtype="int")


def append_positions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.append(a, b, axis=0)


def positions_to_set(a: np.ndarray) -> Set[tuple]:
    return {tuple(pos) for pos in a}


def get_neighboring_positions(positions: np.ndarray) -> np.ndarray:
    neighbor_positions = positions[..., None] + NEIGHBORING_DIRECTIONS_POSITIONS.transpose()[None, ...]
    neighbor_positions = np.swapaxes(neighbor_positions, 1, 2).reshape(-1, 2)
    is_valid_mask = np.logical_and(neighbor_positions >= 0, neighbor_positions < EnvConfig.map_size).all(axis=1)
    neighbor_positions = neighbor_positions[is_valid_mask]
    unique_neighbors = np.unique(neighbor_positions, axis=0)
    return unique_neighbors
