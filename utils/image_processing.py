import numpy as np

from scipy import ndimage
from typing import List


def get_islands(array: np.ndarray) -> List[np.ndarray]:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    islands_array, nr_islands = ndimage.label(array, structure)  # type: ignore
    islands = [np.argwhere(islands_array == i) for i in range(1, nr_islands + 1)]
    return islands


def get_islands_array(array: np.ndarray) -> np.ndarray:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    islands_array, _ = ndimage.label(array, structure)  # type: ignore
    return islands_array


def get_islands_from_islands_array(islands_array: np.ndarray) -> List[np.ndarray]:
    nr_islands = islands_array.max()
    islands = [np.argwhere(islands_array == i) for i in range(1, nr_islands + 1)]
    return islands
