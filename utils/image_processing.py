import numpy as np

from scipy import ndimage
from typing import List


def get_islands(array: np.ndarray) -> List[np.ndarray]:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    islands_array, nr_islands = ndimage.label(array, structure)  # type: ignore
    islands = [np.argwhere(islands_array == i) for i in range(1, nr_islands + 1)]
    return islands
