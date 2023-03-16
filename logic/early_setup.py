import numpy as np

from scipy.signal import convolve2d


from objects.coordinate import Coordinate, CoordinateList


def get_factory_spawn_loc(obs: dict) -> tuple:
    rubble_score = get_rubble_score(obs["board"]["rubble"], obs["factories"]["player_0"], obs["factories"]["player_1"])
    ice_score = get_ice_score(obs["board"]["ice"])
    ore_score = get_ore_score(obs["board"]["ore"])
    scores = get_scores(rubble_score, ice_score, ore_score, valid_spawns=obs["board"]["valid_spawns_mask"])
    spawn_loc = get_coordinate_biggest(scores)
    return spawn_loc


def get_rubble_score(rubble: np.ndarray, factories_player_0: dict, factories_player_1: dict) -> np.ndarray:
    rubble = rubble.copy()
    for factories_player in [factories_player_0, factories_player_1]:
        for factory in factories_player.values():
            x, y = factory["pos"]
            #rubble[x - 1: x + 2, y - 1: y + 2] = 100

    neighbouring_inverted_rubble = sum_neighbouring_inverted_rubble(rubble)
    rubble_score = neighbouring_inverted_rubble / 100000
    return rubble_score


def sum_neighbouring_inverted_rubble(rubble: np.ndarray) -> np.ndarray:
    inverted_rubble = 100 - rubble
    neighbouring_inverted_rubble = sum_closest_numbers(inverted_rubble, r=5)
    return neighbouring_inverted_rubble


def get_ice_score(ice: np.ndarray) -> np.ndarray:
    ice_surrounding = sum_closest_numbers(ice, r=1)
    ice_surrounding = np.clip(ice_surrounding, a_min=0, a_max=1)
    ice_score = ice_surrounding * 10000
    return ice_score


def get_ore_score(ore: np.ndarray) -> np.ndarray:
    ore_surrounding = sum_closest_numbers(ore, r=1)
    ore_surrounding = np.clip(ore_surrounding, a_min=0, a_max=1)
    ore_score = ore_surrounding * 1000
    return ore_score


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


def get_scores(rubble_score: np.ndarray, ice_score: np.ndarray, ore_score: np.ndarray, valid_spawns: np.ndarray
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
