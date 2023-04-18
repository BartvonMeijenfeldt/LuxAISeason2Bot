from enum import Enum
from random import choice
from typing import Iterable


class Direction(Enum):
    CENTER = (0, 0)
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    @property
    def number(self) -> int:
        return DIRECTION_NUMBER[self]


NON_STATIONARY_DIRECTIONS = [direction for direction in Direction if direction != direction.CENTER]


DIRECTION_NUMBER: dict = {
    Direction.CENTER: 0,
    Direction.UP: 1,
    Direction.RIGHT: 2,
    Direction.DOWN: 3,
    Direction.LEFT: 4,
}

NUMBER_DIRECTION = {number: direction for direction, number in DIRECTION_NUMBER.items()}


def get_reversed_direction(direction: Direction) -> Direction:
    if direction == Direction.CENTER:
        return direction.CENTER
    elif direction == Direction.RIGHT:
        return direction.LEFT
    elif direction == Direction.DOWN:
        return direction.UP
    elif direction == Direction.LEFT:
        return direction.RIGHT
    elif direction == Direction.UP:
        return direction.DOWN
    else:
        raise ValueError(f"Unknown direction {direction}")


def get_random_direction(excluded_directions: Iterable[Direction]) -> Direction:
    return choice([d for d in Direction if d not in excluded_directions])
