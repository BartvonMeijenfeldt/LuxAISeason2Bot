from enum import Enum


class Direction(Enum):
    CENTER = (0, 0)
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

    @property
    def number(self) -> int:
        return DIRECTION_NUMBER[self]


DIRECTION_NUMBER: dict = {
    Direction.CENTER: 0,
    Direction.UP: 1,
    Direction.RIGHT: 2,
    Direction.DOWN: 3,
    Direction.LEFT: 4,
}


NUMBER_DIRECTION = {number: direction for direction, number in DIRECTION_NUMBER.items()}