from dataclasses import dataclass


@dataclass
class Coordinate:
    x: int
    y: int

    def __add__(self, other: "Coordinate") -> "Coordinate":
        new_x = self.x + other.x
        new_y = self.y + other.y
        return Coordinate(new_x, new_y)

    def __eq__(self, other: "Coordinate") -> "Coordinate":
        return self.x == other.x and self.y == other.y

    def distance_to(self, c: "Coordinate") -> int:
        """Manhatten distance to point

        Args:
            coordinate: Other coordinate to get the distance to

        Returns:
            Distance
        """
        dis_x = abs(self.x - c.x)
        dis_y = abs(self.y - c.y)
        return dis_x + dis_y

    def direction_to(self, target: "Coordinate") -> int:
        # direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        dx = target.x - self.x
        dy = target.y - self.y

        if dx == 0 and dy == 0:
            return 0
        if abs(dx) > abs(dy):
            if dx > 0:
                return 2
            else:
                return 4
        else:
            if dy > 0:
                return 3
            else:
                return 1


@dataclass
class CoordinateList:
    coordinates: list[Coordinate]

    def dis_to_tiles(self, c: Coordinate) -> list[int]:
        return [c.distance_to(factory_c) for factory_c in self.coordinates]

    def min_dis_to(self, c: Coordinate) -> int:
        return min(self.dis_to_tiles(c))

    def get_all_closest_tiles(self, c: Coordinate) -> "CoordinateList":
        min_dis = self.min_dis_to(c)
        return CoordinateList([c_l for c_l in self.coordinates if min_dis == c_l.distance_to(c)])

    def get_closest_tile(self, c: Coordinate) -> Coordinate:
        return self.get_all_closest_tiles(c)[0]

    def __iter__(self):
        return iter(self.coordinates)

    def __getitem__(self, key):
        return self.coordinates[key]
