from dataclasses import dataclass


@dataclass
class Coordinate:
    x: int
    y: int

    def distance(self, coordinate: "Coordinate") -> int:
        """Manhatten distance to point

        Args:
            coordinate: Other coordinate to get the distance to

        Returns:
            Distance
        """
        dis_x = abs(self.x - coordinate.x)
        dis_y = abs(self.y - coordinate.y)
        return dis_x + dis_y